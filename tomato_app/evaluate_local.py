"""Measure counting + ripeness accuracy of the two-stage model on the labelled images it never trained on.

    py tomato_app/evaluate_local.py                      # all three sets, writes results/evaluation/
    py tomato_app/evaluate_local.py --iou 0.5 --out results/eval_iou05
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tomato_inspect as ti  # noqa: E402

GROUP = {"raw": "unripe", "half_ripe": "unripe", "ripe": "ripe", "rotten": "rotten"}


def evaluate(insp, key: str) -> dict:
    cfg = ti.SETS[key]
    names = cfg["classes"]
    tp = fp = fn = 0
    pred_n, gt_n, abs_err, exact, within1, within10 = [], [], [], 0, 0, 0
    conf = np.zeros((4, 4), int)          # rows = label, cols = prediction (ti.CLASSES order)
    order = list(ti.CLASSES)
    per_cls_pred = {k: 0 for k in order}
    per_cls_gt = {k: 0 for k in order}
    ms = []
    pred_rotten = 0
    for p in cfg["images"]():
        img = ti.imread(p)
        dets, t = insp(img)
        ms.append(t)
        pred_rotten += sum(d.cls == "rotten" for d in dets)
        gts = ti.read_gt(p, img.shape[1], img.shape[0], names)
        pairs, ud, ug = ti.match(dets, gts)
        tp, fp, fn = tp + len(pairs), fp + len(ud), fn + len(ug)
        n, g = len(dets), len(gts)
        pred_n.append(n)
        gt_n.append(g)
        abs_err.append(abs(n - g))
        exact += n == g
        within1 += abs(n - g) <= 1
        within10 += abs(n - g) <= max(1, round(0.1 * g))
        if names[0] != ti.GT_ONLY_TOMATO:
            for i, j in pairs:
                conf[order.index(gts[j]["cls"]), order.index(dets[i].cls)] += 1
            for d in dets:
                per_cls_pred[d.cls] += 1
            for gg in gts:
                per_cls_gt[gg["cls"]] += 1
    P = tp / (tp + fp) if tp + fp else 0
    R = tp / (tp + fn) if tp + fn else 0
    n_img = len(pred_n)
    res = {"set": key, "name_fa": cfg["fa"], "images": n_img, "gt_tomatoes": int(sum(gt_n)),
           "pred_tomatoes": int(sum(pred_n)), "tp": tp, "fp": fp, "fn": fn,
           "precision": P, "recall": R, "f1": 2 * P * R / (P + R) if P + R else 0,
           "total_count_accuracy": 1 - abs(sum(pred_n) - sum(gt_n)) / sum(gt_n),
           "mean_abs_count_error": float(np.mean(abs_err)),
           "mean_count_accuracy_per_image": float(np.mean([1 - min(1, e / g) if g else float(e == 0)
                                                           for e, g in zip(abs_err, gt_n)])),
           "images_exact_count": exact / n_img, "images_within_1": within1 / n_img,
           "images_within_10pct": within10 / n_img, "pred_rotten": pred_rotten, "ms_per_image_cpu": float(np.mean(ms))}
    if conf.sum():
        res["ripeness"] = {
            "classes": order, "confusion": conf.tolist(),
            "accuracy_4class": float(np.trace(conf) / conf.sum()),
            "per_class_recall": {k: float(conf[i, i] / conf[i].sum()) if conf[i].sum() else None
                                 for i, k in enumerate(order)},
            "accuracy_3group": float(sum(conf[i, j] for i, a in enumerate(order) for j, b in enumerate(order)
                                         if GROUP[a] == GROUP[b]) / conf.sum()),
            "accuracy_sound_vs_rotten": float(sum(conf[i, j] for i, a in enumerate(order)
                                                  for j, b in enumerate(order)
                                                  if (a == "rotten") == (b == "rotten")) / conf.sum()),
            "count_pred": per_cls_pred, "count_gt": per_cls_gt}
    return res


def plot_confusion(r: dict, dst: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = np.array(r["confusion"], float)
    norm = c / c.sum(1, keepdims=True).clip(min=1)
    labels = [ti.CLASSES[k][1] for k in r["classes"]]
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=150)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{norm[i, j] * 100:.0f}%\n({int(c[i, j])})", ha="center", va="center", fontsize=10,
                    color="white" if norm[i, j] > 0.55 else "#1e293b")
    ax.set_xticks(range(4), labels)
    ax.set_yticks(range(4), labels)
    ax.set_xlabel("Model prediction")
    ax.set_ylabel("True label")
    ax.set_title(f"Ripeness on matched tomatoes  (accuracy {r['accuracy_4class'] * 100:.1f}%)", fontsize=11)
    fig.tight_layout()
    fig.savefig(dst)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU")
    ap.add_argument("--conf", type=float, default=ti.DET_CONF)
    ap.add_argument("--dedup", type=float, default=ti.DEDUP_IOU, help="duplicate suppression IoU (0 = off)")
    ap.add_argument("--tiles", type=int, default=ti.TILES, help="NxN tiled inference (1 = off)")
    ap.add_argument("--tile-conf", type=float, default=ti.TILE_CONF)
    ap.add_argument("--tile-max-px", type=float, default=ti.TILE_MAX_PX)
    ap.add_argument("--rotten-check", type=float, default=ti.ROTTEN_CHECK_T, help="threshold (0 = check off)")
    ap.add_argument("--sets", nargs="+", default=list(ti.SETS))
    ap.add_argument("--out", default=str(ti.ROOT / "results" / "evaluation"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    insp = ti.TomatoInspector(conf=a.conf, iou=a.iou, dedup=a.dedup, tiles=a.tiles,
                              tile_conf=a.tile_conf, tile_max_px=a.tile_max_px, rotten_check=ti.ROTTEN_CHECK if a.rotten_check else None,
                              rotten_check_t=a.rotten_check)
    allr = {}
    for k in a.sets:
        r = evaluate(insp, k)
        allr[k] = r
        print(f"{k:9s} imgs {r['images']:3d} | GT {r['gt_tomatoes']:4d} pred {r['pred_tomatoes']:4d} | "
              f"P {r['precision']:.3f} R {r['recall']:.3f} F1 {r['f1']:.3f} | count acc {r['total_count_accuracy']:.3f}"
              f" per-img {r['mean_count_accuracy_per_image']:.3f} MAE {r['mean_abs_count_error']:.2f} | "
              f"exact {r['images_exact_count']:.2f} ±1 {r['images_within_1']:.2f} | rotten {r['pred_rotten']}")
        if "ripeness" in r:
            rr = r["ripeness"]
            print(f"          ripeness 4-class {rr['accuracy_4class']:.3f} | ripe/unripe/rotten "
                  f"{rr['accuracy_3group']:.3f} | sound-vs-rotten {rr['accuracy_sound_vs_rotten']:.3f} | "
                  f"recall {json.dumps({k: round(v, 3) for k, v in rr['per_class_recall'].items()})}")
            plot_confusion(rr, out / f"confusion_{k}.png")
    (out / "metrics.json").write_text(json.dumps({"conf": a.conf, "dedup_iou": a.dedup, "tiles": a.tiles, "rotten_check": a.rotten_check, "sets": allr}, indent=1,
                                                 ensure_ascii=False), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
