"""Tests for phase5.metrics: COCO equivalence vs pycocotools on random scenarios + unit tests.

Run: py -3.10 -m phase5.tests.test_metrics   (or pytest phase5/tests/test_metrics.py)
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

from phase5 import common, metrics  # noqa: E402

IOU10 = np.linspace(0.5, 0.95, 10)


# ----------------------------------------------------------------------------- pycocotools reference
def _coco_objs(gts, preds, area_scales, n_cats):
    imgs = [{"id": i + 1, "width": 2000, "height": 2000, "file_name": f"{i}.jpg"} for i in range(len(gts))]
    cats = [{"id": c + 1, "name": f"c{c}"} for c in range(n_cats)]
    ga, da = [], []
    for i, (g, p) in enumerate(zip(gts, preds)):
        s2 = area_scales[i] ** 2
        for r in g:
            w, h = r[3] - r[1], r[4] - r[2]
            ga.append({"id": len(ga) + 1, "image_id": i + 1, "category_id": int(r[0]) + 1,
                       "bbox": [r[1], r[2], w, h], "area": w * h * s2, "iscrowd": 0})
        for r in p:
            w, h = r[2] - r[0], r[3] - r[1]
            # built manually (loadRes would overwrite 'area' with w*h)
            da.append({"id": len(da) + 1, "image_id": i + 1, "category_id": int(r[5]) + 1,
                       "bbox": [r[0], r[1], w, h], "area": w * h * s2, "score": r[4], "iscrowd": 0})
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO()
        cg.dataset = {"images": imgs, "categories": cats, "annotations": ga}
        cg.createIndex()
        cd = COCO()
        cd.dataset = {"images": imgs, "categories": cats, "annotations": da}
        cd.createIndex()
    return cg, cd


def coco_ap_per_iou(gts, preds, *, iou_thrs, area_rng, area_scales, max_det, cat=None, n_cats=1):
    """Per-threshold AP from COCOeval.eval['precision'] (nan where COCO reports -1)."""
    cg, cd = _coco_objs(gts, preds, area_scales, n_cats)
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(cg, cd, "bbox")
        E.params.imgIds = list(range(1, len(gts) + 1))
        E.params.catIds = [cat + 1] if cat is not None else [1]
        E.params.iouThrs = np.asarray(iou_thrs, dtype=np.float64)
        E.params.maxDets = [max_det]
        E.params.areaRng = [list(area_rng)]
        E.params.areaRngLbl = ["x"]
        E.evaluate()
        E.accumulate()
    prec = E.eval["precision"][:, :, 0, 0, 0]
    out = []
    for t in range(prec.shape[0]):
        q = prec[t]
        out.append(float("nan") if (q < 0).all() else float(q[q > -1].mean()))
    return out


# ----------------------------------------------------------------------------- random scenarios
def _rand_scenario(rng: np.random.Generator, n_cats=1):
    n_img = int(rng.integers(0, 7))
    gts, preds = [], []
    base_scale = rng.choice([4.0, 20.0, 60.0, 150.0])
    for _ in range(n_img):
        kind = rng.random()
        ng = 0 if kind < 0.15 else int(rng.integers(1, 12))
        g = []
        cx0, cy0 = rng.uniform(100, 900, 2)
        clustered = rng.random() < 0.5
        for _ in range(ng):
            if clustered:
                cx, cy = cx0 + rng.normal(0, base_scale * 0.6, 2)
            else:
                cx, cy = rng.uniform(0, 1000, 2)
            w, h = rng.uniform(0.3, 2.0, 2) * base_scale * rng.uniform(0.3, 2.5)
            g.append([rng.integers(0, n_cats), cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        g = np.asarray(g, dtype=np.float64).reshape(-1, 5)
        p = []
        if not (0.15 <= kind < 0.25):  # sometimes no dets
            for r in g:  # jittered copies (0-3 each)
                for _ in range(int(rng.integers(0, 4))):
                    w, h = r[3] - r[1], r[4] - r[2]
                    j = rng.normal(0, rng.choice([0.03, 0.1, 0.25]), 4) * np.array([w, h, w, h])
                    b = r[1:5] + j
                    if b[2] <= b[0] or b[3] <= b[1]:
                        continue
                    cls = r[0] if rng.random() < 0.85 else rng.integers(0, n_cats)
                    p.append([*b, 0.0, cls])
            for _ in range(int(rng.integers(0, 8))):  # background dets
                cx, cy = rng.uniform(0, 1000, 2)
                w, h = rng.uniform(0.3, 2.0, 2) * base_scale
                p.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, 0.0, rng.integers(0, n_cats)])
        p = np.asarray(p, dtype=np.float64).reshape(-1, 6)
        if len(p):
            if rng.random() < 0.4:  # many ties
                p[:, 4] = rng.choice([0.1, 0.3, 0.5, 0.9], len(p))
            else:
                p[:, 4] = rng.random(len(p))
            p = p[rng.permutation(len(p))]
        gts.append(g)
        preds.append(p)
    return gts, preds


def _close(a, b, tol=1e-6):
    return (np.isnan(a) and np.isnan(b)) or abs(a - b) <= tol


def test_coco_equivalence(n=260, seed=123):
    rng = np.random.default_rng(seed)
    ranges = list(common.AREA_RANGES.values()) + [(500.0, 5000.0)]
    n_checked = 0
    for k in range(n):
        gts, preds = _rand_scenario(rng)
        rngA = ranges[k % len(ranges)]
        max_det = int(rng.choice([1, 3, 5, 100]))
        scales = [1.0] * len(gts) if rng.random() < 0.5 else list(rng.uniform(0.2, 3.0, len(gts)))
        thrs = IOU10 if rng.random() < 0.7 else np.array([0.5])
        ref = coco_ap_per_iou(gts, preds, iou_thrs=thrs, area_rng=rngA, area_scales=scales, max_det=max_det)
        res = metrics.evaluate(gts, preds, iou_thrs=thrs, area_rng=rngA, area_scales=scales, max_det=max_det)
        ours = [res["ap_per_iou"][float(t)] for t in thrs]
        for t, a, b in zip(thrs, ours, ref):
            assert _close(a, b), f"scenario {k} thr {t}: ours {a} vs coco {b} (rng {rngA}, maxdet {max_det})"
        if not np.isnan(ref[0]):
            assert _close(res["ap"], float(np.mean(ref)))
            n_checked += 1
    assert n_checked > n // 3, n_checked


def test_class_id_vs_coco(n=60, seed=7):
    rng = np.random.default_rng(seed)
    for k in range(n):
        gts, preds = _rand_scenario(rng, n_cats=3)
        sc = [1.0] * len(gts)
        for c in range(3):
            ref = coco_ap_per_iou(gts, preds, iou_thrs=IOU10, area_rng=(0, 1e10), area_scales=sc,
                                  max_det=100, cat=c, n_cats=3)
            res = metrics.evaluate(gts, preds, iou_thrs=IOU10, max_det=100, class_id=c)
            for t, b in zip(IOU10, ref):
                assert _close(res["ap_per_iou"][float(t)], b), (k, c, t)


def _brute_pr(gts, preds, thr, op_conf):
    tp = fp = ng = 0
    for g, p in zip(gts, preds):
        p = p[p[:, 4] >= op_conf]
        p = p[np.argsort(-p[:, 4], kind="mergesort")]
        ng += len(g)
        used = set()
        for d in p:
            best, bi = thr - 1e-12, -1
            for j, r in enumerate(g):
                if j in used:
                    continue
                iou = metrics.box_iou(d[None, :4], r[None, 1:5])[0, 0]
                if iou >= min(thr, 1 - 1e-10) and iou >= best:
                    best, bi = iou, j
            if bi >= 0:
                used.add(bi)
                tp += 1
            else:
                fp += 1
    return tp, fp, ng - tp


def test_pr_at_op_conf(n=80, seed=11):
    rng = np.random.default_rng(seed)
    for k in range(n):
        gts, preds = _rand_scenario(rng)
        op = float(rng.choice([0.0, 0.25, 0.5, 0.95]))
        res = metrics.evaluate(gts, preds, iou_thrs=IOU10, op_conf=op, max_det=10_000)
        tp, fp, fn = _brute_pr(gts, preds, 0.5, op)
        if sum(len(g) for g in gts) == 0:
            assert np.isnan(res["recall"])
            continue
        assert (res["tp"], res["fp"], res["fn"]) == (tp, fp, fn), (k, res["tp"], res["fp"], res["fn"], tp, fp, fn)
        assert _close(res["recall"], tp / (tp + fn))
        if tp + fp:
            assert _close(res["precision"], tp / (tp + fp))
        # best F1 >= F1 at any operating point; max recall = recall at conf 0
        assert res["best_f1"] >= res["f1"] - 1e-12
        tp0, _, fn0 = _brute_pr(gts, preds, 0.5, -1.0)
        assert _close(res["max_recall"], tp0 / (tp0 + fn0))


def test_match_details_and_edges():
    gt = np.array([[0, 0, 0, 10, 10], [1, 100, 100, 120, 120], [0, 200, 200, 210, 210]], float)
    pred = np.array([
        [0, 0, 10, 10, 0.9, 0],        # tp gt0
        [0.5, 0, 10.5, 10, 0.8, 0],    # duplicate of gt0
        [100, 100, 115, 120, 0.7, 0],  # IoU .75 with gt1 (class 1)
        [205, 205, 215, 215, 0.6, 0],  # IoU ~0.14 with gt2 -> localization
        [500, 500, 510, 510, 0.5, 0],  # background
        [0, 0, 10, 10, 0.1, 0],        # below op_conf
    ], float)
    r = metrics.match_details(gt, pred, op_conf=0.25)
    assert r["pred_status"] == ["tp", "duplicate", "tp", "localization", "background"], r["pred_status"]
    assert list(r["pred_index"]) == [0, 1, 2, 3, 4]
    assert list(r["gt_matched"]) == [True, True, False]
    assert r["gt_match_conf"][0] == 0.9 and np.isnan(r["gt_match_conf"][2])
    assert abs(r["gt_best_iou"][2] - 25 / 175) < 1e-12
    r2 = metrics.match_details(gt, pred, op_conf=0.25, class_agnostic=False)
    assert r2["pred_status"][2] == "background" and not r2["gt_matched"][1]
    e = metrics.match_details(common.EMPTY_GT, pred)
    assert e["pred_status"] == ["background"] * 5 and len(e["gt_matched"]) == 0
    e = metrics.match_details(gt, common.EMPTY_PRED)
    assert e["pred_status"] == [] and not e["gt_matched"].any()
    # edges of evaluate
    res = metrics.evaluate([common.EMPTY_GT], [common.EMPTY_PRED])
    assert np.isnan(res["ap50"]) and np.isnan(res["recall"])
    res = metrics.evaluate([], [])
    assert np.isnan(res["ap"])
    res = metrics.evaluate([gt], [pred], iou_thrs=IOU10)
    assert 0 < res["ap50"] <= 1 and res["n_gt"] == 3
    perfect = np.c_[gt[:, 1:5], np.ones(3), gt[:, 0]]
    res = metrics.evaluate([gt], [perfect], iou_thrs=IOU10)
    assert abs(res["ap"] - 1) < 1e-9 and res["precision"] == 1 and res["recall"] == 1
    s = metrics.evaluate_sizes([gt], [perfect])
    assert set(s) == {"all", "small", "medium", "large"} and s["small"]["n_gt"] == 3
    assert np.isnan(s["large"]["ap50"])
    s = metrics.evaluate_sizes([gt], [perfect], area_scales=[5.0])  # x25: 2500, 10000, 2500 px^2
    assert s["small"]["n_gt"] == 0 and s["medium"]["n_gt"] == 2 and s["large"]["n_gt"] == 1
    json.dumps(metrics.evaluate([gt], [pred]))  # serializable


def test_unicode_tempdir_roundtrip():
    """Synthetic YOLO labels in a Persian-named temp dir -> read_yolo_labels -> evaluate."""
    d = tempfile.mkdtemp(prefix="تست_")
    try:
        lbl = os.path.join(d, "گوجه.txt")
        with open(lbl, "w", encoding="utf-8") as f:
            f.write("0 0.5 0.5 0.2 0.2\n2 0.1 0.1 0.05 0.05\n")
        g = common.to_agnostic_gt(common.read_yolo_labels(lbl, 640, 480), None)
        p = common.to_agnostic_pred(np.c_[g[:, 1:5], [0.9, 0.3], [3, 1]])
        res = metrics.evaluate([g], [p])
        assert res["ap50"] == 1.0 and res["recall"] == 1.0
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _dense_match_ref(ious, gt_ig, thr):
    """Original per-det dense matcher (reference for the sparse fast path)."""
    nd, ng = ious.shape
    t = min(thr, 1 - 1e-10)
    dtm = np.full(nd, -1, dtype=np.int64)
    if ng == 0:
        return dtm
    taken = np.zeros(ng, dtype=bool)
    nonig = ~gt_ig.astype(bool)
    for d in range(nd):
        row = ious[d]
        m = -1
        for grp in (nonig, ~nonig):
            cand = grp & ~taken & (row >= t)
            if cand.any():
                v = np.where(cand, row, -np.inf)
                m = int(len(v) - 1 - np.argmax(v[::-1]))
                break
        if m >= 0:
            taken[m] = True
            dtm[d] = m
    return dtm


def test_fast_match_equals_dense(n=400, seed=5):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        nd, ng = int(rng.integers(0, 15)), int(rng.integers(0, 10))
        ious = rng.choice([0.0, 0.3, 0.5, 0.6, 0.75, 0.9, 1.0], size=(nd, ng))  # many exact ties
        ious = np.where(rng.random((nd, ng)) < 0.3, rng.random((nd, ng)), ious)
        ig = (rng.random(ng) < 0.3).astype(np.int64)
        thrs = np.linspace(0.5, 0.95, 10)
        cands = metrics._candidates(ious, 0.5)
        for t in thrs:
            ref = _dense_match_ref(ious, ig, t)
            assert np.array_equal(metrics._coco_match(ious, ig, t), ref)
            assert np.array_equal(metrics._coco_match(ious, ig, t, cands), ref)


def test_speed_large(n_img=5000, n_gt=20, n_det=300, budget_s=30.0):
    import time
    rng = np.random.default_rng(0)
    gts, preds = [], []
    for _ in range(n_img):
        xy = rng.uniform(0, 600, (n_gt, 2)); wh = rng.uniform(8, 80, (n_gt, 2))
        g = np.c_[np.zeros(n_gt), xy, xy + wh]
        k = n_gt  # one jittered det per GT, rest are background clutter
        jit = g[:, 1:5] + rng.normal(0, 3, (k, 4))
        bxy = rng.uniform(0, 600, (n_det - k, 2)); bwh = rng.uniform(5, 60, (n_det - k, 2))
        boxes = np.r_[jit, np.c_[bxy, bxy + bwh]]
        preds.append(np.c_[boxes, rng.random(n_det), np.zeros(n_det)])
        gts.append(g)
    t0 = time.perf_counter()
    r = metrics.evaluate(gts, preds, iou_thrs=np.linspace(0.5, 0.95, 10))
    dt = time.perf_counter() - t0
    print(f"  evaluate 5000x300 x10 thr: {dt:.1f}s ap50={r['ap50']:.3f}")
    assert dt < budget_s, dt


def main() -> int:
    tests = [test_coco_equivalence, test_class_id_vs_coco, test_pr_at_op_conf, test_match_details_and_edges,
             test_unicode_tempdir_roundtrip, test_fast_match_equals_dense, test_speed_large]
    for t in tests:
        t()
        print("OK", t.__name__)
    print("OK test_metrics")
    return 0


if __name__ == "__main__":
    sys.exit(main())
