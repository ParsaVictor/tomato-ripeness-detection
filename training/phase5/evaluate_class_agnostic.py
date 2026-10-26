"""Phase 5 main entry point: class-agnostic "tomato vs background" evaluation.

All 4 classes are merged into one class "tomato" and mAP50 / mAP50-95 / precision / recall are reported on
three eval sets (defects_test, sxsc7_holdout, conveyor_holdout), by object size (original and network-input
pixels), plus the usual 4-class metrics on defects_test, a recall gate on the conveyor holdout and a
failure analysis with example images.

    py -3.10 -m phase5.evaluate_class_agnostic --config phase5_config.yaml [--crosscheck] [--failures-all]
        [--op-conf 0.25] [--no-cache] [--strict]
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import logging
import math
import os
import re
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import yaml

from phase5 import common
from phase5.common import EVAL_SETS, ImageRecord

LOG = logging.getLogger("phase5.evaluate_class_agnostic")

IOU_THRS = np.linspace(0.5, 0.95, 10)
SET_LABELS = {
    "defects_test": "(a) Tomato Defects - test split",
    "sxsc7_holdout": "(b) tomato-sxsc7 - held-out subset",
    "conveyor_holdout": "(c) tomato_count_project (conveyor) - held-out images",
}
FN_TAGS = ("small_object", "motion_blur", "occlusion", "belt_background", "other")
FN_TAG_TITLES = {
    "small_object": "Small objects (< 32x32 px)",
    "motion_blur": "Motion blur",
    "occlusion": "Occlusion / truncation",
    "belt_background": "Belt background (low contrast with belt)",
    "other": "Other (no tag applied)",
}
FP_TAGS = ("belt_background_fp", "localization_fp", "duplicate_fp")


class ManifestMissingError(RuntimeError):
    """Raised when the frozen holdout manifest does not exist."""


# ----------------------------------------------------------------------------- eval sets


def _manifest_path(cfg: dict) -> Path:
    return Path(cfg["splits_dir"]) / "manifest.json"


def _require_manifest(cfg: dict) -> Path:
    mp = _manifest_path(cfg)
    if not mp.is_file():
        raise ManifestMissingError(
            f"Holdout manifest not found: {mp}\n"
            "Run `py -3.10 -m phase5.make_holdouts --config phase5_config.yaml` BEFORE training, then train "
            "only on splits/conveyor_train.txt and splits/sxsc7_train.txt. The evaluator never creates the "
            "holdout itself, because a model trained on the raw folders may already have seen those images."
        )
    return mp


def build_eval_sets(cfg: dict) -> dict[str, dict]:
    """{set_key: {"records": [ImageRecord], "keep_ids": set|None, "names": [str], "dataset": key}}."""
    _require_manifest(cfg)
    from phase5 import make_holdouts  # lazy: sibling module

    out: dict[str, dict] = {}
    dcfg = cfg["datasets"]["defects"]
    ds = common.load_dataset(dcfg["root"], "defects")
    split = dcfg.get("eval_split", "test")
    if split not in ds["splits"]:
        raise FileNotFoundError(
            f"defects dataset has no '{split}' split (available: {sorted(ds['splits'])}) at {ds['root']}"
        )
    out["defects_test"] = {
        "records": list(ds["splits"][split]),
        "keep_ids": common.tomato_class_ids(ds["names"], dcfg.get("tomato_classes", "all")),
        "names": ds["names"],
        "dataset": "defects",
    }
    for set_key, key in (("sxsc7_holdout", "sxsc7"), ("conveyor_holdout", "conveyor")):
        kc = cfg["datasets"][key]
        names = common.load_names(kc["root"])
        recs = list(make_holdouts.load_holdout_records(cfg, key))
        out[set_key] = {
            "records": recs,
            "keep_ids": common.tomato_class_ids(names, kc.get("tomato_classes", "all")),
            "names": names,
            "dataset": key,
        }
    return out


# ----------------------------------------------------------------------------- helpers


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(x, nd: int = 3) -> str:
    x = _f(x)
    return "n/a" if math.isnan(x) else f"{x:.{nd}f}"


def _norm_name(n: str) -> str:
    return re.sub(r"[\s_\-]+", " ", str(n).strip().lower())


def _rel_link(path: str | Path, base_dir: Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = common.PROJECT_ROOT / p
    try:
        return os.path.relpath(p.resolve(), base_dir.resolve()).replace("\\", "/")
    except ValueError:  # different drive
        return p.resolve().as_posix()


def _load_gt(records: list[ImageRecord]):
    """Decode every image (EXIF-correct size) and read its YOLO labels. Unreadable images are skipped."""
    kept, gts, shapes, skipped = [], {}, {}, []
    for r in records:
        img = common.imread(r.image)
        if img is None:
            warnings.warn(f"unreadable image skipped from evaluation: {r.image}")
            skipped.append(r.image)
            continue
        h, w = img.shape[:2]
        gts[r.image] = common.read_yolo_labels(r.label, w, h)
        shapes[r.image] = (h, w)
        kept.append(r)
    return kept, gts, shapes, skipped


def _default_predict_factory(cfg: dict, use_cache: bool):
    from phase5 import predict  # lazy: needs torch / ultralytics

    def factory(set_key: str):
        def fn(records, *, agnostic_nms: bool):
            clf = cfg.get("classifier")
            if not agnostic_nms and clf and Path(clf).is_file():  # two-stage: detector boxes + crop classifier
                dets = fn(records, agnostic_nms=True)
                cache = str(Path(cfg["cache_dir"]) / f"{set_key}_twostage.json") if use_cache else None
                return predict.classify_detections(clf, records, dets, imgsz=int(cfg.get("cls_imgsz", 160)),
                                                   device=cfg["device"], cache_path=cache)
            cache = None
            if use_cache:
                cache = str(Path(cfg["cache_dir"]) / f"{set_key}{'' if agnostic_nms else '_classnms'}.json")
            return predict.run_predictions(
                cfg["model"], records, imgsz=cfg["imgsz"], conf=cfg["pred_conf"], iou=cfg["nms_iou"],
                agnostic_nms=agnostic_nms, max_det=cfg["max_det"], batch=cfg["batch"], device=cfg["device"],
                half=cfg["half"], cache_path=cache,
            )
        return fn

    return factory


def _model_names(cfg: dict, predict_fn) -> tuple[dict[int, str] | None, str]:
    names = getattr(predict_fn, "names", None) if predict_fn is not None else None
    if names:
        return {int(k): str(v) for k, v in dict(names).items()}, "predict_fn.names"
    clf = cfg.get("classifier")
    if clf and Path(clf).is_file():
        from ultralytics import YOLO

        return {int(k): str(v) for k, v in YOLO(clf).names.items()}, "classifier (two-stage)"
    if Path(cfg["model"]).is_file():
        try:
            from phase5 import predict

            return {int(k): str(v) for k, v in predict.model_info(cfg["model"])["names"].items()}, "model"
        except Exception as e:  # noqa: BLE001
            LOG.warning("could not read model class names: %s", e)
    return None, "none"


def _summ(m: dict) -> dict:
    keys = ("n_images", "n_gt", "n_pred", "ap50", "ap", "op_conf", "tp", "fp", "fn", "precision", "recall", "f1",
            "best_f1", "best_f1_conf", "precision_at_best_f1", "recall_at_best_f1", "max_recall")
    d = {k: m.get(k) for k in keys}
    d["ap50_95"] = m.get("ap")
    return d


def _small_sentence(sizes: dict) -> str:
    s = sizes["small"]
    others = [(b, sizes[b]) for b in ("medium", "large") if (sizes[b].get("n_gt") or 0) > 0]
    if not (s.get("n_gt") or 0):
        return "No small (< 32x32 px) tomatoes in this set in this pixel space, so small-object recall cannot be measured here."
    rs = _f(s["recall"])
    if not others:
        return f"Only small tomatoes are present; small recall = {_fmt(rs)}."
    ro = {b: _f(m["recall"]) for b, m in others}
    best = max(ro.values())
    cmp = ", ".join(f"{b} {_fmt(v)}" for b, v in ro.items())
    if rs < best - 0.10:
        return (f"**Yes - small tomatoes are missed more often**: small recall {_fmt(rs)} vs {cmp} "
                f"({s['n_gt']} small GT, {s.get('fn')} missed at op_conf).")
    return f"Small tomatoes are not disproportionately missed: small recall {_fmt(rs)} vs {cmp}."


# ----------------------------------------------------------------------------- crosscheck


def _crosscheck(cfg: dict, records: list[ImageRecord], gts_raw: dict) -> dict:
    from ultralytics import YOLO  # lazy

    max_id = max((int(g[:, 0].max()) for g in gts_raw.values() if len(g)), default=0)
    tmp = Path(tempfile.mkdtemp(prefix="phase5_crosscheck_"))
    lst = tmp / "images.txt"
    common.write_list(lst, [r.image for r in records])
    data = {"path": str(tmp), "train": str(lst), "val": str(lst), "names": {i: f"class{i}" for i in range(max_id + 1)}}
    yml = tmp / "data.yaml"
    with open(yml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True)
    res = YOLO(cfg["model"]).val(
        data=str(yml), single_cls=True, agnostic_nms=True, plots=False, save_json=False, verbose=False,
        conf=cfg["pred_conf"], iou=cfg["nms_iou"], imgsz=cfg["imgsz"], batch=cfg["batch"], split="val",
        device=cfg["device"] or None, half=cfg["half"], project=str(tmp), name="val", exist_ok=True,
    )
    return {"map50": float(res.box.map50), "map50_95": float(res.box.map), "precision": float(res.box.mp),
            "recall": float(res.box.mr), "tmp_dir": str(tmp)}


# ----------------------------------------------------------------------------- main API


def run(cfg: dict, predict_fn=None, crosscheck: bool = False, failures_all: bool = False) -> dict:
    """Evaluate all eval sets and write JSON / CSV / markdown report into cfg['out_dir']."""
    from phase5 import metrics  # lazy sibling import

    manifest_path = _require_manifest(cfg)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    op_conf = float(cfg["op_conf"])
    max_det = int(cfg["max_det"])
    imgsz = int(cfg["imgsz"])
    agn = bool(cfg.get("agnostic_nms", True))
    gate_thr = float(cfg["recall_gate"])
    fcfg = dict(cfg.get("failure", {}))

    if predict_fn is None:
        factory = _default_predict_factory(cfg, not cfg.get("_no_cache", False))
    else:
        def factory(_set_key):
            return predict_fn

    sets = build_eval_sets(cfg)
    results: dict = {
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "model": common.to_project_rel(cfg["model"]),
        "params": {k: cfg[k] for k in ("imgsz", "pred_conf", "op_conf", "nms_iou", "max_det", "iou_match",
                                        "recall_gate")},
        "agnostic_nms": agn,
        "manifest": {"path": common.to_project_rel(manifest_path), "sha1": common.file_sha1(manifest_path)},
        "sets": {},
        "warnings": [],
    }

    for set_key in EVAL_SETS:
        info = sets[set_key]
        recs, gts_raw, shapes, skipped = _load_gt(info["records"])
        if skipped:
            results["warnings"].append(f"{set_key}: {len(skipped)} unreadable images skipped")
        pfn = factory(set_key)
        preds_raw = pfn(recs, agnostic_nms=agn)
        gl = [common.to_agnostic_gt(gts_raw[r.image], info["keep_ids"]) for r in recs]
        pl = [common.to_agnostic_pred(np.asarray(preds_raw.get(r.image, common.EMPTY_PRED), dtype=np.float64)
                                      .reshape(-1, 6)) for r in recs]
        scales = [common.letterbox_ratio(*shapes[r.image], imgsz) for r in recs]
        m = metrics.evaluate(gl, pl, iou_thrs=IOU_THRS, max_det=max_det, op_conf=op_conf)
        sz_orig = metrics.evaluate_sizes(gl, pl, iou_thrs=IOU_THRS, max_det=max_det, op_conf=op_conf)
        sz_inp = metrics.evaluate_sizes(gl, pl, area_scales=scales, iou_thrs=IOU_THRS, max_det=max_det,
                                        op_conf=op_conf)
        n_tom = int(sum(len(g) for g in gl))
        sres = {
            "label": SET_LABELS[set_key],
            "dataset": info["dataset"],
            "n_images": len(recs),
            "n_gt_tomatoes": n_tom,
            "n_skipped_unreadable": len(skipped),
            "metrics": m,
            "summary": _summ(m),
            "sizes_original_px": {b: _summ(v) for b, v in sz_orig.items()},
            "sizes_input_px": {b: _summ(v) for b, v in sz_inp.items()},
            "small_sentence_original": _small_sentence(sz_orig),
            "small_sentence_input": _small_sentence(sz_inp),
            "images": [common.to_project_rel(r.image) for r in recs],
        }

        if set_key == "defects_test":
            sres["four_class"] = _four_class(cfg, metrics, recs, gts_raw, info["names"], pfn, predict_fn,
                                             max_det, op_conf, results["warnings"])

        if crosscheck:
            try:
                cc = _crosscheck(cfg, recs, gts_raw)
                cc["delta_map50"] = cc["map50"] - _f(m["ap50"])
                cc["flag"] = abs(cc["delta_map50"]) > 0.02
                sres["crosscheck"] = cc
            except Exception as e:  # noqa: BLE001
                sres["crosscheck"] = {"error": f"{type(e).__name__}: {e}"}
                results["warnings"].append(f"{set_key}: crosscheck failed: {e}")

        if set_key == "conveyor_holdout" or failures_all:
            from phase5 import failure_analysis

            sres["failures"] = failure_analysis.analyze_failures(
                recs, {r.image: g for r, g in zip(recs, gl)}, {r.image: p for r, p in zip(recs, pl)},
                out_dir=str(out_dir / "failures" / set_key), op_conf=op_conf, iou_thr=float(cfg["iou_match"]),
                max_examples=int(fcfg.get("max_examples", 8)), blur_rel=float(fcfg.get("blur_rel", 0.5)),
                contrast_de=float(fcfg.get("contrast_de", 12.0)), occl_ioa=float(fcfg.get("occl_ioa", 0.2)),
                border_px=int(fcfg.get("border_px", 2)), bg_iou=float(fcfg.get("bg_iou", 0.1)), set_name=set_key,
            )
        results["sets"][set_key] = sres

    conv_r = _f(results["sets"]["conveyor_holdout"]["metrics"]["recall"])
    passed = (not math.isnan(conv_r)) and conv_r >= gate_thr
    results["gate"] = {"set": "conveyor_holdout", "recall": conv_r, "threshold": gate_thr,
                       "passed": bool(passed), "verdict": "PASS" if passed else "FAIL"}
    if math.isnan(conv_r):
        results["gate"]["verdict_line"] = (f"**FAIL - conveyor class-agnostic recall is undefined (no GT tomatoes "
                                           f"in the conveyor holdout) - target {gate_thr:.2f}**")
    elif passed:
        results["gate"]["verdict_line"] = (f"**PASS - conveyor class-agnostic recall {conv_r:.3f} >= "
                                           f"{gate_thr:.2f} (target)**")
    else:
        results["gate"]["verdict_line"] = (f"**FAIL - conveyor class-agnostic recall {conv_r:.3f} < "
                                           f"{gate_thr:.2f} (target)**")

    lk = out_dir / "leakage_report.json"
    if lk.is_file():
        try:
            with open(lk, encoding="utf-8") as f:
                rep = json.load(f)
            results["leakage"] = {"found": True, "ok": bool(rep.get("ok")), "n_train": rep.get("n_train"),
                                  "n_leaks": len(rep.get("leaks", [])), "n_warnings": len(rep.get("warnings", [])),
                                  "min_hamming_per_set": rep.get("min_hamming_per_set")}
        except Exception as e:  # noqa: BLE001
            results["leakage"] = {"found": True, "error": str(e)}
    else:
        results["leakage"] = {"found": False}

    rows = _table_rows(results, crosscheck)
    results["acceptance_table"] = rows
    _write_csv(out_dir / "class_agnostic_table.csv", rows)
    report_path = out_dir / "PHASE5_class_agnostic_report.md"
    results["report_path"] = str(report_path)
    results["table_text"] = _table_md(rows, crosscheck)
    with open(report_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(_report_md(results, cfg, report_path.parent, crosscheck))
    common.dump_json(out_dir / "class_agnostic_results.json", results)
    return results


def _four_class(cfg, metrics, recs, gts_raw, ds_names, pfn, predict_fn, max_det, op_conf, warn_list) -> dict:
    model_names, src = _model_names(cfg, predict_fn)
    notes = []
    if model_names is None:
        model_names = dict(enumerate(ds_names))
        src = "dataset (model names unavailable - identity mapping assumed)"
        notes.append("Model class names unavailable; assumed model class ids == dataset class ids.")
    inv = {}
    for mid, n in model_names.items():
        inv.setdefault(_norm_name(n), mid)
    ds2m = {}
    for did, n in enumerate(ds_names):
        mid = inv.get(_norm_name(n))
        if mid is None:
            msg = f"defects class '{n}' (id {did}) has no matching model class; skipped in 4-class metrics"
            notes.append(msg)
            warn_list.append(msg)
        else:
            ds2m[did] = mid
    preds_cls = pfn(recs, agnostic_nms=False)
    gl, pl = [], []
    for r in recs:
        g = gts_raw[r.image]
        if len(g):
            ids = g[:, 0].astype(int)
            keep = np.array([i in ds2m for i in ids], dtype=bool)
            g = g[keep].copy()
            g[:, 0] = [ds2m[int(i)] for i in g[:, 0].astype(int)] if len(g) else g[:, 0]
        gl.append(g)
        pl.append(np.asarray(preds_cls.get(r.image, common.EMPTY_PRED), dtype=np.float64).reshape(-1, 6))
    per = {}
    for did, mid in ds2m.items():
        mm = metrics.evaluate(gl, pl, iou_thrs=IOU_THRS, max_det=max_det, op_conf=op_conf, class_id=mid)
        per[ds_names[did]] = {"model_id": mid, "model_name": model_names[mid], **_summ(mm)}

    def mean(k):
        v = [_f(p[k]) for p in per.values() if not math.isnan(_f(p[k]))]
        return float(np.mean(v)) if v else float("nan")

    return {"nms": "per-class (agnostic_nms=False)", "names_source": src, "per_class": per,
            "mean": {k: mean(k) for k in ("ap50", "ap50_95", "precision", "recall")}, "notes": notes}


# ----------------------------------------------------------------------------- tables / report

TABLE_COLS = ["eval_set", "images", "gt_tomatoes", "mAP50", "mAP50_95", "P@op_conf", "R@op_conf", "F1",
              "P@bestF1", "R@bestF1", "bestF1_conf", "max_recall", "crosscheck_mAP50"]


def _table_rows(results: dict, crosscheck: bool) -> list[dict]:
    rows = []
    for k in EVAL_SETS:
        s = results["sets"][k]
        m = s["summary"]
        cc = s.get("crosscheck", {})
        rows.append({
            "eval_set": k, "images": s["n_images"], "gt_tomatoes": s["n_gt_tomatoes"],
            "mAP50": _f(m["ap50"]), "mAP50_95": _f(m["ap50_95"]), "P@op_conf": _f(m["precision"]),
            "R@op_conf": _f(m["recall"]), "F1": _f(m["f1"]), "P@bestF1": _f(m["precision_at_best_f1"]),
            "R@bestF1": _f(m["recall_at_best_f1"]), "bestF1_conf": _f(m["best_f1_conf"]),
            "max_recall": _f(m["max_recall"]),
            "crosscheck_mAP50": _f(cc.get("map50")) if crosscheck else float("nan"),
        })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TABLE_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(v, float) and math.isnan(v) else
                            (round(v, 6) if isinstance(v, float) else v)) for k, v in r.items()})


def _table_md(rows: list[dict], crosscheck: bool) -> str:
    hdr = ["Eval set", "Images", "GT tomatoes", "mAP50", "mAP50-95", "P@op_conf", "R@op_conf", "F1",
           "P / R @ best-F1 (conf)", "Max recall"] + (["Crosscheck mAP50"] if crosscheck else [])
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        cells = [f"`{r['eval_set']}`", str(r["images"]), str(r["gt_tomatoes"]), _fmt(r["mAP50"]),
                 _fmt(r["mAP50_95"]), _fmt(r["P@op_conf"]), _fmt(r["R@op_conf"]), _fmt(r["F1"]),
                 f"{_fmt(r['P@bestF1'])} / {_fmt(r['R@bestF1'])} ({_fmt(r['bestF1_conf'], 2)})",
                 _fmt(r["max_recall"])]
        if crosscheck:
            cells.append(_fmt(r["crosscheck_mAP50"]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _size_table(sz: dict) -> str:
    lines = ["| Size | n_gt | AP50 | AP50-95 | Recall@op_conf | Max recall |", "|---|---|---|---|---|---|"]
    for b in ("all", "small", "medium", "large"):
        m = sz[b]
        lines.append(f"| {b} | {m['n_gt']} | {_fmt(m['ap50'])} | {_fmt(m['ap50_95'])} | {_fmt(m['recall'])} | "
                     f"{_fmt(m['max_recall'])} |")
    return "\n".join(lines)


def _img_md(path: str, base: Path, alt: str) -> str:
    return f"![{alt}](<{_rel_link(path, base)}>)"


def _failure_md(fa: dict, base: Path, gate_failed: bool, op_conf: float) -> list[str]:
    L = [f"FN = ground-truth tomatoes not detected at conf >= {op_conf} (IoU 0.5); FP = detections at conf >= "
         f"{op_conf} not matching any tomato. One FN can carry several tags, so shares can sum to > 100 %.", "",
         f"- Missed tomatoes (FN): **{fa.get('n_fn', 0)}**; false positives (FP): **{fa.get('n_fp', 0)}**", "",
         "| Failure type (FN) | Count | Share of FN |", "|---|---|---|"]
    tags, share = fa.get("fn_tags", {}), fa.get("fn_tag_share", {})
    for t in FN_TAGS:
        L.append(f"| {FN_TAG_TITLES[t]} | {tags.get(t, 0)} | {_fmt(_f(share.get(t, 0)) * 100, 1)} % |")
    L += ["", "| FP type | Count |", "|---|---|"]
    for t in FP_TAGS:
        L.append(f"| {t} | {fa.get('fp_tags', {}).get(t, 0)} |")
    L.append("")
    if gate_failed:
        L.append("Failure types observed, with examples (red = missed tomato with tag, orange = false positive, "
                 "green = detected):")
        L.append("")
    ex, sheets = fa.get("examples", {}), fa.get("contact_sheets", {})
    for t in list(FN_TAGS) + list(FP_TAGS):
        imgs = ex.get(t, [])
        if not imgs and t not in sheets:
            continue
        title = FN_TAG_TITLES.get(t, t)
        L.append(f"#### {title} - {tags.get(t, fa.get('fp_tags', {}).get(t, 0))} cases")
        L.append("")
        if t in sheets and sheets[t]:
            L.append(f"Crops: {_img_md(sheets[t], base, t + ' crops')}")
            L.append("")
        for p in imgs:
            L.append(_img_md(p, base, t))
        L.append("")
    return L


def _report_md(res: dict, cfg: dict, base: Path, crosscheck: bool) -> str:
    g = res["gate"]
    op = cfg["op_conf"]
    L = [g["verdict_line"], "", "# Phase 5 - class-agnostic tomato-vs-background evaluation", "",
         f"Model: `{res['model']}` - generated {res['created']}", "", "## Acceptance table (class-agnostic, all "
         "classes merged into one class 'tomato')", "", res["table_text"], "",
         f"P / R / F1 at op_conf = {op}, IoU >= 0.5. mAP50 / mAP50-95 computed COCO-style (101-point) from "
         f"predictions down to conf {cfg['pred_conf']}.", ""]
    if crosscheck:
        L.append("Crosscheck (Ultralytics `val`, single_cls=True):")
        L.append("")
        for k in EVAL_SETS:
            cc = res["sets"][k].get("crosscheck", {})
            if "error" in cc:
                L.append(f"- `{k}`: crosscheck failed: {cc['error']}")
            else:
                flag = " **(|delta mAP50| > 0.02 - check!)**" if cc.get("flag") else ""
                L.append(f"- `{k}`: ultralytics mAP50 {_fmt(cc.get('map50'))} (P {_fmt(cc.get('precision'))}, "
                         f"R {_fmt(cc.get('recall'))} at Ultralytics best-F1 conf, not op_conf; compare with "
                         f"P@bestF1/R@bestF1); delta mAP50 vs ours {_fmt(cc.get('delta_map50'), 4)}{flag}")
        L.append("")
    if not g["passed"]:
        L += [f"**Recall is below {g['threshold']:.2f}** on the conveyor holdout (recall {_fmt(g['recall'])}). "
              "See the failure analysis below for the failure types (small objects, motion blur, occlusion, "
              "belt background) with example images.", ""]

    L += ["## Detection by object size", "",
          "Small < 32x32 px area <= medium < 96x96 <= large (COCO definition, inclusive bounds like pycocotools). "
          f"Two spaces: original image pixels, and network-input pixels (after letterbox resize to imgsz="
          f"{cfg['imgsz']}).", ""]
    for k in EVAL_SETS:
        s = res["sets"][k]
        L += [f"### {s['label']} (`{k}`)", "", "Original image pixels:", "", _size_table(s["sizes_original_px"]),
              "", s["small_sentence_original"], "", f"Network-input pixels (imgsz {cfg['imgsz']}):", "",
              _size_table(s["sizes_input_px"]), "", s["small_sentence_input"], ""]

    fc = res["sets"]["defects_test"].get("four_class")
    L += ["## 4-class metrics on defects_test", ""]
    if fc:
        L += [f"Predictions from a second pass with {fc['nms']}; class names from {fc['names_source']}.", "",
              "| Class | n_gt | AP50 | AP50-95 | P@op_conf | R@op_conf |", "|---|---|---|---|---|---|"]
        for n, p in fc["per_class"].items():
            L.append(f"| {n} | {p['n_gt']} | {_fmt(p['ap50'])} | {_fmt(p['ap50_95'])} | {_fmt(p['precision'])} | "
                     f"{_fmt(p['recall'])} |")
        mm = fc["mean"]
        L.append(f"| **mean** | | {_fmt(mm['ap50'])} | {_fmt(mm['ap50_95'])} | {_fmt(mm['precision'])} | "
                 f"{_fmt(mm['recall'])} |")
        for n in fc["notes"]:
            L.append(f"\n- Note: {n}")
        L.append("")

    L += ["## Failure analysis", ""]
    for k in EVAL_SETS:
        fa = res["sets"][k].get("failures")
        if not fa:
            continue
        L += [f"### {res['sets'][k]['label']} (`{k}`)", ""]
        if k == "conveyor_holdout" and not g["passed"]:
            L += [f"**Recall is below {g['threshold']:.2f}.** Failure types:", ""]
        L += _failure_md(fa, base, k == "conveyor_holdout" and not g["passed"], op)

    lk = res["leakage"]
    if not lk.get("found"):
        lk_txt = ("not run (no leakage_report.json in out_dir) - run `py -3.10 -m phase5.check_leakage --config "
                  "phase5_config.yaml --train <data.yaml>`")
    elif "error" in lk:
        lk_txt = f"leakage_report.json unreadable: {lk['error']}"
    else:
        lk_txt = (f"{'OK - no leaks' if lk['ok'] else 'LEAKS FOUND'}: {lk['n_leaks']} leaks, {lk['n_warnings']} "
                  f"warnings over {lk['n_train']} train/val images")
    s_in = cfg["imgsz"]
    L += ["## Method notes", "",
          "- Class-agnostic: every GT class listed in `tomato_classes` (default all 4) becomes 'tomato'; every "
          "prediction becomes 'tomato' regardless of predicted class.",
          f"- NMS for class-agnostic metrics: agnostic_nms={res['agnostic_nms']}, NMS IoU {cfg['nms_iou']}, "
          f"max_det {cfg['max_det']}, prediction conf >= {cfg['pred_conf']}.",
          f"- Matching: COCO greedy, IoU >= 0.5 for mAP50 / P / R; mAP50-95 averages IoU 0.50:0.05:0.95. "
          f"Precision / recall / F1 at op_conf = {op}; best-F1 scans all confidence thresholds; max recall uses "
          "all predictions.",
          f"- Size thresholds: original px small < 1024 px^2 (32x32), large >= 9216 (96x96); input px: same "
          f"thresholds after scaling each box area by letterbox_ratio^2 = (min({s_in}/h, {s_in}/w))^2.",
          f"- Recall gate: conveyor_holdout recall at op_conf >= {g['threshold']:.2f}.",
          f"- Holdout manifest: `{res['manifest']['path']}` sha1 `{res['manifest']['sha1']}`.",
          f"- Leakage check: {lk_txt}.",
          "- (c) is computed on the >= 30 held-out conveyor images, the only conveyor images never seen in "
          "training (the remaining ~70 conveyor images are used for training).",
          ]
    for w in res["warnings"]:
        L.append(f"- Warning: {w}")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- CLI


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None, help="config YAML (default: phase5_config.yaml if present)")
    ap.add_argument("--crosscheck", action="store_true", help="also run ultralytics val(single_cls=True)")
    ap.add_argument("--failures-all", action="store_true", help="failure analysis on all sets, not only conveyor")
    ap.add_argument("--op-conf", type=float, default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--strict", action="store_true", help="exit code 3 when the recall gate fails")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if a.config is not None:
        if not Path(a.config).is_file():
            print(f"ERROR: config file not found: {a.config}", file=sys.stderr)
            return 2
        cfg_path = a.config
    elif Path("phase5_config.yaml").is_file():
        cfg_path = "phase5_config.yaml"
    else:
        logging.warning("phase5_config.yaml not found in %s; using built-in defaults", Path.cwd())
        cfg_path = None
    try:
        cfg = common.load_config(cfg_path)
    except (OSError, ValueError) as e:
        print(f"ERROR: cannot load config {cfg_path}: {e}", file=sys.stderr)
        return 2
    if a.op_conf is not None:
        cfg["op_conf"] = a.op_conf
    cfg["_no_cache"] = a.no_cache
    try:
        res = run(cfg, crosscheck=a.crosscheck, failures_all=a.failures_all)
    except (ManifestMissingError, FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(res["table_text"])
    print()
    v = res["gate"]["verdict_line"].replace("**", "")
    print(v.upper() if not res["gate"]["passed"] else v)
    print(f"Report: {res['report_path']}")
    if not res["gate"]["passed"] and a.strict:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
