"""Why tomatoes are missed: tag every false negative / false positive of a class-agnostic eval set.

FN tags : small_object, motion_blur, occlusion, belt_background, other (no other tag applied).
FP tags : belt_background_fp (pred on background), localization_fp, duplicate_fp.
Matching is done per image with `metrics.match_details` (conf >= op_conf, greedy COCO matching).
Writes annotated example images per tag and a contact sheet of crops per tag (always via common.imwrite).
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import cv2
import numpy as np

from phase5 import common
from phase5.common import EMPTY_GT, EMPTY_PRED, SMALL_MAX_AREA, ImageRecord

LOG = logging.getLogger(__name__)

FN_TAGS = ("small_object", "motion_blur", "occlusion", "belt_background", "other")
FP_TAGS = ("belt_background_fp", "localization_fp", "duplicate_fp")
_FP_STATUS_TO_TAG = {"background": "belt_background_fp", "localization": "localization_fp",
                     "duplicate": "duplicate_fp"}
BLUR_ABS_FALLBACK = 60.0
MIN_MATCHED_FOR_REL = 5
MAX_SIDE = 1600
SHEET_MAX = 48
TILE = 128
SHEET_COLS = 8
_TAG_ABBR = {"small_object": "small", "motion_blur": "blur", "occlusion": "occl",
             "belt_background": "belt", "other": "other"}


def _match_details(*args, **kwargs):
    from phase5.metrics import match_details  # imported lazily: metrics.py is a sibling module

    return match_details(*args, **kwargs)


# ----------------------------------------------------------------------------- per-box features


def _int_box(box, w, h):
    x1 = int(np.floor(max(0.0, box[0])))
    y1 = int(np.floor(max(0.0, box[1])))
    x2 = int(np.ceil(min(float(w), box[2])))
    y2 = int(np.ceil(min(float(h), box[3])))
    return x1, y1, x2, y2


def blur_features(img: np.ndarray, box) -> tuple[float, float]:
    """(variance of Laplacian on the 64x64 gray crop, gradient anisotropy max/min of mean |Sobel x|,|Sobel y|)."""
    h, w = img.shape[:2]
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        return float("nan"), float("nan")
    x1, y1, x2, y2 = _int_box(box, w, h)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return float("nan"), float("nan")
    crop = img[y1:y2, x1:x2]
    g = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA if min(g.shape) >= 64 else cv2.INTER_LINEAR)
    g = g.astype(np.float64)
    lap = float(cv2.Laplacian(g, cv2.CV_64F).var())
    gx = float(np.abs(cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)).mean())
    gy = float(np.abs(cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)).mean())
    lo, hi = min(gx, gy), max(gx, gy)
    aniso = float(hi / lo) if lo > 1e-9 else (float("inf") if hi > 1e-9 else float("nan"))
    return lap, aniso


def contrast_de(img: np.ndarray, box) -> float:
    """CIELAB ΔE76 between the mean of the inner 60 % of the box and a ring of width 25 % of the box size.

    Lab is computed from float32 BGR/255 (L in 0..100), not from 8-bit Lab.
    """
    h, w = img.shape[:2]
    bx1, by1, bx2, by2 = (float(v) for v in box[:4])
    bw, bh = bx2 - bx1, by2 - by1
    if bw < 2 or bh < 2:
        return float("nan")
    rx, ry = 0.25 * bw, 0.25 * bh
    ox1, oy1, ox2, oy2 = _int_box((bx1 - rx, by1 - ry, bx2 + rx, by2 + ry), w, h)
    if ox2 <= ox1 or oy2 <= oy1:
        return float("nan")
    region = img[oy1:oy2, ox1:ox2]
    if region.ndim == 2:
        region = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(region.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB).astype(np.float64)
    yy, xx = np.mgrid[oy1:oy2, ox1:ox2]
    cx, cy = xx + 0.5, yy + 0.5  # pixel centers
    inside_box = (cx >= bx1) & (cx < bx2) & (cy >= by1) & (cy < by2)
    ix1, ix2 = bx1 + 0.2 * bw, bx2 - 0.2 * bw
    iy1, iy2 = by1 + 0.2 * bh, by2 - 0.2 * bh
    inner = (cx >= ix1) & (cx < ix2) & (cy >= iy1) & (cy < iy2)
    ring = ~inside_box
    if inner.sum() == 0:  # tiny box: fall back to the nearest pixel to the center
        inner = np.zeros_like(inner)
        inner[min(max(int((by1 + by2) / 2) - oy1, 0), inner.shape[0] - 1),
              min(max(int((bx1 + bx2) / 2) - ox1, 0), inner.shape[1] - 1)] = True
    if ring.sum() == 0:
        return float("nan")
    return float(np.linalg.norm(lab[inner].mean(0) - lab[ring].mean(0)))


def max_ioa(gt_boxes: np.ndarray, i: int) -> float:
    """Max over other GT of intersection(gt_i, gt_j) / area(gt_i)."""
    if len(gt_boxes) < 2:
        return 0.0
    b = gt_boxes[i]
    o = np.delete(gt_boxes, i, axis=0)
    iw = (np.minimum(b[2], o[:, 2]) - np.maximum(b[0], o[:, 0])).clip(0)
    ih = (np.minimum(b[3], o[:, 3]) - np.maximum(b[1], o[:, 1])).clip(0)
    area = max((b[2] - b[0]) * (b[3] - b[1]), 1e-12)
    return float((iw * ih).max() / area)


def _truncated(box, w, h, border_px) -> bool:
    return bool(box[0] <= border_px or box[1] <= border_px or box[2] >= w - border_px or box[3] >= h - border_px)


# ----------------------------------------------------------------------------- drawing


def _fmt_box(b):
    return [float(v) for v in b[:4]]


LEGEND_LINES = ("green=GT hit  red=missed GT (FN)  orange=FP  thin blue=TP",)


def _legend_text():
    abbr = " ".join(f"{v}={k}" for k, v in _TAG_ABBR.items())
    return (LEGEND_LINES[0], "FN tags: " + abbr + " | FP <kind> <conf>")


def _add_bars(out, header):
    """Prepend a header bar (tag + counts) and append an ASCII legend strip."""
    w = out.shape[1]
    lines = [header] if header else []
    top = np.full((18 * max(1, len(lines)), w, 3), 30, np.uint8) if lines else np.zeros((0, w, 3), np.uint8)
    for k, t in enumerate(lines):
        cv2.putText(top, t, (4, 13 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    leg = _legend_text()
    bot = np.full((16 * len(leg) + 4, w, 3), 255, np.uint8)
    for k, t in enumerate(leg):
        cv2.putText(bot, t, (4, 13 + 16 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([top, out, bot], axis=0)


def _annotate(img, gt_boxes, gt_matched, fn_tags_by_idx, preds_considered, statuses, header=""):
    h, w = img.shape[:2]
    s = min(1.0, MAX_SIDE / max(h, w))
    out = img if s == 1.0 else cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                                          interpolation=cv2.INTER_AREA)
    out = out.copy()
    lw = max(1, round(2 * max(out.shape[:2]) / 800))

    def P(v):
        return int(round(v * s))

    for p, st in zip(preds_considered, statuses):
        x1, y1, x2, y2 = map(P, p[:4])
        if st == "tp":
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 120, 0), 1)
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), lw)
            cv2.putText(out, f"FP {st[:3]} {p[4]:.2f}", (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 165, 255), 1, cv2.LINE_AA)
    for i, b in enumerate(gt_boxes):
        x1, y1, x2, y2 = map(P, b[:4])
        if gt_matched[i]:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), lw)
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), lw)
            txt = "FN " + ",".join(_TAG_ABBR[t] for t in fn_tags_by_idx.get(i, ["other"]))
            cv2.putText(out, txt, (x1, min(out.shape[0] - 3, y2 + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 255), 1, cv2.LINE_AA)
    return _add_bars(out, header)


def _crop_tile(img, box, color):
    h, w = img.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    x1, y1, x2, y2 = _int_box((box[0] - 0.5 * bw, box[1] - 0.5 * bh, box[2] + 0.5 * bw, box[3] + 0.5 * bh), w, h)
    tile = np.full((TILE, TILE, 3), 40, np.uint8)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return tile
    crop = img[y1:y2, x1:x2].copy()
    bx1, by1, bx2, by2 = (int(round(v)) for v in (box[0] - x1, box[1] - y1, box[2] - x1, box[3] - y1))
    cv2.rectangle(crop, (bx1, by1), (max(bx1, bx2 - 1), max(by1, by2 - 1)), color, 1)
    s = TILE / max(crop.shape[:2])
    nw, nh = max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST if s > 1 else cv2.INTER_AREA)
    oy, ox = (TILE - nh) // 2, (TILE - nw) // 2
    tile[oy:oy + nh, ox:ox + nw] = crop
    return tile


def _contact_sheet(tiles):
    n = len(tiles)
    cols = min(SHEET_COLS, n)
    rows = (n + cols - 1) // cols
    sheet = np.full((rows * TILE, cols * TILE, 3), 255, np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        sheet[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = t
    return sheet


# ----------------------------------------------------------------------------- main API


def _safe_name(path: str) -> str:
    stem = Path(path).stem
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in stem)[:80] or "img"


def _READ_IMAGE(p):
    """Indirection so tests can count decodes."""
    return common.imread(p)


def analyze_failures(records: list[ImageRecord], gts: dict[str, np.ndarray], preds: dict[str, np.ndarray],
                     *, out_dir: str, op_conf=0.25, iou_thr=0.5, max_examples=8, blur_rel=0.5,
                     contrast_de=12.0, occl_ioa=0.2, border_px=2, bg_iou=0.1, set_name: str = "") -> dict:
    """Tag every FN / FP of a class-agnostic set and write example images. See CONTRACT.md."""
    out_dir = Path(out_dir)
    de_thr = float(contrast_de)
    images = list(dict.fromkeys(r.image for r in records))

    # pass 1: matching + features
    per_img = {}
    matched_blur = []
    for p in images:
        gt = gts.get(p, EMPTY_GT)
        gt = EMPTY_GT if gt is None or len(gt) == 0 else np.asarray(gt, np.float64).reshape(-1, 5)
        pr = preds.get(p, EMPTY_PRED)
        pr = EMPTY_PRED if pr is None or len(pr) == 0 else np.asarray(pr, np.float64).reshape(-1, 6)
        md = _match_details(gt, pr, iou_thr=iou_thr, op_conf=op_conf, bg_iou=bg_iou, class_agnostic=True)
        gt_matched = np.asarray(md["gt_matched"], bool).reshape(-1)
        best_iou = np.asarray(md["gt_best_iou"], np.float64).reshape(-1)
        statuses = list(md["pred_status"])
        pidx = np.asarray(md["pred_index"], int).reshape(-1)
        img = common.imread(p) if (len(gt) or len(statuses)) else None
        if img is None and (len(gt) or len(statuses)):
            warnings.warn(f"failure analysis: unreadable image {p}; pixel features set to nan")
        boxes = gt[:, 1:5] if len(gt) else np.zeros((0, 4))
        feats = []
        for i, b in enumerate(boxes):
            if img is not None:
                bl, an = blur_features(img, b)
                de = contrast_de_fn(img, b)
                h, w = img.shape[:2]
                trunc = _truncated(b, w, h, border_px)
            else:
                bl = an = de = float("nan")
                trunc = False
            feats.append({"blur": bl, "anisotropy": an, "de": de, "ioa": max_ioa(boxes, i), "truncated": trunc,
                          "area": float((b[2] - b[0]) * (b[3] - b[1]))})
            if gt_matched[i] and np.isfinite(bl):
                matched_blur.append(bl)
        per_img[p] = {"readable": img is not None, "boxes": boxes, "gt_matched": gt_matched, "best_iou": best_iou,
                      "statuses": statuses, "pcons": pr[pidx] if len(pidx) else EMPTY_PRED, "feats": feats}

    img = None  # drop the last decoded image; pixels are re-read only for chosen examples
    if len(matched_blur) >= MIN_MATCHED_FOR_REL:
        blur_thr = float(blur_rel) * float(np.median(matched_blur))
        blur_mode = "relative"
    else:
        blur_thr = BLUR_ABS_FALLBACK
        blur_mode = "absolute_fallback"

    # pass 2: tags
    fn_tags = {t: 0 for t in FN_TAGS}
    fp_tags = {t: 0 for t in FP_TAGS}
    per_fn = []
    items = {t: [] for t in FN_TAGS + FP_TAGS}  # (image, box, severity)
    img_counts = {t: {} for t in FN_TAGS + FP_TAGS}
    n_fn = n_fp = 0
    for p, d in per_img.items():
        d["fn_tags"] = {}
        for i, b in enumerate(d["boxes"]):
            if d["gt_matched"][i]:
                continue
            n_fn += 1
            f = d["feats"][i]
            tags = []
            if f["area"] < SMALL_MAX_AREA:
                tags.append("small_object")
            if np.isfinite(f["blur"]) and f["blur"] < blur_thr:
                tags.append("motion_blur")
            if f["ioa"] >= occl_ioa or f["truncated"]:
                tags.append("occlusion")
            if np.isfinite(f["de"]) and f["de"] < de_thr:
                tags.append("belt_background")
            if not tags:
                tags.append("other")
            d["fn_tags"][i] = tags
            sev = {"small_object": -f["area"], "motion_blur": -f["blur"] if np.isfinite(f["blur"]) else 0.0,
                   "occlusion": f["ioa"] + (1.0 if f["truncated"] else 0.0),
                   "belt_background": -f["de"] if np.isfinite(f["de"]) else 0.0,
                   "other": -float(d["best_iou"][i]) if np.isfinite(d["best_iou"][i]) else 0.0}
            for t in tags:
                fn_tags[t] += 1
                items[t].append((p, b, sev[t]))
                img_counts[t][p] = img_counts[t].get(p, 0) + 1
            per_fn.append({"image": common.to_project_rel(p), "box": _fmt_box(b), "area": f["area"],
                           "tags": tags, "blur": f["blur"], "anisotropy": f["anisotropy"], "de": f["de"],
                           "ioa": f["ioa"], "truncated": f["truncated"],
                           "best_iou": float(d["best_iou"][i]) if len(d["best_iou"]) > i else float("nan")})
        for pr, st in zip(d["pcons"], d["statuses"]):
            if st == "tp":
                continue
            n_fp += 1
            t = _FP_STATUS_TO_TAG.get(st)
            if t is None:
                LOG.warning("unknown pred_status %r", st)
                continue
            fp_tags[t] += 1
            items[t].append((p, pr[:4], float(pr[4])))
            img_counts[t][p] = img_counts[t].get(p, 0) + 1

    # outputs
    examples, sheets = {}, {}
    for t in FN_TAGS + FP_TAGS:
        if not img_counts[t]:
            continue
        order = sorted(img_counts[t].items(), key=lambda kv: (-kv[1], kv[0]))[: int(max_examples)]
        its = sorted(items[t], key=lambda x: -x[2])[:SHEET_MAX]
        color = (0, 0, 255) if t in FN_TAGS else (0, 165, 255)
        ex_rank = {p: (rank, cnt) for rank, (p, cnt) in enumerate(order)}
        crops_by_img = {}
        for k, (p, b, _) in enumerate(its):
            crops_by_img.setdefault(p, []).append((k, b))
        need = [p for p in dict.fromkeys(list(ex_rank) + list(crops_by_img)) if per_img[p]["readable"]]
        ex_files, tile_at = {}, {}
        for p in need:  # one image decoded at a time, freed right after use
            im = _READ_IMAGE(p)
            if im is None:
                continue
            d = per_img[p]
            if p in ex_rank:
                rank, cnt = ex_rank[p]
                nfn = int((~np.asarray(d["gt_matched"], bool)).sum()) if len(d["gt_matched"]) else 0
                nfp = sum(1 for st in d["statuses"] if st != "tp")
                hdr = f"{t}: {cnt} in image | FN={nfn} FP={nfp} | op_conf={op_conf} iou={iou_thr}"
                ann = _annotate(im, d["boxes"], d["gt_matched"], d["fn_tags"], d["pcons"], d["statuses"], header=hdr)
                dst = out_dir / t / f"{rank:02d}_{cnt}x_{_safe_name(p)}.jpg"
                if common.imwrite(dst, ann):
                    ex_files[rank] = common.to_project_rel(dst)
                del ann
            for k, b in crops_by_img.get(p, []):
                tile_at[k] = _crop_tile(im, b, color)
            del im
        examples[t] = [ex_files[r] for r in sorted(ex_files)]
        tiles = [tile_at[k] for k in sorted(tile_at)]
        if tiles:
            dst = out_dir / f"{t}_crops.jpg"
            if common.imwrite(dst, _contact_sheet(tiles)):
                sheets[t] = common.to_project_rel(dst)

    return {
        "set": set_name,
        "n_fn": n_fn,
        "n_fp": n_fp,
        "fn_tags": fn_tags,
        "fp_tags": fp_tags,
        "fn_tag_share": {t: (fn_tags[t] / n_fn if n_fn else 0.0) for t in FN_TAGS},
        "examples": examples,
        "contact_sheets": sheets,
        "per_fn": per_fn,
        "params": {"op_conf": op_conf, "iou_thr": iou_thr, "blur_rel": blur_rel, "blur_threshold": blur_thr,
                   "blur_mode": blur_mode, "n_matched_blur": len(matched_blur), "contrast_de": de_thr,
                   "occl_ioa": occl_ioa, "border_px": border_px, "bg_iou": bg_iou},
    }


contrast_de_fn = contrast_de  # alias: the analyze_failures keyword `contrast_de` shadows the function name
