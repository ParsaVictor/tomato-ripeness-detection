"""COCO-faithful detection metrics in pure numpy (no IO).

`evaluate` reproduces pycocotools COCOeval bbox AP (iscrowd=0, one category, maxDets=[max_det],
areaRng=[area_rng]) to floating-point precision, and adds P/R/F1 at an operating confidence, a best-F1
scan, max recall and the raw PR curve. `evaluate_sizes` runs it for all/small/medium/large;
`match_details` gives per-box match status for one image (failure analysis).

Conventions (see phase5/common.py): GT (N, 5) = [cls, x1, y1, x2, y2]; preds (M, 6) = [x1, y1, x2, y2, conf, cls].
"""
from __future__ import annotations

import math

import numpy as np

from . import common

_REC_THRS = np.linspace(0.0, 1.0, 101)
_MAX_PR_POINTS = 1000


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes (N, 4) x (M, 4) -> (N, M). Same formula as pycocotools (no +1)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    aw = (a[:, 2] - a[:, 0]).clip(0)
    ah = (a[:, 3] - a[:, 1]).clip(0)
    bw = (b[:, 2] - b[:, 0]).clip(0)
    bh = (b[:, 3] - b[:, 1]).clip(0)
    iw = (np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])).clip(0)
    ih = (np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])).clip(0)
    inter = iw * ih
    union = (aw * ah)[:, None] + (bw * bh)[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def _last_argmax(v: np.ndarray) -> int:
    """Index of the maximum, ties resolved to the LAST occurrence (COCO's `ious < iou: continue` rule)."""
    return int(len(v) - 1 - np.argmax(v[::-1]))


def _candidates(ious: np.ndarray, tmin: float):
    """Sparse candidate lists computed once per image: for every det whose best IoU >= tmin, the GT
    columns with IoU >= tmin and their IoUs (plain Python lists, ascending column order)."""
    if ious.size == 0:
        return []
    rows = np.nonzero(ious.max(axis=1) >= tmin)[0]
    out = []
    for d in rows.tolist():
        cols = np.nonzero(ious[d] >= tmin)[0]
        out.append((d, cols.tolist(), ious[d, cols].tolist()))
    return out


def _coco_match(ious: np.ndarray, gt_ig: np.ndarray, thr: float, cands=None):
    """COCO evaluateImg greedy matching for one threshold. Dets must be in conf order. Returns
    dt_matched_gt_index (D,) int, -1 = unmatched. Among untaken GT with IoU >= t, non-ignored GT are
    preferred; within a group the max IoU wins with ties to the LAST index (COCO's `ious < iou` rule).
    Dets with no GT at IoU >= t are skipped without per-det work. `cands` (from _candidates with a
    threshold <= t) can be passed to share the sparse structure across thresholds."""
    nd, ng = ious.shape
    t = min(thr, 1 - 1e-10)
    dtm = np.full(nd, -1, dtype=np.int64)
    if ng == 0 or nd == 0:
        return dtm
    if cands is None:
        cands = _candidates(ious, t)
    ig = np.asarray(gt_ig).astype(bool).tolist()
    taken = [False] * ng
    for d, cols, vals in cands:
        bn, bnv, bi, biv = -1, -1.0, -1, -1.0
        for c, v in zip(cols, vals):
            if v < t or taken[c]:
                continue
            if ig[c]:
                if v >= biv:
                    bi, biv = c, v
            elif v >= bnv:
                bn, bnv = c, v
        m = bn if bn >= 0 else bi
        if m >= 0:
            taken[m] = True
            dtm[d] = m
    return dtm


def _f(x) -> float:
    return float(x)


def _empty_result(n_images, iou_thrs, op_conf):
    nan = float("nan")
    return {
        "n_images": n_images, "n_gt": 0, "n_pred": 0, "ap50": nan, "ap": nan,
        "ap_per_iou": {float(t): nan for t in iou_thrs}, "op_conf": float(op_conf),
        "tp": 0, "fp": 0, "fn": 0, "precision": nan, "recall": nan, "f1": nan,
        "best_f1": nan, "best_f1_conf": nan, "precision_at_best_f1": nan, "recall_at_best_f1": nan,
        "max_recall": nan, "pr_curve": {"recall": [], "precision": [], "conf": []},
    }


def evaluate(gts, preds, *, iou_thrs=(0.5,), area_rng=(0.0, 1e10), area_scales=None,
             max_det: int = 300, op_conf: float = 0.25, class_id=None) -> dict:
    """COCO bbox evaluation over a list of images (see CONTRACT.md for the returned keys)."""
    if len(gts) != len(preds):
        raise ValueError(f"gts ({len(gts)}) and preds ({len(preds)}) differ in length")
    iou_thrs = [float(t) for t in np.atleast_1d(np.asarray(iou_thrs, dtype=np.float64))]
    if not iou_thrs:
        raise ValueError("iou_thrs is empty")
    lo, hi = float(area_rng[0]), float(area_rng[1])
    n_img = len(gts)
    if area_scales is None:
        area_scales = [1.0] * n_img
    if len(area_scales) != n_img:
        raise ValueError("area_scales length must equal number of images")
    nt = len(iou_thrs)

    scores_l, dtm_l, dtig_l, gtig_l = [], [], [], []
    for i in range(n_img):
        g = np.asarray(gts[i], dtype=np.float64).reshape(-1, 5)
        p = np.asarray(preds[i], dtype=np.float64).reshape(-1, 6)
        if class_id is not None:
            g = g[g[:, 0] == class_id]
            p = p[p[:, 5] == class_id]
        if len(g) == 0 and len(p) == 0:
            continue  # COCO: evaluateImg returns None
        s2 = float(area_scales[i]) ** 2
        g_area = common.box_areas(g[:, 1:5]) * s2
        g_ig = ((g_area < lo) | (g_area > hi)).astype(np.int64)
        gi = np.argsort(g_ig, kind="mergesort")
        g, g_area, g_ig = g[gi], g_area[gi], g_ig[gi]
        di = np.argsort(-p[:, 4], kind="mergesort")[:max_det]
        p = p[di]
        d_area = common.box_areas(p[:, 0:4]) * s2
        d_out = (d_area < lo) | (d_area > hi)
        ious = box_iou(p[:, 0:4], g[:, 1:5])
        dtm = np.zeros((nt, len(p)), dtype=bool)
        dtig = np.zeros((nt, len(p)), dtype=bool)
        cands = _candidates(ious, min(min(iou_thrs), 1 - 1e-10)) if len(g) else []
        for ti, t in enumerate(iou_thrs):
            m = _coco_match(ious, g_ig, t, cands)
            matched = m >= 0
            dtm[ti] = matched
            if len(g):
                dtig[ti] = np.where(matched, g_ig[np.maximum(m, 0)].astype(bool), False)
            dtig[ti] |= (~matched) & d_out
        scores_l.append(p[:, 4])
        dtm_l.append(dtm)
        dtig_l.append(dtig)
        gtig_l.append(g_ig)

    if not gtig_l:
        return _empty_result(n_img, iou_thrs, op_conf)
    scores = np.concatenate(scores_l)
    order = np.argsort(-scores, kind="mergesort")
    scores = scores[order]
    dtm = np.concatenate(dtm_l, axis=1)[:, order]
    dtig = np.concatenate(dtig_l, axis=1)[:, order]
    gtig = np.concatenate(gtig_l)
    npig = int(np.count_nonzero(gtig == 0))
    if npig == 0:
        r = _empty_result(n_img, iou_thrs, op_conf)
        r["n_pred"] = int(np.count_nonzero(~dtig[0]))
        r["fp"] = int(np.count_nonzero((~dtm[0]) & (~dtig[0]) & (scores >= op_conf)))
        r["tp"] = 0
        return r

    tps = dtm & ~dtig
    fps = ~dtm & ~dtig
    tp_sum = np.cumsum(tps, axis=1).astype(np.float64)
    fp_sum = np.cumsum(fps, axis=1).astype(np.float64)
    ap_per = {}
    rc0 = pr0 = None
    for ti in range(nt):
        tp, fp = tp_sum[ti], fp_sum[ti]
        nd = len(tp)
        rc = tp / npig
        pr = tp / (fp + tp + np.spacing(1))
        if ti == 0:
            rc0, pr0 = rc.copy(), pr.copy()
        q = np.zeros(len(_REC_THRS))
        if nd:
            pr_env = np.maximum.accumulate(pr[::-1])[::-1]
            inds = np.searchsorted(rc, _REC_THRS, side="left")
            valid = inds < nd
            q[valid] = pr_env[inds[valid]]
        ap_per[iou_thrs[ti]] = float(np.mean(q))

    # ---- operating point / best F1 / PR curve at iou_thrs[0] (non-ignored dets only)
    keep = ~dtig[0]
    s_k = scores[keep]
    tp_k = dtm[0][keep]
    n_pred = int(len(s_k))
    sel = s_k >= op_conf
    tp_op = int(np.count_nonzero(tp_k[sel]))
    fp_op = int(np.count_nonzero(~tp_k[sel]))
    fn_op = npig - tp_op
    prec = tp_op / (tp_op + fp_op) if (tp_op + fp_op) else float("nan")
    rec = tp_op / npig
    f1 = 2 * tp_op / (2 * tp_op + fp_op + fn_op)

    best = dict(best_f1=0.0, best_f1_conf=float("nan"), precision_at_best_f1=float("nan"), recall_at_best_f1=0.0)
    if n_pred:
        ctp = np.cumsum(tp_k).astype(np.float64)
        cfp = np.cumsum(~tp_k).astype(np.float64)
        last_of_tie = np.r_[s_k[1:] != s_k[:-1], True]  # a cut is only possible after a group of equal scores
        cut = np.nonzero(last_of_tie)[0]
        f1s = 2 * ctp[cut] / (ctp[cut] + cfp[cut] + npig)
        j = int(np.argmax(f1s))
        k = cut[j]
        best = dict(best_f1=_f(f1s[j]), best_f1_conf=_f(s_k[k]),
                    precision_at_best_f1=_f(ctp[k] / (ctp[k] + cfp[k])), recall_at_best_f1=_f(ctp[k] / npig))
        max_recall = _f(ctp[-1] / npig)
    else:
        max_recall = 0.0

    # raw PR curve (all non-ignored dets, conf-sorted)
    rcv, prv, cv = rc0[keep], pr0[keep], s_k
    if len(rcv) > _MAX_PR_POINTS:
        idx = np.unique(np.linspace(0, len(rcv) - 1, _MAX_PR_POINTS).round().astype(int))
        rcv, prv, cv = rcv[idx], prv[idx], cv[idx]

    return {
        "n_images": n_img, "n_gt": npig, "n_pred": n_pred,
        "ap50": ap_per[0.5] if 0.5 in ap_per else float("nan"),
        "ap": float(np.mean(list(ap_per.values()))),
        "ap_per_iou": ap_per, "op_conf": float(op_conf),
        "tp": tp_op, "fp": fp_op, "fn": fn_op,
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        **best, "max_recall": max_recall,
        "pr_curve": {"recall": [float(x) for x in rcv], "precision": [float(x) for x in prv],
                     "conf": [float(x) for x in cv]},
    }


def evaluate_sizes(gts, preds, *, area_scales=None, iou_thrs=(0.5,), max_det=300, op_conf=0.25,
                   class_id=None) -> dict:
    """{"all"|"small"|"medium"|"large": evaluate(...)} using common.AREA_RANGES."""
    return {k: evaluate(gts, preds, iou_thrs=iou_thrs, area_rng=rng, area_scales=area_scales,
                        max_det=max_det, op_conf=op_conf, class_id=class_id)
            for k, rng in common.AREA_RANGES.items()}


def match_details(gt: np.ndarray, pred: np.ndarray, *, iou_thr=0.5, op_conf=0.25, bg_iou=0.1,
                  class_agnostic=True) -> dict:
    """Per-box COCO greedy matching for one image (preds with conf >= op_conf, no area ignore)."""
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 5)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 6)
    cand = np.nonzero(pred[:, 4] >= op_conf)[0]
    order = cand[np.argsort(-pred[cand, 4], kind="mergesort")]
    p = pred[order]
    ious = box_iou(p[:, 0:4], gt[:, 1:5])
    if not class_agnostic and ious.size:
        ious = np.where(p[:, 5][:, None] == gt[:, 0][None, :], ious, 0.0)
    m = _coco_match(ious, np.zeros(len(gt), dtype=np.int64), iou_thr)
    ng = len(gt)
    gt_matched = np.zeros(ng, dtype=bool)
    gt_conf = np.full(ng, np.nan)
    status = []
    for d in range(len(p)):
        if m[d] >= 0:
            gt_matched[m[d]] = True
            gt_conf[m[d]] = p[d, 4]
            status.append("tp")
            continue
        mx = float(ious[d].max()) if ng else 0.0
        if mx >= min(iou_thr, 1 - 1e-10):
            status.append("duplicate")
        elif mx >= bg_iou:
            status.append("localization")
        else:
            status.append("background")
    gt_best = ious.max(axis=0) if len(p) and ng else np.zeros(ng)
    return {"gt_matched": gt_matched, "gt_match_conf": gt_conf, "gt_best_iou": gt_best.astype(np.float64),
            "pred_status": status, "pred_index": order.astype(np.int64)}
