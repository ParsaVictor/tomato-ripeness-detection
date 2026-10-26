"""Synthetic conveyor-belt test for phase5.failure_analysis (run: py -3.10 -m phase5.tests.test_failure_analysis)."""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from phase5 import common
from phase5.common import ImageRecord

try:  # real implementation (written in parallel); stand-in only if it is missing
    from phase5.metrics import match_details  # noqa: F401
except ImportError:  # pragma: no cover
    import types

    def _iou(a, b):
        if len(a) == 0 or len(b) == 0:
            return np.zeros((len(a), len(b)))
        iw = (np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])).clip(0)
        ih = (np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])).clip(0)
        inter = iw * ih
        aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        return inter / (aa[:, None] + ab[None, :] - inter)

    def match_details(gt, pred, *, iou_thr=0.5, op_conf=0.25, bg_iou=0.1, class_agnostic=True):
        cand = np.nonzero(pred[:, 4] >= op_conf)[0]
        order = cand[np.argsort(-pred[cand, 4], kind="mergesort")]
        ious = _iou(pred[order, :4], gt[:, 1:5])
        used = np.zeros(len(gt), bool)
        st = []
        for d in range(len(order)):
            best, bi = min(iou_thr, 1 - 1e-10), -1
            for g in range(len(gt)):
                if not used[g] and ious[d, g] >= best:
                    best, bi = ious[d, g], g
            if bi >= 0:
                used[bi] = True
                st.append("tp")
            else:
                mx = ious[d].max() if len(gt) else 0.0
                st.append("duplicate" if mx >= iou_thr else ("localization" if mx >= bg_iou else "background"))
        return {"gt_matched": used, "gt_match_conf": np.full(len(gt), np.nan),
                "gt_best_iou": ious.max(0) if ious.size else np.zeros(len(gt)), "pred_status": st,
                "pred_index": order}

    sys.modules["phase5.metrics"] = types.SimpleNamespace(match_details=match_details)

from phase5 import failure_analysis as fa  # noqa: E402

RNG = np.random.default_rng(0)
H, W = 480, 640
TOMATO = (40, 60, 210)  # BGR red/orange


def belt(h=H, w=W):
    base = np.full((h, w, 3), 70, np.float64)
    base += RNG.normal(0, 12, (h, w, 1))  # textured gray belt
    base[:, ::40] -= 25  # belt seams
    return base


def disc(img, cx, cy, r, color, noise=12.0):
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(r), 1, -1)
    m = mask.astype(bool)
    tex = np.asarray(color, np.float64)[None, :] + RNG.normal(0, noise, (m.sum(), 1))
    img[m] = tex
    return [0, cx - r, cy - r, cx + r, cy + r]


def finish(img):
    return np.clip(img, 0, 255).astype(np.uint8)


def build_scene(root: Path):
    """Two images. Returns records, gts, preds and the expected tags per missed GT."""
    expected = {}  # (image, gt_index) -> tag that must be present
    must_not = {}
    gts, preds, recs = {}, {}, []

    # ---------- image A: belt with sharp detected tomatoes + chosen misses
    img = belt()
    g = []
    for k in range(6):  # 6 sharp detected tomatoes -> relative blur threshold
        g.append(disc(img, 60 + k * 95, 90, 30, TOMATO))
    i_small = len(g); g.append(disc(img, 600, 300, 8, TOMATO))  # tiny far tomato (area 256)
    i_border = len(g); g.append([0, 0, 260, 40, 320])  # cut by the left border
    disc(img, 10, 290, 30, TOMATO)
    i_other = len(g); g.append(disc(img, 300, 260, 30, TOMATO))  # sharp, clearly visible, simply missed
    i_belt = len(g); g.append(disc(img, 450, 260, 30, (72, 72, 72)))  # recoloured to belt colour
    # overlapping pair: the front one detected, the hidden one missed
    i_back = len(g); g.append(disc(img, 160, 400, 32, TOMATO))
    i_front = len(g); g.append(disc(img, 200, 400, 32, (30, 80, 230)))
    a = root / "صحنه_الف.jpg"
    common.imwrite(a, finish(img), quality=98)
    gt = np.asarray(g, np.float64)
    p = [[*gt[k, 1:], 0.9, 0] for k in range(6)]
    p.append([*gt[i_front, 1:], 0.8, 0])
    p.append([500, 420, 560, 470, 0.85, 0])  # fires on empty belt -> belt_background_fp
    b0 = gt[0, 1:]
    p.append([b0[0] + 20, b0[1] + 20, b0[2] + 20, b0[3] + 20, 0.7, 0])  # IoU ~0.1-0.5 -> localization
    b1 = gt[1, 1:]
    p.append([b1[0] + 1, b1[1], b1[2] + 1, b1[3], 0.6, 0])  # second box on a matched tomato -> duplicate
    p.append([*gt[i_other, 1:], 0.1, 0])  # below op_conf: must be ignored (tomato stays FN)
    gts[str(a)] = gt
    preds[str(a)] = np.asarray(p, np.float64)
    recs.append(ImageRecord(str(a), None, "conveyor", "all"))
    expected[(str(a), i_small)] = "small_object"
    expected[(str(a), i_border)] = "occlusion"
    expected[(str(a), i_other)] = "other"
    expected[(str(a), i_belt)] = "belt_background"
    expected[(str(a), i_back)] = "occlusion"
    must_not[(str(a), i_belt)] = {"motion_blur", "small_object", "occlusion"}
    must_not[(str(a), i_other)] = {"motion_blur", "small_object", "occlusion", "belt_background"}

    # ---------- image B: motion-blurred tomato (linear kernel) + a sharp detected one
    img = belt()
    g = [disc(img, 120, 120, 30, TOMATO)]
    i_blur = len(g)
    tmp = img.copy()
    disc(tmp, 400, 240, 34, TOMATO)
    k = np.zeros((31, 31)); k[15, :] = 1.0 / 31  # horizontal linear motion kernel
    blurred = cv2.filter2D(tmp, -1, k, borderType=cv2.BORDER_REFLECT)
    img[180:300, 330:470] = blurred[180:300, 330:470]
    g.append([0, 366, 206, 434, 274])
    b = root / "صحنه_ب.png"
    common.imwrite(b, finish(img))
    gt = np.asarray(g, np.float64)
    gts[str(b)] = gt
    preds[str(b)] = np.asarray([[*gt[0, 1:], 0.95, 0]], np.float64)
    recs.append(ImageRecord(str(b), None, "conveyor", "all"))
    expected[(str(b), i_blur)] = "motion_blur"
    must_not[(str(b), i_blur)] = {"belt_background", "small_object", "occlusion"}

    # ---------- image C: nothing to see, nothing predicted (must not crash / not be counted)
    c = root / "خالی.jpg"
    common.imwrite(c, finish(belt()))
    gts[str(c)] = common.EMPTY_GT
    preds[str(c)] = common.EMPTY_PRED
    recs.append(ImageRecord(str(c), None, "conveyor", "all"))
    return recs, gts, preds, expected, must_not


def test_lab_is_float_based():
    img = np.zeros((40, 40, 3), np.uint8)
    img[:] = (70, 70, 70)
    img[10:30, 10:30] = (40, 60, 210)
    de = fa.contrast_de(img, [12, 12, 28, 28])
    lab_in = cv2.cvtColor(np.float32([[[40, 60, 210]]]) / 255, cv2.COLOR_BGR2LAB)[0, 0]
    lab_out = cv2.cvtColor(np.float32([[[70, 70, 70]]]) / 255, cv2.COLOR_BGR2LAB)[0, 0]
    # ring of a 16 px box extends 4 px -> pixels 8..12 & 28..32, mostly the red square: compute directly
    assert de > 0
    full = np.linalg.norm(lab_in - lab_out)
    de2 = fa.contrast_de(img, [10, 10, 30, 30])  # ring fully on the gray background
    assert abs(de2 - full) < 1e-3, (de2, full)
    assert np.isnan(fa.blur_features(img, [5, 5, 6.5, 20])[0])  # < 2 px wide -> nan


def test_failure_tags():
    tmp = Path(tempfile.mkdtemp(prefix="تست_خطا_"))
    try:
        recs, gts, preds, expected, must_not = build_scene(tmp)
        out = tmp / "خروجی"
        reads = []
        orig = fa._READ_IMAGE
        fa._READ_IMAGE = lambda q: (reads.append(q), orig(q))[1]
        try:
            res = fa.analyze_failures(recs, gts, preds, out_dir=str(out), set_name="conveyor_holdout")
        finally:
            fa._READ_IMAGE = orig
        # output pass re-reads only images used as examples/crops, never the empty image
        assert reads and all(gts[q] is not None for q in reads)
        assert not any(Path(q).name == "خالی.jpg" for q in reads)
        assert res["set"] == "conveyor_holdout"
        assert res["n_fn"] == len(expected), (res["n_fn"], res["per_fn"])
        assert res["params"]["blur_mode"] == "relative", res["params"]
        by_key = {}
        for f in res["per_fn"]:
            by_key[(f["image"], tuple(round(v, 3) for v in f["box"]))] = f
        for (img, i), tag in expected.items():
            box = tuple(round(float(v), 3) for v in gts[img][i, 1:5])
            f = by_key[(common.to_project_rel(img), box)]
            assert tag in f["tags"], (Path(img).name, i, tag, f)
            bad = must_not.get((img, i), set()) & set(f["tags"])
            assert not bad, (Path(img).name, i, bad, f)
        assert res["fp_tags"] == {"belt_background_fp": 1, "localization_fp": 1, "duplicate_fp": 1}, res["fp_tags"]
        assert res["n_fp"] == 3
        assert res["fn_tags"]["occlusion"] == 2 and res["fn_tags"]["other"] == 1, res["fn_tags"]
        assert abs(sum(res["fn_tags"].values()) / res["n_fn"] - sum(res["fn_tag_share"].values())) < 1e-9
        for t in ("small_object", "motion_blur", "occlusion", "belt_background", "other",
                  "belt_background_fp", "localization_fp", "duplicate_fp"):
            assert res["examples"].get(t), t
            for p in res["examples"][t]:
                ex = common.imread(common.resolve_project_path(p))
                assert ex is not None, p
                src = [r.image for r in recs if fa._safe_name(r.image) in Path(p).name][0]
                assert ex.shape[0] > common.imread(src).shape[0], "header/legend bars missing"
            sheet = common.resolve_project_path(res["contact_sheets"][t])
            assert sheet.name == f"{t}_crops.jpg" and common.imread(sheet) is not None
        # the occlusion examples: image A (2 occlusion FNs) is the only one and comes first
        assert "2x" in Path(res["examples"]["occlusion"][0]).name
        common.dump_json(out / "res.json", res)  # JSON-serialisable
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_no_pixels_kept_and_legend():
    img = np.full((60, 80, 3), 90, np.uint8)
    ann = fa._annotate(img, np.array([[5, 5, 20, 20.]]), np.array([False]), {0: ["small_object"]},
                       common.EMPTY_PRED, [], header="small_object: 1")
    assert ann.shape[1] == 80 and ann.shape[0] > 60
    import inspect
    assert '"img": img' not in inspect.getsource(fa.analyze_failures)


def main() -> int:
    test_no_pixels_kept_and_legend()
    test_lab_is_float_based()
    test_failure_tags()
    print("OK test_failure_analysis")
    return 0


if __name__ == "__main__":
    sys.exit(main())
