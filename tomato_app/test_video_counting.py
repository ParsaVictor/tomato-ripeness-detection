"""Measure video counting with a known answer.

A real belt clip with "how many tomatoes passed" is not available, so this builds smooth moving-camera clips
from labelled holdout photos (never trained on): a window slides across the photo, the way a belt slides under
a fixed camera. Every labelled tomato passes through the window, so the true number of distinct tomatoes is
known from the label file. The clip goes through the same tracker + counter as a real video.

  unique count   = tracks seen in >= 3 frames            (truth: every labelled tomato)
  line count     = tracks whose centre crossed the middle line (truth: tomatoes whose centre the line sweeps)
  per-frame sum  = what counting every frame without tracking would report (shows why tracking is needed)

    py tomato_app/test_video_counting.py
"""
from __future__ import annotations

import json
import random
import time
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tomato_inspect as ti  # noqa: E402
import tomato_video as tv  # noqa: E402

OUT = ti.ROOT / "results" / "video_counting_test"


def pan_frames(img, frac_win=0.45, step_frac=0.02, out_w=None):
    """Horizontal pan; yields (frame, x0, scale)."""
    h, w = img.shape[:2]
    ww = int(w * frac_win)
    step = max(1, int(ww * step_frac))
    xs = list(range(0, w - ww + 1, step))
    if xs[-1] != w - ww:
        xs.append(w - ww)
    s = (out_w / ww) if out_w else 1.0
    for x0 in xs:
        f = img[:, x0:x0 + ww]
        yield (cv2.resize(f, (round(ww * s), round(h * s)), interpolation=cv2.INTER_AREA) if s != 1 else f), x0, ww


def run_clip(insp, img, gts, step_frac, demo_dir=None, assoc_scale=tv.ASSOC_SCALE):
    h, w = img.shape[:2]
    vt = tv.TomatoVideo(insp, line=[0.5, 0, 0.5, 1], assoc_scale=assoc_scale)
    vt._tracker(30)
    per_frame_sum, writer = 0, None
    xs_win = None
    for fi, (f, x0, ww) in enumerate(pan_frames(img, step_frac=step_frac, out_w=640)):
        xs_win = ww
        t0 = time.perf_counter()
        live, shown, ids = vt.step(fi, f)
        ms = (time.perf_counter() - t0) * 1000
        per_frame_sum += len(shown)
        if demo_dir is not None:
            fr = tv.draw_frame(f, shown, ids, vt, {"ms": ms, "name": "pan", "frame": str(fi + 1),
                                                         "fps": 1000 / max(ms, 1)}, H=900)
            bgr = cv2.cvtColor(np.asarray(fr), cv2.COLOR_RGB2BGR)
            if writer is None:
                demo_dir.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(demo_dir / "pan_demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15,
                                         (bgr.shape[1], bgr.shape[0]))
            writer.write(bgr)
            if fi % 20 == 0:
                cv2.imencode(".jpg", bgr)[1].tofile(str(demo_dir / f"pan_{fi:04d}.jpg"))
    if writer:
        writer.release()
    cx = [(g["box"][0] + g["box"][2]) / 2 for g in gts]
    truth_line = sum(xs_win / 2 < c < w - xs_win / 2 for c in cx)
    return {"truth": len(gts), "truth_line": truth_line,
            "unique": sum(t.hits >= 3 for t in vt.tracks.values()),
            "line": sum(t.counted for t in vt.tracks.values()),
            "per_frame_sum": per_frame_sum, "frames": fi + 1,
            "cls_calls": vt.cls_calls, "boxes": vt.det_boxes}


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--assoc-scale", type=float, default=tv.ASSOC_SCALE)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    rng = random.Random(1)
    insp = ti.TomatoInspector()
    clips = []
    for key, n in (("sxsc7", 8), ("conveyor", 8)):
        imgs = ti.SETS[key]["images"]()
        rng.shuffle(imgs)
        picked = []
        for p in imgs:
            im = ti.imread(p)
            g = ti.read_gt(p, im.shape[1], im.shape[0], ti.SETS[key]["classes"])
            if len(g) >= 8:
                picked.append((p, im, g))
            if len(picked) == n:
                break
        clips += [(key, *x) for x in picked]
    rows = []
    for speed in (0.02, 0.05):
        for k, (key, p, im, g) in enumerate(clips):
            demo = out / "demo" if (speed == 0.02 and key == "sxsc7" and k == 0) else None
            r = run_clip(insp, im, g, speed, demo, a.assoc_scale)
            r.update(set=key, image=p.name, speed=speed)
            rows.append(r)
            print(f"{key:8s} speed {speed:.2f} | truth {r['truth']:3d} unique {r['unique']:3d} | "
                  f"line truth {r['truth_line']:3d} line {r['line']:3d} | per-frame sum {r['per_frame_sum']:5d} | "
                  f"cls {r['cls_calls']}/{r['boxes']}", flush=True)
    summ = {}
    for key in ("sxsc7", "conveyor"):
        for speed in (0.02, 0.05):
            rr = [r for r in rows if r["set"] == key and r["speed"] == speed]
            T, U = sum(r["truth"] for r in rr), sum(r["unique"] for r in rr)
            TL, L = sum(r["truth_line"] for r in rr), sum(r["line"] for r in rr)
            summ[f"{key}@{speed}"] = {
                "clips": len(rr), "truth": T, "unique": U, "unique_accuracy": 1 - abs(U - T) / T,
                "unique_mean_clip_acc": float(np.mean([1 - min(1, abs(r["unique"] - r["truth"]) / r["truth"])
                                                        for r in rr])),
                "truth_line": TL, "line": L, "line_accuracy": 1 - abs(L - TL) / TL if TL else None,
                "line_mean_clip_acc": float(np.mean([1 - min(1, abs(r["line"] - r["truth_line"]) /
                                                             max(1, r["truth_line"])) for r in rr])),
                "per_frame_sum_vs_truth": sum(r["per_frame_sum"] for r in rr) / T,
                "classifier_saving": 1 - sum(r["cls_calls"] for r in rr) / sum(r["boxes"] for r in rr)}
            print(key, speed, json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                          for k, v in summ[f"{key}@{speed}"].items()}))
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"assoc_scale": a.assoc_scale, "summary": summ, "clips": rows}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
