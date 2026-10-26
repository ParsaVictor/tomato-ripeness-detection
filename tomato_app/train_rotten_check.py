"""Train the "rotten check": a second opinion that decides rotten vs sound for every crop the ripeness
classifier calls rotten.

Why: the ripeness classifier saw rotten tomatoes only in Tomato Defects, where they are dark red/brown. On the
conveyor and in tomato-sxsc7 it calls healthy deep-red tomatoes "rotten" (checked by eye: none of those images
contain rotten fruit). The check is a logistic regression on the classifier's own 512-d embedding, trained on
  rotten = Tomato Defects / extra train crops labelled rotten
  sound  = Tomato Defects / extra train crops (raw, half_ripe, ripe)
         + every labelled tomato of conveyor and sxsc7 TRAIN images (never a holdout image or a near-copy of one)
Output: weights/rotten_check.npz (+ a JSON with the validation numbers).

    py tomato_app/train_rotten_check.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tomato_inspect as ti  # noqa: E402

CLS_DIR = ti.ROOT / "datasets/cls_tomato"
OUT = ti.WEIGHTS / "rotten_check.npz"
SOUND = ["raw", "half_ripe", "ripe"]


def dhash(img, n=8) -> int:
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (n + 1, n), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def frame_no(p: Path):
    m = re.search(r"frame_0*(\d+)_", p.name)
    return int(m.group(1)) if m else None


def domain_images(key: str, root: str) -> list[Path]:
    """Train images of conveyor / sxsc7 that are far from every holdout image."""
    hold = ti.SETS[key]["images"]()
    hold_set = {p.resolve() for p in hold}
    hold_hash = [dhash(ti.imread(p)) for p in hold]
    hold_frames = {frame_no(p) for p in hold} - {None}
    keep = []
    for p in sorted((ti.ROOT / root).glob("*/images/*.jpg")):
        if p.resolve() in hold_set:
            continue
        f = frame_no(p)
        if f is not None and any(abs(f - h) <= 3 for h in hold_frames):     # conveyor: neighbouring video frames
            continue
        hsh = dhash(ti.imread(p))
        if min(bin(hsh ^ q).count("1") for q in hold_hash) <= 12:          # near-copy of a holdout image
            continue
        keep.append(p)
    return keep


def domain_crops(paths: list[Path]):
    crops, groups = [], []
    for p in paths:
        img = ti.imread(p)
        for g in ti.read_gt(p, img.shape[1], img.shape[0], [ti.GT_ONLY_TOMATO]):
            c = ti.crop_box(img, g["box"])
            if c is not None:
                crops.append(c)
                groups.append(p.stem)
    return crops, groups


def folder_crops(split: str, cls: str, limit: int | None, rng):
    fs = sorted((CLS_DIR / split / cls).glob("*.jpg"))
    rng.shuffle(fs)
    fs = fs[:limit] if limit else fs
    return [ti.imread(f) for f in fs]


def embed(model, crops, bs=256):
    out = []
    for i in range(0, len(crops), bs):
        e = model.embed(crops[i:i + bs], imgsz=ti.CLS_IMGSZ, verbose=False, device="cpu")
        out += [t.cpu().numpy() for t in e]
    x = np.asarray(out, np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-6)


def main():
    from sklearn.linear_model import LogisticRegression
    from ultralytics import YOLO

    rng = random.Random(0)
    model = YOLO(str(ti.CLS_WEIGHTS))
    print("collecting conveyor / sxsc7 train crops (holdouts and their near-copies excluded) ...", flush=True)
    dom = {}
    for key, root in (("conveyor", "datasets/unified/u_conveyor"), ("sxsc7", "datasets/unified/u_sxsc7")):
        imgs = domain_images(key, root)
        crops, groups = domain_crops(imgs)
        uniq = sorted(set(groups))
        rng.shuffle(uniq)
        val_g = set(uniq[: max(1, len(uniq) // 5)])                       # 20% of images for validation
        dom[key] = {"imgs": len(imgs), "train": [c for c, g in zip(crops, groups) if g not in val_g],
                    "val": [c for c, g in zip(crops, groups) if g in val_g]}
        print(f"  {key}: {len(imgs)} images, {len(crops)} tomato crops", flush=True)

    print("embedding crops ...", flush=True)
    Xr = embed(model, folder_crops("train", "rotten", None, rng))
    Xs = embed(model, sum((folder_crops("train", c, 2500, rng) for c in SOUND), []))
    Xd = embed(model, dom["conveyor"]["train"] + dom["sxsc7"]["train"])
    X = np.concatenate([Xr, Xs, Xd])
    y = np.r_[np.ones(len(Xr)), np.zeros(len(Xs) + len(Xd))]
    w = np.r_[np.full(len(Xr), 1.0), np.full(len(Xs), 1.0), np.full(len(Xd), 2.0)]   # weight the new domain
    clf = LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced")
    clf.fit(X, y, sample_weight=w)

    # validation: Tomato Defects val crops (rotten recall) + held-back conveyor / sxsc7 crops (false rotten)
    Vr = embed(model, folder_crops("val", "rotten", None, rng))
    Vs = embed(model, sum((folder_crops("val", c, 800, rng) for c in SOUND), []))
    Vd = {k: embed(model, dom[k]["val"]) for k in dom}
    rep = {"train": {"rotten": len(Xr), "sound_defects": len(Xs), "sound_conveyor_sxsc7": len(Xd)},
           "domain_images": {k: dom[k]["imgs"] for k in dom}, "thresholds": {}}
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        rep["thresholds"][t] = {
            "defects_val_rotten_recall": float((clf.predict_proba(Vr)[:, 1] >= t).mean()),
            "defects_val_sound_false_rotten": float((clf.predict_proba(Vs)[:, 1] >= t).mean()),
            **{f"{k}_val_false_rotten": float((clf.predict_proba(Vd[k])[:, 1] >= t).mean()) for k in Vd}}
        print(t, json.dumps({k: round(v, 3) for k, v in rep["thresholds"][t].items()}))
    np.savez(OUT, coef=clf.coef_[0].astype(np.float32), intercept=np.float32(clf.intercept_[0]))
    OUT.with_suffix(".json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("->", OUT)


if __name__ == "__main__":
    main()
