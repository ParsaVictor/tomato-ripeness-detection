"""Download the Roboflow tomato datasets once and build the data for the two-stage model.

Stage 1 - detector: one class "tomato" (every tomato box of every dataset, incl. conveyor and sxsc7).
Stage 2 - classifier on tomato crops: raw / half_ripe / ripe / rotten (the Tomato Defects classes);
only datasets with real ripeness labels feed it (no guessing from color).

Unified folders (Roboflow-style exports: <split>/images, <split>/labels, data.yaml):
  u_defects  : Tomato Defects (4 classes)              -> eval set (a): its test split
  u_sxsc7    : tomato-sxsc7 ("tomato")                 -> eval set (b): held-out subset
  u_conveyor : tomato_count_project (conveyor belt)    -> eval set (c): >= 30 held-out images
  u_extra    : papapeto + tomato-detection-sorting + tomato-classification, mapped to the 4 classes
               (extra training data for both stages, never evaluated)
File names get a source prefix ("def__IMG_1_jpg.rf.<hash>.jpg"); Roboflow's augmented copies keep one
shared source stem, so make_holdouts keeps them on one side of the split.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from phase5 import common as c

CLASSES = ["raw", "half_ripe", "ripe", "rotten"]  # stage-2 classes (Tomato Defects)
TOMATO = "tomato"  # ripeness unknown: detector only

# key, workspace, project, version (None = latest), unified folder, {source class (lower) -> CLASSES | TOMATO | None}
SOURCES = [
    ("def", "universitas-hasanuddin-lq2k8", "tomato-defects-80syu", None, "u_defects",
     {"raw": "raw", "half_ripe": "half_ripe", "ripe": "ripe", "rotten": "rotten"}),
    ("sx", "dai-tran-van", "tomato-sxsc7", None, "u_sxsc7", {"tomato": TOMATO}),
    ("cnv", "canberk-qi0vx", "tomato_count_project", None, "u_conveyor",
     {"tomato": TOMATO, "tomatob": TOMATO, "b": TOMATO, "bn": TOMATO}),
    ("pap", "papapeto", "tomato-v1-2mdez", None, "u_extra",
     {"fully-ripe": "ripe", "semi-ripe": "half_ripe", "unripe": "raw"}),
    ("srt", "tomato-6fguu", "tomato-detection-sorting", None, "u_extra",
     {"fresh": "ripe", "rotten": "rotten", "unripe": "raw"}),
    ("cls", "ai-24uje", "tomato-classification-2kd0n", None, "u_extra",
     {"ripe tomato": "ripe", "unripe tomato": "raw", "healthy ripe": "ripe", "damaged": "rotten", "unripe": "raw"}),
]
UNIFIED = ("u_defects", "u_sxsc7", "u_conveyor", "u_extra")
EVAL_UNIFIED = {"u_defects": CLASSES, "u_sxsc7": [TOMATO], "u_conveyor": [TOMATO], "u_extra": CLASSES + [TOMATO]}


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def _valid_export(d: Path) -> bool:
    return (d / "data.yaml").exists() and any((d / s / "images").is_dir() for s in c.SPLIT_DIRS)


def _api(path: str, key: str) -> dict:
    with urllib.request.urlopen(f"https://api.roboflow.com/{path}?api_key={key}", timeout=120) as r:
        return json.loads(r.read().decode())


def download_one(key: str, ws: str, proj: str, ver: int | None, dest: Path, log=print) -> str:
    """yolov8 export of one Universe project into dest (retries while Roboflow builds the export)."""
    if ver is None:
        info = _api(f"{ws}/{proj}", key)
        ver = max(int(v["id"].split("/")[-1]) for v in info["versions"])
    for attempt in range(10):
        try:
            link = (_api(f"{ws}/{proj}/{ver}/yolov8", key).get("export") or {}).get("link")
            if link:
                tmp = dest.parent / f"{dest.name}.zip"
                urllib.request.urlretrieve(link, tmp)
                if zipfile.is_zipfile(tmp):
                    shutil.rmtree(dest, ignore_errors=True)
                    with zipfile.ZipFile(tmp) as z:
                        z.extractall(dest)
                    tmp.unlink()
                    if _valid_export(dest):
                        return f"{ws}/{proj} v{ver}"
                tmp.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            log(f"   تلاش {attempt + 1}: {type(e).__name__}")
        time.sleep(15)
    raise SystemExit(f"دانلود {ws}/{proj} نسخه‌ی {ver} ممکن نشد. اینترنت/کلید را بررسی کن و دوباره Run all بزن.")


def download_all(raw_dir: str | Path, api_key: str, log=print) -> dict:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    done = {}
    for k, ws, proj, ver, _, _ in SOURCES:
        dest = raw_dir / k
        if _valid_export(dest):
            done[k] = "cached"
            continue
        log(f"دانلود {ws}/{proj} ...")
        done[k] = download_one(api_key, ws, proj, ver, dest, log)
        log(f"   ✅ {done[k]}")
    return done


def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def build_unified(raw_dir: str | Path, out_dir: str | Path, log=print) -> dict:
    """out_dir/u_* from raw_dir/<key>. Class ids follow EVAL_UNIFIED[u]. Idempotent (.ready marker)."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    stats = {}
    for u in UNIFIED:
        dst = out_dir / u
        if (dst / ".ready").exists():
            stats[u] = json.loads((dst / ".ready").read_text(encoding="utf-8"))
            continue
        shutil.rmtree(dst, ignore_errors=True)
        names_out = EVAL_UNIFIED[u]
        st = {"images": 0, "boxes": {n: 0 for n in names_out}, "dropped": 0, "sources": []}
        for k, ws, proj, _, unified, cmap in SOURCES:
            if unified != u:
                continue
            ds = c.load_dataset(raw_dir / k, k)
            names = [_norm(n) for n in ds["names"]]
            lookup = {_norm(a): b for a, b in cmap.items()}
            unknown = [n for n in names if n not in lookup]
            if unknown:
                raise SystemExit(f"{ws}/{proj}: کلاس‌های ناشناخته {unknown}")
            st["sources"].append(f"{ws}/{proj}: {ds['names']}")
            for split, recs in ds["splits"].items():
                sp = "valid" if split == "val" else split
                (dst / sp / "images").mkdir(parents=True, exist_ok=True)
                (dst / sp / "labels").mkdir(parents=True, exist_ok=True)
                for r in recs:
                    img = c.imread(r.image)
                    if img is None:
                        continue
                    hh, ww = img.shape[:2]
                    rows = []
                    for cls, x1, y1, x2, y2 in c.read_yolo_labels(r.label, ww, hh):
                        tgt = lookup.get(names[int(cls)]) if int(cls) < len(names) else None
                        if tgt is None:
                            st["dropped"] += 1
                            continue
                        st["boxes"][tgt] += 1
                        rows.append(f"{names_out.index(tgt)} {(x1 + x2) / 2 / ww:.6f} {(y1 + y2) / 2 / hh:.6f} "
                                    f"{(x2 - x1) / ww:.6f} {(y2 - y1) / hh:.6f}")
                    name = f"{k}__{Path(r.image).name}"
                    shutil.copyfile(r.image, dst / sp / "images" / name)
                    (dst / sp / "labels" / (Path(name).stem + ".txt")).write_text(
                        "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
                    st["images"] += 1
        yaml.safe_dump({"path": str(dst), "train": "train/images", "val": "valid/images", "test": "test/images",
                        "nc": len(names_out), "names": names_out}, open(dst / "data.yaml", "w", encoding="utf-8"),
                       sort_keys=False)
        (dst / ".ready").write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
        stats[u] = st
        log(f"   ✅ {u}: {st['images']} عکس، کادرها {st['boxes']}")
    return stats


def _put(img: str, split: str, out: Path, relabel=None):
    """Hard-link image into out/images/split; write its label (relabel(rows) -> rows) into out/labels/split."""
    d, ld = out / "images" / split, out / "labels" / split
    d.mkdir(parents=True, exist_ok=True)
    ld.mkdir(parents=True, exist_ok=True)
    t = d / Path(img).name
    if not t.exists():
        try:
            os.link(img, t)
        except OSError:
            shutil.copyfile(img, t)
    src = Path(c.img2label_path(img))
    rows = [r for r in src.read_text(encoding="utf-8").splitlines() if r.strip()] if src.exists() else []
    if relabel:
        rows = relabel(rows)
    (ld / (Path(img).stem + ".txt")).write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _hash(path: str) -> int | None:
    im = c.imread(path)
    return None if im is None else c.dhash(im)


def clean_train_images(unified_dir: Path, thr: int = 6, cache: Path | None = None) -> set[str]:
    """Training candidates (Tomato Defects train + u_extra) that are near-duplicates (dHash <= thr) or
    same-source copies of a Tomato Defects valid/test image. Roboflow's own split leaks such pairs;
    they are removed from BOTH stages. Result cached as JSON."""
    if cache and Path(cache).exists():
        return set(json.loads(Path(cache).read_text(encoding="utf-8")))
    ds = c.load_dataset(unified_dir / "u_defects", "defects")
    refs = [r.image for s, recs in ds["splits"].items() if s != "train" for r in recs]
    cands = [r.image for r in ds["splits"].get("train", [])]
    cands += [r.image for r in c.all_records(c.load_dataset(unified_dir / "u_extra", "extra"))]
    ref_stems = {c.source_stem(p) for p in refs}
    rh = [h for h in (_hash(p) for p in refs) if h is not None]
    bad = {p for p in cands if c.source_stem(p) in ref_stems}
    ch = [(p, _hash(p)) for p in cands if p not in bad]
    ch = [(p, h) for p, h in ch if h is not None]
    if rh and ch:
        d = c.hamming_matrix([h for _, h in ch], rh).min(1)
        bad |= {p for (p, _), v in zip(ch, d) if v <= thr}
    # validation images must not be near-copies of test images either (model selection would see test)
    tests = [r.image for r in ds["splits"].get("test", [])]
    vals = [r.image for s, recs in ds["splits"].items() if s not in ("train", "test") for r in recs]
    th = [h for h in (_hash(p) for p in tests) if h is not None]
    vh = [(p, _hash(p)) for p in vals]
    vh = [(p, h) for p, h in vh if h is not None]
    t_stems = {c.source_stem(p) for p in tests}
    bad |= {p for p, _ in vh if c.source_stem(p) in t_stems}
    if th and vh:
        d = c.hamming_matrix([h for _, h in vh], th).min(1)
        bad |= {p for (p, _), v in zip(vh, d) if v <= thr}
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps(sorted(bad)), encoding="utf-8")
    return bad


def _members(unified_dir: Path, splits_dir: Path, bad: set[str]):
    """(image, split) pairs of the detector data; holdout images and their near-duplicates never included."""
    ds = c.load_dataset(unified_dir / "u_defects", "defects")
    for split, recs in ds["splits"].items():
        for r in recs:
            if split != "test" and r.image in bad:
                continue
            yield r.image, {"train": "train", "test": "test"}.get(split, "val")
    for key in ("sxsc7", "conveyor"):
        for img in c.read_list(splits_dir / f"{key}_train.txt"):
            yield img, "train"
    for r in c.all_records(c.load_dataset(unified_dir / "u_extra", "extra")):
        if r.image not in bad:  # extra datasets' own valid/test splits widen the (small) validation set
            yield r.image, "train" if r.split == "train" else "val"


def build_detector_set(unified_dir, splits_dir, out, bad: set[str] = frozenset()) -> dict:
    """Stage-1 data: every box -> class 0 'tomato'. val / test = Tomato Defects valid / test."""
    unified_dir, splits_dir, out = Path(unified_dir), Path(splits_dir), Path(out)
    if (out / ".ready").exists():
        return json.loads((out / ".ready").read_text(encoding="utf-8"))
    shutil.rmtree(out, ignore_errors=True)
    for img, split in _members(unified_dir, splits_dir, bad):
        _put(img, split, out, lambda rows: ["0 " + " ".join(r.split()[1:]) for r in rows])
    counts = {s: len(list((out / "images" / s).glob("*"))) for s in ("train", "val", "test")}
    counts["removed_near_duplicates"] = len(bad)
    yaml.safe_dump({"path": str(out), "train": "images/train", "val": "images/val", "test": "images/test",
                    "names": {0: TOMATO}}, open(out / "data.yaml", "w", encoding="utf-8"), sort_keys=False)
    (out / ".ready").write_text(json.dumps(counts), encoding="utf-8")
    return counts


def crop_box(img: np.ndarray, box, pad: float = 0.1, size: int | None = None) -> np.ndarray | None:
    """Square-ish crop with `pad` context on each side (same rule at train and inference time)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return None
    xa, ya = int(max(0, x1 - pad * bw)), int(max(0, y1 - pad * bh))
    xb, yb = int(min(w, x2 + pad * bw)), int(min(h, y2 + pad * bh))
    crop = img[ya:yb, xa:xb]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA) if size else crop


def build_classifier_set(unified_dir, out, size: int = 160, bad: set[str] = frozenset()) -> dict:
    """Stage-2 data: out/{train,val,test}/<class>/*.jpg crops. train = Tomato Defects train + u_extra;
    val / test = Tomato Defects valid / test (the same images as the detector's val / test)."""
    unified_dir, out = Path(unified_dir), Path(out)
    if (out / ".ready").exists():
        return json.loads((out / ".ready").read_text(encoding="utf-8"))
    shutil.rmtree(out, ignore_errors=True)
    counts = {s: {n: 0 for n in CLASSES} for s in ("train", "val", "test")}
    jobs = [(r, {"train": "train", "test": "test"}.get(s, "val"), CLASSES)
            for s, recs in c.load_dataset(unified_dir / "u_defects", "d")["splits"].items() for r in recs]
    jobs += [(r, "train" if r.split == "train" else "val", CLASSES + [TOMATO])
             for r in c.all_records(c.load_dataset(unified_dir / "u_extra", "e"))]
    for r, split, names in jobs:
        if r.image in bad and not (split == "test" and r.dataset == "d"):
            continue
        img = c.imread(r.image)
        if img is None:
            continue
        hh, ww = img.shape[:2]
        for j, (cls, x1, y1, x2, y2) in enumerate(c.read_yolo_labels(r.label, ww, hh)):
            n = names[int(cls)]
            if n not in CLASSES:
                continue
            crop = crop_box(img, (x1, y1, x2, y2), size=size)
            if crop is None:
                continue
            c.imwrite(out / split / n / f"{Path(r.image).stem}_{j}.jpg", crop, quality=95)
            counts[split][n] += 1
            if split == "train" and n == "rotten":  # rare class: 3 extra rotated / flipped copies
                for t, aug in enumerate((cv2.flip(crop, 1), cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE),
                                         cv2.rotate(crop, cv2.ROTATE_180))):
                    c.imwrite(out / split / n / f"{Path(r.image).stem}_{j}_a{t}.jpg", aug, quality=95)
                    counts[split][n] += 1
    (out / ".ready").write_text(json.dumps(counts), encoding="utf-8")
    return counts
