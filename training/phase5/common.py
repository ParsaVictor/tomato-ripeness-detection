"""Shared helpers for Phase 5 evaluation: config, dataset IO, YOLO labels, hashing, size buckets.

The project lives under a non-ASCII (Persian) folder, so:
  * images are always read/written through numpy buffers (cv2.imread/imwrite fail on such paths on Windows),
  * every text file is opened with encoding="utf-8".

Box convention everywhere in phase5: absolute pixel xyxy in the image *as decoded by imread()*
(EXIF orientation applied, same as Ultralytics).
  GT array   : float64 (N, 5)  -> [cls, x1, y1, x2, y2]
  pred array : float64 (M, 6)  -> [x1, y1, x2, y2, conf, cls]
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_DIRS = ("train", "valid", "val", "test")

# COCO size definition, in squared pixels: small < 32^2 <= medium < 96^2 <= large.
SMALL_MAX_AREA = 32.0**2
MEDIUM_MAX_AREA = 96.0**2
SIZE_BUCKETS = ("small", "medium", "large")
# Area ranges used for size-restricted AP. Bounds are inclusive on both ends, exactly like pycocotools.
AREA_RANGES = {
    "all": (0.0, 1e10),
    "small": (0.0, SMALL_MAX_AREA),
    "medium": (SMALL_MAX_AREA, MEDIUM_MAX_AREA),
    "large": (MEDIUM_MAX_AREA, 1e10),
}

EMPTY_GT = np.zeros((0, 5), dtype=np.float64)
EMPTY_PRED = np.zeros((0, 6), dtype=np.float64)

DEFAULT_CONFIG = {
    "model": "runs/detect/train/weights/best.pt",
    "imgsz": 640,
    "device": "",
    "batch": 16,
    "half": False,
    "pred_conf": 0.001,  # low threshold used to build the PR curve / mAP
    "op_conf": 0.25,  # operating point for precision / recall / failure analysis
    "nms_iou": 0.7,
    "agnostic_nms": True,
    "max_det": 300,
    "iou_match": 0.5,
    "recall_gate": 0.85,
    "seed": 0,
    "out_dir": "reports/phase5",
    "splits_dir": "splits",
    "cache_dir": "reports/phase5/pred_cache",
    "near_dup_hamming": 6,
    "failure": {
        "max_examples": 8,
        "blur_rel": 0.5,
        "contrast_de": 12.0,
        "occl_ioa": 0.2,
        "border_px": 2,
        "bg_iou": 0.1,
    },
    "datasets": {
        "defects": {"root": "datasets/tomato-defects", "eval_split": "test", "tomato_classes": "all"},
        "sxsc7": {
            "root": "datasets/tomato-sxsc7",
            "tomato_classes": "all",
            "holdout": {"use_existing_split": "test", "min_images": 30, "frac": 0.2},
        },
        "conveyor": {
            "root": "datasets/tomato_count_project",
            "tomato_classes": "all",
            "holdout": {"n_images": 30},
        },
    },
}

EVAL_SETS = ("defects_test", "sxsc7_holdout", "conveyor_holdout")


@dataclass
class ImageRecord:
    image: str  # absolute path to the image
    label: str | None  # absolute path to the YOLO label .txt (missing file => image has no objects)
    dataset: str  # dataset key from the config, e.g. "conveyor"
    split: str  # split folder in the export: train / valid / val / test / all


# ----------------------------------------------------------------------------- config / paths


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def resolve_project_path(p: str | Path, base: Path | None = None) -> Path:
    """Absolute paths are kept; relative ones are resolved against `base` (default: project root)."""
    p = Path(p)
    return p if p.is_absolute() else ((base or PROJECT_ROOT) / p).resolve()


def to_project_rel(p: str | Path) -> str:
    """Posix path relative to the project root when possible (used in manifests), else absolute posix."""
    p = Path(p).resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def load_config(path: str | Path | None = None) -> dict:
    """Load the YAML config on top of DEFAULT_CONFIG. Relative paths resolve against the config's folder.

    Path-valued keys (model, out_dir, splits_dir, cache_dir, datasets.*.root) are returned as absolute str.
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    base = PROJECT_ROOT
    if path is not None:
        path = Path(path).resolve()
        base = path.parent
        with open(path, encoding="utf-8") as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    for k in ("model", "out_dir", "splits_dir", "cache_dir"):
        cfg[k] = str(resolve_project_path(cfg[k], base))
    for d in cfg["datasets"].values():
        d["root"] = str(resolve_project_path(d["root"], base))
    cfg["_config_dir"] = str(base)
    return cfg


# ----------------------------------------------------------------------------- image / text IO


def imread(path: str | Path) -> np.ndarray | None:
    """Unicode-safe BGR read (same approach as ultralytics.utils.patches.imread). Returns None on failure."""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite(path: str | Path, img: np.ndarray, quality: int = 92) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    ok, enc = cv2.imencode(ext, img, params)
    if ok:
        enc.tofile(str(path))
    return bool(ok)


def write_list(path: str | Path, paths) -> None:
    """One absolute path per line, UTF-8 (Ultralytics 8.4 reads list files as UTF-8)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(str(Path(p).resolve()) for p in paths) + "\n")


def read_list(path: str | Path) -> list[str]:
    """Read a list file; './x' entries are relative to the file's folder (Ultralytics rule)."""
    path = Path(path)
    out = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            out.append(str((path.parent / line).resolve() if (line.startswith("./") or not p.is_absolute()) else p))
    return out


def dump_json(path: str | Path, obj) -> None:
    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"not JSON serializable: {type(o)}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=default)


def file_sha1(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- datasets


def img2label_path(img_path: str | Path) -> str:
    """Ultralytics rule: replace the LAST '/images/' path component with '/labels/' and the suffix with .txt."""
    parts = list(Path(img_path).parts)
    idx = max((i for i, s in enumerate(parts[:-1]) if s == "images"), default=None)
    if idx is None:
        return str(Path(img_path).with_suffix(".txt"))
    parts[idx] = "labels"
    return str(Path(*parts).with_suffix(".txt"))


def load_names(root: str | Path) -> list[str]:
    """Class names from <root>/data.yaml (list or {id: name} dict). Empty list when missing."""
    for fn in ("data.yaml", "data.yml"):
        p = Path(root) / fn
        if p.exists():
            with open(p, encoding="utf-8") as f:
                d = yaml.safe_load(f) or {}
            names = d.get("names", [])
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names, key=int)]
            return [str(n) for n in names]
    return []


def list_images(folder: str | Path) -> list[str]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(str(p.resolve()) for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def load_dataset(root: str | Path, key: str) -> dict:
    """Scan a Roboflow / Ultralytics export: <root>/<split>/images/*.jpg (+ <root>/<split>/labels/*.txt).

    Falls back to <root>/images when there are no split folders (split name "all").
    Returns {"key", "root", "names": [str], "splits": {split: [ImageRecord]}}.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset '{key}' not found at {root}")
    splits: dict[str, list[ImageRecord]] = {}
    for s in SPLIT_DIRS:
        imgs = list_images(root / s / "images")
        if imgs:
            splits[s] = [ImageRecord(p, img2label_path(p), key, s) for p in imgs]
    if not splits:
        imgs = list_images(root / "images")
        if imgs:
            splits["all"] = [ImageRecord(p, img2label_path(p), key, "all") for p in imgs]
    if not splits:
        raise FileNotFoundError(f"dataset '{key}': no images under {root}/<split>/images or {root}/images")
    return {"key": key, "root": str(root), "names": load_names(root), "splits": splits}


def all_records(ds: dict) -> list[ImageRecord]:
    return [r for s in sorted(ds["splits"]) for r in ds["splits"][s]]


def read_yolo_labels(label_path: str | Path | None, w: int, h: int) -> np.ndarray:
    """Parse a YOLO label file into absolute-pixel GT boxes (N, 5) = [cls, x1, y1, x2, y2].

    Row formats: 'cls cx cy w h' (box) or 'cls x1 y1 x2 y2 x3 y3 ...' (polygon, >= 3 points, converted
    to its bounding box). Malformed rows are skipped with a warning. Boxes are clipped to the image and
    zero-area boxes dropped. Missing / empty file -> (0, 5).
    """
    if label_path is None or not Path(label_path).is_file():
        return EMPTY_GT.copy()
    rows = []
    with open(label_path, encoding="utf-8-sig") as f:
        for ln, line in enumerate(f, 1):
            vals = line.split()
            if not vals:
                continue
            try:
                nums = [float(v) for v in vals]
            except ValueError:
                warnings.warn(f"{label_path}:{ln}: non-numeric row skipped")
                continue
            cls, coords = nums[0], nums[1:]
            if len(coords) == 4:
                cx, cy, bw, bh = coords
                x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                xs, ys = coords[0::2], coords[1::2]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            else:
                warnings.warn(f"{label_path}:{ln}: unsupported row with {len(vals)} values skipped")
                continue
            rows.append([int(cls), x1 * w, y1 * h, x2 * w, y2 * h])
    if not rows:
        return EMPTY_GT.copy()
    a = np.asarray(rows, dtype=np.float64)
    a[:, [1, 3]] = a[:, [1, 3]].clip(0, w)
    a[:, [2, 4]] = a[:, [2, 4]].clip(0, h)
    keep = (a[:, 3] - a[:, 1] > 0) & (a[:, 4] - a[:, 2] > 0)
    return a[keep]


def tomato_class_ids(names: list[str], tomato_classes) -> set[int] | None:
    """Which label class ids count as 'tomato'. 'all' / None -> None (every class counts)."""
    if tomato_classes in (None, "all"):
        return None
    wanted = {str(n).strip().lower() for n in tomato_classes}
    ids = {i for i, n in enumerate(names) if n.strip().lower() in wanted}
    missing = wanted - {names[i].strip().lower() for i in ids}
    if missing:
        raise ValueError(f"tomato_classes {sorted(missing)} not in dataset names {names}")
    return ids


def to_agnostic_gt(gt: np.ndarray, keep_ids: set[int] | None) -> np.ndarray:
    """Keep only 'tomato' rows and set their class to 0."""
    if keep_ids is not None and len(gt):
        gt = gt[np.isin(gt[:, 0].astype(int), sorted(keep_ids))]
    gt = gt.copy()
    gt[:, 0] = 0
    return gt


def to_agnostic_pred(pred: np.ndarray) -> np.ndarray:
    pred = pred.copy()
    if len(pred):
        pred[:, 5] = 0
    return pred


# ----------------------------------------------------------------------------- grouping helpers

_RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]{16,}$", re.IGNORECASE)
_VIDEO_FRAME = re.compile(r"^(?P<video>.+?_(?:mp4|avi|mov|mkv|webm|m4v))[-_](?P<frame>\d+)", re.IGNORECASE)


def source_stem(path: str | Path) -> str:
    """Original image name before Roboflow export: 'IMG_1_jpg.rf.<hash>.jpg' -> 'img_1_jpg'.

    Augmented copies of one source image share this stem, so they must always land on the same side
    of a train / holdout split.
    """
    return _RF_SUFFIX.sub("", Path(path).stem).lower()


def video_frame(path: str | Path) -> tuple[str, int] | None:
    """('clip1_mp4', 12) for Roboflow video-frame names like 'clip1_mp4-0012_jpg.rf.<hash>.jpg', else None."""
    m = _VIDEO_FRAME.match(source_stem(path))
    return (m["video"], int(m["frame"])) if m else None


def dhash(img: np.ndarray, size: int = 8) -> int:
    """64-bit difference hash of a BGR or gray image (near-duplicate detection)."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def hamming_matrix(a: list[int] | np.ndarray, b: list[int] | np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Pairwise Hamming distances (len(a), len(b)) between 64-bit hashes, uint8, computed in chunks."""
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    bb = np.unpackbits(b.view(np.uint8).reshape(-1, 8), axis=1).astype(np.int32)  # (nb, 64)
    out = np.empty((len(a), len(b)), dtype=np.uint8)
    nb1 = bb.sum(1)
    for i in range(0, len(a), chunk):
        aa = np.unpackbits(a[i : i + chunk].view(np.uint8).reshape(-1, 8), axis=1).astype(np.int32)
        out[i : i + chunk] = (aa.sum(1)[:, None] + nb1[None, :] - 2 * aa @ bb.T).astype(np.uint8)
    return out


# ----------------------------------------------------------------------------- size helpers


def box_areas(boxes_xyxy: np.ndarray) -> np.ndarray:
    if len(boxes_xyxy) == 0:
        return np.zeros(0)
    return (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clip(0) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).clip(0)


def size_bucket(area: float) -> str:
    """Single bucket per object: small < 32^2 <= medium < 96^2 <= large."""
    return "small" if area < SMALL_MAX_AREA else ("medium" if area < MEDIUM_MAX_AREA else "large")


def letterbox_ratio(h: int, w: int, imgsz: int) -> float:
    """Resize ratio Ultralytics' LetterBox applies before the network (scaleup allowed, as in predict)."""
    return min(imgsz / h, imgsz / w)
