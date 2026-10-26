"""Ultralytics YOLO inference for Phase 5 with a JSON prediction cache.

Images are always decoded with `common.imread` (unicode-safe) and handed to Ultralytics as BGR numpy
arrays, so Ultralytics never touches the (Persian) file paths itself.
Output per image: float64 (M, 6) = [x1, y1, x2, y2, conf, cls] in the decoded image's pixel frame.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import warnings
from pathlib import Path

import numpy as np

from phase5 import common
from phase5.common import EMPTY_PRED, ImageRecord

LOG = logging.getLogger(__name__)
_BAD_TASKS = {"classify", "pose", "obb"}
CACHE_VERSION = 2


def _load_model(model_path: str):
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    task = getattr(model, "task", None)
    if task in _BAD_TASKS:
        raise ValueError(f"model task '{task}' is not supported (need detect or segment): {model_path}")
    return model


def _model_imgsz(model):
    imgsz = None
    ckpt = getattr(model, "ckpt", None)
    if isinstance(ckpt, dict):
        ta = ckpt.get("train_args") or {}
        if isinstance(ta, dict):
            imgsz = ta.get("imgsz")
    if not imgsz:
        imgsz = (getattr(model, "overrides", None) or {}).get("imgsz")
    if isinstance(imgsz, (list, tuple)):
        imgsz = max(int(v) for v in imgsz) if imgsz else None
    return int(imgsz) if imgsz else None


def _info_from_model(model, model_path: str) -> dict:
    names = getattr(model, "names", None) or {}
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    names = {int(k): str(v) for k, v in names.items()}
    return {
        "names": names,
        "nc": len(names),
        "task": str(getattr(model, "task", "detect")),
        "imgsz": _model_imgsz(model),
        "sha1": common.file_sha1(model_path),
    }


def model_info(model_path: str) -> dict:
    """{"names": {int: str}, "nc", "task", "imgsz" (train imgsz or None), "sha1" (of the model / yaml file)}."""
    model = _load_model(model_path)
    return _info_from_model(model, model_path)


def _images_sha1(paths: list[str]) -> str:
    h = hashlib.sha1()
    for p in sorted(paths):
        try:
            size = os.path.getsize(p)
        except OSError:
            size = -1
        h.update(f"{p}\t{size}\n".encode("utf-8"))
    return h.hexdigest()


def _read_cache(cache_path: str, meta: dict, images: list[str]):
    p = Path(cache_path)
    if not p.is_file():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        LOG.warning("prediction cache %s unreadable (%s); recomputing", p, e)
        return None
    if not isinstance(data, dict) or data.get("meta") != meta:
        LOG.info("prediction cache %s metadata mismatch; recomputing", p)
        return None
    preds = data.get("preds") or {}
    if set(preds) != set(images):
        return None
    out = {}
    for k in images:
        a = np.asarray(preds[k], dtype=np.float64)
        out[k] = a.reshape(-1, 6) if a.size else EMPTY_PRED.copy()
    return out


def _result_to_array(r) -> np.ndarray:
    b = getattr(r, "boxes", None)
    if b is None or len(b) == 0:
        return EMPTY_PRED.copy()
    xyxy = b.xyxy.detach().cpu().numpy().astype(np.float64).reshape(-1, 4)
    conf = b.conf.detach().cpu().numpy().astype(np.float64).reshape(-1, 1)
    cls = b.cls.detach().cpu().numpy().astype(np.float64).reshape(-1, 1)
    return np.concatenate([xyxy, conf, cls], axis=1)


def run_predictions(model_path: str, records: list[ImageRecord], *, imgsz=640, conf=0.001, iou=0.7,
                    agnostic_nms=True, max_det=300, batch=16, device="", half=False,
                    cache_path: str | None = None) -> dict[str, np.ndarray]:
    """Predict every record once; returns {record.image: float64 (M, 6)}. Uses/refreshes `cache_path`."""
    images = list(dict.fromkeys(r.image for r in records))
    is_yaml = str(model_path).lower().endswith((".yaml", ".yml"))
    if cache_path and str(model_path).lower().endswith((".yaml", ".yml")):
        warnings.warn(f"model {model_path} is a .yaml (random weights per load); prediction cache disabled")
        LOG.warning("prediction cache disabled for yaml model %s", model_path)
        cache_path = None
    meta = {
        "version": CACHE_VERSION,
        "model_sha1": None if is_yaml else common.file_sha1(model_path),
        "imgsz": int(imgsz),
        "conf": float(conf),
        "iou": float(iou),
        "agnostic_nms": bool(agnostic_nms),
        "max_det": int(max_det),
        "half": bool(half),
        "device": str(device),
        "images_sha1": _images_sha1(images),
    }
    if cache_path:
        cached = _read_cache(cache_path, meta, images)
        if cached is not None:
            LOG.info("using prediction cache %s (%d images)", cache_path, len(cached))
            return cached

    model = _load_model(model_path)
    out: dict[str, np.ndarray] = {}
    batch = max(1, int(batch))
    for i in range(0, len(images), batch):
        chunk = images[i : i + batch]
        arrs, keys = [], []
        for p in chunk:
            img = common.imread(p)
            if img is None:
                warnings.warn(f"unreadable image, empty prediction used: {p}")
                LOG.warning("unreadable image: %s", p)
                out[p] = EMPTY_PRED.copy()
                continue
            arrs.append(img)
            keys.append(p)
        if not arrs:
            continue
        results = model.predict(source=arrs, imgsz=imgsz, conf=conf, iou=iou, agnostic_nms=agnostic_nms,
                                max_det=max_det, device=(None if device == "" else device), half=half,
                                verbose=False, stream=False)
        if len(results) != len(keys):
            raise RuntimeError(f"Ultralytics returned {len(results)} results for {len(keys)} images")
        for k, r in zip(keys, results):
            out[k] = _result_to_array(r)
    out = {k: out[k] for k in images}
    if cache_path:
        common.dump_json(cache_path, {"meta": meta, "preds": {k: v.tolist() for k, v in out.items()}})
    return out


def classify_detections(cls_model_path: str, records: list[ImageRecord], dets: dict[str, np.ndarray], *,
                        min_conf: float = 0.01, imgsz: int = 160, batch: int = 128, device="",
                        cache_path: str | None = None) -> dict[str, np.ndarray]:
    """Stage 2: classify every detected box (conf >= min_conf) with a YOLO classify model.

    Returns {image: (M, 6)} with cls = top-1 class of the crop and conf = det_conf * p(top-1), the usual
    two-stage score, so 4-class AP can be computed. Crops use prepare_data.crop_box (same rule as training).
    """
    from phase5.prepare_data import crop_box

    images = list(dict.fromkeys(r.image for r in records))
    meta = {"version": CACHE_VERSION, "cls_sha1": common.file_sha1(cls_model_path), "min_conf": float(min_conf),
            "imgsz": int(imgsz), "images_sha1": _images_sha1(images),
            "dets_sha1": hashlib.sha1(b"".join(np.ascontiguousarray(dets.get(i, common.EMPTY_PRED)).tobytes()
                                                for i in images)).hexdigest()}
    if cache_path:
        cached = _read_cache(cache_path, meta, images)
        if cached is not None:
            return cached
    from ultralytics import YOLO

    model = YOLO(cls_model_path)
    if model.task != "classify":
        raise ValueError(f"{cls_model_path} is a '{model.task}' model, expected a classify model")
    out: dict[str, np.ndarray] = {}
    for im_path in images:
        d = np.asarray(dets.get(im_path, common.EMPTY_PRED), dtype=np.float64).reshape(-1, 6)
        d = d[d[:, 4] >= min_conf]
        img = common.imread(im_path) if len(d) else None
        if img is None or not len(d):
            out[im_path] = common.EMPTY_PRED.copy()
            continue
        crops, keep = [], []
        for i, b in enumerate(d):
            cr = crop_box(img, b[:4], size=imgsz)
            if cr is not None:
                crops.append(cr)
                keep.append(i)
        rows = []
        for s in range(0, len(crops), batch):
            res = model.predict(crops[s:s + batch], imgsz=imgsz, verbose=False,
                                device=None if device == "" else device)
            for k, r in enumerate(res):
                p = r.probs
                b = d[keep[s + k]]
                rows.append([b[0], b[1], b[2], b[3], b[4] * float(p.top1conf), int(p.top1)])
        out[im_path] = np.asarray(rows, dtype=np.float64).reshape(-1, 6) if rows else common.EMPTY_PRED.copy()
    if cache_path:
        common.dump_json(cache_path, {"meta": meta, "preds": {k: v.tolist() for k, v in out.items()}})
    return out
