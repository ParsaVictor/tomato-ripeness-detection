"""Offline test for phase5.predict (random-weight yolov8n built from yaml; no downloads).

Run: py -3.10 -m phase5.tests.test_predict
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

from phase5 import common, predict
from phase5.common import ImageRecord


def _make_model(tmp: Path) -> str:
    from ultralytics import YOLO

    m = YOLO("yolov8n.yaml")  # built locally, random weights
    pt = tmp / "مدل.pt"
    try:
        m.save(str(pt))
        if pt.is_file():
            YOLO(str(pt))  # must be loadable
            return str(pt)
    except Exception as e:  # pragma: no cover
        warnings.warn(f"model.save failed ({e}); using the yaml path")
    return "yolov8n.yaml"


def test_predict():
    tmp = Path(tempfile.mkdtemp(prefix="تست_پیش‌بینی_"))
    try:
        model = _make_model(tmp)
        info = predict.model_info(model)
        assert info["task"] == "detect" and info["nc"] == len(info["names"]) and info["nc"] > 0, info
        assert all(isinstance(k, int) for k in info["names"])
        assert isinstance(info["sha1"], str) and len(info["sha1"]) == 40
        assert info["imgsz"] is None or isinstance(info["imgsz"], int)

        rng = np.random.default_rng(0)
        recs = []
        for i, (h, w) in enumerate([(240, 320), (300, 200), (128, 128)]):
            p = tmp / f"تصویر_{i}.jpg"
            common.imwrite(p, rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
            recs.append(ImageRecord(str(p), None, "conveyor", "all"))
        bad = tmp / "خراب.jpg"
        bad.write_bytes(b"not an image")
        recs.append(ImageRecord(str(bad), None, "conveyor", "all"))

        cache = tmp / "کش" / "pred.json"
        kw = dict(imgsz=128, conf=0.0001, iou=0.7, batch=2, cache_path=str(cache))
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            out = predict.run_predictions(model, recs, **kw)
        assert any("unreadable" in str(w.message) for w in wl)
        assert list(out) == [r.image for r in recs]
        for k, v in out.items():
            assert v.dtype == np.float64 and v.ndim == 2 and v.shape[1] == 6, (k, v.shape)
            assert len(v) <= 300
        assert len(out[str(bad)]) == 0
        n_total = sum(len(v) for v in out.values())
        assert n_total > 0, "random model at conf 1e-4 should produce some boxes"
        for r in recs[:3]:
            v = out[r.image]
            if len(v):
                h, w = common.imread(r.image).shape[:2]
                assert (v[:, 4] >= 1e-4 - 1e-9).all() and (v[:, 4] <= 1).all()
                assert (v[:, [0, 2]] >= -1e-3).all() and (v[:, [0, 2]] <= w + 1e-3).all()
                assert (v[:, [1, 3]] >= -1e-3).all() and (v[:, [1, 3]] <= h + 1e-3).all()
        assert cache.is_file()

        # cache reuse: tamper the cached predictions; an identical call must return the tampered values
        with open(cache, encoding="utf-8") as f:
            data = json.load(f)
        k0 = recs[0].image
        data["preds"][k0] = [[1.0, 2.0, 3.0, 4.0, 0.5, 0.0]]
        common.dump_json(cache, data)
        out2 = predict.run_predictions(model, recs, **kw)
        assert np.array_equal(out2[k0], np.array([[1.0, 2.0, 3.0, 4.0, 0.5, 0.0]]))
        assert out2[k0].dtype == np.float64

        # invalidation: any param change -> recomputed (tampered row disappears)
        for change in (dict(conf=0.001), dict(iou=0.6), dict(agnostic_nms=False), dict(max_det=50), dict(imgsz=160)):
            common.dump_json(cache, data)
            out3 = predict.run_predictions(model, recs, **{**kw, **change})
            assert not np.array_equal(out3[k0], out2[k0]), change
            with open(cache, encoding="utf-8") as f:
                meta = json.load(f)["meta"]
            for kk, vv in change.items():
                assert meta[kk] == vv, (kk, meta)
        # image set change -> invalidated
        common.dump_json(cache, data)
        out4 = predict.run_predictions(model, recs[:2], **kw)
        assert list(out4) == [r.image for r in recs[:2]]
        assert not np.array_equal(out4[k0], out2[k0])
        # half is part of the cache key
        common.dump_json(cache, data)
        out6 = predict.run_predictions(model, recs, **{**kw, "half": True})
        assert not np.array_equal(out6[k0], out2[k0])
        with open(cache, encoding="utf-8") as f:
            assert json.load(f)["meta"]["half"] is True
        # a .yaml model has new random weights every load -> caching disabled (warns, no file)
        ycache = tmp / "کش" / "yaml.json"
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            predict.run_predictions("yolov8n.yaml", recs[:1], **{**kw, "cache_path": str(ycache)})
        assert any("cache disabled" in str(w.message) for w in wl)
        assert not ycache.exists()
        # deterministic: same as first run
        out5 = predict.run_predictions(model, recs, **{**kw, "cache_path": None})
        assert np.allclose(out5[k0], out[k0], atol=1e-4)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    test_predict()
    print("OK test_predict")
    return 0


if __name__ == "__main__":
    sys.exit(main())
