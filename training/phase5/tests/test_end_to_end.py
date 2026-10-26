"""End-to-end test of phase5.evaluate_class_agnostic on synthetic Roboflow-style datasets (no downloads).

    py -3.10 -m phase5.tests.test_end_to_end
"""
from __future__ import annotations

import codecs
import csv
import json
import math
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from phase5 import common

W, H = 320, 240
DEFECT_NAMES = ["healthy", "cracked", "rotten", "spotted"]


def _belt(rng):
    base = rng.integers(40, 120)
    img = np.full((H, W, 3), base, np.uint8)
    noise = rng.integers(-40, 40, (H, W, 1))
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    for x in range(0, W, int(rng.integers(15, 40))):  # belt stripes -> distinct dHash per image
        cv2.line(img, (x, 0), (x + int(rng.integers(-30, 30)), H), (int(rng.integers(0, 255)),) * 3, 2)
    return img


def _draw(img, cx, cy, r, color):
    cv2.circle(img, (int(cx), int(cy)), int(r), color, -1)
    return [cx - r, cy - r, cx + r, cy + r]


def _save(root: Path, split: str, name: str, img, boxes):
    common.imwrite(root / split / "images" / f"{name}.jpg", img)
    lp = root / split / "labels" / f"{name}.txt"
    lp.parent.mkdir(parents=True, exist_ok=True)
    with open(lp, "w", encoding="utf-8") as f:
        for c, (x1, y1, x2, y2) in boxes:
            x1, x2 = max(0, x1), min(W, x2)
            y1, y2 = max(0, y1), min(H, y2)
            f.write(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")


def _rf(i, rng):
    return f"_jpg.rf.{rng.integers(0, 16**8):08x}{i:08x}"


def make_data(base: Path, rng) -> dict:
    roots = {k: base / k for k in ("tomato-defects", "tomato-sxsc7", "tomato_count_project")}
    # (a) defects: 4 classes, train + test
    for split, n in (("train", 8), ("test", 12)):
        for i in range(n):
            img, boxes = _belt(rng), []
            for j in range(3):
                c = int(rng.integers(0, 4))
                r = float(rng.choice([12, 30, 60]))
                cx, cy = 70 + 90 * j, float(rng.uniform(70, 170))
                boxes.append((c, _draw(img, cx, cy, r, (30, 30 + 50 * c, 220))))
            _save(roots["tomato-defects"], split, f"def_{split}_{i}{_rf(i, rng)}", img, boxes)
    # (b) sxsc7: train/valid/test, one class
    for split, n in (("train", 10), ("valid", 4), ("test", 12)):
        for i in range(n):
            img, boxes = _belt(rng), []
            for j in range(2):
                r = float(rng.choice([12, 35, 55]))
                boxes.append((0, _draw(img, 80 + 150 * j, float(rng.uniform(70, 170)), r, (40, 40, 210))))
            _save(roots["tomato-sxsc7"], split, f"sx_{split}_{i}{_rf(i, rng)}", img, boxes)
    # (c) conveyor: 100 images, small / blurred / occluded tomatoes
    for i in range(100):
        split = "train" if i < 80 else ("valid" if i < 90 else "test")
        img, boxes = _belt(rng), []
        boxes.append((0, _draw(img, 60, 60, 9, (30, 30, 200))))  # small (18x18)
        boxes.append((0, _draw(img, 260, 60, 10, (30, 30, 200))))  # small
        boxes.append((0, _draw(img, 90, 160, 52, (20, 20, 230))))  # large (104x104)
        boxes.append((0, _draw(img, 140, 175, 40, (25, 25, 190))))  # medium, occludes the large one
        if i % 3 == 0:
            boxes.append((0, _draw(img, 260, 170, 30, (30, 30, 200))))
            img = cv2.filter2D(img, -1, np.full((1, 15), 1 / 15.0))  # motion blur
        _save(roots["tomato_count_project"], split, f"conv_{i:03d}{_rf(i, rng)}", img, boxes)
    for k, names in (("tomato-defects", DEFECT_NAMES), ("tomato-sxsc7", ["tomato"]),
                     ("tomato_count_project", ["tomato"])):
        with open(roots[k] / "data.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump({"names": names, "nc": len(names)}, f)
    return roots


def make_fake(drop_small: float, fp_rate: float, seed: int):
    def fake(records, *, agnostic_nms: bool):
        rng = np.random.default_rng(seed)
        out = {}
        for r in records:
            img = common.imread(r.image)
            h, w = img.shape[:2]
            gt = common.read_yolo_labels(r.label, w, h)
            rows = []
            for c, x1, y1, x2, y2 in gt:
                small = (x2 - x1) * (y2 - y1) < common.SMALL_MAX_AREA
                if small and rng.random() < drop_small:
                    if rng.random() < 0.5:  # sometimes a low-confidence detection below op_conf
                        rows.append([x1, y1, x2, y2, 0.05, c])
                    continue
                j = rng.normal(0, 0.8, 4)
                rows.append([x1 + j[0], y1 + j[1], x2 + j[2], y2 + j[3], float(rng.uniform(0.6, 0.99)), c])
            if rng.random() < fp_rate:
                rows.append([5, h - 40, 35, h - 10, 0.5, 0])  # belt FP
            out[r.image] = np.asarray(rows, np.float64).reshape(-1, 6)
        return out

    fake.names = dict(enumerate(DEFECT_NAMES))
    return fake


def _check_numbers(res):
    for k in common.EVAL_SETS:
        s = res["sets"][k]["summary"]
        for key in ("ap50", "ap50_95", "precision", "recall", "f1", "max_recall"):
            v = float(s[key])
            assert 0.0 <= v <= 1.0, (k, key, v)
        assert s["ap50_95"] <= s["ap50"] + 1e-9, (k, s)
        assert s["recall"] <= s["max_recall"] + 1e-9


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="تست_ارزیابی_"))
    try:
        rng = np.random.default_rng(0)
        roots = make_data(tmp / "داده", rng)
        cfgd = {
            "model": str(tmp / "مدل" / "best.pt"),
            "out_dir": str(tmp / "گزارش"),
            "splits_dir": str(tmp / "splits"),
            "cache_dir": str(tmp / "گزارش" / "cache"),
            "datasets": {
                "defects": {"root": str(roots["tomato-defects"]), "eval_split": "test"},
                "sxsc7": {"root": str(roots["tomato-sxsc7"]),
                          "holdout": {"use_existing_split": "test", "min_images": 10, "frac": 0.2}},
                "conveyor": {"root": str(roots["tomato_count_project"]), "holdout": {"n_images": 30}},
            },
        }
        cp = tmp / "config.yaml"
        with open(cp, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfgd, f, allow_unicode=True)
        cfg = common.load_config(cp)

        from phase5 import evaluate_class_agnostic as ev

        # manifest missing -> clear error, nothing auto-created
        try:
            ev.run(cfg, predict_fn=make_fake(0, 0, 1))
            raise AssertionError("expected ManifestMissingError")
        except ev.ManifestMissingError as e:
            assert "make_holdouts" in str(e)
        assert ev.main(["--config", str(cp)]) == 2
        # regression: explicit --config that does not exist -> exit 2, no silent fallback to defaults
        assert ev.main(["--config", str(tmp / "nonexist_تست.yaml")]) == 2

        from phase5 import make_holdouts

        man = make_holdouts.build_holdouts(cfg)
        conv_hold = make_holdouts.load_holdout_records(cfg, "conveyor")
        assert len(conv_hold) >= 30, len(conv_hold)
        train_list = set(common.read_list(Path(cfg["splits_dir"]) / "conveyor_train.txt"))
        assert not (train_list & {r.image for r in conv_hold}), "conveyor holdout leaks into train list"
        assert man
        # regression: missing dataset split -> clean exit 2 instead of a traceback
        bad = json.loads(json.dumps(cfgd))
        bad["datasets"]["defects"]["eval_split"] = "nosuch_split"
        bp = tmp / "bad_config.yaml"
        with open(bp, "w", encoding="utf-8") as f:
            yaml.safe_dump(bad, f, allow_unicode=True)
        assert ev.main(["--config", str(bp)]) == 2

        # ---- FAIL case: most small tomatoes dropped + belt FPs
        res = ev.run(cfg, predict_fn=make_fake(drop_small=0.9, fp_rate=0.3, seed=1))
        out = Path(cfg["out_dir"])
        rep = out / "PHASE5_class_agnostic_report.md"
        for p in (rep, out / "class_agnostic_table.csv", out / "class_agnostic_results.json"):
            assert p.is_file(), p
        # regression: CSV has a UTF-8 BOM so Excel shows Persian paths correctly
        assert (out / "class_agnostic_table.csv").read_bytes().startswith(codecs.BOM_UTF8)
        with open(out / "class_agnostic_table.csv", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        assert [r["eval_set"] for r in rows] == list(common.EVAL_SETS), rows
        with open(out / "class_agnostic_results.json", encoding="utf-8") as f:
            js = json.load(f)
        assert js["gate"]["verdict"] == "FAIL"
        text = rep.read_text(encoding="utf-8")
        first = text.splitlines()[0]
        assert first.startswith("**FAIL"), first
        assert "Recall is below 0.85" in text
        for t in ("Small objects", "Motion blur", "Occlusion", "Belt background"):
            assert t in text, t
        assert "the only conveyor images never seen in training" in text
        assert text.count("| `") >= 3
        conv = res["sets"]["conveyor_holdout"]
        r = conv["summary"]["recall"]
        assert 0.4 < r < 0.85, r
        so = conv["sizes_original_px"]
        assert so["small"]["recall"] < so["large"]["recall"], so
        assert so["small"]["n_gt"] > 0 and so["large"]["n_gt"] > 0
        assert "missed more often" in conv["small_sentence_original"]
        fa = conv["failures"]
        assert fa["n_fn"] > 0 and fa["fn_tags"].get("small_object", 0) > 0, fa["fn_tags"]
        ex = [p for v in fa["examples"].values() for p in v]
        assert ex, "no failure example images written"
        assert "![" in text
        fc = res["sets"]["defects_test"]["four_class"]
        assert len(fc["per_class"]) == 4, fc
        _check_numbers(res)
        d = res["sets"]["defects_test"]["summary"]
        assert d["ap50"] > 0.5, d

        # ---- PASS case: near-perfect predictions
        res2 = ev.run(cfg, predict_fn=make_fake(drop_small=0.0, fp_rate=0.0, seed=2))
        assert res2["gate"]["verdict"] == "PASS", res2["gate"]
        assert (out / "PHASE5_class_agnostic_report.md").read_text(encoding="utf-8").startswith("**PASS")
        for k in common.EVAL_SETS:
            s = res2["sets"][k]["summary"]
            assert s["recall"] > 0.97 and s["ap50"] > 0.95, (k, s)
            assert not math.isnan(s["ap50_95"])
        _check_numbers(res2)
        print("OK test_end_to_end")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_end_to_end():
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
