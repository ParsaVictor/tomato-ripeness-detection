"""Tests for make_holdouts / check_leakage on synthetic Roboflow-style exports in a Persian-named temp dir.

    py -3.10 -m phase5.tests.test_splits      (or: py -3.10 -m pytest phase5/tests/test_splits.py)
"""
from __future__ import annotations

import copy
import json
import logging
import secrets
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from phase5 import check_leakage as CL
from phase5 import common
from phase5 import make_holdouts as MH

RNG = np.random.default_rng(1234)


def _img(seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    g = r.integers(0, 256, (8, 9, 3), dtype=np.uint8)
    img = cv2.resize(g, (180, 160), interpolation=cv2.INTER_LINEAR)
    return np.clip(img.astype(np.int16) + r.integers(-3, 4, img.shape), 0, 255).astype(np.uint8)


def _rf(stem: str) -> str:
    return f"{stem}.rf.{secrets.token_hex(16)}.jpg"


def _put(root: Path, split: str, name: str, img: np.ndarray) -> Path:
    p = root / split / "images" / name
    assert common.imwrite(p, img)
    lp = root / split / "labels" / (Path(name).stem + ".txt")
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    return p


def _yaml(root: Path):
    (root / "data.yaml").write_text(
        "train: ../train/images\nval: ../valid/images\ntest: ../test/images\nnc: 1\nnames: ['tomato']\n",
        encoding="utf-8")


def make_world(base: Path) -> dict:
    seed = [0]

    def nxt():
        seed[0] += 1
        return _img(seed[0] * 7919)

    info = {}
    # conveyor: 60 still images (train with 3 aug copies, valid, test) + one 40-frame video
    cv = base / "datasets" / "tomato_count_project"
    for k in range(60):
        im = nxt()
        stem = f"IMG_{k:04d}_jpg"
        split = "train" if k < 40 else ("valid" if k < 50 else "test")
        if split == "train":
            for a in range(3):
                aug = cv2.flip(im, 1) if a == 1 else np.clip(im.astype(int) + 10 * a, 0, 255).astype(np.uint8)
                _put(cv, split, _rf(stem), aug)
        else:
            _put(cv, split, _rf(stem), im)
    for f in range(1, 41):
        _put(cv, "train", _rf(f"belt_mp4-{f:04d}_jpg"), nxt())
    # exact duplicate under a different name + near duplicate (re-encoded, resized)
    src = next(iter(sorted((cv / "test" / "images").iterdir())))
    shutil.copy(src, cv / "train" / "images" / _rf("COPY_0001_jpg"))
    im = common.imread(src)
    _put(cv, "valid", _rf("NEAR_0001_jpg"), cv2.resize(im, (162, 144)))
    _yaml(cv)

    # sxsc7: train 40 x3 aug, valid 10, test 35, with leaks of test images planted in train
    sx = base / "datasets" / "tomato-sxsc7"
    test_imgs = []
    for k in range(85):
        im = nxt()
        stem = f"sx_{k:04d}_jpg"
        split = "train" if k < 40 else ("valid" if k < 50 else "test")
        n = 3 if split == "train" else 1
        for a in range(n):
            p = _put(sx, split, _rf(stem), np.clip(im.astype(int) + 8 * a, 0, 255).astype(np.uint8))
        if split == "test":
            test_imgs.append((stem, im, p))
    planted = []
    s0, im0, p0 = test_imgs[0]
    q = sx / "train" / "images" / _rf("dup_of_test_jpg")
    shutil.copy(p0, q)
    planted.append(q)
    s1, im1, _ = test_imgs[1]
    planted.append(_put(sx, "train", _rf("near_of_test_jpg"), cv2.resize(im1, (170, 150))))
    s2, im2, _ = test_imgs[2]
    planted.append(_put(sx, "train", _rf(s2), cv2.flip(im2, 0)))  # augmented copy (same stem)
    # an augmented copy of the near-dup leak (same stem) must go too
    planted.append(_put(sx, "valid", _rf(common.source_stem(planted[1])), cv2.flip(cv2.resize(im1, (170, 150)), 1)))
    _yaml(sx)
    info["sx_planted"] = [str(p.resolve()) for p in planted]

    df = base / "datasets" / "tomato-defects"
    for k in range(20):
        _put(df, "train" if k < 10 else "test", _rf(f"def_{k:03d}_jpg"), nxt())
    _yaml(df)
    return info


def make_cfg(base: Path, splits="splits", seed=0) -> dict:
    cfg = copy.deepcopy(common.DEFAULT_CONFIG)
    cfg["seed"] = seed
    cfg["splits_dir"] = str(base / splits)
    cfg["out_dir"] = str(base / "out")
    cfg["datasets"]["conveyor"]["root"] = str(base / "datasets" / "tomato_count_project")
    cfg["datasets"]["sxsc7"]["root"] = str(base / "datasets" / "tomato-sxsc7")
    cfg["datasets"]["defects"]["root"] = str(base / "datasets" / "tomato-defects")
    return cfg


def _paths(man, key, part):
    return {str(common.resolve_project_path(e["path"])) for e in man[key][part]}


def check_world(base: Path, info: dict):
    cfg = make_cfg(base)
    man = MH.build_holdouts(cfg)
    for key in ("conveyor", "sxsc7"):
        ds = common.load_dataset(cfg["datasets"][key]["root"], key)
        allp = {r.image for r in common.all_records(ds)}
        h, t, x = _paths(man, key, "holdout"), _paths(man, key, "train"), _paths(man, key, "excluded")
        assert not (h & t) and not (h & x) and not (t & x)
        assert h | t | x == allp, key
        # no group split: train shares no stem / sha1 / near-dup with the holdout
        hs = {common.source_stem(p) for p in h}
        assert not ({common.source_stem(p) for p in t} & hs), key
        hsha = {e["sha1"] for e in man[key]["holdout"]} | {e["sha1"] for e in man[key]["excluded"]}
        assert not ({e["sha1"] for e in man[key]["train"]} & {e["sha1"] for e in man[key]["holdout"]})
        hd = [common.dhash(common.imread(p)) for p in h]
        td = [common.dhash(common.imread(p)) for p in t]
        assert common.hamming_matrix(td, hd).min() > cfg["near_dup_hamming"], key
        # holdout deduplicated by stem
        assert len(hs) == len(h)
        # lists on disk
        assert set(common.read_list(Path(cfg["splits_dir"]) / f"{key}_holdout.txt")) == h
        assert set(common.read_list(Path(cfg["splits_dir"]) / f"{key}_train.txt")) == t
        del hsha
    # conveyor: >= 30 unique; video group not split; train ~= rest
    c = man["conveyor"]
    assert c["counts"]["holdout_unique_stems"] >= 30
    h, t = _paths(man, "conveyor", "holdout"), _paths(man, "conveyor", "train")
    vh = {common.video_frame(p) is not None for p in h}
    vt = {common.video_frame(p) is not None for p in t}
    assert not (True in vh and True in vt), "video split across train/holdout"
    n_stem = len({common.source_stem(p) for p in h | t | _paths(man, "conveyor", "excluded")})
    t_stems = len({common.source_stem(p) for p in t})
    assert t_stems >= n_stem - c["counts"]["holdout_unique_stems"] - 2, (t_stems, n_stem)
    # exact dup / near dup of a holdout image never in train
    # sxsc7: holdout = existing test split; planted leaks removed from train
    s = man["sxsc7"]
    assert s["params"]["mode"] == "existing_split:test"
    assert all(e["split"] == "test" for e in s["holdout"]) and len(s["holdout"]) == 35
    xs = _paths(man, "sxsc7", "excluded")
    for p in info["sx_planted"]:
        assert p in xs, p
    assert len(_paths(man, "sxsc7", "train")) == 40 * 3 + 10
    # manifest json on disk round-trips; project-rel or absolute posix paths
    with open(Path(cfg["splits_dir"]) / "manifest.json", encoding="utf-8") as f:
        assert json.load(f) == json.loads(json.dumps(man))
    assert all("\\" not in e["path"] for e in c["holdout"])

    # determinism
    man2 = MH.build_holdouts(make_cfg(base, "splits_b"))
    for key in ("conveyor", "sxsc7"):
        assert _paths(man2, key, "holdout") == _paths(man, key, "holdout")
        assert _paths(man2, key, "train") == _paths(man, key, "train")
    # freeze: second call returns identical manifest
    assert MH.build_holdouts(cfg) == man
    recs = MH.load_holdout_records(cfg, "conveyor")
    assert {r.image for r in recs} == h
    # leakage check: clean train lists pass
    clean = common.read_list(Path(cfg["splits_dir"]) / "conveyor_train.txt") + \
        common.read_list(Path(cfg["splits_dir"]) / "sxsc7_train.txt")
    rep = CL.check_leakage(cfg, clean)
    assert rep["ok"], rep["leaks"][:3]
    # injected leaks: copied holdout file + re-encoded resized copy
    inj = base / "inject"
    hp = sorted(h)[0]
    inj.mkdir()
    shutil.copy(hp, inj / "copied.jpg")
    common.imwrite(inj / "reenc.jpg", cv2.resize(common.imread(sorted(h)[1]), (150, 133)), quality=70)
    dt = common.load_dataset(cfg["datasets"]["defects"]["root"], "d")["splits"]["test"][0].image
    shutil.copy(dt, inj / "defects_copy.png")
    lst = base / "leaky_train.txt"
    common.write_list(lst, clean + [str(inj / "copied.jpg"), str(inj / "reenc.jpg"), str(inj / "defects_copy.png")])
    rep = CL.check_leakage(cfg, CL.collect_train_images(str(lst)))
    reasons = {(Path(l["train"]).name, l["reason"], l["eval_set"]) for l in rep["leaks"]}
    assert ("copied.jpg", "identical_sha1", "conveyor_holdout") in reasons, reasons
    assert ("reenc.jpg", "near_duplicate_dhash", "conveyor_holdout") in reasons, reasons
    assert ("defects_copy.png", "identical_sha1", "defects_test") in reasons, reasons
    assert not rep["ok"] and len({l["train"] for l in rep["leaks"]}) == 3
    # CLI: exit 1 + report + --write-clean
    cfg_y = base / "cfg.yaml"
    import yaml
    with open(cfg_y, "w", encoding="utf-8") as f:
        yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, f, allow_unicode=True)
    cl_out = base / "clean.txt"
    rc = CL.main(["--config", str(cfg_y), "--train", str(lst), "--write-clean", str(cl_out)])
    assert rc == 1 and (Path(cfg["out_dir"]) / "leakage_report.json").is_file()
    assert CL.main(["--config", str(cfg_y), "--train", str(cl_out)]) == 0
    # data.yaml '../train/images' quirk + val included
    yimgs = CL.collect_train_images(str(Path(cfg["datasets"]["sxsc7"]["root"]) / "data.yaml"))
    ds = common.load_dataset(cfg["datasets"]["sxsc7"]["root"], "sx")
    want = {r.image for s in ("train", "valid") for r in ds["splits"][s]}
    assert set(yimgs) == want
    assert not CL.check_leakage(cfg, yimgs)["ok"]  # raw export train contains planted leaks
    # yaml with 'path' + list entries
    yp = base / "y2" / "d.yaml"
    yp.parent.mkdir()
    yp.write_text(f"path: ..\ntrain: [splits/conveyor_train.txt]\nval: splits/sxsc7_train.txt\nnames: [t]\n",
                  encoding="utf-8")
    assert set(CL.collect_train_images(str(yp))) == set(clean)

    # freeze: sha1 change raises; --force re-splits
    with open(hp, "ab") as f:
        f.write(b"\0")
    try:
        MH.build_holdouts(cfg)
        raise AssertionError("sha1 change not detected")
    except RuntimeError:
        pass
    man3 = MH.build_holdouts(cfg, force=True)
    assert man3["conveyor"]["counts"]["holdout_unique_stems"] >= 30
    MH.build_holdouts(cfg)  # frozen again, verifies

    # errors: missing root, too few images
    bad = make_cfg(base, "splits_bad")
    bad["datasets"]["conveyor"]["root"] = str(base / "nope")
    try:
        MH.build_holdouts(bad)
        raise AssertionError
    except FileNotFoundError:
        pass
    small = base / "small"
    for k in range(30):
        _put(small, "train", _rf(f"s_{k}_jpg"), _img(10_000 + k))
    bad["datasets"]["conveyor"]["root"] = str(small)
    try:
        MH.build_holdouts(bad)
        raise AssertionError
    except ValueError:
        pass


def check_video_block(base: Path):
    """One long video covering everything -> contiguous block + gap buffer."""
    root = base / "vid"
    for f in range(1, 101):
        _put(root, "train", _rf(f"belt_mp4-{f:04d}_jpg"), _img(50_000 + f))
    cfg = make_cfg(base, "splits_vid")
    cfg["datasets"]["conveyor"]["root"] = str(root)
    man = {"conveyor": MH._split_conveyor(cfg, Path(cfg["splits_dir"]))}
    c = man["conveyor"]
    assert c["params"]["mode"] == "video_block"
    hf = sorted(common.video_frame(e["path"])[1] for e in c["holdout"])
    tf = sorted(common.video_frame(e["path"])[1] for e in c["train"])
    assert len(hf) >= 30 and hf == list(range(hf[0], hf[0] + len(hf)))
    assert all(f < hf[0] - 10 or f > hf[-1] + 10 for f in tf)
    assert len(tf) + len(hf) + len(c["excluded"]) == 100


def _variant(base_grid: np.ndarray, k: int, n_cells: int) -> np.ndarray:
    """Base 8x9 grid with n_cells cells perturbed (deterministic per k) -> small, varying dHash distances."""
    r = np.random.default_rng(90_000 + k)
    g = base_grid.copy()
    for _ in range(n_cells):
        y, x = int(r.integers(0, 8)), int(r.integers(0, 9))
        g[y, x] = r.integers(0, 256, 3)
    return cv2.resize(g, (180, 160), interpolation=cv2.INTER_LINEAR)


def _no_leak_at_cfg_thr(cfg, man, key):
    h = _paths(man, key, "holdout")
    t = _paths(man, key, "train")
    assert t, key
    hd = [common.dhash(common.imread(p)) for p in h]
    td = [common.dhash(common.imread(p)) for p in t]
    assert common.hamming_matrix(td, hd).min() > cfg["near_dup_hamming"], key
    rep = CL.check_leakage(cfg, common.read_list(Path(cfg["splits_dir"]) / f"{key}_train.txt"))
    assert rep["ok"], rep["leaks"][:3]


def check_threshold_guard(base: Path):
    """Regression: the >50 % guard may lower the grouping threshold, but train must still be free of
    near-dups of the holdout at the CONFIGURED threshold; existing-split mode must never lower it."""
    bg = np.random.default_rng(777).integers(0, 256, (8, 9, 3), dtype=np.uint8)
    # conveyor: 100 look-alike belt stills (uniform background) -> chaining guard kicks in
    cv = base / "guard_cv"
    for k in range(100):
        _put(cv, "train", _rf(f"still_{k:04d}_jpg"), _variant(bg, k, 1 + k % 6))
    sx = base / "guard_sx"
    for k in range(40):
        g = np.random.default_rng(5000 + k).integers(0, 256, (8, 9, 3), dtype=np.uint8)
        _put(sx, "test", _rf(f"t_{k:04d}_jpg"), cv2.resize(g, (180, 160)))
        if k < 30:  # near-dup of a test frame in train (distinct stem): >50 % of train candidates
            _put(sx, "train", _rf(f"tr_{k:04d}_jpg"), cv2.resize(g, (170, 150)))
    for k in range(10):
        _put(sx, "train", _rf(f"ok_{k:04d}_jpg"), _img(70_000 + k))
    cfg = make_cfg(base, "splits_guard")
    cfg["datasets"]["conveyor"]["root"] = str(cv)
    cfg["datasets"]["sxsc7"]["root"] = str(sx)
    man = MH.build_holdouts(cfg)
    c = man["conveyor"]
    assert c["params"]["near_dup_hamming_effective"] < cfg["near_dup_hamming"]  # guard did kick in
    assert any("near-duplicate at configured threshold" in e["reason"] for e in c["excluded"])
    assert any("near-duplicates (hamming<=" in w for w in c["warnings"])
    _no_leak_at_cfg_thr(cfg, man, "conveyor")
    s = man["sxsc7"]
    assert s["params"]["mode"] == "existing_split:test"
    assert s["params"]["near_dup_hamming_effective"] == cfg["near_dup_hamming"]
    assert len(s["train"]) == 10 and len(s["excluded"]) == 30
    _no_leak_at_cfg_thr(cfg, man, "sxsc7")


def check_norm():
    p = Path(tempfile.gettempdir()) / "a" / ".." / "b.jpg"
    assert CL._norm(str(p).lower()) == CL._norm(str(p).upper()) or not __import__('os').name == "nt"
    assert CL._norm(str(p).replace("\\", "/")) == CL._norm(str(Path(tempfile.gettempdir()) / "b.jpg"))


def test_splits():
    logging.basicConfig(level=logging.WARNING)
    base = Path(tempfile.mkdtemp(prefix="تست_"))
    try:
        info = make_world(base)
        check_world(base, info)
        check_video_block(base)
        check_threshold_guard(base)
        check_norm()
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    test_splits()
    print("OK test_splits")
