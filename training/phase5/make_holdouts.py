"""Freeze the held-out evaluation sets (conveyor, sxsc7) and the matching leak-free training lists.

Images are grouped with union-find before any split decision (identical sha1, same Roboflow source stem,
video frames within `video_gap`, dHash near-duplicates), so a group never straddles train / holdout.
The result is written once to `<splits_dir>/manifest.json` and is then frozen: later calls only verify
it (sha1 of every holdout file) unless `force=True` / `--force` is given.

    py -3.10 -m phase5.make_holdouts --config phase5_config.yaml [--force]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import logging
from pathlib import Path

import numpy as np

from phase5 import common
from phase5.common import ImageRecord

log = logging.getLogger("phase5.make_holdouts")

MANIFEST_VERSION = 1
HOLDOUT_KEYS = ("conveyor", "sxsc7")
SPLIT_PREF = {"test": 0, "valid": 1, "val": 1, "train": 2, "all": 3}
DEFAULT_VIDEO_GAP = 10


# ----------------------------------------------------------------------------- union-find


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.p[rb] = ra
            else:
                self.p[ra] = rb

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for i in range(len(self.p)):
            out.setdefault(self.find(i), []).append(i)
        return out


# ----------------------------------------------------------------------------- scanning


def _scan(root: str, key: str) -> list[dict]:
    """All images of a dataset with sha1 / source stem / video frame / dHash."""
    root_p = Path(root)
    if not root_p.is_dir():
        raise FileNotFoundError(
            f"dataset '{key}' root not found: {root_p} (set datasets.{key}.root in the config)")
    ds = common.load_dataset(root_p, key)
    items = []
    for rec in common.all_records(ds):
        img = common.imread(rec.image)
        dh = None
        if img is None:
            log.warning("[%s] unreadable image (no near-dup check possible): %s", key, rec.image)
        else:
            dh = common.dhash(img)
        items.append({
            "rec": rec,
            "sha1": common.file_sha1(rec.image),
            "stem": common.source_stem(rec.image),
            "video": common.video_frame(rec.image),
            "dhash": dh,
        })
    items.sort(key=lambda d: d["rec"].image)
    return items


def _hard_unions(items: list[dict], idx: list[int] | None = None) -> UnionFind:
    """Union by identical sha1 and identical source stem."""
    uf = UnionFind(len(items))
    by_sha, by_stem = {}, {}
    for i, it in enumerate(items):
        if idx is not None and i not in idx:
            continue
        for d, k in ((by_sha, it["sha1"]), (by_stem, it["stem"])):
            if k in d:
                uf.union(d[k], i)
            else:
                d[k] = i
    return uf


def _near_dup_pairs(items: list[dict], thr: int, rows: list[int] | None = None,
                    cols: list[int] | None = None) -> list[tuple[int, int, int]]:
    """(i, j, hamming) for dHash Hamming <= thr. rows/cols restrict the comparison (default: all vs all)."""
    valid = [i for i, it in enumerate(items) if it["dhash"] is not None]
    rows = [i for i in (rows if rows is not None else valid) if items[i]["dhash"] is not None]
    cols = [i for i in (cols if cols is not None else valid) if items[i]["dhash"] is not None]
    if thr < 0 or not rows or not cols:
        return []
    hc = [items[j]["dhash"] for j in cols]
    out = []
    step = 1024
    for s in range(0, len(rows), step):
        rr = rows[s:s + step]
        m = common.hamming_matrix([items[i]["dhash"] for i in rr], hc)
        ii, jj = np.nonzero(m <= thr)
        for a, b in zip(ii.tolist(), jj.tolist()):
            i, j = rr[a], cols[b]
            if i != j:
                out.append((i, j, int(m[a, b])))
    return out


def _atoms(items: list[dict], thr: int, key: str, warnings: list[str]) -> tuple[UnionFind, int]:
    """sha1 + stem + near-dup unions with the >50 % chaining guard. Returns (uf, effective threshold)."""
    n = len(items)
    base = _hard_unions(items)
    base_max = max(len(g) for g in base.groups().values())
    pairs_all = _near_dup_pairs(items, thr)
    t = thr
    while True:
        uf = _hard_unions(items)
        for i, j, h in pairs_all:
            if h <= t:
                uf.union(i, j)
        biggest = max(len(g) for g in uf.groups().values())
        if t < 0 or biggest <= 0.5 * n or biggest <= base_max:
            break
        msg = (f"[{key}] near-dup chaining at hamming<={t} makes one group of {biggest}/{n} images "
               f"(>50 %); lowering threshold to {t - 1}")
        log.warning(msg)
        warnings.append(msg)
        t -= 1
    if t < 0:
        msg = f"[{key}] near-dup grouping disabled (still >50 % at hamming 0)"
        log.warning(msg)
        warnings.append(msg)
    return uf, t


def _add_video_unions(items: list[dict], uf: UnionFind, gap: int) -> None:
    vids: dict[str, list[tuple[int, int]]] = {}
    for i, it in enumerate(items):
        if it["video"]:
            vids.setdefault(it["video"][0], []).append((it["video"][1], i))
    for frames in vids.values():
        frames.sort()
        for (fa, ia), (fb, ib) in zip(frames, frames[1:]):
            if fb - fa <= gap:
                uf.union(ia, ib)


# ----------------------------------------------------------------------------- manifest helpers


def _entry(it: dict, group: int, reason: str | None = None) -> dict:
    e = {"path": common.to_project_rel(it["rec"].image), "sha1": it["sha1"], "source_stem": it["stem"],
         "split": it["rec"].split, "group": int(group)}
    if reason:
        e["reason"] = reason
    return e


def _dedup_holdout(items: list[dict], idx: list[int]) -> tuple[list[int], list[int]]:
    """One image per source stem (and per sha1): prefer test > valid > train, then lexicographic."""
    order = sorted(idx, key=lambda i: (SPLIT_PREF.get(items[i]["rec"].split, 9), items[i]["rec"].image))
    keep, drop, seen_stem, seen_sha = [], [], set(), set()
    for i in order:
        if items[i]["stem"] in seen_stem or items[i]["sha1"] in seen_sha:
            drop.append(i)
        else:
            keep.append(i)
            seen_stem.add(items[i]["stem"])
            seen_sha.add(items[i]["sha1"])
    return sorted(keep, key=lambda i: items[i]["rec"].image), drop


def _n_unique(items: list[dict], idx) -> int:
    """Number of images left after holdout deduplication (unique source stem AND unique sha1)."""
    seen_stem, seen_sha, n = set(), set(), 0
    for i in idx:
        if items[i]["stem"] not in seen_stem and items[i]["sha1"] not in seen_sha:
            n += 1
        seen_stem.add(items[i]["stem"])
        seen_sha.add(items[i]["sha1"])
    return n


def _finish(key: str, root: str, items: list[dict], group_of: dict[int, int], hold: list[int],
            excluded: dict[int, str], params: dict, warnings: list[str], splits_dir: Path) -> dict:
    hold_set = set(hold)
    keep, dup = _dedup_holdout(items, hold)
    for i in dup:
        excluded.setdefault(i, "duplicate / augmented copy of a holdout image (holdout deduplicated)")
    # Final guard: whatever threshold the grouping used, no train image may be a near-duplicate of a
    # holdout image at the CONFIGURED threshold (that is what check_leakage verifies).
    thr = int(params.get("near_dup_hamming", 6))
    cand = [i for i in range(len(items)) if i not in hold_set and i not in excluded]
    best: dict[int, tuple[int, int]] = {}
    for i, j, h in _near_dup_pairs(items, thr, rows=cand, cols=sorted(hold_set)):
        if i not in best or h < best[i][0]:
            best[i] = (h, j)
    for i, (h, j) in sorted(best.items()):
        excluded[i] = (f"near-duplicate at configured threshold (dHash hamming {h} <= {thr}) of holdout image "
                       f"{common.to_project_rel(items[j]['rec'].image)}")
    if best:
        bad_stems = {items[i]["stem"] for i in best}
        bad_sha = {items[i]["sha1"] for i in best}
        for i in cand:
            if i not in excluded and (items[i]["stem"] in bad_stems or items[i]["sha1"] in bad_sha):
                excluded[i] = "same source/sha1 as an image excluded as near-duplicate of the holdout"
        msg = (f"[{key}] excluded {len(best)} train images that are near-duplicates (hamming<={thr}) of "
               f"holdout images: " + "; ".join(
                   f"{common.to_project_rel(items[i]['rec'].image)} ~ "
                   f"{common.to_project_rel(items[j]['rec'].image)} (h={h})" for i, (h, j) in sorted(best.items())))
        log.warning(msg)
        warnings.append(msg)
    train = [i for i in range(len(items)) if i not in hold_set and i not in excluded]
    if not train:
        raise ValueError(f"[{key}] no training images left after removing holdout near-duplicates")
    assert not (set(train) & hold_set)
    for i, why in sorted(excluded.items()):
        log.info("[%s] excluded %s: %s", key, items[i]["rec"].image, why)
    log.info("[%s] holdout %d unique images (%d files in holdout groups), train %d, excluded %d",
             key, len(keep), len(hold), len(train), len(excluded))
    common.write_list(splits_dir / f"{key}_holdout.txt", [items[i]["rec"].image for i in keep])
    common.write_list(splits_dir / f"{key}_train.txt", [items[i]["rec"].image for i in train])
    return {
        "dataset_root": common.to_project_rel(root),
        "params": params,
        "n_images": len(items),
        "n_groups": len(set(group_of.values())),
        "holdout": [_entry(items[i], group_of[i]) for i in keep],
        "train": [_entry(items[i], group_of[i]) for i in train],
        "excluded": [_entry(items[i], group_of[i], why) for i, why in sorted(excluded.items())],
        "counts": {"holdout": len(keep), "holdout_unique_stems": _n_unique(items, keep),
                   "train": len(train), "excluded": len(excluded), "total": len(items)},
        "holdout_list": common.to_project_rel(splits_dir / f"{key}_holdout.txt"),
        "train_list": common.to_project_rel(splits_dir / f"{key}_train.txt"),
        "warnings": warnings,
    }


# ----------------------------------------------------------------------------- conveyor


def _split_conveyor(cfg: dict, splits_dir: Path) -> dict:
    dcfg = cfg["datasets"]["conveyor"]
    hcfg = dict(dcfg.get("holdout") or {})
    warnings: list[str] = []
    n_req = int(hcfg.get("n_images", 30))
    if n_req < 30:
        msg = f"[conveyor] holdout.n_images={n_req} < 30: clamped up to 30"
        log.warning(msg)
        warnings.append(msg)
        n_req = 30
    gap = int(hcfg.get("video_gap", DEFAULT_VIDEO_GAP))
    thr = int(cfg.get("near_dup_hamming", 6))
    items = _scan(dcfg["root"], "conveyor")
    n_unique = _n_unique(items, range(len(items)))
    if n_unique <= n_req:
        raise ValueError(f"[conveyor] only {n_unique} unique source images in {dcfg['root']}; need more than "
                         f"{n_req} to hold out {n_req} and still train on the rest")
    if n_unique < 60:
        msg = f"[conveyor] only {n_unique} unique source images (< 60): training set will be small"
        log.warning(msg)
        warnings.append(msg)

    uf, t_eff = _atoms(items, thr, "conveyor", warnings)
    atom_of = {i: uf.find(i) for i in range(len(items))}
    _add_video_unions(items, uf, gap)
    groups = sorted(uf.groups().values(), key=lambda g: items[g[0]]["rec"].image)
    group_of = {i: gi for gi, g in enumerate(groups) for i in g}
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    order = rng.permutation(len(groups)).tolist()

    hold: list[int] = []
    excluded: dict[int, str] = {}
    for gi in order:
        if _n_unique(items, hold) >= n_req:
            break
        hold.extend(groups[gi])
    mode = "whole_groups"
    if _n_unique(items, hold) > 0.5 * n_unique:
        # Whole groups overshoot (one long video): hold out ONE contiguous frame block instead.
        big = max(range(len(groups)), key=lambda g: len(groups[g]))
        vid_members = [i for i in groups[big] if items[i]["video"]]
        if not vid_members:
            raise ValueError("[conveyor] cannot build a holdout: one non-video group covers >50 % of images")
        mode = "video_block"
        hold, excluded = _video_block(items, groups, big, atom_of, n_req, gap, rng, warnings)
    hold.sort()
    params = {"n_images": n_req, "seed": int(cfg.get("seed", 0)), "video_gap": gap,
              "near_dup_hamming": thr, "near_dup_hamming_effective": t_eff, "mode": mode}
    return _finish("conveyor", dcfg["root"], items, group_of, hold, excluded, params, warnings, splits_dir)


def _video_block(items, groups, big, atom_of, n_req, gap, rng, warnings):
    """Contiguous block of atoms from the largest video group + a `gap` buffer removed from training."""
    g = groups[big]
    atoms: dict[int, list[int]] = {}
    for i in g:
        atoms.setdefault(atom_of[i], []).append(i)

    def akey(a):
        fr = [items[i]["video"] for i in atoms[a] if items[i]["video"]]
        return min(fr) if fr else ("~", 0)

    seq = sorted(atoms, key=akey)
    start = int(rng.integers(0, len(seq)))
    # walk forward (wrapping by starting earlier if we hit the end) until enough unique stems
    hold: list[int] = []
    for k in list(range(start, len(seq))) + list(range(start - 1, -1, -1)):
        if _n_unique(items, hold) >= n_req:
            break
        hold.extend(atoms[seq[k]])
    hold_set = set(hold)
    ranges: dict[str, list[int]] = {}
    for i in hold:
        if items[i]["video"]:
            v, f = items[i]["video"]
            ranges.setdefault(v, []).append(f)
    excluded = {}
    for a, mem in atoms.items():
        if any(i in hold_set for i in mem):
            continue
        for i in mem:
            vf = items[i]["video"]
            if vf and vf[0] in ranges and min(ranges[vf[0]]) - gap <= vf[1] <= max(ranges[vf[0]]) + gap:
                for j in mem:
                    excluded[j] = f"video buffer: within {gap} frames of the held-out block of {vf[0]}"
                break
    msg = (f"[conveyor] whole-group holdout would exceed 50 % of the data; held out one contiguous frame "
           f"block instead ({len(hold)} files) with a {gap}-frame buffer ({len(excluded)} files) removed "
           f"from training")
    log.warning(msg)
    warnings.append(msg)
    return hold, excluded


# ----------------------------------------------------------------------------- sxsc7


def _split_sxsc7(cfg: dict, splits_dir: Path) -> dict:
    dcfg = cfg["datasets"]["sxsc7"]
    hcfg = dict(dcfg.get("holdout") or {})
    warnings: list[str] = []
    use_split = hcfg.get("use_existing_split", "test")
    min_images = int(hcfg.get("min_images", 30))
    frac = float(hcfg.get("frac", 0.2))
    gap = int(hcfg.get("video_gap", DEFAULT_VIDEO_GAP))
    thr = int(cfg.get("near_dup_hamming", 6))
    seed = int(cfg.get("seed", 0))
    items = _scan(dcfg["root"], "sxsc7")
    n = len(items)
    n_unique = _n_unique(items, range(n))
    if n_unique <= min_images:
        raise ValueError(f"[sxsc7] only {n_unique} unique source images in {dcfg['root']}; "
                         f"need more than holdout.min_images={min_images}")
    split_idx = [i for i, it in enumerate(items) if use_split and it["rec"].split == use_split]
    excluded: dict[int, str] = {}
    if split_idx and _n_unique(items, split_idx) >= min_images:
        mode = f"existing_split:{use_split}"
        hold_seed = set(split_idx)
        rest = [i for i in range(n) if i not in hold_seed]
        uf = _hard_unions(items)
        _add_video_unions(items, uf, gap)
        groups = uf.groups()
        group_of = {i: uf.find(i) for i in range(n)}
        hold_roots = {group_of[i] for i in hold_seed}
        for i in rest:
            if group_of[i] in hold_roots:
                r = next(j for j in groups[group_of[i]] if j in hold_seed)
                why = ("identical sha1 to holdout image" if items[i]["sha1"] == items[r]["sha1"] else
                       "same source stem as holdout image" if items[i]["stem"] == items[r]["stem"] else
                       f"video frame within {gap} frames of holdout image")
                excluded[i] = f"{why} {common.to_project_rel(items[r]['rec'].image)}"
        cand = [i for i in rest if i not in excluded]
        # Leak removal against a FIXED holdout: never lower the threshold (the >50 % guard is only for
        # group chaining while carving). Every near-dup at the configured threshold is excluded.
        t = thr
        pairs = _near_dup_pairs(items, thr, rows=cand, cols=sorted(hold_seed))
        leaked = {i for i, _, _ in pairs}
        if len(leaked) > 0.5 * max(len(cand), 1):
            msg = (f"[sxsc7] {len(leaked)}/{len(cand)} train candidates are near-duplicates "
                   f"(hamming<={thr}) of the '{use_split}' split; all of them are excluded from training")
            log.warning(msg)
            warnings.append(msg)
        best: dict[int, tuple[int, int]] = {}
        for i, j, h in pairs:
            if i not in best or h < best[i][0]:
                best[i] = (h, j)
        for i, (h, j) in sorted(best.items()):
            excluded[i] = f"near-duplicate (dHash hamming {h}) of holdout image {common.to_project_rel(items[j]['rec'].image)}"
        # propagate: everything grouped (sha1 / stem / video) with a near-dup leak goes too
        for i in sorted(best):
            for j in groups[uf.find(i)]:
                if j not in hold_seed and j not in excluded:
                    excluded[j] = f"grouped with near-duplicate {common.to_project_rel(items[i]['rec'].image)}"
        hold = sorted(hold_seed)
        # group ids for the manifest: hard groups, plus near-dup partners share their holdout's group
        for i, (h, j) in best.items():
            uf.union(i, j)
        group_of = {i: uf.find(i) for i in range(n)}
        remap = {g: k for k, g in enumerate(sorted(set(group_of.values())))}
        group_of = {i: remap[g] for i, g in group_of.items()}
        log.info("[sxsc7] removed %d training images that leak into the existing '%s' split",
                 len(excluded), use_split)
        t_eff = t
    else:
        if use_split:
            msg = (f"[sxsc7] split '{use_split}' missing or has < {min_images} unique images: "
                   f"carving a group-aware holdout")
            log.warning(msg)
            warnings.append(msg)
        mode = "carved"
        uf, t_eff = _atoms(items, thr, "sxsc7", warnings)
        _add_video_unions(items, uf, gap)
        groups = sorted(uf.groups().values(), key=lambda g: items[g[0]]["rec"].image)
        group_of = {i: gi for gi, g in enumerate(groups) for i in g}
        need = max(min_images, int(np.ceil(frac * n_unique)))
        rng = np.random.default_rng(seed)
        hold = []
        for gi in rng.permutation(len(groups)).tolist():
            if _n_unique(items, hold) >= need:
                break
            hold.extend(groups[gi])
        if _n_unique(items, hold) >= n_unique:
            raise ValueError("[sxsc7] grouping leaves nothing for training")
        hold.sort()
    params = {"use_existing_split": use_split, "min_images": min_images, "frac": frac, "seed": seed,
              "video_gap": gap, "near_dup_hamming": thr, "near_dup_hamming_effective": t_eff, "mode": mode}
    return _finish("sxsc7", dcfg["root"], items, group_of, hold, excluded, params, warnings, splits_dir)


# ----------------------------------------------------------------------------- public API


def _manifest_path(cfg: dict) -> Path:
    return Path(cfg["splits_dir"]) / "manifest.json"


def load_manifest(cfg: dict) -> dict:
    p = _manifest_path(cfg)
    if not p.is_file():
        raise FileNotFoundError(f"holdout manifest not found: {p} - run `py -3.10 -m phase5.make_holdouts "
                                f"--config phase5_config.yaml` BEFORE training")
    import json
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _verify(man: dict) -> None:
    bad = []
    for key in HOLDOUT_KEYS:
        for e in man.get(key, {}).get("holdout", []):
            p = common.resolve_project_path(e["path"])
            if not p.is_file():
                bad.append(f"{key}: missing {p}")
            elif common.file_sha1(p) != e["sha1"]:
                bad.append(f"{key}: sha1 changed {p}")
    if bad:
        raise RuntimeError("frozen holdout manifest no longer matches the files on disk (leakage risk; "
                           "re-split only deliberately with --force):\n  " + "\n  ".join(bad[:20]))


def build_holdouts(cfg: dict, force: bool = False) -> dict:
    splits_dir = Path(cfg["splits_dir"])
    mp = _manifest_path(cfg)
    if mp.is_file() and not force:
        man = load_manifest(cfg)
        _verify(man)
        log.info("holdout manifest is frozen (%s); verified, not re-splitting (use --force to re-split)", mp)
        return man
    if mp.is_file():
        log.warning("--force: re-splitting and overwriting %s", mp)
    man = {"version": MANIFEST_VERSION, "seed": int(cfg.get("seed", 0)),
           "created": _dt.datetime.now().isoformat(timespec="seconds"),
           "near_dup_hamming": int(cfg.get("near_dup_hamming", 6))}
    man["conveyor"] = _split_conveyor(cfg, splits_dir)
    man["sxsc7"] = _split_sxsc7(cfg, splits_dir)
    common.dump_json(mp, man)
    return man


def load_holdout_records(cfg: dict, key: str) -> list[ImageRecord]:
    if key not in HOLDOUT_KEYS:
        raise ValueError(f"key must be one of {HOLDOUT_KEYS}, got {key!r}")
    man = load_manifest(cfg)
    out = []
    for e in man[key]["holdout"]:
        p = str(common.resolve_project_path(e["path"]))
        out.append(ImageRecord(p, common.img2label_path(p), key, e["split"]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true", help="re-split even if a frozen manifest exists")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = common.load_config(a.config)
    try:
        man = build_holdouts(cfg, force=a.force)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        log.error("%s", e)
        return 2
    for k in HOLDOUT_KEYS:
        c = man[k]["counts"]
        print(f"{k}: holdout={c['holdout']} train={c['train']} excluded={c['excluded']} total={c['total']}")
    print(f"manifest: {_manifest_path(cfg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
