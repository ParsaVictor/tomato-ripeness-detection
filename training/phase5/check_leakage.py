"""Prove the held-out evaluation images were never trained on (nor used for validation / model selection).

Every train + val image of a training spec (data.yaml / list .txt / images dir) is compared against every
holdout image of conveyor_holdout, sxsc7_holdout (frozen manifest) and defects_test:
LEAK = identical sha1 or dHash Hamming <= threshold; WARNING = same Roboflow source stem only.

    py -3.10 -m phase5.check_leakage --config phase5_config.yaml --train data.yaml [--train more.txt]
        [--write-clean clean_train.txt] [--out-dir DIR]          # exit code 1 if any LEAK
"""
from __future__ import annotations

import argparse
import os
import logging
from pathlib import Path

import numpy as np
import yaml

from phase5 import common
from phase5 import make_holdouts

log = logging.getLogger("phase5.check_leakage")


# ----------------------------------------------------------------------------- collecting train images


def _norm(p) -> str:
    return os.path.normcase(os.path.normpath(str(Path(p).resolve())))


def _glob_images(d: Path) -> list[str]:
    return sorted(str(p.resolve()) for p in d.rglob("*") if p.is_file() and p.suffix.lower() in common.IMG_EXTS)


def _resolve_entry(base: Path, entry: str) -> list[str]:
    x = (base / entry).resolve()
    if not x.exists() and str(entry).startswith("../"):
        x = (base / str(entry)[3:]).resolve()  # Roboflow export quirk ('../train/images')
    if x.is_dir():
        return _glob_images(x)
    if x.is_file() and x.suffix.lower() == ".txt":
        return common.read_list(x)
    if x.is_file() and x.suffix.lower() in common.IMG_EXTS:
        return [str(x)]
    raise FileNotFoundError(f"training entry '{entry}' not found (resolved to {x})")


def yaml_entries(yaml_path: str | Path, keys=("train", "val")) -> dict[str, list[str]]:
    """Images per key of a data.yaml, resolved exactly like Ultralytics (base = 'path' key or yaml folder)."""
    yp = Path(yaml_path).resolve()
    with open(yp, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    base = yp.parent
    if d.get("path"):
        p = Path(str(d["path"]))
        base = p if p.is_absolute() else (yp.parent / p).resolve()
    out = {}
    for k in keys:
        v = d.get(k)
        if not v:
            continue
        ents = v if isinstance(v, (list, tuple)) else [v]
        imgs = []
        for e in ents:
            imgs.extend(_resolve_entry(base, str(e)))
        out[k] = imgs
    return out


def collect_train_images(spec: str) -> list[str]:
    p = Path(spec).resolve()
    if not p.exists():
        raise FileNotFoundError(f"training spec not found: {p}")
    if p.is_dir():
        imgs = _glob_images(p)
    elif p.suffix.lower() in (".yaml", ".yml"):
        imgs = [i for v in yaml_entries(p, ("train", "val")).values() for i in v]
    elif p.suffix.lower() == ".txt":
        imgs = common.read_list(p)
    else:
        raise ValueError(f"unsupported training spec (want data.yaml, list .txt or images dir): {p}")
    seen, out = set(), []
    for i in imgs:
        i = str(Path(i).resolve())
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


# ----------------------------------------------------------------------------- checking


def _holdout_sets(cfg: dict) -> tuple[dict[str, list[str]], list[str]]:
    warns = []
    man = make_holdouts.load_manifest(cfg)
    sets = {
        "conveyor_holdout": [str(common.resolve_project_path(e["path"])) for e in man["conveyor"]["holdout"]],
        "sxsc7_holdout": [str(common.resolve_project_path(e["path"])) for e in man["sxsc7"]["holdout"]],
    }
    d = cfg["datasets"]["defects"]
    try:
        ds = common.load_dataset(d["root"], "defects")
        recs = ds["splits"].get(d.get("eval_split", "test"), [])
        if not recs:
            warns.append(f"defects: split '{d.get('eval_split', 'test')}' not found - defects_test not checked")
        sets["defects_test"] = [r.image for r in recs]
    except FileNotFoundError as e:
        warns.append(f"defects_test not checked: {e}")
        sets["defects_test"] = []
    return sets, warns


def _hashes(paths: list[str]) -> tuple[list[str], list[int | None]]:
    sh, dh = [], []
    for p in paths:
        try:
            sh.append(common.file_sha1(p))
        except OSError:
            sh.append("")
        img = common.imread(p)
        dh.append(None if img is None else common.dhash(img))
    return sh, dh


def check_leakage(cfg: dict, train_images: list[str], near_dup_hamming: int | None = None) -> dict:
    thr = int(cfg.get("near_dup_hamming", 6) if near_dup_hamming is None else near_dup_hamming)
    sets, warnings = _holdout_sets(cfg)
    train_images = [str(Path(p).resolve()) for p in train_images]
    t_sha, t_dh = _hashes(train_images)
    for p, s, h in zip(train_images, t_sha, t_dh):
        if not s:
            warnings.append(f"train image missing: {p}")
        elif h is None:
            warnings.append(f"train image unreadable (sha1 check only): {p}")
    t_stem = [common.source_stem(p) for p in train_images]
    leaks, minh = [], {}
    for name, hold in sets.items():
        if not hold:
            minh[name] = None
            continue
        h_sha, h_dh = _hashes(hold)
        sha_idx = {}
        for j, s in enumerate(h_sha):
            sha_idx.setdefault(s, j)
        stem_idx = {}
        for j, p in enumerate(hold):
            stem_idx.setdefault(common.source_stem(p), j)
        ti = [i for i, h in enumerate(t_dh) if h is not None]
        hj = [j for j, h in enumerate(h_dh) if h is not None]
        best = {}
        if ti and hj:
            m = common.hamming_matrix([t_dh[i] for i in ti], [h_dh[j] for j in hj])
            minh[name] = int(m.min())
            am = m.argmin(1)
            for r, i in enumerate(ti):
                best[i] = (int(m[r, am[r]]), hj[int(am[r])])
        else:
            minh[name] = None
        for i, p in enumerate(train_images):
            if t_sha[i] and t_sha[i] in sha_idx:
                j = sha_idx[t_sha[i]]
                leaks.append({"train": p, "holdout": hold[j], "eval_set": name, "reason": "identical_sha1",
                              "hamming": 0})
            elif i in best and best[i][0] <= thr:
                leaks.append({"train": p, "holdout": hold[best[i][1]], "eval_set": name,
                              "reason": "near_duplicate_dhash", "hamming": best[i][0]})
            elif t_stem[i] in stem_idx:
                warnings.append({"train": p, "holdout": hold[stem_idx[t_stem[i]]], "eval_set": name,
                                 "reason": "same_source_stem"})
    return {"ok": not leaks, "n_train": len(train_images), "near_dup_hamming": thr, "leaks": leaks,
            "warnings": warnings, "min_hamming_per_set": minh,
            "n_holdout_per_set": {k: len(v) for k, v in sets.items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--train", action="append", required=True,
                    help="data.yaml / list .txt / images dir (repeatable)")
    ap.add_argument("--write-clean", default=None, help="write the train list with leaked images removed")
    ap.add_argument("--near-dup-hamming", type=int, default=None)
    ap.add_argument("--out-dir", default=None, help="default: config out_dir")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = common.load_config(a.config)
    try:
        imgs, seen = [], set()
        for spec in a.train:
            for p in collect_train_images(spec):
                if p not in seen:
                    seen.add(p)
                    imgs.append(p)
        rep = check_leakage(cfg, imgs, a.near_dup_hamming)
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 2
    # yaml 'test' pointing at holdout images is not training, but worth flagging
    hold_all = set()
    for k in make_holdouts.HOLDOUT_KEYS:
        hold_all |= {_norm(common.resolve_project_path(e["path"])) for e in make_holdouts.load_manifest(cfg)[k]["holdout"]}
    for spec in a.train:
        if Path(spec).suffix.lower() in (".yaml", ".yml"):
            try:
                tst = yaml_entries(spec, ("test",)).get("test", [])
            except FileNotFoundError:
                tst = []
            n = len({_norm(p) for p in tst} & hold_all)
            if n:
                rep["warnings"].append(f"{spec}: 'test' entry points at {n} holdout images (ok only for testing)")
    out_dir = Path(a.out_dir or cfg["out_dir"])
    common.dump_json(out_dir / "leakage_report.json", rep)
    if a.write_clean:
        bad = {l["train"] for l in rep["leaks"]}
        common.write_list(a.write_clean, [p for p in imgs if p not in bad])
        print(f"clean list: {a.write_clean} ({len(imgs) - len(bad)} of {len(imgs)} images)")
    print(f"leakage check: {'OK' if rep['ok'] else 'LEAK'} - {len(rep['leaks'])} leaks, "
          f"{len(rep['warnings'])} warnings, n_train={rep['n_train']}; report {out_dir / 'leakage_report.json'}")
    for l in rep["leaks"][:20]:
        print(f"  LEAK [{l['eval_set']}] {l['reason']} (h={l['hamming']}): {l['train']} ~ {l['holdout']}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
