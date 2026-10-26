# Phase 5 add-on — interface contract (all modules must follow this exactly)

Goal (from the user's spec): besides the 4-class metrics, compute a CLASS-AGNOSTIC "tomato vs background"
detection score (all classes merged into one class "tomato") — mAP50, precision, recall — separately on:

| eval set key       | images                                                                 |
|--------------------|------------------------------------------------------------------------|
| `defects_test`     | Tomato Defects dataset, its `test` split (config `datasets.defects.eval_split`) |
| `sxsc7_holdout`    | held-out subset of tomato-sxsc7 (never trained on)                     |
| `conveyor_holdout` | >= 30 held-out images of tomato_count_project (conveyor); remaining ~70 go to training |

Plus: metrics by object size (small < 32^2 px area, medium, large >= 96^2), a recall gate
(conveyor class-agnostic recall < 0.85 => say so clearly + failure types small objects / motion blur /
occlusion / belt background with example images), and the acceptance table.

Runtime: Windows 11, `py -3.10` (Python 3.10.11), ultralytics 8.4.14, torch 2.10 CPU, numpy 2.2, opencv 5.0,
pycocotools 2.0.11, PyYAML, pandas, matplotlib. Project root path contains Persian characters
(`C:\1\1_پروژه\9_Tomato`) — never use `cv2.imread/imwrite` on paths, always `common.imread/imwrite`;
always open text files with `encoding="utf-8"`. Run modules as `py -3.10 -m phase5.<module>` from the
project root. Python 3.10 syntax only (no `match` on 3.10 is fine, but no 3.11+ features like `tomllib`,
`ExceptionGroup`, `Self`). Every module starts with a short docstring; no prints inside library functions
except via `logging` / an explicit `verbose` flag; CLIs use `argparse` and `main(argv=None) -> int`.

## Shared module (already written — read it, do NOT edit it): `phase5/common.py`

Conventions:
* GT per image: `np.ndarray float64 (N, 5)` = `[cls, x1, y1, x2, y2]` absolute pixels.
* Predictions per image: `np.ndarray float64 (M, 6)` = `[x1, y1, x2, y2, conf, cls]` absolute pixels.
* Pixel frame = image as decoded by `common.imread` (EXIF applied, same as Ultralytics).
* `ImageRecord(image, label, dataset, split)` dataclass; dict keys for per-image data are `record.image` (absolute str path).
* Helpers: `load_config, resolve_project_path, to_project_rel, imread, imwrite, write_list, read_list,
  dump_json, file_sha1, img2label_path, load_names, list_images, load_dataset, all_records,
  read_yolo_labels, tomato_class_ids, to_agnostic_gt, to_agnostic_pred, source_stem, video_frame, dhash,
  hamming_matrix, box_areas, size_bucket, letterbox_ratio`, constants `PROJECT_ROOT, SMALL_MAX_AREA,
  MEDIUM_MAX_AREA, SIZE_BUCKETS, AREA_RANGES, EMPTY_GT, EMPTY_PRED, EVAL_SETS, DEFAULT_CONFIG`.
* Config keys and defaults: see `DEFAULT_CONFIG` in common.py. `load_config()` makes path keys absolute.

## `phase5/metrics.py` — pure numpy, no IO

```python
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray          # (N,4),(M,4) xyxy -> (N,M)

def evaluate(gts: list[np.ndarray], preds: list[np.ndarray], *,
             iou_thrs=(0.5,), area_rng=(0.0, 1e10), area_scales: list[float] | None = None,
             max_det: int = 300, op_conf: float = 0.25, class_id: int | None = None) -> dict
```
* COCO-faithful (must match pycocotools `COCOeval` bbox numbers to 1e-6 on the same inputs, iscrowd=0,
  with `params.maxDets=[max_det]` and `params.areaRng=[area_rng]`):
  per image, keep only rows of `class_id` (None = class-agnostic: use every row, ignore the class column),
  sort dets by conf desc (stable), keep top `max_det`; GT area = bbox area * area_scales[i]**2
  (area_scales default 1.0 — used to express sizes in network-input pixels), det area likewise;
  gtIgnore = area outside `area_rng` (inclusive bounds, like COCO); COCO greedy matching per IoU threshold
  (best IoU, non-ignored GT preferred, `iou >= thr` with COCO's `min(t, 1-1e-10)` rule);
  dets matched to ignored GT are ignored; unmatched dets with area outside area_rng are ignored;
  accumulate across images sorted by conf with **mergesort** like COCO; precision envelope; AP = mean
  precision over the 101 recall points `np.linspace(0, 1, 101)` (COCO `searchsorted(..., side='left')`).
  If there are no non-ignored GT, AP = nan (COCO uses -1; we use nan), recall = nan.
* Returns dict (all floats are python floats, nan allowed):
  `n_images, n_gt (non-ignored), n_pred (non-ignored, all confs), ap50, ap (mean over iou_thrs; equals
  mAP50-95 when iou_thrs = np.linspace(.5,.95,10)), ap_per_iou {thr: ap}, op_conf, tp, fp, fn,
  precision, recall, f1` (at IoU = iou_thrs[0] for dets with conf >= op_conf), `best_f1, best_f1_conf,
  precision_at_best_f1, recall_at_best_f1` (scanning conf thresholds at iou_thrs[0]), `max_recall`
  (recall using all dets), and `pr_curve: {"recall": [...], "precision": [...], "conf": [...]}` at
  iou_thrs[0] (raw, un-interpolated, after sorting; may be subsampled to <= 1000 points).
  `ap50` = AP at IoU 0.5 if 0.5 in iou_thrs else nan.

```python
def evaluate_sizes(gts, preds, *, area_scales=None, iou_thrs=(0.5,), max_det=300, op_conf=0.25,
                   class_id=None) -> dict     # {"all": evaluate(...), "small": ..., "medium": ..., "large": ...}
def match_details(gt: np.ndarray, pred: np.ndarray, *, iou_thr=0.5, op_conf=0.25, bg_iou=0.1,
                  class_agnostic=True) -> dict
```
`match_details` (one image, used by failure analysis) — consider only preds with conf >= op_conf, sorted by
conf desc, greedy COCO matching at iou_thr (no area ignore). Returns
`gt_matched: bool (N,)`, `gt_match_conf: float (N,)` (nan if unmatched), `gt_best_iou: float (N,)` (max IoU
with any considered pred), `pred_status: list[str] (M_considered)` each in `{"tp","duplicate","localization","background"}`
(duplicate: unmatched but IoU >= iou_thr with an already-matched GT; localization: bg_iou <= max IoU < iou_thr;
background: max IoU < bg_iou), `pred_index: int array` mapping considered preds back to rows of `pred`.
With `class_agnostic=False` a pred may only match a GT of the same class.

## `phase5/make_holdouts.py` — freeze the held-out sets (CLI + API)

```python
def build_holdouts(cfg: dict, force: bool = False) -> dict      # returns the manifest dict
def load_holdout_records(cfg: dict, key: str) -> list[ImageRecord]   # key in {"conveyor", "sxsc7"}
def load_manifest(cfg: dict) -> dict
def main(argv=None) -> int     # py -3.10 -m phase5.make_holdouts --config phase5_config.yaml [--force]
```
* Writes `<splits_dir>/manifest.json` and, per key, `<key>_holdout.txt`, `<key>_train.txt`
  (absolute paths, `common.write_list`) — `<key>_train.txt` is what the training phase must use for that
  dataset (it already excludes every holdout image, all its Roboflow-augmented copies, and near-duplicates).
* Frozen: if manifest.json exists and `force` is False -> do not re-split; verify every holdout file still
  exists with the same sha1 (raise on mismatch) and return it. A changing holdout = leakage risk.
* Grouping (union-find) before splitting, so a group is never split across train/holdout:
  same sha1; same `common.source_stem` (Roboflow augmented copies); same video + frame within
  `video_gap` (default 10) frames when `common.video_frame` matches; dHash Hamming <= `near_dup_hamming`.
  Guard: if near-dup chaining makes one group > 50 % of the images, lower the hamming threshold step by
  step (log it) until it does not; if video frames are detected, prefer holding out whole videos or
  one contiguous frame block, with a `video_gap` buffer of frames removed from training on each side.
* conveyor: pool ALL splits of tomato_count_project; holdout = randomly chosen whole groups (seeded,
  `cfg["seed"]`) until >= `holdout.n_images` (default 30, minimum 30 — clamp up and warn) UNIQUE source
  images; everything else -> train. Warn if total unique images < 60.
* sxsc7: if split `holdout.use_existing_split` (default "test") exists with >= `holdout.min_images`
  unique images -> holdout = that split; else carve out `max(min_images, frac * n)` group-aware.
  Train = all other images MINUS anything grouped with a holdout image (log how many were removed).
  For big datasets compute near-dup only holdout-vs-rest (chunked `common.hamming_matrix`).
* Holdout evaluation list is deduplicated: one image per source stem (prefer test > valid > train split,
  then lexicographic) so augmented copies are not double-counted.
* Manifest per key: `dataset_root` (project-rel), `params`, `n_groups`, `holdout`/`train`/`excluded`
  lists of `{"path": project-rel posix, "sha1", "source_stem", "split", "group"}`, counts, and warnings.
  Top level: `version`, `seed`, `created` (ISO time), `near_dup_hamming`.

## `phase5/check_leakage.py` — prove holdouts are never trained on (CLI + API)

```python
def collect_train_images(spec: str) -> list[str]   # spec = data.yaml (reads train AND val entries, each a
                                                    # dir / list .txt / list of those, relative to yaml 'path'
                                                    # or the yaml folder), a list .txt, or an images dir
def check_leakage(cfg: dict, train_images: list[str], near_dup_hamming: int | None = None) -> dict
def main(argv=None) -> int   # py -3.10 -m phase5.check_leakage --config phase5_config.yaml --train data.yaml
                             #   [--train more.txt ...] [--write-clean out.txt] ; exit 1 if any LEAK
```
* Compares every training/validation image against every holdout image of every eval set
  (`conveyor_holdout`, `sxsc7_holdout` from the manifest, and `defects_test` from the dataset).
  LEAK: identical sha1, or dHash Hamming <= threshold. WARNING (not leak): same source stem only.
  Report: `{"ok": bool, "n_train": int, "leaks": [{"train", "holdout", "eval_set", "reason", "hamming"}],
  "warnings": [...], "min_hamming_per_set": {...}}`. Using held-out images for validation / model
  selection counts as leakage too (that is why yaml `val` is included).

## `phase5/predict.py` — Ultralytics inference with caching

```python
def model_info(model_path: str) -> dict   # {"names": {int: str}, "nc": int, "task": str, "imgsz": int|None, "sha1": str}
def run_predictions(model_path: str, records: list[ImageRecord], *, imgsz=640, conf=0.001, iou=0.7,
                    agnostic_nms=True, max_det=300, batch=16, device="", half=False,
                    cache_path: str | None = None) -> dict[str, np.ndarray]    # record.image -> (M, 6)
```
* Load `ultralytics.YOLO(model_path)` once; read images with `common.imread`, pass BGR numpy arrays in
  batches to `model.predict(..., verbose=False, stream=False)`; take `r.boxes.xyxy, conf, cls` (CPU float64).
  Task must be detect / segment / obb-free (raise for classify/pose/obb).
* Cache (JSON, `common.dump_json`): metadata `{model_sha1, imgsz, conf, iou, agnostic_nms, max_det,
  images_sha1 (sha1 of sorted image paths + file sizes)}`; reuse only if metadata matches exactly.
* Unreadable image -> warn and return `EMPTY_PRED` for it (never crash the whole run).

## `phase5/failure_analysis.py` — why tomatoes are missed (API, used by the evaluator)

```python
def analyze_failures(records: list[ImageRecord], gts: dict[str, np.ndarray], preds: dict[str, np.ndarray],
                     *, out_dir: str, op_conf=0.25, iou_thr=0.5, max_examples=8, blur_rel=0.5,
                     contrast_de=12.0, occl_ioa=0.2, border_px=2, bg_iou=0.1, set_name: str = "") -> dict
```
* gts/preds are CLASS-AGNOSTIC (cls already 0). Uses `metrics.match_details` per image.
* Each missed GT (false negative) gets every tag that applies:
  `small_object` (area < 32^2 px, original pixels), `motion_blur` (variance of Laplacian of the gray crop
  resized to 64x64 < `blur_rel` * median of the same score over matched GT crops of this set; absolute
  fallback 60.0 if < 5 matched crops; also record gradient anisotropy = max/min of mean |Sobel x|, |Sobel y|),
  `occlusion` (IoA with another GT >= `occl_ioa`, or box within `border_px` of the image edge = truncated),
  `belt_background` (CIELAB ΔE76 between the inner 60 % of the box and a surrounding ring of width 25 %
  of the box size < `contrast_de` — tomato blends into the belt), `other` (no tag applied).
* Each false positive at op_conf gets `pred_status` from match_details; `background` FPs are reported as
  `belt_background_fp` (model fires on belt/background), plus `localization_fp`, `duplicate_fp`.
* Writes example images to `<out_dir>/<tag>/` (up to `max_examples` images per tag, images with most
  failures of that tag first): full image annotated (TP GT green, FN red with tag text, FP orange with
  conf, matched preds thin blue), downscaled so the long side <= 1600 px, plus `<out_dir>/<tag>_crops.jpg`
  contact sheet of up to 48 crops (with 50 % context) sorted by severity. Always `common.imwrite`.
* Returns `{"set": set_name, "n_fn": int, "n_fp": int, "fn_tags": {tag: count}, "fp_tags": {tag: count},
  "fn_tag_share": {tag: fraction of FN}, "examples": {tag: [project-rel posix paths]},
  "contact_sheets": {tag: path}, "per_fn": [ {image, box, area, tags, blur, anisotropy, de, ioa} ... ]}`.

## `phase5/evaluate_class_agnostic.py` — main entry point (CLI + API)

```python
def build_eval_sets(cfg) -> dict[str, dict]
    # {set_key: {"records": [ImageRecord], "keep_ids": set|None, "names": [str], "dataset": key}}
def run(cfg: dict, predict_fn=None, crosscheck: bool = False, failures_all: bool = False) -> dict
def main(argv=None) -> int
    # py -3.10 -m phase5.evaluate_class_agnostic --config phase5_config.yaml
    #     [--crosscheck] [--failures-all] [--op-conf 0.25] [--no-cache]
```
* `predict_fn(records, *, agnostic_nms: bool) -> {image: (M,6)}`; default = `predict.run_predictions`
  with config params and cache file `<cache_dir>/<set_key>[_classnms].json`. Tests inject a fake one.
  Class-agnostic metrics use `agnostic_nms=cfg["agnostic_nms"]` (default True); the 4-class metrics on
  `defects_test` use a second call with `agnostic_nms=False` (normal per-class NMS, like `yolo val`).
* Holdout manifest missing -> exit with a clear error telling the user to run make_holdouts BEFORE
  training (never auto-create it at evaluation time: the model may already have seen those images).
* `--strict`: return exit code 3 when the recall gate fails (default: 0, but print the FAIL loudly).
* For each eval set: load GT (`read_yolo_labels` with the decoded image size — get h, w cheaply but
  correctly: decode with `common.imread`; it is fine to decode twice), class-agnostic GT/preds, then
  `metrics.evaluate` with iou_thrs = np.linspace(.5,.95,10) -> ap50, ap50_95, P/R/F1 at op_conf,
  best-F1 P/R; `metrics.evaluate_sizes` twice: original pixels and network-input pixels
  (`area_scales = letterbox_ratio(h, w, imgsz)`); per-size also report n_gt and recall at op_conf.
* 4-class metrics on `defects_test`: map dataset class names to model class names (case-insensitive,
  strip, '_' == ' ' == '-'); per class `evaluate(class_id=...)` after remapping GT ids into model ids with
  per-class NMS predictions (if agnostic NMS predictions were used, note that in the report); report
  per-class AP50, AP50-95, P, R and their mean; unmapped names -> warning, skipped.
* Recall gate: `conveyor_holdout` recall at op_conf < `recall_gate` (0.85) -> verdict FAIL, printed in
  capitals and as the first line of the report; run `failure_analysis.analyze_failures` on conveyor
  ALWAYS (and on all sets with `failures_all`), output under `<out_dir>/failures/<set_key>/`.
* `--crosscheck`: also run `ultralytics YOLO(model).val(single_cls=True, agnostic_nms=True, plots=False,
  conf=pred_conf, iou=nms_iou, imgsz, batch, split="val")` per set using a temp data yaml whose `val` is a
  list file of that set's images and whose `names` cover max(label id)+1 classes; report its mAP50/P/R
  next to ours and flag |ΔmAP50| > 0.02. (Note: crosscheck ignores `tomato_classes` filtering.)
* Outputs in `out_dir`: `class_agnostic_results.json` (everything), `class_agnostic_table.csv`
  (acceptance table), `PHASE5_class_agnostic_report.md` with, in order: gate verdict line; ACCEPTANCE
  TABLE (rows = 3 eval sets; columns = images, GT tomatoes, mAP50, mAP50-95, Precision@op_conf,
  Recall@op_conf, F1, P/R at best-F1 (conf), max recall, [crosscheck mAP50]); size tables (original px,
  input px) with n_gt / AP50 / recall@op_conf per bucket; 4-class table; failure analysis section
  (counts + share per tag + embedded example images with relative links); method notes (definitions,
  thresholds, NMS mode, holdout manifest sha1, interpretation note: "(c) is computed on the >= 30
  held-out conveyor images, the only conveyor images never seen in training"). Paths in markdown are
  relative to the report file and use forward slashes.
* Log the leakage check result if `check_leakage` was run (read `<out_dir>/leakage_report.json` if present).

## `phase5_config.yaml` (project root) — documented template of every key in DEFAULT_CONFIG.

## Tests: `phase5/tests/` (pytest-free: plain `python -m phase5.tests.<name>` scripts with asserts that
print `OK <name>`; pytest may also be used if installed). Synthetic data goes to a temp dir whose name
contains Persian characters (to catch unicode-path bugs).
