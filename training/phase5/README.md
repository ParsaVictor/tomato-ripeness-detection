# Phase 5 — class-agnostic "tomato vs background" evaluation

## فارسی (خلاصه)

هدف: علاوه بر معیارهای ۴ کلاسه، همهٔ کلاس‌ها یک کلاس «گوجه» در نظر گرفته می‌شوند و mAP50، Precision و Recall
روی سه مجموعه گزارش می‌شود: (a) تست Tomato Defects، (b) زیرمجموعهٔ کنار گذاشته‌شدهٔ tomato-sxsc7، (c) حداقل ۳۰ تصویر
کنار گذاشته‌شدهٔ نوار نقاله (tomato_count_project) که هرگز در آموزش استفاده نمی‌شوند (~۷۰ تصویر دیگر برای آموزش).

ترتیب مراحل (از ریشهٔ پروژه):

1. دیتاست‌ها را در `datasets/<name>` دانلود کنید (همان نام‌های `phase5_config.yaml`:
   `datasets/tomato-defects`، `datasets/tomato-sxsc7`، `datasets/tomato_count_project`).
2. **قبل از آموزش**: `py -3.10 -m phase5.make_holdouts --config phase5_config.yaml`
3. آموزش را با `splits/conveyor_train.txt` و `splits/sxsc7_train.txt` به جای پوشه‌های خام انجام دهید —
   هرگز از فایل‌های `*_holdout.txt` برای train یا val استفاده نکنید.
4. بررسی نشت داده: `py -3.10 -m phase5.check_leakage --config phase5_config.yaml --train <your data.yaml>`
5. ارزیابی: `py -3.10 -m phase5.evaluate_class_agnostic --config phase5_config.yaml [--crosscheck]`
6. گزارش را بخوانید: `reports/phase5/PHASE5_class_agnostic_report.md`

اگر Recall کلاس‌آگنوستیک روی نوار نقاله کمتر از 0.85 باشد، خط اول گزارش FAIL است و انواع خطا (اشیای کوچک،
تاری حرکت، هم‌پوشانی، زمینهٔ نوار) با تصاویر نمونه آورده می‌شود.

## English

Steps (run from the project root):

1. Download the datasets into `datasets/<name>` using the names in `phase5_config.yaml`.
2. **Before training**: `py -3.10 -m phase5.make_holdouts --config phase5_config.yaml` — freezes the holdouts
   (`splits/manifest.json`, `splits/<key>_holdout.txt`, `splits/<key>_train.txt`). Re-running never re-splits
   unless `--force` (a changing holdout is a leakage risk).
3. Train using `splits/conveyor_train.txt` and `splits/sxsc7_train.txt` instead of the raw folders — never the
   holdout lists (not for `train`, not for `val`/model selection either).
4. `py -3.10 -m phase5.check_leakage --config phase5_config.yaml --train <your data.yaml>` (exit 1 on any leak;
   writes `reports/phase5/leakage_report.json`, which the report then quotes).
5. `py -3.10 -m phase5.evaluate_class_agnostic --config phase5_config.yaml [--crosscheck] [--failures-all]
   [--op-conf 0.25] [--no-cache] [--strict]` (`--strict` → exit code 3 when the recall gate fails).
6. Read `reports/phase5/PHASE5_class_agnostic_report.md`.

### Outputs (`out_dir`, default `reports/phase5/`)

| File | Meaning |
|---|---|
| `PHASE5_class_agnostic_report.md` | First line = recall-gate verdict; acceptance table; size tables; 4-class table; failure analysis with example images; method notes |
| `class_agnostic_table.csv` | The acceptance table (3 rows: defects_test, sxsc7_holdout, conveyor_holdout) |
| `class_agnostic_results.json` | Everything (all metrics, PR curves, size buckets, per-FN failure records) |
| `failures/<set>/<tag>/*.jpg`, `<tag>_crops.jpg` | Annotated failure examples (green = detected, red = missed with tag, orange = false positive) |
| `pred_cache/*.json` | Cached predictions (reused only if model / params / images are identical) |

### Definitions

- **Class-agnostic**: all classes (or `tomato_classes`) merged into "tomato"; predicted class ignored.
- **mAP50**: COCO 101-point AP at IoU 0.5 (pycocotools-equivalent); **mAP50-95**: mean over IoU 0.50:0.05:0.95.
- **P / R / F1 @ op_conf**: detections with conf >= `op_conf` (0.25), greedy matching at IoU >= 0.5.
- **P/R @ best-F1**: the confidence threshold maximising F1; **max recall**: recall using every prediction.
- **Sizes**: small < 32² px² ≤ medium < 96² ≤ large, both in original pixels and in network-input pixels
  (box area × letterbox ratio², ratio = min(imgsz/h, imgsz/w)).
- **Recall gate**: conveyor_holdout recall @ op_conf must be >= 0.85.
- **Failure tags** (missed tomatoes): `small_object`, `motion_blur`, `occlusion` (incl. truncated at the edge),
  `belt_background` (low colour contrast to the belt), `other`; FPs: `belt_background_fp`, `localization_fp`,
  `duplicate_fp`.
- (c) is computed on the >= 30 held-out conveyor images — the only conveyor images never seen in training.

Tests (synthetic data, no downloads): `py -3.10 -m phase5.tests.test_end_to_end`
