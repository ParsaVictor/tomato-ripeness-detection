# Model family

One **complete pipeline** and its **components**, each usable on its own.

| | Model | Input → output | Key result | Download |
|---|---|---|---|---|
| 🍅 **Complete pipeline** | detector + classifier + rotten check (+ ByteTrack in `tomato_app/`) | photo / video → every tomato, its ripeness, a once-per-tomato count | 95–99% video counting accuracy | [🤗 tomato-ripeness-yolo26](https://huggingface.co/ParsaVictor/tomato-ripeness-yolo26) · [GitHub Release](https://github.com/ParsaVictor/tomato-ripeness-detection/releases/latest) |
| 🔍 **Detector** | YOLO26s, 640 px, 1 class | image → tomato boxes | mAP50 **0.988**, recall 99% (Tomato Defects test) | [🤗 tomato-detector-yolo26s](https://huggingface.co/ParsaVictor/tomato-detector-yolo26s) |
| 🎨 **Ripeness classifier** | YOLO26s-cls, 160 px crop | one tomato → unripe / half-ripe / ripe / rotten | 80.8% 4-class · **98.6%** sound vs rotten | [🤗 tomato-ripeness-classifier-yolo26s](https://huggingface.co/ParsaVictor/tomato-ripeness-classifier-yolo26s) |
| 🛡️ Rotten check (module of the classifier) | logistic regression on the 512-d embedding | "rotten" crop → keep / reject | false rotten on conveyor 32 → **1** | shipped with the classifier |

## Why two stages instead of one 4-class detector?

- **More data for detection.** Only ~2,500 of the 8,497 images carry ripeness labels. A one-class detector can learn from all of them (conveyor and sxsc7 included), which is what makes it robust on unseen scenes.
- **Better ripeness judgement.** The classifier sees a tight 160×160 crop of one fruit instead of a small region of a 640 px image.
- **Cheaper video.** With tracking, the classifier runs only on a few good views per tomato (70–88% fewer calls) while the detector runs every frame.
- **Independent upgrades.** Either stage can be retrained or swapped without touching the other.

## Files

| File | Size | Used by |
|---|---:|---|
| `weights/detector_yolo26s_best.pt` | 20 MB | detector |
| `weights/ripeness_yolo26s-cls_best.pt` | 11 MB | classifier |
| `weights/rotten_check.npz` | 3 KB | rotten check |
| `weights/thresholds.json` | <1 KB | detection threshold chosen on validation |

All are released under AGPL-3.0 (trained with Ultralytics YOLO).
