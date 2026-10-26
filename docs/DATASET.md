# Unified Tomato Dataset — detection + ripeness

**8,497 images · 74,486 tomato boxes · 4 ripeness classes · leakage-free holdouts**

Six public tomato datasets from Roboflow Universe, merged into **one consistent YOLO schema**, cleaned and split so that results are honest:

- one label schema: `raw` (unripe), `half_ripe`, `ripe`, `rotten`, and `tomato` where the source has no ripeness label
- fixed **holdouts** for the conveyor and sxsc7 sources that no model ever trains on
- **near-duplicate removal** (perceptual dHash): 1,234 train images that were near-copies of test images were dropped, and 221 images too similar to a holdout image were set aside
- ready-made **classifier crops** (160×160) for ripeness: 50,776 train / 8,579 val / 768 test

## Sources

| Part | Source (Roboflow Universe) | Images | Boxes | Original classes → unified |
|---|---|---:|---:|---|
| Tomato Defects | [universitas-hasanuddin-lq2k8/tomato-defects-80syu](https://universe.roboflow.com/universitas-hasanuddin-lq2k8/tomato-defects-80syu) | 2,501 | 21,168 | half_ripe, raw, ripe, rotten |
| tomato-sxsc7 | [dai-tran-van/tomato-sxsc7](https://universe.roboflow.com/dai-tran-van/tomato-sxsc7) | 600 | 5,318 | tomato |
| Conveyor belt | [canberk-qi0vx/tomato_count_project](https://universe.roboflow.com/canberk-qi0vx/tomato_count_project) | 269 | 3,928 | B, Tomato, Tomatob → tomato |
| Extra (ripeness) | [papapeto/tomato-v1-2mdez](https://universe.roboflow.com/papapeto/tomato-v1-2mdez) | 5,127 (3 sources) | 44,072 | fully-ripe → ripe, semi-ripe → half_ripe, unripe → raw |
| | [tomato-6fguu/tomato-detection-sorting](https://universe.roboflow.com/tomato-6fguu/tomato-detection-sorting) | | | Fresh → ripe, Rotten → rotten, Unripe → raw |
| | [ai-24uje/tomato-classification-2kd0n](https://universe.roboflow.com/ai-24uje/tomato-classification-2kd0n) | | | ripe tomato → ripe, unripe tomato → raw |
| **Total** | | **8,497** | **74,486** | |

Box counts per class: raw 31,406 · half_ripe 16,952 · ripe 15,307 · rotten 1,575 · tomato (no ripeness) 9,246.

## Splits

| Use | Train | Val | Test / holdout |
|---|---:|---:|---:|
| Detector (one class) | 5,783 | 1,073 | 86 (Tomato Defects test) |
| Conveyor holdout | — | — | 43 images (frames 50–70, 91–113) |
| sxsc7 holdout | — | — | 57 images |
| Ripeness crops | 50,776 | 8,579 | 768 |

Holdout lists: [`tomato_app/holdouts/`](../tomato_app/holdouts). Rotten crops in train are ×4 (flip / rotations) because the class is rare.

## Layout

```
unified/
  u_defects/{train,valid,test}/{images,labels}   classes: raw, half_ripe, ripe, rotten
  u_sxsc7/...                                    class: tomato
  u_conveyor/...                                 class: tomato   (frames 39–139 of one video)
  u_extra/...                                    classes: raw, half_ripe, ripe, rotten, tomato
cls_tomato/{train,val,test}/{raw,half_ripe,ripe,rotten}/*.jpg   160×160 crops
```

Labels are standard YOLO txt (`class cx cy w h`, normalised).

## Download

🤗 **Hugging Face: [ParsaVictor/tomato-ripeness-detection](https://huggingface.co/datasets/ParsaVictor/tomato-ripeness-detection)** — images + YOLO labels, ripeness crops, holdout lists and demo videos.

## License & attribution

All six sources are published under **CC BY 4.0**; this unified version is released under **CC BY 4.0** as well. If you use it, please credit the original authors above and this repository.
