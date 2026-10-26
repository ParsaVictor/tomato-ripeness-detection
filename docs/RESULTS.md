# Detailed results

Settings: detector conf 0.25 · duplicate suppression IoU 0.55 (YOLO26 is NMS-free) · rotten check threshold 0.5 · CPU timing.

## Photos — detection and counting

| Test set | Images | GT | Predicted | Recall | Precision | F1 | Count acc. (per image) | Mean abs. count error | Images within ±1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Tomato Defects test | 86 | 768 | 815 | 99.0% | 93.3% | 96.0% | 93.8% | 0.66 | 83.7% |
| Conveyor holdout | 43 | 650 | 613 | 84.9% | 90.0% | 87.4% | 89.7% | 1.56 | 55.8% |
| sxsc7 holdout | 57 | 736 | 725 | 93.3% | 94.8% | 94.0% | 96.3% | 0.72 | 86.0% |

Original Colab evaluation of the detector alone (Ultralytics val, Tomato Defects test): mAP50 0.988 · mAP50-95 0.919.
Conveyor misses are mostly small / far tomatoes (93% of misses), motion blur (57%) and occlusion at the frame edge (42%).

## Ripeness (Tomato Defects test, IoU ≥ 0.5 matches)

| Class | GT | Predicted | Recall |
|---|---:|---:|---:|
| ripe | 196 | 221 | 80.7% |
| half_ripe | 339 | 363 | 80.7% |
| raw (unripe) | 186 | 182 | 79.5% |
| rotten | 47 | 49 | 87.0% |

4-class accuracy 80.8% · ripe / unripe / rotten 88.7% · sound vs rotten 98.6%.

### Rotten check (before → after)

| | Without | With |
|---|---:|---:|
| Healthy conveyor tomatoes called rotten (holdout, 613 detections) | 32 | **1** |
| Healthy sxsc7 tomatoes called rotten (holdout, 725 detections) | 20 | **3** |
| Truly rotten tomatoes kept (validation) | — | 97.5% |

## Video — tracking and counting

Moving-camera clips: a window slides across labelled holdout photos (8 clips per set), so the true number of distinct tomatoes is known. Slow = 2% of the window per frame, fast = 5%.

| Set | Speed | Truth | Counted | Accuracy | Line-count accuracy | Sum of per-frame counts / truth | Ripeness calls saved |
|---|---|---:|---:|---:|---:|---:|---:|
| sxsc7 | slow | 125 | 123 | 98.4% | 89.0% | 42× | 88% |
| sxsc7 | fast | 125 | 119 | 95.2% | 89.9% | 16× | 76% |
| conveyor | slow | 135 | 134 | 99.3% | 91.0% | 53× | 87% |
| conveyor | fast | 135 | 115 | 85.2% | 80.6% | 19× | 70% |

Enlarging the association boxes ×2 raised fast-conveyor counting from 65% to 85%.

## Tried, not enabled by default

**Tiled inference (SAHI-style, 2×2 tiles + full frame).** Conveyor recall 84.9% → 88.5%, but precision fell by the same amount and count accuracy did not improve; sxsc7 got slightly worse. Available as `--tiles 2` for cameras far from the belt.
