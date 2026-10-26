# Model weights

Put the trained files here:

- `detector_yolo26s_best.pt` — tomato detector (YOLO26s, 1 class)
- `ripeness_yolo26s-cls_best.pt` — ripeness classifier (YOLO26s-cls, 4 classes)
- `rotten_check.npz` — rotten second-opinion head

They are available on request: [open an issue](https://github.com/ParsaVictor/tomato-ripeness-detection/issues/new?title=Model%20weights%20request).
You can also train your own with `training/Tomato_Model_Colab.ipynb` and `tomato_app/train_rotten_check.py`.
