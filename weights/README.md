# Model weights

Put the trained files here:

- `detector_yolo26s_best.pt` — tomato detector (YOLO26s, 1 class)
- `ripeness_yolo26s-cls_best.pt` — ripeness classifier (YOLO26s-cls, 4 classes)
- `rotten_check.npz` — rotten second-opinion head

Download them from Hugging Face: [https://huggingface.co/ParsaVictor/tomato-ripeness-yolo26](https://huggingface.co/ParsaVictor/tomato-ripeness-yolo26)

    hf download ParsaVictor/tomato-ripeness-yolo26 --include "weights/*" --local-dir .
You can also train your own with `training/Tomato_Model_Colab.ipynb` and `tomato_app/train_rotten_check.py`.
