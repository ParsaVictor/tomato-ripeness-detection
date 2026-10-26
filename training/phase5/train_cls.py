"""Stage-2 training: YOLO classify model on tomato crops (raw / half_ripe / ripe / rotten).

Trains straight into <project>/run on Google Drive (small model: a checkpoint every epoch is cheap), resumes
from last.pt after a disconnect, writes DONE when finished (then a rerun does nothing) and test.json with
the accuracy on the Tomato Defects test crops.

  python -m phase5.train_cls --data /content/work/cls_tomato --project /content/drive/.../runs/cls \
      --model yolo26s-cls.pt --imgsz 160 --epochs 60
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--model", default="yolo26s-cls.pt")
    ap.add_argument("--imgsz", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    run = Path(a.project) / "run"
    if (run / "DONE").exists():
        print("[cls] training already finished:", run / "weights" / "best.pt", flush=True)
    else:
        last = run / "weights" / "last.pt"
        if last.exists():
            print("[cls] resuming from", last, flush=True)
            try:
                YOLO(str(last)).train(resume=True)
            except AssertionError:
                print("[cls] nothing to resume; treating as finished", flush=True)
        else:
            print(f"[cls] new run: {a.model}, imgsz {a.imgsz}, {a.epochs} epochs", flush=True)
            YOLO(a.model).train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, patience=a.patience,
                                project=str(Path(a.project)), name="run", exist_ok=True, workers=a.workers,
                                fraction=a.fraction, device=a.device, plots=True, seed=0,
                                fliplr=0.5, flipud=0.5, hsv_h=0.01, hsv_s=0.4, hsv_v=0.3, erasing=0.2)
        (run / "DONE").write_text("ok")
    best = run / "weights" / "best.pt"
    m = YOLO(str(best)).val(data=a.data, split="test", imgsz=a.imgsz, batch=a.batch, plots=True,
                            project=str(run), name="test_eval", exist_ok=True, device=a.device, verbose=False)
    res = {"top1": float(m.top1), "top5": float(m.top5)}
    (run / "test.json").write_text(json.dumps(res), encoding="utf-8")
    print(f"[cls] TEST accuracy (Tomato Defects test crops): top1 {res['top1']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
