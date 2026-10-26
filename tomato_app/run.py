"""One command for everything: photo, folder of photos, video file, folder of video frames, or a camera.

    py tomato_app/run.py photo.jpg
    py tomato_app/run.py folder_of_photos/
    py tomato_app/run.py belt.mp4 --line 0,0.6,1,0.6
    py tomato_app/run.py 0 --show                         # camera 0, live window
    py tomato_app/run.py frames_folder/ --video --fps 25  # a folder holding consecutive video frames

Results go to results/run_<name>/ unless --out is given.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tomato_inspect as ti  # noqa: E402
import tomato_video as tv  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("--out")
    ap.add_argument("--video", action="store_true", help="treat a folder as consecutive video frames")
    ap.add_argument("--line", help="video: counting line x1,y1,x2,y2 (fractions of width / height)")
    ap.add_argument("--fps", type=float)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--tiles", type=int, default=1, help="photos: 2 = extra tiled pass for far, small tomatoes")
    a = ap.parse_args()
    src = Path(a.source)
    out = Path(a.out) if a.out else ti.ROOT / "results" / f"run_{src.stem or 'camera'}"
    is_video = a.source.isdigit() or src.suffix.lower() in tv.VIDEO_EXT or (src.is_dir() and a.video)
    if is_video:
        line = [float(v) for v in a.line.split(",")] if a.line else None
        res = tv.run(a.source, out, fps_out=a.fps, line=line, show=a.show)
        print(f"counted {res['counted']} tomatoes | sound {res['sound']} | ripe {res['ripe']} | "
              f"unripe {res['unripe']} | rotten {res['rotten']}  ->  {out}")
    else:
        ti.main(["--source", a.source, "--out", str(out), "--tiles", str(a.tiles)])


if __name__ == "__main__":
    main()
