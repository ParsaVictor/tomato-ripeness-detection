"""Rebuild the conveyor clip from tomato_count_project: its images are frames 39..139 of one video, each frame
present in up to 3 Roboflow copies (some augmented: rotated / flipped / recoloured). For every frame number keep
the copy that looks most like the previously kept frame, then write the frames + an mp4.

    py tomato_app/make_conveyor_video.py      -> datasets/conveyor_sequence/{frames/, conveyor.mp4, frames.txt}
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tomato_inspect as ti  # noqa: E402

OUT = ti.ROOT / "datasets/conveyor_sequence"


def small(img):
    return cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32)


def main(fps=10):
    by_frame = defaultdict(list)
    for p in (ti.ROOT / "datasets/unified/u_conveyor").glob("*/images/*.jpg"):
        m = re.search(r"frame_0*(\d+)_", p.name)
        if m:
            by_frame[int(m.group(1))].append(p)
    nums = sorted(by_frame)
    # start: the copy of the first frame that best matches the copies of the next frame (unaugmented ones agree)
    nxt = [small(ti.imread(p)) for p in by_frame[nums[1]]]
    cur = min(by_frame[nums[0]], key=lambda p: min(np.abs(small(ti.imread(p)) - q).mean() for q in nxt))
    chosen, prev = [cur], small(ti.imread(cur))
    for n in nums[1:]:
        cands = [(np.abs(small(ti.imread(p)) - prev).mean(), p) for p in by_frame[n]]
        d, p = min(cands, key=lambda t: t[0])
        chosen.append(p)
        prev = small(ti.imread(p))
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    first = ti.imread(chosen[0])
    vw = cv2.VideoWriter(str(OUT / "conveyor.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (first.shape[1], first.shape[0]))
    for n, p in zip(nums, chosen):
        img = ti.imread(p)
        cv2.imencode(".jpg", img)[1].tofile(str(OUT / "frames" / f"frame_{n:04d}.jpg"))
        vw.write(img)
    vw.release()
    (OUT / "frames.txt").write_text("\n".join(f"{n}\t{p.relative_to(ti.ROOT).as_posix()}"
                                              for n, p in zip(nums, chosen)), encoding="utf-8")
    print(f"{len(chosen)} frames ({nums[0]}..{nums[-1]}) -> {OUT}")


if __name__ == "__main__":
    main()
