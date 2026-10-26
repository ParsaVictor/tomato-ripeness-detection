"""Video / production-line mode: track every tomato across frames, count each one once, grade it once.

Per frame:  detector + ByteTrack  -> every tomato keeps one ID while it is in view (a tomato missed in a few
            frames keeps its ID, so a miss does not cost a count)
Per track:  the ripeness classifier runs only on a few good views of the tomato (at most --cls-views, spaced
            --cls-gap frames apart, confident box not cut by the frame edge); the label is the average of
            those views (+ the rotten check). This is far cheaper and steadier than classifying every frame.
Counting:   a track is counted once it has been seen in --min-hits frames, or, with --line, when its centre
            crosses the counting line (recommended on a real belt: a tomato is counted when it passes).

Sources: a video file, a folder of frames (sorted by name), or a camera index (0, 1, ...).

    py tomato_app/tomato_video.py --source belt.mp4 --out results/video_belt
    py tomato_app/tomato_video.py --source 0 --line 0,0.6,1,0.6          # live camera, horizontal line at 60%
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tomato_inspect as ti  # noqa: E402

TRACKER_CFG = Path(__file__).resolve().parent / "bytetrack_tomato.yaml"
ASSOC_SCALE = 2.0   # boxes are enlarged by this for frame-to-frame matching (fast belts / small tomatoes)
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


@dataclass
class Track:
    tid: int
    first: int
    last: int = 0
    hits: int = 0
    box: list = field(default_factory=list)
    conf: float = 0.0
    prob_sum: dict = field(default_factory=dict)
    n_views: int = 0
    last_view: int = -10 ** 9
    counted: bool = False
    side: float | None = None           # side of the counting line the centre was on last frame

    @property
    def label(self) -> str | None:
        if not self.n_views:
            return None
        return max(self.prob_sum, key=self.prob_sum.get)

    @property
    def label_prob(self) -> float:
        return self.prob_sum[self.label] / self.n_views if self.n_views else 0.0


def frames(source: str):
    """Yields (index, frame BGR); also returns fps and frame count through the first item."""
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        fps, n = cap.get(cv2.CAP_PROP_FPS) or 25, None
    elif Path(source).is_dir():
        files = sorted(p for p in Path(source).iterdir() if p.suffix.lower() in ti.IMG_EXT)
        yield {"fps": None, "n": len(files)}
        for i, p in enumerate(files):
            yield i, ti.imread(p)
        return
    else:
        cap = cv2.VideoCapture(source)
        fps, n = cap.get(cv2.CAP_PROP_FPS) or 25, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    yield {"fps": fps, "n": n}
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield i, f
        i += 1
    cap.release()


class TomatoVideo:
    def __init__(self, insp: ti.TomatoInspector, min_hits=3, cls_views=3, cls_gap=4, cls_min_conf=0.4,
                 line=None, assoc_scale=ASSOC_SCALE):
        self.insp, self.min_hits, self.cls_views, self.cls_gap = insp, min_hits, cls_views, cls_gap
        self.cls_min_conf, self.line, self.assoc_scale = cls_min_conf, line, assoc_scale
        self.tracks: dict[int, Track] = {}
        self.cls_calls = 0
        self.det_boxes = 0

    def _line_side(self, x, y, w, h):
        x1, y1, x2, y2 = self.line[0] * w, self.line[1] * h, self.line[2] * w, self.line[3] * h
        return np.sign((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1))

    def _tracker(self, fps):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import YAML, IterableSimpleNamespace

        self.bt = BYTETracker(IterableSimpleNamespace(**YAML.load(TRACKER_CFG)), frame_rate=int(round(fps or 30)))

    def step(self, fi: int, frame: np.ndarray) -> tuple[list[Track], list[ti.Detection], list]:
        """Returns live tracks, every detection to draw this frame, and its track id (None = not tracked yet)."""
        from ultralytics.engine.results import Boxes

        if not hasattr(self, "bt"):
            self._tracker(None)
        h, w = frame.shape[:2]
        boxes, confs = self.insp._predict(frame, 0.1)        # low boxes feed ByteTrack's second association
        if self.insp.dedup and len(boxes):                  # YOLO26 is NMS-free: drop duplicate boxes
            k = ti.dedup_boxes(boxes, confs, self.insp.dedup)
            boxes, confs = boxes[k], confs[k]
        # the class column carries each detection's index: the tracker copies it onto the track it matched,
        # so every returned track points at its exact detection (its own box is Kalman-smoothed)
        ab = boxes
        if self.assoc_scale != 1 and len(boxes):             # enlarged copies keep overlapping between frames
            c, half = (boxes[:, :2] + boxes[:, 2:]) / 2, (boxes[:, 2:] - boxes[:, :2]) / 2 * self.assoc_scale
            ab = np.concatenate([c - half, c + half], 1)
        data = np.concatenate([ab, confs[:, None], np.arange(len(boxes))[:, None]], 1) if len(boxes) else             np.zeros((0, 6))
        out = self.bt.update(Boxes(data, (h, w)), frame)
        det_tid = [None] * len(boxes)
        for row in out:
            det_tid[int(round(row[6]))] = int(row[4])
        live, todo = [], []
        for j, (b, c) in enumerate(zip(boxes, confs)):
            tid = det_tid[j]
            if tid is None:
                continue
            t = self.tracks.setdefault(tid, Track(tid, fi))
            t.last, t.box, t.conf = fi, [float(v) for v in b], float(c)
            t.hits += 1
            live.append(t)
            edge = b[0] <= 2 or b[1] <= 2 or b[2] >= w - 2 or b[3] >= h - 2
            if t.n_views == 0 or (t.n_views < self.cls_views and fi - t.last_view >= self.cls_gap
                                  and c >= self.cls_min_conf and not edge):
                todo.append(t)
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if self.line is not None:
                side = self._line_side(cx, cy, w, h)
                if t.side is not None and side != 0 and side != t.side and not t.counted:
                    t.counted = True
                if side != 0:
                    t.side = side
            elif t.hits >= self.min_hits:
                t.counted = True
        # confident detections without a track yet (just appeared) are drawn and graded too, not counted
        loose = [ti.Detection([float(v) for v in boxes[j]], float(confs[j]), "", 0.0)
                 for j in range(len(boxes)) if det_tid[j] is None and confs[j] >= self.insp.conf]
        self.det_boxes += len(live) + len(loose)
        self._classify(frame, todo, fi, loose)
        shown = [ti.Detection(t.box, t.conf, t.label, t.label_prob) for t in live if t.label] +             [d for d in loose if d.cls]
        ids = [t.tid for t in live if t.label] + [None] * sum(bool(d.cls) for d in loose)
        return live, shown, ids

    def _classify(self, frame, todo: list[Track], fi: int, loose=()):
        crops, ts = [], []
        for t in list(todo) + list(loose):
            c = ti.crop_box(frame, t.box)
            if c is not None:
                crops.append(c)
                ts.append(t)
        if not crops:
            return
        res = self.insp.cls.predict(crops, imgsz=ti.CLS_IMGSZ, device=self.insp.device, verbose=False)
        self.cls_calls += len(crops)
        probs = []
        for rr in res:
            p = rr.probs.data.cpu().numpy()
            probs.append({self.insp.cls_names[k]: float(p[k]) for k in range(len(p))})
        rot = [k for k, p in enumerate(probs) if max(p, key=p.get) == "rotten"]
        if self.insp.rotten_check and rot:
            q = self.insp._rotten_prob([crops[k] for k in rot])
            for k, v in zip(rot, q):
                if v < self.insp.rotten_check[2]:              # check disagrees: drop the rotten vote
                    p = probs[k]
                    rest = 1 - p["rotten"]
                    probs[k] = {c: (0.0 if c == "rotten" else p[c] / max(rest, 1e-6)) for c in p}
        for t, p in zip(ts, probs):
            if isinstance(t, ti.Detection):
                t.cls = max(p, key=p.get)
                t.cls_prob = p[t.cls]
                continue
            for c, v in p.items():
                t.prob_sum[c] = t.prob_sum.get(c, 0.0) + v
            t.n_views += 1
            t.last_view = fi

    def counted(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.counted and t.label]


def summary_of(tracks: list[Track]) -> dict:
    dets = [ti.Detection(t.box, t.conf, t.label, t.label_prob) for t in tracks]
    return ti.summarize(dets)


def draw_frame(frame, shown, ids, vt: TomatoVideo, info: dict, H=900) -> Image.Image:
    h, w = frame.shape[:2]
    scale = H / h
    pic = ti.draw_detections(ti._resize(frame, scale), shown, scale, "id",
                             ids=[("" if i is None else i) for i in ids], tag_px=16)
    if vt.line is not None:
        d = ImageDraw.Draw(pic)
        x1, y1, x2, y2 = (vt.line[0] * w * scale, vt.line[1] * h * scale, vt.line[2] * w * scale,
                          vt.line[3] * h * scale)
        d.line([x1, y1, x2, y2], fill=(0, 0, 0), width=6)
        d.line([x1, y1, x2, y2], fill=(56, 189, 248), width=3)
    s = summary_of(vt.counted())
    panel = ti.draw_panel(max(H, 900), s, {
        "title": "شمارش گوجه روی خط تولید", "ms": info["ms"], "name": info["name"],
        "total_label": "گوجه‌های شمرده‌شده (هر گوجه یک بار)" if vt.line is None else "گوجه‌های عبورکرده از خط",
        "rows": [("گوجه در تصویر الان", str(len(shown))),
                 ("فریم", info["frame"]),
                 ("سرعت پردازش", f"{info['fps']:.1f} fps (CPU)"),
                 ("فراخوانی مدل رسیدگی / کادر", f"{vt.cls_calls} / {vt.det_boxes}")]}, pw=440)
    canvas = Image.new("RGB", (pic.width + panel.width, panel.height), ti.BG)
    canvas.paste(pic, (0, (panel.height - pic.height) // 2))
    canvas.paste(panel, (pic.width, 0))
    return canvas


def run(source: str, out: Path, insp=None, fps_out=None, line=None, show=False, every=1, **kw) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    insp = insp or ti.TomatoInspector()
    vt = TomatoVideo(insp, line=line, **kw)
    it = frames(source)
    meta = next(it)
    vt._tracker(fps_out or meta["fps"])
    fps_out = fps_out or meta["fps"] or 10
    writer, t_all, n_done = None, time.perf_counter(), 0
    per_frame = []
    for fi, frame in it:
        if frame is None or fi % every:
            continue
        t0 = time.perf_counter()
        live, shown, ids = vt.step(fi, frame)
        ms = (time.perf_counter() - t0) * 1000
        n_done += 1
        per_frame.append({"frame": fi, "detected": len(shown), "tracked": len(live), "counted": len(vt.counted())})
        img = draw_frame(frame, shown, ids, vt, {"ms": ms, "name": Path(source).name,
                                           "frame": f"{fi + 1}" + (f" / {meta['n']}" if meta["n"] else ""),
                                           "fps": 1000 / max(ms, 1)})
        bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        if writer is None:
            writer = cv2.VideoWriter(str(out / "tracked.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_out,
                                     (bgr.shape[1], bgr.shape[0]))
        writer.write(bgr)
        if fi in (0, 10) or fi % 25 == 0:
            cv2.imencode(".jpg", bgr)[1].tofile(str(out / f"frame_{fi:05d}.jpg"))
        if show:
            cv2.imshow("tomato", bgr)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    if writer:
        writer.release()
    if n_done:
        cv2.imencode(".jpg", bgr)[1].tofile(str(out / "last_frame.jpg"))
    counted = vt.counted()
    with open(out / "tracks.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["track_id", "first_frame", "last_frame", "frames_seen", "counted", "label", "label_prob",
                     "classifier_views"])
        for t in sorted(vt.tracks.values(), key=lambda t: t.tid):
            wr.writerow([t.tid, t.first, t.last, t.hits, t.counted, t.label, round(t.label_prob, 3), t.n_views])
    s = summary_of(counted)
    res = {"source": source, "frames": n_done, "seconds": round(time.perf_counter() - t_all, 1),
           "tracks_total": len(vt.tracks), "counted": s["total"], "sound": s["sound"], "ripe": s["ripe"],
           "unripe": s["unripe"], "rotten": s["rotten"], "per_class": s["per_class"],
           "classifier_calls": vt.cls_calls, "detected_boxes": vt.det_boxes,
           "classifier_saving": round(1 - vt.cls_calls / max(vt.det_boxes, 1), 3),
           "per_frame": per_frame}
    (out / "summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file, folder of frames, or camera index")
    ap.add_argument("--out", default=str(ti.ROOT / "results" / "video"))
    ap.add_argument("--line", help="counting line x1,y1,x2,y2 as fractions of width/height, e.g. 0,0.6,1,0.6")
    ap.add_argument("--min-hits", type=int, default=3, help="frames a track needs before it is counted")
    ap.add_argument("--cls-views", type=int, default=3, help="classifier runs per tomato")
    ap.add_argument("--cls-gap", type=int, default=4, help="frames between two classifier runs of one tomato")
    ap.add_argument("--fps", type=float, help="output fps (default: source fps, or 10 for a frame folder)")
    ap.add_argument("--every", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--show", action="store_true", help="show a live window (Esc stops)")
    a = ap.parse_args(argv)
    line = [float(v) for v in a.line.split(",")] if a.line else None
    res = run(a.source, Path(a.out), fps_out=a.fps, line=line, show=a.show, every=a.every, min_hits=a.min_hits,
              cls_views=a.cls_views, cls_gap=a.cls_gap)
    print(json.dumps({k: v for k, v in res.items() if k != "per_frame"}, ensure_ascii=False, indent=1))
    print("->", a.out)


if __name__ == "__main__":
    main()
