"""Two-stage tomato inspection: count every tomato in a photo and grade its ripeness.

Stage 1  detector  (YOLO26s, one class "tomato")          -> where is every tomato
Stage 2  classifier (YOLO26s-cls on a crop of each tomato) -> raw / half_ripe / ripe / rotten

Every result image gets a clean annotation (thin outlined boxes, small numbered tags, one colour per class)
and a side dashboard with the counts: total, sound, ripe, unripe, rotten and the per-class breakdown.

    py tomato_app/tomato_inspect.py --source path/to/photo_or_folder --out results/my_run
    py tomato_app/tomato_inspect.py --random 6 --set conveyor --out results/random_conveyor
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
DET_WEIGHTS = WEIGHTS / "detector_yolo26s_best.pt"
CLS_WEIGHTS = WEIGHTS / "ripeness_yolo26s-cls_best.pt"
DET_CONF = 0.25             # standard operating point; 0.334 = F1-max on val (weights/thresholds.json)
CLS_IMGSZ = 160             # the classifier was trained on 160x160 crops
UNSURE_PROB = 0.55          # ripeness probability below this -> tag gets a "?"
TILES = 1                   # >1 = also run an NxN grid of overlapping tiles (small far tomatoes)
TILE_CONF = None            # confidence for tile boxes (None = same as DET_CONF)
TILE_MAX_PX = 1e9           # keep a tile box only if the tomato is smaller than this in the whole-image 640 input
DEDUP_IOU = 0.55            # YOLO26 is NMS-free; drop a box that overlaps a stronger one this much (0 = off)
ROTTEN_CHECK = WEIGHTS / "rotten_check.npz"   # second opinion on "rotten" (tomato_app/train_rotten_check.py)
ROTTEN_CHECK_T = 0.5        # keep "rotten" only if the check's probability >= this
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# class -> (Persian name, English label, RGB colour). Order = display order.
CLASSES = {
    "ripe":      ("رسیده",      "Ripe",      (239, 68, 68)),
    "half_ripe": ("نیمه‌رسیده", "Half-ripe", (245, 158, 11)),
    "raw":       ("نارس",       "Unripe",    (34, 197, 94)),
    "rotten":    ("خراب",       "Rotten",    (168, 85, 247)),
}
GT_ONLY_TOMATO = "tomato"   # datasets whose labels have no ripeness (conveyor, sxsc7)

SETS = {  # evaluation sets with ground-truth labels, never used for training
    "defects":  {"fa": "تست دیتاست Tomato Defects", "classes": ["raw", "half_ripe", "ripe", "rotten"],
                 "images": lambda: sorted((ROOT / "datasets/unified/u_defects/test/images").glob("*.jpg"))},
    "conveyor": {"fa": "نوار نقاله (holdout)", "classes": [GT_ONLY_TOMATO],
                 "images": lambda: _list_file(ROOT / "tomato_app/holdouts/conveyor_holdout.txt")},
    "sxsc7":    {"fa": "tomato-sxsc7 (holdout)", "classes": [GT_ONLY_TOMATO],
                 "images": lambda: _list_file(ROOT / "tomato_app/holdouts/sxsc7_holdout.txt")},
}


def _list_file(p: Path) -> list[Path]:
    return [ROOT / s.strip() for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]


def imread(path) -> np.ndarray | None:
    """Unicode-safe BGR read (the project lives under a Persian path)."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None


def crop_box(img, box, pad=0.1, size=CLS_IMGSZ):
    """Same crop rule the classifier was trained with (phase5/prepare_data.crop_box)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return None
    xa, ya = int(max(0, x1 - pad * bw)), int(max(0, y1 - pad * bh))
    xb, yb = int(min(w, x2 + pad * bw)), int(min(h, y2 + pad * bh))
    crop = img[ya:yb, xa:xb]
    if not crop.size:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA) if size else crop


def label_path(img_path: Path) -> Path:
    return img_path.parent.parent / "labels" / (img_path.stem + ".txt")


def read_gt(img_path: Path, w: int, h: int, names: list[str]) -> list[dict]:
    lp = label_path(img_path)
    out = []
    if lp.exists():
        for line in lp.read_text().splitlines():
            v = line.split()
            if len(v) < 5:
                continue
            c, cx, cy, bw, bh = int(v[0]), *map(float, v[1:5])
            out.append({"cls": names[c] if c < len(names) else GT_ONLY_TOMATO,
                        "box": [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]})
    return out


# ----------------------------------------------------------------------------------------------- model
@dataclass
class Detection:
    box: list[float]
    det_conf: float
    cls: str
    cls_prob: float
    probs: dict = field(default_factory=dict)


class TomatoInspector:
    def __init__(self, det_weights=DET_WEIGHTS, cls_weights=CLS_WEIGHTS, conf=DET_CONF, iou=0.7, device="cpu",
                 dedup=DEDUP_IOU, tiles=TILES, tile_overlap=0.25, tile_conf=TILE_CONF,
                 tile_max_px=TILE_MAX_PX, rotten_check=ROTTEN_CHECK,
                 rotten_check_t=ROTTEN_CHECK_T):
        from ultralytics import YOLO

        self.det, self.cls = YOLO(str(det_weights)), YOLO(str(cls_weights))
        self.conf, self.iou, self.device, self.dedup = conf, iou, device, dedup
        self.tiles, self.tile_overlap, self.tile_max_px = tiles, tile_overlap, tile_max_px
        self.tile_conf = conf if tile_conf is None else tile_conf
        self.cls_names = {int(k): v for k, v in self.cls.names.items()}
        self.rotten_check = None
        if rotten_check and Path(rotten_check).exists():
            z = np.load(rotten_check)
            self.rotten_check = (z["coef"], float(z["intercept"]), rotten_check_t)
            self.emb = YOLO(str(cls_weights))   # own instance: embed() leaves its predictor in embed mode

    def _rotten_prob(self, crops) -> np.ndarray:
        coef, b, _ = self.rotten_check
        e = np.asarray([t.cpu().numpy() for t in self.emb.embed(crops, imgsz=CLS_IMGSZ, device=self.device,
                                                                 verbose=False)], np.float32)
        e /= np.linalg.norm(e, axis=1, keepdims=True).clip(1e-6)
        return 1 / (1 + np.exp(-(e @ coef + b)))

    def _predict(self, img, conf):
        r = self.det.predict(img, imgsz=640, conf=conf, iou=self.iou, agnostic_nms=True, max_det=300,
                             device=self.device, verbose=False)[0]
        if r.boxes is None or not len(r.boxes):
            return np.zeros((0, 4)), np.zeros(0)
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()

    def detect(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Whole image + (optionally) an NxN grid of overlapping tiles, each upscaled to 640, so small far
        tomatoes get more pixels (the idea of SAHI). Tile boxes cut by an inner tile edge are dropped: the
        neighbouring tile or the whole-image pass sees that tomato complete."""
        boxes, confs = self._predict(img, self.conf)
        if self.tiles > 1:
            h, w = img.shape[:2]
            n, ov = self.tiles, self.tile_overlap
            tw, th = int(np.ceil(w / (n - (n - 1) * ov))), int(np.ceil(h / (n - (n - 1) * ov)))
            xs = np.linspace(0, w - tw, n).round().astype(int)
            ys = np.linspace(0, h - th, n).round().astype(int)
            all_b, all_c = [boxes], [confs]
            for y0 in ys:
                for x0 in xs:
                    b, c = self._predict(img[y0:y0 + th, x0:x0 + tw], self.tile_conf)
                    if not len(b):
                        continue
                    edge = 3
                    cut = (((b[:, 0] <= edge) & (x0 > 0)) | ((b[:, 2] >= tw - edge) & (x0 + tw < w)) |
                           ((b[:, 1] <= edge) & (y0 > 0)) | ((b[:, 3] >= th - edge) & (y0 + th < h)))
                    side = np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) * 640 / max(h, w)
                    small = side < self.tile_max_px          # small in the whole-image 640 input
                    b, c = b[~cut & small], c[~cut & small]
                    all_b.append(b + [x0, y0, x0, y0])
                    all_c.append(c)
            boxes, confs = np.concatenate(all_b), np.concatenate(all_c)
        return boxes, confs

    def __call__(self, img: np.ndarray) -> tuple[list[Detection], float]:
        t0 = time.perf_counter()
        boxes, confs = self.detect(img)
        if self.dedup and len(boxes):
            keep_i = dedup_boxes(boxes, confs, self.dedup)
            boxes, confs = boxes[keep_i], confs[keep_i]
        crops, keep = [], []
        for i, b in enumerate(boxes):
            c = crop_box(img, b)
            if c is not None:
                crops.append(c)
                keep.append(i)
        dets = []
        if crops:
            res = self.cls.predict(crops, imgsz=CLS_IMGSZ, device=self.device, verbose=False)
            for i, rr in zip(keep, res):
                p = rr.probs.data.cpu().numpy()
                probs = {self.cls_names[k]: float(p[k]) for k in range(len(p))}
                top = int(p.argmax())
                dets.append(Detection([float(v) for v in boxes[i]], float(confs[i]), self.cls_names[top],
                                      float(p[top]), probs))
            rot = [k for k, d in enumerate(dets) if d.cls == "rotten"]
            if self.rotten_check and rot:
                pr = self._rotten_prob([crops[k] for k in rot])
                for k, q in zip(rot, pr):
                    d = dets[k]
                    d.probs["rotten_check"] = float(q)
                    if q < self.rotten_check[2]:              # check disagrees: best sound class instead
                        alt = max((c for c in d.probs if c in CLASSES and c != "rotten"), key=d.probs.get)
                        d.cls, d.cls_prob = alt, d.probs[alt] / max(1e-6, 1 - d.probs["rotten"])
        dets.sort(key=lambda d: (round(d.box[1] / 40), d.box[0]))   # reading order: rows, then left to right
        return dets, (time.perf_counter() - t0) * 1000


def dedup_boxes(boxes: np.ndarray, confs: np.ndarray, thr: float) -> list[int]:
    """Greedy suppression: a box is a duplicate when IoU > thr or it lies >85% inside a stronger box."""
    keep: list[int] = []
    for i in np.argsort(-confs):
        b = boxes[i]
        area = (b[2] - b[0]) * (b[3] - b[1])
        dup = False
        for k in keep:
            q = boxes[k]
            inter = max(0, min(b[2], q[2]) - max(b[0], q[0])) * max(0, min(b[3], q[3]) - max(b[1], q[1]))
            qa = (q[2] - q[0]) * (q[3] - q[1])
            if inter / (area + qa - inter) > thr or inter / min(area, qa) > 0.85:
                dup = True
                break
        if not dup:
            keep.append(int(i))
    return sorted(keep)


def summarize(dets: list[Detection]) -> dict:
    n = {k: sum(d.cls == k for d in dets) for k in CLASSES}
    return {"total": len(dets), "per_class": n, "sound": len(dets) - n["rotten"], "ripe": n["ripe"],
            "unripe": n["raw"] + n["half_ripe"], "rotten": n["rotten"],
            "mean_det_conf": float(np.mean([d.det_conf for d in dets])) if dets else None,
            "unsure": sum(d.cls_prob < UNSURE_PROB for d in dets)}


# ----------------------------------------------------------------------------------------------- drawing
FONT_FA = "C:/Windows/Fonts/tahoma.ttf"
FONT_FA_B = "C:/Windows/Fonts/tahomabd.ttf"
FONT_EN = "C:/Windows/Fonts/segoeui.ttf"
FONT_EN_B = "C:/Windows/Fonts/segoeuib.ttf"
BG, CARD, CARD2, MUTED, TEXT = (11, 15, 25), (24, 30, 44), (32, 40, 58), (148, 163, 184), (241, 245, 249)
_fonts: dict = {}


def _fallback(path: str) -> str:
    """Windows fonts when present, else DejaVu Sans (ships with matplotlib, has Persian glyphs)."""
    if Path(path).exists():
        return path
    import matplotlib

    bold = path.endswith(("bd.ttf", "b.ttf"))
    return str(Path(matplotlib.get_data_path()) / "fonts/ttf" / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"))


def font(path, size):
    key = (path, size)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(_fallback(path), size)
    return _fonts[key]


def fa(s: str) -> str:
    import arabic_reshaper
    from bidi.algorithm import get_display

    return get_display(arabic_reshaper.reshape(s))


def text_w(d: ImageDraw.ImageDraw, s: str, f) -> int:
    l, _, r, _ = d.textbbox((0, 0), s, font=f)
    return r - l


def draw_rtl(d, x_right, y, s, f, fill):
    """Persian text whose right edge is at x_right."""
    t = fa(s)
    d.text((x_right - text_w(d, t, f), y), t, font=f, fill=fill)


def _overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def draw_detections(img_rgb: Image.Image, dets: list[Detection], scale: float, label_mode="auto",
                    gt_missing: list[list[float]] | None = None, offset=(0, 0), tag_px=None,
                    ids: list | None = None) -> Image.Image:
    """Boxes are in original-image pixels; drawn at (box - offset) * scale."""
    W, H = img_rgb.size
    ox, oy = offset
    base = img_rgb.convert("RGBA")
    fill_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    fd = ImageDraw.Draw(fill_layer)
    lw = max(2, round(max(W, H) / 700))
    tag_f = font(FONT_EN_B, tag_px or max(15, round(max(W, H) / 80)))
    boxes = [[(d.box[0] - ox) * scale, (d.box[1] - oy) * scale, (d.box[2] - ox) * scale, (d.box[3] - oy) * scale]
             for d in dets]
    gt_missing = [[(b[0] - ox) * scale, (b[1] - oy) * scale, (b[2] - ox) * scale, (b[3] - oy) * scale]
                  for b in gt_missing or []]
    for d, b in zip(dets, boxes):
        fd.rectangle(b, fill=CLASSES[d.cls][2] + (34,))
    out = Image.alpha_composite(base, fill_layer)
    dr = ImageDraw.Draw(out)
    for b in gt_missing:   # tomatoes in the label file the model did not find: dashed white
        _dashed_rect(dr, b, (255, 255, 255, 255), lw)
    for d, b in zip(dets, boxes):
        col = CLASSES[d.cls][2]
        x1, y1, x2, y2 = b
        dr.rectangle([x1 - 1, y1 - 1, x2 + 1, y2 + 1], outline=(0, 0, 0, 150), width=lw + 2)   # dark halo
        dr.rectangle(b, outline=col + (255,), width=lw)
        k = max(6, min(x2 - x1, y2 - y1) * 0.22)                                          # corner accents
        for (cx, cy, sx, sy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            dr.line([(cx, cy), (cx + sx * k, cy)], fill=col + (255,), width=lw + 2)
            dr.line([(cx, cy), (cx, cy + sy * k)], fill=col + (255,), width=lw + 2)
    placed: list[tuple] = []
    for i, (d, b) in enumerate(zip(dets, boxes), 1):
        x1, y1, x2, y2 = b
        unsure = "?" if d.cls_prob < UNSURE_PROB else ""
        full = label_mode == "full" or (label_mode == "auto" and len(dets) <= 8)
        i = ids[i - 1] if ids else i
        s = f"{i} {CLASSES[d.cls][1]} {d.cls_prob * 100:.0f}%{unsure}" if full else f"{i}{unsure}"
        if label_mode == "none" or not s:
            continue
        tw = text_w(dr, s, tag_f)
        th = tag_f.size + 4
        pw, ph = tw + 10, th
        cands = [(x1, y1 - ph - 1), (x1, y1 + 1), (x1, y2 - ph - 1), (x2 - pw, y1 - ph - 1), (x1, y2 + 1)]
        best, best_cost = None, None
        for (px, py) in cands:
            px = min(max(0, px), W - pw)
            py = min(max(0, py), H - ph)
            r = (px, py, px + pw, py + ph)
            cost = sum(_overlap(r, q) for q in placed)
            if best_cost is None or cost < best_cost:
                best, best_cost = r, cost
            if cost == 0:
                break
        placed.append(best)
        col = CLASSES[d.cls][2]
        dr.rounded_rectangle(best, radius=ph // 2 - 1, fill=col + (235,), outline=(0, 0, 0, 160), width=1)
        dr.text((best[0] + 5, best[1] + 1), s, font=tag_f, fill=(255, 255, 255, 255),
                stroke_width=1, stroke_fill=(0, 0, 0, 120))
    return out.convert("RGB")


def _dashed_rect(dr, b, col, lw, dash=8):
    x1, y1, x2, y2 = b
    for (a, c) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        L = max(abs(c[0] - a[0]), abs(c[1] - a[1]))
        n = max(1, int(L // dash))
        for k in range(0, n, 2):
            t0, t1 = k / n, min(1, (k + 1) / n)
            dr.line([(a[0] + (c[0] - a[0]) * t0, a[1] + (c[1] - a[1]) * t0),
                     (a[0] + (c[0] - a[0]) * t1, a[1] + (c[1] - a[1]) * t1)], fill=col, width=lw)


def draw_panel(height: int, s: dict, info: dict, gt: dict | None = None, pw=470) -> Image.Image:
    """info: ms, name; optional title, total_label, rows (footer [(label, value)] replacing the default)."""
    P = Image.new("RGB", (pw, height), BG)
    d = ImageDraw.Draw(P)
    R, L, y = pw - 24, 24, 22
    draw_rtl(d, R, y, info.get("title", "گزارش بازرسی گوجه"), font(FONT_FA_B, 24), TEXT)
    y += 38
    d.text((L, y), "YOLO26s detector  +  YOLO26s-cls ripeness", font=font(FONT_EN, 14), fill=MUTED)
    y += 30

    # total
    d.rounded_rectangle([L, y, R, y + 112], radius=14, fill=CARD)
    draw_rtl(d, R - 18, y + 14, info.get("total_label", "تعداد کل گوجه‌های تشخیص داده‌شده"), font(FONT_FA, 16), MUTED)
    d.text((L + 18, y + 26), str(s["total"]), font=font(FONT_EN_B, 64), fill=TEXT)
    if gt is not None:
        diff = s["total"] - gt["total"]
        acc = max(0.0, 100 * (1 - abs(diff) / gt["total"])) if gt["total"] else (100.0 if not diff else 0.0)
        c = (74, 222, 128) if abs(diff) <= max(1, 0.05 * gt["total"]) else (251, 191, 36)
        x_val = R - 150
        for k, (lab, val, col) in enumerate((("تعداد واقعی (برچسب)", str(gt["total"]), TEXT),
                                             ("اختلاف", f"{diff:+d}", c),
                                             ("دقت شمارش", f"{acc:.0f}%", c))):
            draw_rtl(d, R - 18, y + 38 + k * 22, lab, font(FONT_FA, 14), MUTED)
            vf = font(FONT_EN_B, 16)
            d.text((x_val - text_w(d, val, vf), y + 36 + k * 22), val, font=vf, fill=col)
    y += 128

    # KPI grid
    kpis = [("سالم", s["sound"], (45, 212, 191)), ("خراب", s["rotten"], CLASSES["rotten"][2]),
            ("رسیده", s["ripe"], CLASSES["ripe"][2]), ("نرسیده (نارس + نیمه‌رسیده)", s["unripe"], CLASSES["half_ripe"][2])]
    cw, ch, gap = (R - L - 12) // 2, 92, 12
    for k, (name, val, col) in enumerate(kpis):
        cx = R - (k % 2) * (cw + gap) - cw
        cy = y + (k // 2) * (ch + gap)
        d.rounded_rectangle([cx, cy, cx + cw, cy + ch], radius=12, fill=CARD)
        d.rounded_rectangle([cx + cw - 6, cy + 14, cx + cw, cy + ch - 14], radius=3, fill=col)
        f_name = font(FONT_FA, 14 if len(name) > 12 else 16)
        draw_rtl(d, cx + cw - 16, cy + 12, name, f_name, MUTED)
        d.text((cx + 14, cy + 34), str(val), font=font(FONT_EN_B, 38), fill=col)
        pct = f"{100 * val / s['total']:.0f}%" if s["total"] else "-"
        d.text((cx + cw - 16 - text_w(d, pct, font(FONT_EN, 15)), cy + 56), pct, font=font(FONT_EN, 15), fill=MUTED)
    y += 2 * ch + gap + 22

    # per-class bars
    draw_rtl(d, R, y, "تفکیک کلاس‌ها", font(FONT_FA_B, 17), TEXT)
    y += 34
    for key, (name, en, col) in CLASSES.items():
        n = s["per_class"][key]
        frac = n / s["total"] if s["total"] else 0
        d.rounded_rectangle([R - 16, y + 3, R, y + 19], radius=4, fill=col)
        draw_rtl(d, R - 26, y, name, font(FONT_FA, 16), TEXT)
        d.text((L, y), en, font=font(FONT_EN, 14), fill=MUTED)
        d.text((L + 100, y - 2), str(n), font=font(FONT_EN_B, 17), fill=TEXT)
        if gt and gt.get("per_class"):
            g = str(gt["per_class"][key])
            d.text((L + 150, y), "label " + g, font=font(FONT_EN, 14), fill=MUTED)
        y += 28
        d.rounded_rectangle([L, y, R, y + 8], radius=4, fill=CARD2)
        if frac:
            d.rounded_rectangle([R - max(8, (R - L) * frac), y, R, y + 8], radius=4, fill=col)
        y += 22

    # footer
    rows = info.get("rows") or [("میانگین اطمینان تشخیص", f"{s['mean_det_conf']:.2f}" if s["mean_det_conf"] is not None else "-"),
            ("رسیدگی نامطمئن (?)", str(s["unsure"])),
            ("زمان پردازش", f"{info['ms']:.0f} ms (CPU)"),
            ("فایل", info["name"][:34])]
    y = max(y + 6, height - 32 - 25 * len(rows))
    d.line([L, y, R, y], fill=CARD2, width=1)
    y += 12
    for k, v in rows:
        draw_rtl(d, R, y, k, font(FONT_FA, 14), MUTED)
        d.text((L, y), v, font=font(FONT_EN, 14), fill=TEXT)
        y += 25
    return P


def _resize(img_bgr, scale):
    h, w = img_bgr.shape[:2]
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    big = cv2.resize(img_bgr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=interp)
    return Image.fromarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))


def _roi(dets, gt_missing, w, h, margin=0.12):
    """Region holding every tomato, padded and kept between 3:5 and 5:3, or None if it is most of the image."""
    bs = [d.box for d in dets] + list(gt_missing or [])
    if not bs:
        return None
    x1, y1 = min(b[0] for b in bs), min(b[1] for b in bs)
    x2, y2 = max(b[2] for b in bs), max(b[3] for b in bs)
    mx, my = (x2 - x1) * margin + 12, (y2 - y1) * margin + 12
    x1, y1, x2, y2 = x1 - mx, y1 - my, x2 + mx, y2 + my
    bw, bh = x2 - x1, y2 - y1
    if bw / bh < 0.6:
        cx, bw = (x1 + x2) / 2, 0.6 * bh
        x1, x2 = cx - bw / 2, cx + bw / 2
    elif bw / bh > 1.67:
        cy, bh = (y1 + y2) / 2, bw / 1.67
        y1, y2 = cy - bh / 2, cy + bh / 2
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if (x2 - x1) * (y2 - y1) > 0.30 * w * h:   # zoom only when it magnifies >= ~1.8x
        return None
    return [int(x1), int(y1), int(x2), int(y2)]


def render(img_bgr, dets, s, info, gt=None, gt_missing=None, label_mode="auto", H=1200, zoom=True) -> Image.Image:
    """[full frame] [zoom on the tomatoes, when they fill only part of the frame] [dashboard]."""
    h, w = img_bgr.shape[:2]
    roi = _roi(dets, gt_missing, w, h) if zoom else None
    views = []
    if roi is None:
        scale = min(H / h, 1700 / w)
        pic = draw_detections(_resize(img_bgr, scale), dets, scale, label_mode, gt_missing)
        views.append(pic)
    else:
        scale = H / h
        over = draw_detections(_resize(img_bgr, scale), dets, scale, "none", gt_missing)
        od = ImageDraw.Draw(over)
        rb = [v * scale for v in roi]
        od.rectangle([rb[0] - 2, rb[1] - 2, rb[2] + 2, rb[3] + 2], outline=(0, 0, 0), width=2)
        _dashed_rect(od, rb, (255, 255, 255), 3, dash=14)
        x1, y1, x2, y2 = roi
        zs = min(H / (y2 - y1), 1500 / (x2 - x1))
        zoomed = draw_detections(_resize(img_bgr[y1:y2, x1:x2], zs), dets, zs, label_mode, gt_missing,
                                 offset=(x1, y1), tag_px=19)
        zd = ImageDraw.Draw(zoomed)
        tag = fa("نمای بزرگ‌شده")
        zd.rounded_rectangle([12, 12, 24 + text_w(zd, tag, font(FONT_FA_B, 18)), 46], radius=8, fill=(0, 0, 0))
        zd.text((18, 14), tag, font=font(FONT_FA_B, 18), fill=TEXT)
        views += [over, zoomed]
    Hc = max(H, max(v.height for v in views), 860)
    panel = draw_panel(Hc, s, info, gt)
    gap = 8
    canvas = Image.new("RGB", (sum(v.width for v in views) + gap * len(views) + panel.width, Hc), BG)
    x = 0
    for v in views:
        canvas.paste(v, (x, (Hc - v.height) // 2))
        x += v.width + gap
    canvas.paste(panel, (x, 0))
    return canvas


# ----------------------------------------------------------------------------------------------- matching
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def match(dets: list[Detection], gts: list[dict], thr=0.5):
    """Greedy by confidence. Returns [(det_idx, gt_idx)], unmatched det idx, unmatched gt idx."""
    pairs, used = [], set()
    for i in sorted(range(len(dets)), key=lambda k: -dets[k].det_conf):
        best, bj = thr, None
        for j, g in enumerate(gts):
            if j not in used:
                v = iou(dets[i].box, g["box"])
                if v >= best:
                    best, bj = v, j
        if bj is not None:
            used.add(bj)
            pairs.append((i, bj))
    md = {i for i, _ in pairs}
    return pairs, [i for i in range(len(dets)) if i not in md], [j for j in range(len(gts)) if j not in used]


def gt_summary(gts: list[dict]) -> dict:
    out = {"total": len(gts)}
    if gts and all(g["cls"] in CLASSES for g in gts):
        out["per_class"] = {k: sum(g["cls"] == k for g in gts) for k in CLASSES}
    return out


# ----------------------------------------------------------------------------------------------- CLI
def collect(source: str) -> list[Path]:
    p = Path(source)
    if p.is_dir():
        return sorted(q for q in p.rglob("*") if q.suffix.lower() in IMG_EXT)
    return [p]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="image file or folder")
    ap.add_argument("--set", choices=list(SETS), help="labelled evaluation set to sample from")
    ap.add_argument("--random", type=int, default=0, help="pick N random images (from --source or --set)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "inspect"))
    ap.add_argument("--conf", type=float, default=DET_CONF)
    ap.add_argument("--labels", choices=["auto", "full", "id", "none"], default="auto")
    ap.add_argument("--tiles", type=int, default=TILES, help="2 = extra 2x2 tiled pass for small far tomatoes")
    ap.add_argument("--show-missed", action="store_true", help="draw labelled tomatoes the model missed (dashed)")
    a = ap.parse_args(argv)
    if not a.source and not a.set:
        ap.error("give --source or --set")
    imgs = collect(a.source) if a.source else SETS[a.set]["images"]()
    names = SETS[a.set]["classes"] if a.set else None
    if a.random:
        random.Random(a.seed).shuffle(imgs)
        imgs = imgs[: a.random]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    insp = TomatoInspector(conf=a.conf, tiles=a.tiles, tile_conf=0.35 if a.tiles > 1 else None, tile_max_px=32)
    rows = []
    for p in imgs:
        img = imread(p)
        if img is None:
            print("unreadable:", p)
            continue
        dets, ms = insp(img)
        s = summarize(dets)
        gt = gts = missing = None
        if names and label_path(p).exists():
            gts = read_gt(p, img.shape[1], img.shape[0], names)
            gt = gt_summary(gts)
            if a.show_missed:
                _, _, fn = match(dets, gts)
                missing = [gts[j]["box"] for j in fn]
        canvas = render(img, dets, s, {"ms": ms, "name": p.name}, gt, missing, a.labels)
        dst = out / f"{p.stem[:60]}_result.jpg"
        canvas.save(dst, quality=93, subsampling=0)
        rows.append({"image": str(p), "result": str(dst), **{k: v for k, v in s.items() if k != "per_class"},
                     **s["per_class"], "gt_total": gt["total"] if gt else None, "ms": round(ms, 1),
                     "detections": [d.__dict__ for d in dets]})
        print(f"{p.name[:50]:50s} total {s['total']:3d} | sound {s['sound']:3d} | ripe {s['ripe']:3d} | "
              f"unripe {s['unripe']:3d} | rotten {s['rotten']:3d}" + (f" | label {gt['total']}" if gt else ""))
    (out / "results.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    with open(out / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        cols = ["image", "total", "sound", "ripe", "unripe", "rotten", "ripe", "half_ripe", "raw", "gt_total", "ms"]
        wr = csv.DictWriter(f, fieldnames=list(dict.fromkeys(cols)), extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    print("->", out)
    return rows


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
