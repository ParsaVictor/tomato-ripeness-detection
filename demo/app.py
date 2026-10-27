"""Local web demo: tomato detection, ripeness grading and counting (photo + short video).

    pip install -r requirements.txt gradio
    python demo/app.py          # then open http://127.0.0.1:7860
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import gradio as gr
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tomato_app"))
import tomato_inspect as ti  # noqa: E402
import tomato_video as tv  # noqa: E402

REPO = "ParsaVictor/tomato-ripeness-yolo26"
FILES = (("det", "detector_yolo26s_best.pt"), ("cls", "ripeness_yolo26s-cls_best.pt"), ("rot", "rotten_check.npz"))
# local weights/ first, else download once from Hugging Face
W = {k: str(ti.WEIGHTS / f) if (ti.WEIGHTS / f).exists() else hf_hub_download(REPO, f"weights/{f}") for k, f in FILES}
INSP = ti.TomatoInspector(W["det"], W["cls"], rotten_check=W["rot"])
MAX_VIDEO_FRAMES = 150
EX = Path(__file__).parent / "examples"


def ffmpeg_exe() -> str:
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def summary_md(s: dict) -> str:
    return (f"### 🍅 {s['total']} tomatoes\n"
            f"| Sound | Ripe | Unripe (half + raw) | Rotten |\n|:-:|:-:|:-:|:-:|\n"
            f"| **{s['sound']}** | **{s['ripe']}** | **{s['unripe']}** | **{s['rotten']}** |\n\n"
            f"🔴 ripe {s['per_class']['ripe']} · 🟠 half-ripe {s['per_class']['half_ripe']} · "
            f"🟢 unripe {s['per_class']['raw']} · 🟣 rotten {s['per_class']['rotten']}")


def photo(img_rgb, conf):
    if img_rgb is None:
        return None, "Upload a photo first."
    INSP.conf = float(conf)
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    dets, ms = INSP(bgr)
    s = ti.summarize(dets)
    out = ti.render(bgr, dets, s, {"ms": ms, "name": "upload"}, H=1000)
    return out, summary_md(s)


def video(path, line_y):
    if not path:
        return None, "Upload a video first."
    tmp = Path(tempfile.mkdtemp())
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = tmp / "frames"
    frames.mkdir()
    n = 0
    while n < MAX_VIDEO_FRAMES:
        ok, f = cap.read()
        if not ok:
            break
        cv2.imwrite(str(frames / f"{n:05d}.jpg"), f)
        n += 1
    cap.release()
    line = [0, line_y, 1, line_y] if line_y > 0 else None
    INSP.conf = ti.DET_CONF
    res = tv.run(str(frames), tmp / "out", insp=INSP, fps_out=fps, line=line)
    mp4 = tmp / "result.mp4"
    subprocess.run([ffmpeg_exe(), "-v", "error", "-y", "-i", str(tmp / "out" / "tracked.mp4"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(mp4)], check=True)
    s = {k: res[k] for k in ("sound", "ripe", "unripe", "rotten", "per_class")}
    s["total"] = res["counted"]
    note = f"\n\n_{res['frames']} frames processed (max {MAX_VIDEO_FRAMES}); each tomato counted once._"
    return str(mp4), summary_md(s) + note


with gr.Blocks(title="Tomato Ripeness Detection & Counting", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🍅 Tomato Ripeness Detection & Counting\n"
        "Find every tomato, grade it **unripe / half-ripe / ripe / rotten**, and count each one once on a conveyor. "
        "YOLO26s detector + YOLO26s-cls classifier + ByteTrack · "
        "[Model](https://huggingface.co/ParsaVictor/tomato-ripeness-yolo26) · "
        "[Dataset](https://huggingface.co/datasets/ParsaVictor/tomato-ripeness-detection) · "
        "[Code](https://github.com/ParsaVictor/tomato-ripeness-detection)")
    with gr.Tab("📷 Photo"):
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="numpy", label="Photo")
                conf = gr.Slider(0.1, 0.6, value=ti.DET_CONF, step=0.05, label="Detection confidence")
                btn = gr.Button("Analyse", variant="primary")
                gr.Examples([[str(EX / f)] for f in ("cherry_tomatoes.jpg", "mixed_ripeness.jpg", "conveyor.jpg",
                                                     "ripe_and_unripe.jpg")], inputs=[inp])
            with gr.Column(scale=2):
                out_img = gr.Image(type="pil", label="Result")
                out_md = gr.Markdown()
        btn.click(photo, [inp, conf], [out_img, out_md])
    with gr.Tab("🎞️ Video / conveyor"):
        with gr.Row():
            with gr.Column(scale=1):
                vin = gr.Video(label=f"Short video (first {MAX_VIDEO_FRAMES} frames are used)")
                line_y = gr.Slider(0, 0.95, value=0.6, step=0.05,
                                   label="Counting line height (0 = count every tracked tomato once)")
                vbtn = gr.Button("Track & count", variant="primary")
                gr.Examples([[str(EX / "conveyor_clip.mp4")]], inputs=[vin])
            with gr.Column(scale=2):
                vout = gr.Video(label="Result")
                vmd = gr.Markdown()
        vbtn.click(video, [vin, line_y], [vout, vmd])
    gr.Markdown("Runs on a free CPU: a photo takes ~1 s, a 150-frame video about 1–2 minutes. "
                "Colours: 🔴 ripe · 🟠 half-ripe · 🟢 unripe · 🟣 rotten.")

if __name__ == "__main__":
    demo.launch()
