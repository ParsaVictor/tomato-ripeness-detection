"""Build results/report.html from results/evaluation/metrics.json and the rendered results/samples_* images."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FA = {"ripe": "رسیده", "half_ripe": "نیمه‌رسیده", "raw": "نارس", "rotten": "خراب"}
COL = {"ripe": "#ef4444", "half_ripe": "#f59e0b", "raw": "#22c55e", "rotten": "#a855f7"}


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    m = json.loads((RES / "evaluation/metrics.json").read_text(encoding="utf-8"))
    rows = ""
    for k, r in m["sets"].items():
        rows += (f"<tr><td>{r['name_fa']}</td><td>{r['images']}</td><td>{r['gt_tomatoes']}</td>"
                 f"<td>{r['pred_tomatoes']}</td><td><b>{pct(r['recall'])}</b></td><td><b>{pct(r['precision'])}</b></td>"
                 f"<td>{pct(r['f1'])}</td><td><b>{pct(r['mean_count_accuracy_per_image'])}</b></td>"
                 f"<td>{r['mean_abs_count_error']:.2f}</td><td>{pct(r['images_within_1'])}</td></tr>")
    rp = m["sets"]["defects"]["ripeness"]
    cls_rows = "".join(
        f"<tr><td><span class=sw style='background:{COL[c]}'></span>{FA[c]}</td><td>{rp['count_gt'][c]}</td>"
        f"<td>{rp['count_pred'][c]}</td><td>{pct(rp['per_class_recall'][c])}</td></tr>" for c in rp["classes"])
    vid = ""
    vt_p, cv_p = RES / "video_counting_test/results.json", RES / "video_conveyor/summary.json"
    if vt_p.exists():
        vs = json.loads(vt_p.read_text(encoding="utf-8"))["summary"]
        vr = "".join(
            f"<tr><td>{'tomato-sxsc7' if k.startswith('sxsc7') else 'نوار نقاله'}</td>"
            f"<td>{'آهسته (۲٪ کادر در هر فریم)' if k.endswith('0.02') else 'سریع (۵٪ کادر در هر فریم)'}</td>"
            f"<td>{v['truth']}</td><td>{v['unique']}</td><td><b>{pct(v['unique_accuracy'])}</b></td>"
            f"<td>{pct(v['line_accuracy'])}</td><td>{v['per_frame_sum_vs_truth']:.0f}×</td>"
            f"<td>{pct(v['classifier_saving'])}</td></tr>" for k, v in vs.items())
        vid += ("<h2>ویدیو و خط تولید — tracking و شمارش هر گوجه فقط یک بار</h2>"
                "<div class='card wrap'><p class=muted>کلیپ‌های متحرک از عکس‌های holdout برچسب‌دار ساخته شده‌اند (پنجره‌ای که "
                "روی عکس حرکت می‌کند، مثل نوار زیر دوربین)؛ پس تعداد واقعی گوجه‌ها معلوم است.</p><table><tr><th>مجموعه</th>"
                "<th>سرعت</th><th>گوجه‌ی واقعی</th><th>شمرده‌شده با tracking</th><th>دقت شمارش</th>"
                "<th>دقت شمارش با خط</th><th>شمارش بدون tracking (جمع فریم‌ها)</th><th>صرفه‌جویی در مدل رسیدگی</th></tr>"
                f"{vr}</table></div>")
        demo = sorted((RES / "video_counting_test/demo").glob("pan_*.jpg"))[1:3]
        vid += "<div class=gal>" + "".join(f"<img src='{p.relative_to(RES).as_posix()}'>" for p in demo) + "</div>"
    if cv_p.exists():
        c = json.loads(cv_p.read_text(encoding="utf-8"))
        shots = sorted((RES / "video_conveyor").glob("frame_*.jpg"))[1:3]
        vid += (f"<h2>ویدیوی واقعی نوار نقاله ({c['frames']} فریم)</h2><div class=card><p>شمرده‌شده: <b>{c['counted']}</b> · "
                f"سالم {c['sound']} · رسیده {c['ripe']} · نرسیده {c['unripe']} · خراب {c['rotten']} · "
                f"<a style='color:#38bdf8' href='video_conveyor/tracked.mp4'>دیدن ویدیو</a></p><p class=muted>فریم‌های این دیتاست "
                "با فاصله‌ی زمانی زیاد گرفته شده‌اند (گوجه‌ها بین دو فریم چند برابر اندازه‌ی خودشان جابه‌جا می‌شوند) و چند "
                "صحنه‌ی مختلف دارند؛ پس این عدد دقیق نیست و فقط کارکرد سیستم را نشان می‌دهد. دوربین خط تولید با ۲۵ تا ۳۰ "
                "فریم در ثانیه این مشکل را ندارد.</p></div><div class=gal>"
                + "".join(f"<img src='{p.relative_to(RES).as_posix()}'>" for p in shots) + "</div>")
    gal = ""
    for k, title in (("conveyor", "نوار نقاله"), ("sxsc7", "tomato-sxsc7"), ("defects", "Tomato Defects")):
        imgs = sorted((RES / f"samples_{k}").glob("*_result.jpg"))
        gal += f"<h2>نمونه‌های تصادفی — {title}</h2><div class=gal>"
        gal += "".join(f"<a href='{p.relative_to(RES).as_posix()}' target=_blank>"
                       f"<img loading=lazy src='{p.relative_to(RES).as_posix()}'></a>" for p in imgs)
        gal += "</div>"
    html = f"""<!doctype html><html lang=fa dir=rtl><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Tomato Inspection Report</title>
<style>
:root{{--bg:#0b0f19;--card:#181e2c;--text:#f1f5f9;--muted:#94a3b8;--line:#263045}}
body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.7 Tahoma,'Segoe UI',sans-serif}}
main{{max-width:1300px;margin:auto;padding:24px 16px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:19px;margin:34px 0 12px}}
.muted{{color:var(--muted)}} .card{{background:var(--card);border-radius:14px;padding:16px 18px;margin:12px 0}}
table{{border-collapse:collapse;width:100%}} td,th{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:center}}
th{{color:var(--muted);font-weight:normal}} td:first-child,th:first-child{{text-align:right}}
.wrap{{overflow-x:auto}} .sw{{display:inline-block;width:12px;height:12px;border-radius:3px;margin-left:8px}}
.gal{{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:10px}}
.gal img{{width:100%;border-radius:10px;display:block}} .two{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
@media(max-width:800px){{.two{{grid-template-columns:1fr}}.gal{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>گزارش مدل دو مرحله‌ای گوجه</h1>
<div class=muted>مرحله‌ی ۱: YOLO26s تشخیص گوجه · مرحله‌ی ۲: YOLO26s-cls رسیدگی ۴ کلاسه · آستانه‌ی تشخیص {m['conf']} · حذف کادر تکراری IoU {m['dedup_iou']} · بررسی دوم خراب: آستانه {m.get('rotten_check')}</div>
<h2>شمارش و تشخیص — روی عکس‌هایی که مدل هرگز ندیده</h2>
<div class="card wrap"><table><tr><th>مجموعه</th><th>عکس</th><th>گوجه‌ی واقعی</th><th>گوجه‌ی شمرده‌شده</th>
<th>Recall (چند درصد گوجه‌ها پیدا شد)</th><th>Precision (چند درصد تشخیص‌ها درست بود)</th><th>F1</th>
<th>دقت شمارش هر عکس</th><th>میانگین خطای شمارش</th><th>عکس‌ها با خطای حداکثر ±۱</th></tr>{rows}</table></div>
<h2>رسیدگی — تست Tomato Defects (تنها مجموعه با برچسب رسیدگی)</h2>
<div class=two><div class=card>
<p>دقت ۴ کلاسه: <b>{pct(rp['accuracy_4class'])}</b><br>دقت ۳ گروهی (رسیده / نرسیده / خراب): <b>{pct(rp['accuracy_3group'])}</b><br>
دقت سالم در برابر خراب: <b>{pct(rp['accuracy_sound_vs_rotten'])}</b><br>
گوجه‌ی سالمی که «خراب» نشان داده شد — نوار نقاله: <b>{m['sets']['conveyor']['pred_rotten']}</b> از {m['sets']['conveyor']['pred_tomatoes']} ·
sxsc7: <b>{m['sets']['sxsc7']['pred_rotten']}</b> از {m['sets']['sxsc7']['pred_tomatoes']} (قبل از «بررسی دوم خراب»: ۳۲ و ۲۰)</p>
<table><tr><th>کلاس</th><th>تعداد واقعی</th><th>تعداد مدل</th><th>درصد درست</th></tr>{cls_rows}</table>
<p class=muted>تقریباً همه‌ی خطاها بین دو مرحله‌ی کنار هم است (رسیده↔نیمه‌رسیده، نیمه‌رسیده↔نارس)؛ رسیده هیچ‌وقت با نارس اشتباه نمی‌شود.</p>
</div><div class=card><img src='evaluation/confusion_defects.png' style='width:100%;border-radius:8px;background:#fff'></div></div>
<div class=card class=muted>راهنمای رنگ‌ها: <span class=sw style='background:#ef4444'></span>رسیده
<span class=sw style='background:#f59e0b'></span>نیمه‌رسیده <span class=sw style='background:#22c55e'></span>نارس
<span class=sw style='background:#a855f7'></span>خراب · کادر سفید خط‌چین = گوجه‌ای که برچسب دارد ولی مدل پیدا نکرد · «?» = رسیدگی نامطمئن</div>
{vid}
{gal}
</main></body></html>"""
    (RES / "report.html").write_text(html, encoding="utf-8")
    print("->", RES / "report.html")
    # one-file copy to send: every image embedded (downscaled JPEG), local video links dropped
    import base64
    import re

    import cv2
    import numpy as np

    def embed(m):
        img = cv2.imdecode(np.fromfile(str(RES / m.group(1)), np.uint8), cv2.IMREAD_COLOR)
        f = min(1.0, 1400 / img.shape[1])
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else img
        b64 = base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()).decode()
        return f"src='data:image/jpeg;base64,{b64}'"

    one = re.sub(r"src='([^']+\.(?:jpg|png))'", embed, html)
    one = re.sub(r"<a href='[^']+\.jpg' target=_blank>(<img[^>]*>)</a>", lambda m: m.group(1), one)
    one = re.sub(r" · <a style='[^']*' href='[^']+\.mp4'>[^<]*</a>", "", one)
    dst = RES / "Tomato_Report_standalone.html"
    dst.write_text(one, encoding="utf-8")
    print("->", dst, f"{dst.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
