<div dir="rtl">

# 🍅 تشخیص، شمارش و تعیین رسیدگی گوجه‌فرنگی با بینایی ماشین

**تشخیص لحظه‌ای گوجه، تعیین رسیدگی (نارس / نیمه‌رسیده / رسیده / خراب) و شمارش روی نوار نقاله با YOLO26 و ByteTrack**

[English](README.md) · [فارسی](README.fa.md)

<p align="center"><img src="assets/demo_tracking_counting.gif" width="820" alt="دموی tracking و شمارش گوجه"/></p>

هر گوجه یک شماره‌ی ثابت می‌گیرد، وقتی از خط رد شود **فقط یک بار** شمرده می‌شود و رسیدگی‌اش تعیین می‌شود؛ داشبورد زنده سمت راست است.

## چه کاری انجام می‌دهد؟

- 🔍 **پیدا کردن همه‌ی گوجه‌ها** — ۹۹٪ روی تست Tomato Defects و ۹۳٪ روی عکس‌هایی که مدل ندیده
- 🎨 **تعیین رسیدگی** — ۴ کلاس با رنگ جدا؛ تشخیص سالم از خراب با دقت **۹۸٫۶٪**
- 🔢 **شمارش روی نوار نقاله** — با tracking هر گوجه یک بار شمرده می‌شود؛ دقت شمارش **۹۵ تا ۹۹٪** در تست با جواب معلوم
- ⚡ **کم‌هزینه** — مدل رسیدگی فقط چند بار برای هر گوجه اجرا می‌شود؛ **۷۰ تا ۸۸٪ پردازش کمتر**
- 📊 **خروجی خوانا** — کادرهای نازک، شماره‌ی هر گوجه، رنگ هر کلاس، زوم خودکار و داشبورد: کل / سالم / رسیده / نرسیده / خراب
- 📷 **عکس، ویدیو، دوربین** — با یک دستور

<p align="center"><img src="assets/demo_conveyor.gif" width="820" alt="شمارش گوجه روی نوار نقاله"/></p>

## نمونه‌ی خروجی‌ها

<p align="center">
<img src="assets/samples/cherry_tomatoes_29.jpg" width="49%"/>
<img src="assets/samples/mixed_ripeness_38.jpg" width="49%"/>
<img src="assets/samples/conveyor_zoom_view.jpg" width="49%"/>
<img src="assets/samples/ripe_vs_unripe_10.jpg" width="49%"/>
</p>

## نتایج (روی عکس‌هایی که مدل هرگز ندیده)

| مجموعه | گوجه | پیدا شده (Recall) | درستی تشخیص (Precision) | دقت شمارش هر عکس |
|---|---:|---:|---:|---:|
| تست Tomato Defects | 768 | 99.0% | 93.3% | 93.8% |
| نوار نقاله | 650 | 84.9% | 90.0% | 89.7% |
| tomato-sxsc7 | 736 | 93.3% | 94.8% | 96.3% |

رسیدگی: ۴ کلاسه ۸۰٫۸٪ · رسیده/نرسیده/خراب ۸۸٫۷٪ · سالم در برابر خراب ۹۸٫۶٪
ویدیو: دقت شمارش با tracking ۹۸٫۴٪ و ۹۹٫۳٪ (سرعت آهسته)، ۹۵٫۲٪ و ۸۵٫۲٪ (سرعت سریع). بدون tracking شمارش ۱۶ تا ۵۳ برابر بیشتر از واقع بود.

جزئیات کامل: [docs/RESULTS.md](docs/RESULTS.md)

## اجرا

</div>

```bash
pip install -r requirements.txt
python tomato_app/run.py photo.jpg
python tomato_app/run.py belt.mp4 --line 0,0.6,1,0.6
python tomato_app/run.py 0 --show
```

<div dir="rtl">

## وزن‌های مدل

وزن‌های آموزش‌دیده روی 🤗 Hugging Face هستند: [ParsaVictor/tomato-ripeness-yolo26](https://huggingface.co/ParsaVictor/tomato-ripeness-yolo26)
برای سفارشی‌سازی روی خط تولید، محصول یا دوربین خودتان [تماس بگیرید](https://github.com/ParsaVictor/tomato-ripeness-detection/issues/new?title=Collaboration).

## دیتاست

**۸٬۴۹۷ عکس و ۷۴٬۴۸۶ کادر گوجه** از ۶ دیتاست عمومی، یکپارچه با یک ساختار برچسب، بدون نشتی بین آموزش و تست، و با حذف عکس‌های تقریباً تکراری. جزئیات: [docs/DATASET.md](docs/DATASET.md) · دانلود از 🤗 [Hugging Face](https://huggingface.co/datasets/ParsaVictor/tomato-ripeness-detection)

## حمایت

اگر این پروژه به کارتان آمد، لطفاً ⭐ بدهید. برای سفارشی‌سازی روی خط تولید، محصول یا دوربین خودتان [تماس بگیرید](https://github.com/ParsaVictor/tomato-ripeness-detection/issues/new?title=Collaboration).

</div>
