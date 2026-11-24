"""Create a printable Persian report from a completed experiment batch."""

import argparse
import base64
import hashlib
import json
from html import escape
from pathlib import Path

import pandas as pd

from src.data import DATA_PATH, ROOT, load_sales
from src.eda import analyze
from src.tracking import configure_tracking


def number(value, decimals=2):
    return f"{float(value):,.{decimals}f}"


def table(headers, rows):
    def cells(values, tag):
        return "".join(f"<{tag}>{escape(str(value))}</{tag}>" for value in values)
    body = "".join(f"<tr>{cells(row, 'td')}</tr>" for row in rows)
    return f"<table><thead><tr>{cells(headers, 'th')}</tr></thead><tbody>{body}</tbody></table>"


def figure(path, caption):
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return (
        f'<figure><img src="data:image/png;base64,{data}" alt="{escape(caption)}">'
        f"<figcaption>{escape(caption)}</figcaption></figure>"
    )


def build_report(summary, comparison, eda, weekly, batch, stage):
    best = summary["best_run"]
    test = summary["test_metrics"]
    split_names = {"train": "آموزش", "validation": "اعتبارسنجی", "test": "آزمون"}
    splits = table(
        ["بخش", "شروع", "پایان", "هفته", "مشاهده"],
        [[split_names[name], s["start"], s["end"], s["weeks"], s["rows"]]
         for name, s in summary["splits"].items()],
    )
    rows = table(
        ["Run", "RMSE", "MAE", "WAPE (%)", "R²", "عدم همگرایی"],
        [[r.run_name, number(r.rmse, 0), number(r.mae, 0), number(r.wape),
          number(r.r2, 3), r.nonconverged_stores]
         for r in comparison.itertuples(index=False)],
    )
    baseline = comparison.loc[comparison["run_name"] == "naive_last_week", "rmse"]
    baseline_note = "خطای مدل منتخب با مدل پایه در جدول قابل مقایسه است."
    if not baseline.empty and float(baseline.iloc[0]) > 0:
        difference = 100 * (float(baseline.iloc[0]) - best["rmse"]) / float(baseline.iloc[0])
        if difference >= 0:
            baseline_note = f"RMSE مدل منتخب نسبت به مبنای هفته قبل {number(difference)} درصد کمتر یا برابر است."
        else:
            baseline_note = f"RMSE مدل منتخب نسبت به مبنای هفته قبل {number(abs(difference))} درصد بیشتر است."

    train = weekly[weekly["Date"] <= pd.Timestamp(eda["eda_training_end"])]
    totals = train.groupby("Date")["Weekly_Sales"].sum()
    monthly = totals.groupby(totals.index.month).mean()
    monthly_note = (
        f"بیشترین میانگین فروش هفتگی کل فروشگاه‌ها در ماه شماره "
        f"{monthly.idxmax()} و کمترین در ماه شماره {monthly.idxmin()} مشاهده شد."
    )
    holiday = train.groupby("IsHoliday")["Weekly_Sales"].mean()
    holiday_note = "مقایسه تعطیلات و هفته‌های عادی به وجود هر دو گروه در داده وابسته است."
    if True in holiday.index and False in holiday.index and holiday.loc[False] != 0:
        holiday_diff = 100 * (holiday.loc[True] / holiday.loc[False] - 1)
        holiday_note = (
            f"میانگین فروش هر فروشگاه در هفته‌های تعطیل نسبت به عادی "
            f"{number(abs(holiday_diff))} درصد "
            f"{'بیشتر' if holiday_diff >= 0 else 'کمتر'} است. این مقایسه توصیفی است و رابطه علّی را اثبات نمی‌کند."
        )
    metrics = table(
        ["معیار", "Validation", "Test"],
        [[key.upper(), number(best[key], 3 if key == "r2" else 2),
          number(test[key], 3 if key == "r2" else 2)]
         for key in ("rmse", "mae", "wape", "r2")],
    )
    gap_note = "خطای آزمون از اعتبارسنجی بیشتر است." if test["rmse"] > best["rmse"] else "خطای آزمون از اعتبارسنجی بیشتر نیست."
    missing = sum(eda["missing_values"].values())
    batch = Path(batch)
    eda_dir = batch / "eda"

    return f"""<!doctype html>
<html lang="fa" dir="rtl">
<head><meta charset="utf-8"><title>گزارش پروژه MLflow</title>
<style>
@page {{ size: A4; margin: 17mm 16mm; }}
* {{ box-sizing: border-box; }}
body {{ font: 14px/1.8 Tahoma, Arial, sans-serif; color: #202020; margin: 0 auto; max-width: 178mm; }}
section {{ padding: 0 0 8mm; }}
h1 {{ font-size: 23px; margin: 0 0 8px; }}
h2 {{ font-size: 19px; border-bottom: 1px solid #777; padding-bottom: 5px; }}
h3 {{ font-size: 15px; margin-bottom: 4px; }}
p {{ margin: 6px 0 10px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 11px; text-align: center; }}
td, th {{ border: 1px solid #aaa; padding: 4px; }}
th {{ background: #eee; }}
td {{ direction: ltr; unicode-bidi: plaintext; }}
figure {{ margin: 9px 0; text-align: center; break-inside: avoid; }}
img {{ max-width: 100%; max-height: 69mm; object-fit: contain; }}
figcaption {{ font-size: 11px; color: #444; }}
code {{ direction: ltr; unicode-bidi: embed; overflow-wrap: anywhere; }}
small {{ color: #444; }}
@media print {{ section:not(:last-child) {{ break-after: page; }} }}
@media screen {{ section {{ margin: 16mm 0; border-bottom: 1px solid #ccc; }} }}
</style></head>
<body>
<section>
<h1>پیش‌بینی فروش فروشگاه‌های خرده‌فروشی با MLflow</h1>
<p>گزارش پروژه پایانی دوره MLflow — شناسه اجرای آزمایش: <code>{escape(summary["batch_id"])}</code></p>
<h2>۱. معرفی مسئله و تحلیل داده</h2>
<p>هدف، پیش‌بینی مجموع فروش هفته آینده هر فروشگاه با استفاده از اطلاعات تاریخی است.
داده آموزشی مسابقه Walmart Recruiting – Store Sales Forecasting در سطح دپارتمان ارائه شده است.
فروش دپارتمان‌ها در هر فروشگاه و هفته جمع شده تا متغیر هدف ساخته شود.
داده آزمون مسابقه به علت نداشتن برچسب فروش استفاده نشده است.</p>
{table(["شاخص", "مقدار"], [
    ["تعداد فروشگاه", eda["stores"]], ["مشاهدات فروشگاه-هفته", eda["store_weeks"]],
    ["شروع داده", eda["first_date"]], ["پایان داده", eda["last_date"]],
    ["مقادیر گمشده پس از اعتبارسنجی", missing],
    ["مشاهدات پرچم‌گذاری شده با IQR در آموزش", eda["training_iqr_outliers"]],
])}
<p>کلیدهای تکراری و هفته‌های گمشده بررسی شدند. فروش منفی در سطح دپارتمان می‌تواند حاصل مرجوعی باشد؛ بنابراین حذف خودکار انجام نشد.
نقاط پرت به کمک فاصله بین چارکی در هر فروشگاه شناسایی شدند، اما برای حفظ جهش‌های واقعی فروش، حذف نشدند.</p>
{figure(eda_dir / "sales_vs_time.png", "شکل ۱. روند فروش کل و میانگین متحرک چهار هفته‌ای")}
<p>{monthly_note} {holiday_note}</p>
</section>
<section>
<h2>۲. مهندسی ویژگی و طراحی ارزیابی</h2>
<p>ویژگی‌های زمانی شامل سال، ماه، شماره هفته، روز هفته، روند زمانی و تعطیلات هستند.
روز هفته در این دیتاست ثابت است و قدرت تفکیک ندارد. شناسه فروشگاه در مدل‌های رگرسیون و درختی به صورت One-hot کدگذاری شده است.</p>
<p>مقادیر فروش ۱، ۲، ۴، ۱۳ و ۵۲ هفته قبل به عنوان Lag، میانگین‌های متحرک ۴ و ۱۳ هفته و اختلاف فروش یک و چهار هفته قبل به عنوان شاخص روند استفاده شدند.
تمام ویژگی‌های وابسته به فروش با انتقال حداقل یک هفته ساخته شده‌اند تا مقدار واقعی فروش هفته هدف وارد ورودی مدل نشود.</p>
{figure(eda_dir / "monthly_seasonality.png", "شکل ۲. الگوی ماهانه در بخش آموزش")}
{splits}
<p>۵۲ هفته نخست برای تشکیل تاریخچه ویژگی‌ها کنار گذاشته شدند. داده‌ها بدون درهم‌ریزی به آموزش، اعتبارسنجی و آزمون تقسیم شدند.
در هر پیش‌بینی یک‌هفته‌ای، فروش واقعی هفته‌های گذشته در دسترس است؛ بنابراین این روش با پیش‌بینی هم‌زمان ۱۳ هفته متفاوت است.
مدل‌ها فقط روی آموزش برازش شدند و انتخاب مدل بر اساس RMSE اعتبارسنجی انجام شد.</p>
</section>
<section>
<h2>۳. آزمایش‌ها و مقایسه مدل‌ها</h2>
<p>آزمایش‌ها تحت نام <code>Retail_Sales_Forecasting</code> ثبت شده‌اند.
مدل‌های پایه، رگرسیون خطی، جنگل تصادفی، XGBoost و SARIMAX با تنظیمات متفاوت مقایسه شدند.
SARIMAX در این پروژه به صورت مدل‌های AR(1) و AR(2) مستقل برای هر فروشگاه پیاده‌سازی شده است.
مدل‌های بدون همگرایی کامل از انتخاب نهایی کنار گذاشته می‌شوند.</p>
{rows}
<p>MAE میانگین قدرمطلق خطا، RMSE ریشه میانگین مربعات خطا، WAPE نسبت مجموع خطای مطلق به مجموع قدرمطلق فروش واقعی (درصد) و R² ضریب تعیین است.
RMSE به دلیل حساسیت بیشتر به خطاهای بزرگ معیار انتخاب در نظر گرفته شد.</p>
{figure(batch / "comparison.png", "شکل ۳. مقایسه RMSE اعتبارسنجی")}
<p>مدل منتخب: <code>{escape(best["run_name"])}</code>. {baseline_note}</p>
</section>
<section>
<h2>۴. ارزیابی نهایی و مدیریت چرخه عمر</h2>
{metrics}
<p>{gap_note} تفاوت این خطاها می‌تواند از تغییر الگوهای فروش در دوره‌های زمانی ناشی شود و به تنهایی اثبات بیش‌برازش نیست.
R² بالا نیز ممکن است بخشی از تفاوت سطح فروش میان فروشگاه‌ها را منعکس کند.</p>
{figure(batch / "selected_test" / "test_forecast.png", "شکل ۴. پیش‌بینی در برابر فروش واقعی آزمون")}
<p>مدل با شناسه Run <code>{escape(best["run_id"])}</code> در Registry با نام
<code>{escape(summary["registered_model"])}</code> و نسخه <code>{escape(str(summary["version"]))}</code>
ثبت شده است. وضعیت فعلی نسخه هنگام ساخت گزارش: <code>{escape(stage)}</code>.
نسخه ابتدا وارد Staging می‌شود و پس از بررسی انسانی قابلیت انتقال به Production را دارد.
پارامترها، متریک‌ها، مدل، امضای ورودی و نمودارها در MLflow قابل مشاهده‌اند.</p>
<h3>جمع‌بندی و محدودیت‌ها</h3>
<p>این پروژه امکان تکرار آزمایش‌ها و ردیابی نسخه مدل را فراهم می‌کند.
محدودیت‌ها شامل بازه تاریخی کوتاه، نبود اطلاعات قیمت و ترویج فروش، ساده بودن مدل سری زمانی و ارزیابی تنها در یک بازه آزمون است.
پیشنهاد ادامه کار استفاده از اعتبارسنجی چندپنجره‌ای و بررسی خطا به تفکیک فروشگاه و تعطیلات است.</p>
<small>منابع: داده Walmart Recruiting – Store Sales Forecasting (Kaggle)؛
<a href="https://mlflow.org/docs/latest/ml/model-registry/workflow/">مستندات MLflow Model Registry</a>.
شناسه SHA-256 داده: <code>{escape(summary["dataset_sha256"])}</code></small>
</section></body></html>"""


def generate_report(data_path=DATA_PATH, root=ROOT, batch_id=None):
    root = Path(root)
    data_path = Path(data_path)
    forecasts = root / "reports" / "forecasts"
    if batch_id is None:
        batch_id = json.loads((forecasts / "latest.json").read_text(encoding="utf-8"))["batch_id"]
    batch = forecasts / batch_id
    summary = json.loads((batch / "summary.json").read_text(encoding="utf-8"))
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != summary["dataset_sha256"]:
        raise ValueError("The input dataset differs from the experiment dataset")
    comparison = pd.read_csv(batch / "comparison.csv")
    weekly = load_sales(data_path)
    eda = analyze(weekly, output=batch / "eda")
    version = configure_tracking(root).get_model_version(summary["registered_model"], summary["version"])
    if version.run_id != summary["best_run"]["run_id"]:
        raise ValueError("Registered model does not match the selected experiment run")
    report = build_report(summary, comparison, eda, weekly, batch, version.current_stage)
    output = batch / "final_report.html"
    output.write_text(report, encoding="utf-8")
    print(f"Report: {output.resolve()}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a report from actual MLflow results")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--batch-id")
    args = parser.parse_args()
    generate_report(args.data, batch_id=args.batch_id)
