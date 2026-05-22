"""Analyze correlation between US 10-Year Treasury Yield (^TNX) and QQQ over 20 years.

This script downloads daily data from Yahoo Finance's public chart endpoint,
aligns trading days, and calculates comprehensive correlation metrics between
the 10-year Treasury yield and QQQ. Analysis includes regime detection,
rolling correlation, lead/lag analysis, and annual breakdowns.

Example:
    python analyze_tnx_qqq_correlation.py --start 2006-01-01 --end 2026-05-22
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
DEFAULT_START_DATE = "2006-01-01"
DEFAULT_ROLLING_WINDOW = 60
DEFAULT_MAX_LAG = 20
TRADING_DAYS_PER_YEAR = 252

REGIMES = [
    ("2006-01-01", "2008-12-31", "Pre-GFC 加息周期", "pre_gfc"),
    ("2009-01-01", "2015-12-31", "后危机零利率时代", "zirp"),
    ("2016-01-01", "2019-12-31", "温和加息周期", "hike_moderate"),
    ("2020-01-01", "2021-12-31", "疫情量化宽松", "qe_covid"),
    ("2022-01-01", "2023-12-31", "激进加息周期", "hike_aggressive"),
    ("2024-01-01", "2026-12-31", "降息转向期", "pivot"),
]


@dataclass(frozen=True)
class AssetConfig:
    symbol: str
    label: str


@dataclass(frozen=True)
class PricePoint:
    trade_date: date
    close: float


def parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"日期必须使用 YYYY-MM-DD 格式，当前输入: {value}"
        ) from exc


def to_unix_seconds(value: date, end_of_day: bool = False) -> int:
    if end_of_day:
        dt = datetime(value.year, value.month, value.day, 23, 59, 59, tzinfo=timezone.utc)
    else:
        dt = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    return int(dt.timestamp())


def download_yahoo_prices(asset: AssetConfig, start: date, end: date) -> list[PricePoint]:
    period1 = to_unix_seconds(start)
    period2 = to_unix_seconds(end, end_of_day=True)
    encoded_symbol = quote(asset.symbol, safe="")
    url = (
        f"{YAHOO_CHART_URL.format(symbol=encoded_symbol)}"
        f"?period1={period1}&period2={period2}&interval=1d&events=history"
        "&includeAdjustedClose=true"
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"下载 {asset.label}({asset.symbol}) 数据失败: {exc}") from exc

    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        raise RuntimeError(f"Yahoo Finance 返回错误 {asset.symbol}: {error}")

    results = chart.get("result") or []
    if not results:
        raise RuntimeError(f"未获取到 {asset.label}({asset.symbol}) 的价格数据")

    result = results[0]
    timestamps = result.get("timestamp") or []
    quote_data = (result.get("indicators", {}).get("quote") or [{}])[0]
    adjclose_data = (result.get("indicators", {}).get("adjclose") or [{}])[0]
    closes = adjclose_data.get("adjclose") or quote_data.get("close") or []

    if not timestamps or not closes:
        raise RuntimeError(f"{asset.label}({asset.symbol}) 数据缺少时间戳或收盘价")

    points: list[PricePoint] = []
    for timestamp, close in zip(timestamps, closes):
        if close is None:
            continue
        trade_date = datetime.fromtimestamp(timestamp, tz=timezone.utc).date()
        points.append(PricePoint(trade_date=trade_date, close=float(close)))

    deduped = {point.trade_date: point.close for point in points}
    cleaned = [PricePoint(trade_date=day, close=deduped[day]) for day in sorted(deduped)]
    if not cleaned:
        raise RuntimeError(f"{asset.label}({asset.symbol}) 清洗后没有有效价格数据")

    return cleaned


def build_aligned_dataset(
    tnx: AssetConfig,
    qqq: AssetConfig,
    start: date,
    end: date,
) -> list[dict]:
    """Download both assets and align on shared trading dates.

    For ^TNX: close = yield level (%), change = yield change in percentage points.
    For QQQ: close = price, return = daily percentage return.
    """
    tnx_prices = {p.trade_date: p.close for p in download_yahoo_prices(tnx, start, end)}
    qqq_prices = {p.trade_date: p.close for p in download_yahoo_prices(qqq, start, end)}
    shared_dates = sorted(set(tnx_prices) & set(qqq_prices))

    if len(shared_dates) < 60:
        raise RuntimeError("对齐交易日少于 60 天，无法进行可靠分析。")

    rows: list[dict] = []
    prev_tnx: float | None = None
    prev_qqq: float | None = None
    for trade_day in shared_dates:
        tnx_close = tnx_prices[trade_day]
        qqq_close = qqq_prices[trade_day]
        # Yield change in basis points (1 bp = 0.01%)
        tnx_change_bp = None if prev_tnx is None else (tnx_close - prev_tnx) * 100
        tnx_change_pct = None if prev_tnx is None else (tnx_close - prev_tnx)
        qqq_return = None if prev_qqq is None else qqq_close / prev_qqq - 1
        rows.append({
            "date": trade_day,
            "year": trade_day.year,
            "tnx_yield": tnx_close,
            "qqq_price": qqq_close,
            "tnx_change_bp": tnx_change_bp,
            "tnx_change_pct": tnx_change_pct,
            "qqq_return": qqq_return,
        })
        prev_tnx = tnx_close
        prev_qqq = qqq_close

    return rows


def paired_values(
    left: Iterable[float | None], right: Iterable[float | None]
) -> tuple[list[float], list[float]]:
    x_vals: list[float] = []
    y_vals: list[float] = []
    for x, y in zip(left, right):
        if x is None or y is None:
            continue
        if math.isnan(x) or math.isnan(y):
            continue
        x_vals.append(float(x))
        y_vals.append(float(y))
    return x_vals, y_vals


def pearson(left: Iterable[float | None], right: Iterable[float | None]) -> float:
    x_vals, y_vals = paired_values(left, right)
    if len(x_vals) < 2:
        return math.nan
    x_mean = fmean(x_vals)
    y_mean = fmean(y_vals)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
    x_den = sum((x - x_mean) ** 2 for x in x_vals)
    y_den = sum((y - y_mean) ** 2 for y in y_vals)
    denom = math.sqrt(x_den * y_den)
    return num / denom if denom else math.nan


def sample_variance(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    mean = fmean(values)
    return sum((v - mean) ** 2 for v in values) / (len(values) - 1)


def annualized_volatility(values: list[float]) -> float:
    var = sample_variance(values)
    return math.sqrt(var) * math.sqrt(TRADING_DAYS_PER_YEAR) if not math.isnan(var) else math.nan


def rolling_pearson(
    left: list[float | None], right: list[float | None], window: int
) -> list[float | None]:
    result: list[float | None] = []
    for i in range(len(left)):
        if i + 1 < window:
            result.append(None)
        else:
            result.append(pearson(left[i + 1 - window: i + 1], right[i + 1 - window: i + 1]))
    return result


def shifted(values: list[float | None], lag: int) -> list[float | None]:
    if lag == 0:
        return values[:]
    if lag > 0:
        return [None] * lag + values[:-lag]
    return values[-lag:] + [None] * (-lag)


def analyze_regime(
    data: list[dict],
    regime_start: str,
    regime_end: str,
    regime_name: str,
) -> dict:
    """Calculate statistics for a specific market regime."""
    s = parse_date(regime_start)
    e = parse_date(regime_end)
    subset = [row for row in data if s <= row["date"] <= e]
    if len(subset) < 20:
        return {"regime": regime_name, "days": 0, "corr": math.nan,
                "qqq_ann_return": math.nan, "tnx_start": math.nan,
                "tnx_end": math.nan, "tnx_change_bp": math.nan}

    tnx_changes = [row["tnx_change_pct"] for row in subset]
    qqq_returns = [row["qqq_return"] for row in subset]
    corr = pearson(tnx_changes, qqq_returns)

    valid_qqq = [r for r in qqq_returns if r is not None]
    qqq_ann_return = sum(math.log(1 + r) for r in valid_qqq) * TRADING_DAYS_PER_YEAR / len(valid_qqq) if valid_qqq else math.nan

    tnx_vals = [row["tnx_yield"] for row in subset]
    tnx_start = tnx_vals[0]
    tnx_end = tnx_vals[-1]

    return {
        "regime": regime_name,
        "start": regime_start,
        "end": regime_end,
        "days": len(subset),
        "yield_change_vs_qqq_return_corr": corr,
        "qqq_annualized_log_return": qqq_ann_return,
        "tnx_yield_start": tnx_start,
        "tnx_yield_end": tnx_end,
        "tnx_net_change_bp": (tnx_end - tnx_start) * 100,
    }


def analyze_directional(data: list[dict]) -> dict:
    """Analyze QQQ performance when yields rise vs fall."""
    rise_days = [row["qqq_return"] for row in data
                 if row["tnx_change_pct"] is not None and row["tnx_change_pct"] > 0
                 and row["qqq_return"] is not None]
    fall_days = [row["qqq_return"] for row in data
                 if row["tnx_change_pct"] is not None and row["tnx_change_pct"] < 0
                 and row["qqq_return"] is not None]

    def pct(lst: list[float], threshold: float = 0) -> float:
        return sum(1 for v in lst if v > threshold) / len(lst) if lst else math.nan

    return {
        "yield_rise_days": len(rise_days),
        "yield_fall_days": len(fall_days),
        "qqq_mean_return_yield_rise": fmean(rise_days) if rise_days else math.nan,
        "qqq_mean_return_yield_fall": fmean(fall_days) if fall_days else math.nan,
        "qqq_positive_pct_yield_rise": pct(rise_days),
        "qqq_positive_pct_yield_fall": pct(fall_days),
    }


def analyze_yield_quartile(data: list[dict]) -> list[dict]:
    """Analyze QQQ returns segmented by yield level quartiles."""
    valid = [(row["tnx_yield"], row["qqq_return"]) for row in data if row["qqq_return"] is not None]
    yields = sorted(v[0] for v in valid)
    q1 = yields[len(yields) // 4]
    q2 = yields[len(yields) // 2]
    q3 = yields[3 * len(yields) // 4]

    def stats(lst: list[float]) -> dict:
        if not lst:
            return {"n": 0, "mean_return": math.nan, "ann_vol": math.nan}
        return {
            "n": len(lst),
            "mean_daily_return": fmean(lst),
            "ann_vol": annualized_volatility(lst),
        }

    buckets = [
        ("Q1: <{:.2f}%".format(q1), [r for y, r in valid if y < q1]),
        ("Q2: {:.2f}%-{:.2f}%".format(q1, q2), [r for y, r in valid if q1 <= y < q2]),
        ("Q3: {:.2f}%-{:.2f}%".format(q2, q3), [r for y, r in valid if q2 <= y < q3]),
        ("Q4: >{:.2f}%".format(q3), [r for y, r in valid if y >= q3]),
    ]
    result = []
    for label, returns in buckets:
        s = stats(returns)
        s["yield_quartile"] = label
        result.append(s)
    return result


def analyze_annual(data: list[dict]) -> list[dict]:
    """Year-by-year correlation and performance."""
    years = sorted(set(row["year"] for row in data))
    result = []
    for year in years:
        subset = [row for row in data if row["year"] == year]
        tnx_changes = [row["tnx_change_pct"] for row in subset]
        qqq_returns = [row["qqq_return"] for row in subset]
        corr = pearson(tnx_changes, qqq_returns)
        valid_qqq = [r for r in qqq_returns if r is not None]
        cum_return = math.exp(sum(math.log(1 + r) for r in valid_qqq)) - 1 if valid_qqq else math.nan
        tnx_yields = [row["tnx_yield"] for row in subset]
        result.append({
            "year": year,
            "trading_days": len(subset),
            "yield_change_vs_qqq_return_corr": corr,
            "qqq_cumulative_return": cum_return,
            "tnx_yield_start": tnx_yields[0],
            "tnx_yield_end": tnx_yields[-1],
            "tnx_net_change_bp": (tnx_yields[-1] - tnx_yields[0]) * 100,
        })
    return result


def calculate_lag_correlations(data: list[dict], max_lag: int) -> list[dict]:
    """Correlation when yield changes lead/lag QQQ returns."""
    tnx_changes = [row["tnx_change_pct"] for row in data]
    qqq_returns = [row["qqq_return"] for row in data]
    result = []
    for lag in range(-max_lag, max_lag + 1):
        result.append({
            "tnx_lag_days": lag,
            "description": f"TNX leads QQQ by {lag}d" if lag > 0 else (f"QQQ leads TNX by {-lag}d" if lag < 0 else "同步"),
            "return_correlation": pearson(qqq_returns, shifted(tnx_changes, lag)),
        })
    return result


def calculate_overall_summary(data: list[dict], rolling_window: int) -> dict:
    tnx_yields = [row["tnx_yield"] for row in data]
    qqq_prices = [row["qqq_price"] for row in data]
    tnx_changes = [row["tnx_change_pct"] for row in data]
    qqq_returns = [row["qqq_return"] for row in data]
    paired_tc, paired_qr = paired_values(tnx_changes, qqq_returns)
    rolling_corr = [v for v in rolling_pearson(tnx_changes, qqq_returns, rolling_window) if v is not None]

    return {
        "start_date": min(row["date"] for row in data).isoformat(),
        "end_date": max(row["date"] for row in data).isoformat(),
        "aligned_trading_days": len(data),
        "price_level_correlation_tnx_vs_qqq": pearson(tnx_yields, qqq_prices),
        "yield_change_vs_qqq_return_corr": pearson(tnx_changes, qqq_returns),
        f"latest_{rolling_window}d_rolling_corr": rolling_corr[-1] if rolling_corr else math.nan,
        "tnx_yield_min": min(tnx_yields),
        "tnx_yield_max": max(tnx_yields),
        "tnx_yield_mean": fmean(tnx_yields),
        "tnx_yield_start": tnx_yields[0],
        "tnx_yield_end": tnx_yields[-1],
        "qqq_price_start": qqq_prices[0],
        "qqq_price_end": qqq_prices[-1],
        "qqq_total_return": qqq_prices[-1] / qqq_prices[0] - 1,
        "qqq_annualized_volatility": annualized_volatility(paired_qr),
        "tnx_annualized_yield_change_vol": annualized_volatility(paired_tc),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plots(
    data: list[dict],
    annual: list[dict],
    lag_corr: list[dict],
    regimes: list[dict],
    quartile: list[dict],
    rolling_window: int,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch

    plt.rcParams["font.family"] = ["DejaVu Sans", "Arial Unicode MS", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    dates = [row["date"] for row in data]
    tnx_yields = [row["tnx_yield"] for row in data]
    qqq_prices = [row["qqq_price"] for row in data]
    tnx_changes = [row["tnx_change_pct"] for row in data]
    qqq_returns = [row["qqq_return"] for row in data]
    rolling_corr = rolling_pearson(tnx_changes, qqq_returns, rolling_window)

    REGIME_COLORS = ["#e8f4fd", "#fef9e7", "#e8f8f5", "#fdf2f8", "#fef0e7", "#f0f4ff"]

    # ─── Figure 1: Main 4-panel overview ───────────────────────────────────────
    fig = plt.figure(figsize=(16, 18))
    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.45)

    ax1a = fig.add_subplot(gs[0])
    ax1b = ax1a.twinx()
    ax1a.plot(dates, qqq_prices, color="#2196F3", linewidth=1.2, label="QQQ Price")
    ax1b.plot(dates, tnx_yields, color="#FF5722", linewidth=1.2, alpha=0.8, label="10Y Yield (%)")
    for i, (rs, re, rname, _) in enumerate(REGIMES):
        s, e = parse_date(rs), parse_date(re)
        ax1a.axvspan(s, e, alpha=0.15, color=REGIME_COLORS[i % len(REGIME_COLORS)], label=rname)
    ax1a.set_ylabel("QQQ Price (USD)", color="#2196F3", fontsize=10)
    ax1b.set_ylabel("10Y Treasury Yield (%)", color="#FF5722", fontsize=10)
    ax1a.set_title("QQQ vs US 10-Year Treasury Yield (20 Years)", fontsize=13, fontweight="bold")
    lines1, labels1 = ax1a.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1a.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8, ncol=2)

    ax2 = fig.add_subplot(gs[1])
    corr_vals = [v if v is not None else float("nan") for v in rolling_corr]
    colors_corr = ["#E53935" if (v is not None and not math.isnan(v) and v < 0) else "#43A047" for v in rolling_corr]
    ax2.plot(dates, corr_vals, color="#7B1FA2", linewidth=1.2, label=f"{rolling_window}-day Rolling Corr (Yield Change vs QQQ Return)")
    ax2.fill_between(dates, corr_vals, 0,
                     where=[v is not None and not math.isnan(v) and v < 0 for v in rolling_corr],
                     alpha=0.3, color="#E53935", label="Negative corr (yield up → QQQ down)")
    ax2.fill_between(dates, corr_vals, 0,
                     where=[v is not None and not math.isnan(v) and v > 0 for v in rolling_corr],
                     alpha=0.3, color="#43A047", label="Positive corr")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.axhline(-0.3, color="#E53935", linewidth=0.6, linestyle=":", alpha=0.7)
    ax2.axhline(0.3, color="#43A047", linewidth=0.6, linestyle=":", alpha=0.7)
    ax2.set_ylabel("Pearson Correlation", fontsize=10)
    ax2.set_title(f"{rolling_window}-Day Rolling Correlation: Yield Change vs QQQ Return", fontsize=12)
    ax2.set_ylim(-1, 1)
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[2])
    sc_x, sc_y = paired_values(tnx_changes, qqq_returns)
    scatter_colors = ["#E53935" if x > 0 else "#43A047" for x in sc_x]
    ax3.scatter(sc_x, sc_y, c=scatter_colors, alpha=0.25, s=8)
    if len(sc_x) > 2:
        coeff = sum((xi - fmean(sc_x)) * (yi - fmean(sc_y)) for xi, yi in zip(sc_x, sc_y))
        slope = coeff / sum((xi - fmean(sc_x)) ** 2 for xi in sc_x)
        intercept = fmean(sc_y) - slope * fmean(sc_x)
        x_line = [min(sc_x), max(sc_x)]
        y_line = [slope * x + intercept for x in x_line]
        ax3.plot(x_line, y_line, "k--", linewidth=1.5, label=f"OLS slope={slope:.4f}")
    ax3.axhline(0, color="gray", linewidth=0.5)
    ax3.axvline(0, color="gray", linewidth=0.5)
    ax3.set_xlabel("10Y Yield Daily Change (%)", fontsize=10)
    ax3.set_ylabel("QQQ Daily Return", fontsize=10)
    ax3.set_title("Scatter: Daily Yield Change vs QQQ Return", fontsize=12)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax3.legend(fontsize=9)

    ax4 = fig.add_subplot(gs[3])
    year_labels = [str(row["year"]) for row in annual]
    year_corrs = [row["yield_change_vs_qqq_return_corr"] for row in annual]
    bar_colors = ["#E53935" if (not math.isnan(c) and c < 0) else "#43A047" for c in year_corrs]
    bars = ax4.bar(year_labels, year_corrs, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax4.axhline(0, color="black", linewidth=0.8)
    ax4.set_xlabel("Year", fontsize=10)
    ax4.set_ylabel("Correlation", fontsize=10)
    ax4.set_title("Annual Correlation: Yield Change vs QQQ Return", fontsize=12)
    ax4.tick_params(axis="x", rotation=45)
    for bar, corr_val in zip(bars, year_corrs):
        if not math.isnan(corr_val):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (0.01 if corr_val >= 0 else -0.04),
                     f"{corr_val:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    fig.savefig(output_dir / "tnx_qqq_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ─── Figure 2: Regime & structural analysis ────────────────────────────────
    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle("TNX vs QQQ — Regime & Structural Analysis", fontsize=14, fontweight="bold")

    ax = axes[0][0]
    valid_regimes = [r for r in regimes if r["days"] > 0]
    r_names = [r["regime"][:12] for r in valid_regimes]
    r_corrs = [r["yield_change_vs_qqq_return_corr"] for r in valid_regimes]
    r_colors = ["#E53935" if (not math.isnan(c) and c < 0) else "#43A047" for c in r_corrs]
    bars = ax.barh(r_names, r_corrs, color=r_colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Correlation")
    ax.set_title("Correlation by Market Regime")
    for bar, val in zip(bars, r_corrs):
        if not math.isnan(val):
            ax.text(val + (0.01 if val >= 0 else -0.01), bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=9, fontweight="bold")

    ax = axes[0][1]
    q_labels = [r["yield_quartile"] for r in quartile]
    q_returns = [r.get("mean_daily_return", math.nan) for r in quartile]
    q_colors = ["#43A047" if (not math.isnan(v) and v > 0) else "#E53935" for v in q_returns]
    bars = ax.bar(q_labels, [v * 100 for v in q_returns if True], color=q_colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean Daily QQQ Return (%)")
    ax.set_title("QQQ Mean Daily Return by Yield Level Quartile")
    ax.tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, q_returns):
        if not math.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, val * 100 + (0.001 if val >= 0 else -0.002),
                    f"{val:.4%}", ha="center", va="bottom" if val >= 0 else "top", fontsize=9)

    ax = axes[1][0]
    lag_vals = [row["tnx_lag_days"] for row in lag_corr]
    lag_corrs = [row["return_correlation"] for row in lag_corr]
    lag_colors = ["#E53935" if (not math.isnan(c) and c < 0) else "#43A047" for c in lag_corrs]
    ax.bar(lag_vals, lag_corrs, color=lag_colors, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="navy", linewidth=1.2, linestyle="--", alpha=0.5)
    ax.set_xlabel("Lag Days (positive = TNX leads QQQ)")
    ax.set_ylabel("Correlation")
    ax.set_title("Lead/Lag Analysis: TNX Change vs QQQ Return")

    ax = axes[1][1]
    qqq_cum = [row["qqq_price"] / qqq_prices[0] * 100 for row in data]
    ax2b = ax.twinx()
    ax.plot(dates, qqq_cum, color="#2196F3", linewidth=1.2, label="QQQ (left)")
    ax2b.plot(dates, tnx_yields, color="#FF5722", linewidth=1.0, alpha=0.7, label="10Y Yield % (right)")
    ax.set_ylabel("QQQ Cumulative Return (base=100)", color="#2196F3")
    ax2b.set_ylabel("10Y Yield (%)", color="#FF5722")
    ax.set_title("QQQ Cumulative Return vs 10Y Yield Level")
    lines_a, labels_a = ax.get_legend_handles_labels()
    lines_b, labels_b = ax2b.get_legend_handles_labels()
    ax.legend(lines_a + lines_b, labels_a + labels_b, fontsize=8)

    fig2.tight_layout()
    fig2.savefig(output_dir / "tnx_qqq_regime_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)


def print_report(
    summary: dict,
    annual: list[dict],
    regimes: list[dict],
    lag_corr: list[dict],
    directional: dict,
    quartile: list[dict],
    rolling_window: int,
) -> None:
    SEP = "=" * 60

    print(f"\n{SEP}")
    print("  美国10年期国债收益率 vs QQQ — 20年相关性深度分析")
    print(SEP)
    print(f"  样本区间: {summary['start_date']}  →  {summary['end_date']}")
    print(f"  对齐交易日: {int(summary['aligned_trading_days'])} 天")
    print(SEP)

    print("\n【一、整体相关性统计】")
    print(f"  价格水平相关性 (TNX 绝对水平 vs QQQ 价格):  {summary['price_level_correlation_tnx_vs_qqq']:+.4f}")
    print(f"  日收益率相关性 (TNX 日变动 vs QQQ 日涨跌):  {summary['yield_change_vs_qqq_return_corr']:+.4f}")
    print(f"  最近 {rolling_window} 日滚动相关性:              {summary[f'latest_{rolling_window}d_rolling_corr']:+.4f}")

    print("\n【二、基本统计数据】")
    print(f"  10Y 收益率范围:  {summary['tnx_yield_min']:.2f}% ~ {summary['tnx_yield_max']:.2f}%  (均值 {summary['tnx_yield_mean']:.2f}%)")
    print(f"  10Y 收益率变化: {summary['tnx_yield_start']:.2f}% → {summary['tnx_yield_end']:.2f}%  ({(summary['tnx_yield_end']-summary['tnx_yield_start'])*100:+.0f} bp)")
    print(f"  QQQ 价格变化:   ${summary['qqq_price_start']:.2f} → ${summary['qqq_price_end']:.2f}  (总收益 {summary['qqq_total_return']:+.1%})")
    print(f"  QQQ 年化波动率: {summary['qqq_annualized_volatility']:.2%}")

    print("\n【三、市场周期分析】")
    print(f"  {'周期':<22} {'交易日':>6}  {'收益率变化':>10}  {'相关性':>8}  {'QQQ年化收益':>10}")
    print("  " + "-" * 62)
    for r in regimes:
        if r["days"] == 0:
            continue
        corr_str = f"{r['yield_change_vs_qqq_return_corr']:+.3f}" if not math.isnan(r["yield_change_vs_qqq_return_corr"]) else "  N/A "
        ann_str = f"{r['qqq_annualized_log_return']:+.1%}" if not math.isnan(r["qqq_annualized_log_return"]) else "  N/A "
        change_str = f"{r['tnx_net_change_bp']:+.0f}bp" if not math.isnan(r["tnx_net_change_bp"]) else "  N/A "
        print(f"  {r['regime'][:22]:<22} {r['days']:>6}  {change_str:>10}  {corr_str:>8}  {ann_str:>10}")

    print("\n【四、方向性分析 — 收益率上升vs下降时QQQ表现】")
    print(f"  收益率上升日:  {int(directional['yield_rise_days'])} 天  |  QQQ 平均日收益: {directional['qqq_mean_return_yield_rise']:+.4%}  |  QQQ 正收益概率: {directional['qqq_positive_pct_yield_rise']:.1%}")
    print(f"  收益率下降日:  {int(directional['yield_fall_days'])} 天  |  QQQ 平均日收益: {directional['qqq_mean_return_yield_fall']:+.4%}  |  QQQ 正收益概率: {directional['qqq_positive_pct_yield_fall']:.1%}")

    print("\n【五、利率水平分位数分析 — 不同利率环境下QQQ表现】")
    print(f"  {'利率区间':<25}  {'样本日':>6}  {'均值日收益':>10}  {'年化波动率':>10}")
    print("  " + "-" * 58)
    for q in quartile:
        mean_str = f"{q.get('mean_daily_return', math.nan):+.5%}" if not math.isnan(q.get("mean_daily_return", math.nan)) else "  N/A  "
        vol_str = f"{q.get('ann_vol', math.nan):.2%}" if not math.isnan(q.get("ann_vol", math.nan)) else "  N/A "
        print(f"  {q['yield_quartile']:<25}  {q['n']:>6}  {mean_str:>10}  {vol_str:>10}")

    best_lag = max(lag_corr, key=lambda r: abs(r["return_correlation"]) if not math.isnan(r["return_correlation"]) else 0)
    print("\n【六、领先/滞后分析】")
    print(f"  绝对相关性最高的时滞: {int(best_lag['tnx_lag_days'])} 天  ({best_lag['description']})  corr={best_lag['return_correlation']:+.4f}")

    print("\n【七、逐年相关性统计】")
    print(f"  {'年份':>5}  {'交易日':>6}  {'相关性':>8}  {'QQQ全年收益':>11}  {'10Y变动':>8}")
    print("  " + "-" * 46)
    for row in annual:
        corr_str = f"{row['yield_change_vs_qqq_return_corr']:+.3f}" if not math.isnan(row["yield_change_vs_qqq_return_corr"]) else "  N/A "
        ret_str = f"{row['qqq_cumulative_return']:+.1%}" if not math.isnan(row["qqq_cumulative_return"]) else " N/A  "
        chg_str = f"{row['tnx_net_change_bp']:+.0f}bp" if not math.isnan(row["tnx_net_change_bp"]) else " N/A "
        print(f"  {int(row['year']):>5}  {int(row['trading_days']):>6}  {corr_str:>8}  {ret_str:>11}  {chg_str:>8}")

    print(f"\n{SEP}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="分析美国10年期国债收益率与QQQ的20年相关性")
    parser.add_argument("--start", type=parse_date, default=parse_date(DEFAULT_START_DATE))
    parser.add_argument("--end", type=parse_date, default=date.today())
    parser.add_argument("--tnx-symbol", default="^TNX")
    parser.add_argument("--qqq-symbol", default="QQQ")
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW)
    parser.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    if args.end <= args.start:
        parser.error("--end 必须晚于 --start")

    tnx = AssetConfig(symbol=args.tnx_symbol, label="TNX")
    qqq = AssetConfig(symbol=args.qqq_symbol, label="QQQ")

    print(f"正在下载 {tnx.label} ({tnx.symbol}) 和 {qqq.label} ({qqq.symbol}) 数据...")
    data = build_aligned_dataset(tnx, qqq, args.start, args.end)
    print(f"数据下载完成，共 {len(data)} 个对齐交易日。")

    summary = calculate_overall_summary(data, args.rolling_window)
    annual = analyze_annual(data)
    regimes_result = [
        analyze_regime(data, rs, re, rname)
        for rs, re, rname, _ in REGIMES
    ]
    lag_corr = calculate_lag_correlations(data, args.max_lag)
    directional = analyze_directional(data)
    quartile = analyze_yield_quartile(data)

    print_report(summary, annual, regimes_result, lag_corr, directional, quartile, args.rolling_window)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "tnx_qqq_aligned_data.csv", data)
    write_csv(args.output_dir / "tnx_qqq_summary.csv", [summary])
    write_csv(args.output_dir / "tnx_qqq_annual.csv", annual)
    write_csv(args.output_dir / "tnx_qqq_regimes.csv", regimes_result)
    write_csv(args.output_dir / "tnx_qqq_lag_correlations.csv", lag_corr)
    write_csv(args.output_dir / "tnx_qqq_quartile.csv", quartile)
    write_csv(args.output_dir / "tnx_qqq_directional.csv", [directional])

    if not args.no_plots:
        try:
            print("\n正在生成图表...")
            save_plots(data, annual, lag_corr, regimes_result, quartile, args.rolling_window, args.output_dir)
            print(f"图表已保存: {args.output_dir / 'tnx_qqq_overview.png'}")
            print(f"图表已保存: {args.output_dir / 'tnx_qqq_regime_analysis.png'}")
        except ImportError:
            print("未检测到 matplotlib，跳过图表生成。")

    print(f"\n所有结果已保存到: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
