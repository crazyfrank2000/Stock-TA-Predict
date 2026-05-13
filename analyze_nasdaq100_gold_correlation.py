"""Analyze correlation between the Nasdaq 100 index and gold.

This script downloads daily prices from Yahoo Finance's public chart endpoint,
aligns trading days, and calculates price-level, return, rolling, and lead/lag
correlations between the Nasdaq 100 (default: ^NDX) and gold futures (default:
GC=F). It saves CSV results under ``reports/`` and can optionally save charts
when matplotlib is installed.

Example:
    python analyze_nasdaq100_gold_correlation.py --start 2015-01-01 --end 2026-05-13
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
DEFAULT_START_DATE = "2015-01-01"
DEFAULT_ROLLING_WINDOW = 60
DEFAULT_MAX_LAG = 10
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class AssetConfig:
    """Configuration for a market series to download and label."""

    symbol: str
    label: str


@dataclass(frozen=True)
class PricePoint:
    """Single daily adjusted close observation."""

    trade_date: date
    close: float


def parse_date(value: str) -> date:
    """Parse an ISO date string for argparse."""

    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"日期必须使用 YYYY-MM-DD 格式，当前输入: {value}"
        ) from exc


def to_unix_seconds(value: date, end_of_day: bool = False) -> int:
    """Convert a date to a UTC Unix timestamp accepted by Yahoo Finance."""

    if end_of_day:
        dt = datetime(value.year, value.month, value.day, 23, 59, 59, tzinfo=timezone.utc)
    else:
        dt = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    return int(dt.timestamp())


def download_yahoo_prices(asset: AssetConfig, start: date, end: date) -> list[PricePoint]:
    """Download adjusted daily close prices from Yahoo Finance.

    The Yahoo chart endpoint returns timestamps, OHLC prices, and adjusted close
    values. Adjusted close is preferred when available so ETF distributions and
    futures adjustments are handled more consistently.
    """

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
    nasdaq: AssetConfig,
    gold: AssetConfig,
    start: date,
    end: date,
) -> list[dict[str, float | date | None]]:
    """Download both assets and align them on shared trading dates."""

    nasdaq_prices = {point.trade_date: point.close for point in download_yahoo_prices(nasdaq, start, end)}
    gold_prices = {point.trade_date: point.close for point in download_yahoo_prices(gold, start, end)}
    shared_dates = sorted(set(nasdaq_prices) & set(gold_prices))

    if len(shared_dates) < 30:
        raise RuntimeError(
            "两个资产可对齐的交易日少于 30 天，无法进行可靠相关性分析。"
        )

    rows: list[dict[str, float | date | None]] = []
    previous_nasdaq: float | None = None
    previous_gold: float | None = None
    for trade_day in shared_dates:
        nasdaq_close = nasdaq_prices[trade_day]
        gold_close = gold_prices[trade_day]
        nasdaq_return = None if previous_nasdaq is None else nasdaq_close / previous_nasdaq - 1
        gold_return = None if previous_gold is None else gold_close / previous_gold - 1
        rows.append(
            {
                "date": trade_day,
                nasdaq.label: nasdaq_close,
                gold.label: gold_close,
                f"{nasdaq.label}_return": nasdaq_return,
                f"{gold.label}_return": gold_return,
            }
        )
        previous_nasdaq = nasdaq_close
        previous_gold = gold_close

    return rows


def paired_values(
    left: Iterable[float | None], right: Iterable[float | None]
) -> tuple[list[float], list[float]]:
    """Return aligned non-null numeric pairs."""

    x_values: list[float] = []
    y_values: list[float] = []
    for x_value, y_value in zip(left, right):
        if x_value is None or y_value is None:
            continue
        if math.isnan(x_value) or math.isnan(y_value):
            continue
        x_values.append(float(x_value))
        y_values.append(float(y_value))
    return x_values, y_values


def pearson(left: Iterable[float | None], right: Iterable[float | None]) -> float:
    """Calculate Pearson correlation, returning NaN for insufficient data."""

    x_values, y_values = paired_values(left, right)
    if len(x_values) < 2:
        return math.nan
    x_mean = fmean(x_values)
    y_mean = fmean(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    x_denominator = sum((x - x_mean) ** 2 for x in x_values)
    y_denominator = sum((y - y_mean) ** 2 for y in y_values)
    denominator = math.sqrt(x_denominator * y_denominator)
    return numerator / denominator if denominator else math.nan


def sample_variance(values: list[float]) -> float:
    """Calculate sample variance."""

    if len(values) < 2:
        return math.nan
    mean = fmean(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def sample_covariance(left: list[float], right: list[float]) -> float:
    """Calculate sample covariance."""

    if len(left) < 2 or len(left) != len(right):
        return math.nan
    left_mean = fmean(left)
    right_mean = fmean(right)
    return sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right)) / (len(left) - 1)


def annualized_volatility(values: list[float]) -> float:
    """Calculate annualized volatility from daily returns."""

    variance = sample_variance(values)
    return math.sqrt(variance) * math.sqrt(TRADING_DAYS_PER_YEAR) if not math.isnan(variance) else math.nan


def rolling_correlations(
    left: list[float | None], right: list[float | None], window: int
) -> list[float | None]:
    """Calculate rolling Pearson correlations."""

    result: list[float | None] = []
    for index in range(len(left)):
        if index + 1 < window:
            result.append(None)
            continue
        result.append(pearson(left[index + 1 - window : index + 1], right[index + 1 - window : index + 1]))
    return result


def shifted(values: list[float | None], lag: int) -> list[float | None]:
    """Shift a list by lag days, preserving length with null padding."""

    if lag == 0:
        return values[:]
    if lag > 0:
        return [None] * lag + values[:-lag]
    return values[-lag:] + [None] * (-lag)


def calculate_summary(
    data: list[dict[str, float | date | None]],
    nasdaq_label: str,
    gold_label: str,
    rolling_window: int,
) -> dict[str, float | int | str]:
    """Create a compact summary of correlation and risk statistics."""

    nasdaq_prices = [row[nasdaq_label] for row in data]
    gold_prices = [row[gold_label] for row in data]
    nasdaq_returns = [row[f"{nasdaq_label}_return"] for row in data]
    gold_returns = [row[f"{gold_label}_return"] for row in data]
    paired_nasdaq_returns, paired_gold_returns = paired_values(nasdaq_returns, gold_returns)
    latest_rolling_values = [
        value for value in rolling_correlations(nasdaq_returns, gold_returns, rolling_window) if value is not None
    ]

    gold_variance = sample_variance(paired_gold_returns)
    beta_to_gold = (
        sample_covariance(paired_nasdaq_returns, paired_gold_returns) / gold_variance
        if gold_variance and not math.isnan(gold_variance)
        else math.nan
    )

    return {
        "start_date": min(row["date"] for row in data).isoformat(),
        "end_date": max(row["date"] for row in data).isoformat(),
        "aligned_trading_days": len(data),
        "daily_return_observations": len(paired_nasdaq_returns),
        "price_level_correlation": pearson(nasdaq_prices, gold_prices),
        "daily_return_correlation": pearson(nasdaq_returns, gold_returns),
        f"latest_{rolling_window}d_return_correlation": latest_rolling_values[-1]
        if latest_rolling_values
        else math.nan,
        "nasdaq_annualized_volatility": annualized_volatility(paired_nasdaq_returns),
        "gold_annualized_volatility": annualized_volatility(paired_gold_returns),
        "nasdaq_beta_to_gold_daily_returns": beta_to_gold,
    }


def calculate_lag_correlations(
    data: list[dict[str, float | date | None]],
    nasdaq_label: str,
    gold_label: str,
    max_lag: int,
) -> list[dict[str, float | int]]:
    """Calculate correlations when gold returns lead/lag Nasdaq returns.

    Positive ``gold_lag_days`` means gold returns are shifted forward and gold is
    tested as a leading indicator for Nasdaq returns. Negative values mean Nasdaq
    leads gold by that many trading days.
    """

    nasdaq_returns = [row[f"{nasdaq_label}_return"] for row in data]
    gold_returns = [row[f"{gold_label}_return"] for row in data]
    rows: list[dict[str, float | int]] = []
    for lag in range(-max_lag, max_lag + 1):
        rows.append(
            {
                "gold_lag_days": lag,
                "return_correlation": pearson(nasdaq_returns, shifted(gold_returns, lag)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int | str | date | None]]) -> None:
    """Write dictionaries to CSV with stable field ordering."""

    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plots(
    data: list[dict[str, float | date | None]],
    lag_correlations: list[dict[str, float | int]],
    nasdaq_label: str,
    gold_label: str,
    output_dir: Path,
    rolling_window: int,
) -> None:
    """Save visualization charts. Requires matplotlib in the runtime environment."""

    import matplotlib.pyplot as plt

    dates = [row["date"] for row in data]
    nasdaq_prices = [float(row[nasdaq_label]) for row in data]
    gold_prices = [float(row[gold_label]) for row in data]
    nasdaq_returns = [row[f"{nasdaq_label}_return"] for row in data]
    gold_returns = [row[f"{gold_label}_return"] for row in data]
    rolling_corr = rolling_correlations(nasdaq_returns, gold_returns, rolling_window)
    normalized_nasdaq = [value / nasdaq_prices[0] * 100 for value in nasdaq_prices]
    normalized_gold = [value / gold_prices[0] * 100 for value in gold_prices]
    scatter_x, scatter_y = paired_values(gold_returns, nasdaq_returns)

    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), constrained_layout=True)

    axes[0].plot(dates, normalized_nasdaq, label=nasdaq_label)
    axes[0].plot(dates, normalized_gold, label=gold_label)
    axes[0].set_title("Nasdaq 100 与黄金价格走势（起点=100）")
    axes[0].set_ylabel("Normalized price")
    axes[0].legend()

    axes[1].plot(dates, rolling_corr, color="tab:purple")
    axes[1].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[1].set_title(f"{rolling_window} 日滚动收益率相关性")
    axes[1].set_ylabel("Correlation")

    axes[2].scatter(scatter_x, scatter_y, alpha=0.35, s=12)
    axes[2].set_title("日收益率散点图")
    axes[2].set_xlabel(f"{gold_label} daily return")
    axes[2].set_ylabel(f"{nasdaq_label} daily return")

    fig.savefig(output_dir / "nasdaq100_gold_correlation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(
        [row["gold_lag_days"] for row in lag_correlations],
        [row["return_correlation"] for row in lag_correlations],
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("黄金收益率领先/滞后相关性")
    ax.set_xlabel("Gold lag days (positive = gold leads Nasdaq)")
    ax.set_ylabel("Correlation")
    fig.savefig(output_dir / "nasdaq100_gold_lag_correlation.png", dpi=150)
    plt.close(fig)


def save_outputs(
    data: list[dict[str, float | date | None]],
    summary: dict[str, float | int | str],
    lag_correlations: list[dict[str, float | int]],
    nasdaq_label: str,
    gold_label: str,
    output_dir: Path,
    rolling_window: int,
    should_save_plots: bool,
) -> None:
    """Save CSV files and optional visualization charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "nasdaq100_gold_aligned_data.csv", data)
    write_csv(output_dir / "nasdaq100_gold_correlation_summary.csv", [summary])
    write_csv(output_dir / "nasdaq100_gold_lag_correlations.csv", lag_correlations)

    if should_save_plots:
        save_plots(data, lag_correlations, nasdaq_label, gold_label, output_dir, rolling_window)


def print_summary(
    summary: dict[str, float | int | str],
    lag_correlations: list[dict[str, float | int]],
    rolling_window: int,
) -> None:
    """Print human-readable results to the terminal."""

    best_lag = max(
        lag_correlations,
        key=lambda row: abs(float(row["return_correlation"]))
        if not math.isnan(float(row["return_correlation"]))
        else -1,
    )

    print("\n=== 纳斯达克100 与 黄金相关性分析 ===")
    print(f"样本区间: {summary['start_date']} 至 {summary['end_date']}")
    print(f"对齐交易日: {int(summary['aligned_trading_days'])}")
    print(f"价格水平相关性: {float(summary['price_level_correlation']):.4f}")
    print(f"日收益率相关性: {float(summary['daily_return_correlation']):.4f}")
    print(
        f"最新 {rolling_window} 日滚动收益率相关性: "
        f"{float(summary[f'latest_{rolling_window}d_return_correlation']):.4f}"
    )
    print(f"纳指100年化波动率: {float(summary['nasdaq_annualized_volatility']):.2%}")
    print(f"黄金年化波动率: {float(summary['gold_annualized_volatility']):.2%}")
    print(f"纳指100相对黄金日收益Beta: {float(summary['nasdaq_beta_to_gold_daily_returns']):.4f}")
    print(
        "绝对值最高的领先/滞后相关性: "
        f"lag={int(best_lag['gold_lag_days'])}, corr={float(best_lag['return_correlation']):.4f}"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description="分析纳斯达克100指数与黄金的相关性")
    parser.add_argument("--start", type=parse_date, default=parse_date(DEFAULT_START_DATE), help="开始日期，格式 YYYY-MM-DD")
    parser.add_argument("--end", type=parse_date, default=date.today(), help="结束日期，格式 YYYY-MM-DD，默认今天")
    parser.add_argument("--nasdaq-symbol", default="^NDX", help="纳斯达克100 Yahoo Finance 代码，默认 ^NDX")
    parser.add_argument("--gold-symbol", default="GC=F", help="黄金 Yahoo Finance 代码，默认 GC=F（COMEX黄金期货）")
    parser.add_argument("--nasdaq-label", default="Nasdaq100", help="纳斯达克100输出列名")
    parser.add_argument("--gold-label", default="Gold", help="黄金输出列名")
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW, help="滚动相关性窗口，默认60个交易日")
    parser.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG, help="领先/滞后分析最大交易日，默认10")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"), help="输出目录，默认 reports")
    parser.add_argument("--no-plots", action="store_true", help="只输出CSV，不生成图片")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    """Run the correlation analysis workflow."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.end <= args.start:
        parser.error("--end 必须晚于 --start")
    if args.rolling_window < 2:
        parser.error("--rolling-window 至少为 2")
    if args.max_lag < 0:
        parser.error("--max-lag 不能为负数")

    nasdaq = AssetConfig(symbol=args.nasdaq_symbol, label=args.nasdaq_label)
    gold = AssetConfig(symbol=args.gold_symbol, label=args.gold_label)

    data = build_aligned_dataset(nasdaq, gold, args.start, args.end)
    summary = calculate_summary(data, nasdaq.label, gold.label, args.rolling_window)
    lag_correlations = calculate_lag_correlations(data, nasdaq.label, gold.label, args.max_lag)
    save_outputs(
        data=data,
        summary=summary,
        lag_correlations=lag_correlations,
        nasdaq_label=nasdaq.label,
        gold_label=gold.label,
        output_dir=args.output_dir,
        rolling_window=args.rolling_window,
        should_save_plots=not args.no_plots,
    )
    print_summary(summary, lag_correlations, args.rolling_window)
    print(f"\n结果已保存到: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
