"""分析香港恒生科技指数（HSTECH）与恒生指数（HSI）的相关性及多恒科空恒生配对策略。

策略定义：每日开盘同时买入HSTECH、卖空HSI（等权），观察不同持仓周期下的
胜率（spread日收益>0）与赔率（平均盈利/平均亏损）。

用法示例:
    python analyze_hk_tech_index.py --start 2022-01-01
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
DEFAULT_START_DATE = "2022-01-01"
DEFAULT_ROLLING_WINDOW = 60
TRADING_DAYS_PER_YEAR = 252


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
        raise argparse.ArgumentTypeError(f"日期格式须为 YYYY-MM-DD，当前输入: {value}") from exc


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
    for ts, close in zip(timestamps, closes):
        if close is None:
            continue
        trade_date = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        points.append(PricePoint(trade_date=trade_date, close=float(close)))

    deduped = {p.trade_date: p.close for p in points}
    return [PricePoint(trade_date=d, close=deduped[d]) for d in sorted(deduped)]


def paired_values(left: list, right: list) -> tuple[list[float], list[float]]:
    x_vals, y_vals = [], []
    for x, y in zip(left, right):
        if x is None or y is None or math.isnan(float(x)) or math.isnan(float(y)):
            continue
        x_vals.append(float(x))
        y_vals.append(float(y))
    return x_vals, y_vals


def pearson(left: list, right: list) -> float:
    x_vals, y_vals = paired_values(left, right)
    if len(x_vals) < 2:
        return math.nan
    xm = fmean(x_vals)
    ym = fmean(y_vals)
    num = sum((x - xm) * (y - ym) for x, y in zip(x_vals, y_vals))
    denom = math.sqrt(sum((x - xm) ** 2 for x in x_vals) * sum((y - ym) ** 2 for y in y_vals))
    return num / denom if denom else math.nan


def sample_variance(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    m = fmean(values)
    return sum((v - m) ** 2 for v in values) / (len(values) - 1)


def annualized_volatility(values: list[float]) -> float:
    v = sample_variance(values)
    return math.sqrt(v * TRADING_DAYS_PER_YEAR) if not math.isnan(v) else math.nan


def rolling_corr(left: list, right: list, window: int) -> list[float | None]:
    result: list[float | None] = []
    for i in range(len(left)):
        if i + 1 < window:
            result.append(None)
        else:
            result.append(pearson(left[i + 1 - window : i + 1], right[i + 1 - window : i + 1]))
    return result


def rolling_mean(values: list[float | None], window: int) -> list[float | None]:
    result: list[float | None] = []
    for i in range(len(values)):
        if i + 1 < window:
            result.append(None)
        else:
            window_vals = [v for v in values[i + 1 - window : i + 1] if v is not None]
            result.append(fmean(window_vals) if window_vals else None)
    return result


def build_aligned_data(
    tech: AssetConfig, hsi: AssetConfig, start: date, end: date
) -> list[dict]:
    tech_prices = {p.trade_date: p.close for p in download_yahoo_prices(tech, start, end)}
    hsi_prices = {p.trade_date: p.close for p in download_yahoo_prices(hsi, start, end)}
    shared_dates = sorted(set(tech_prices) & set(hsi_prices))

    if len(shared_dates) < 30:
        raise RuntimeError("可对齐交易日不足30天，请扩大数据范围。")

    rows: list[dict] = []
    prev_tech = prev_hsi = None
    for d in shared_dates:
        tc = tech_prices[d]
        hc = hsi_prices[d]
        tr = None if prev_tech is None else tc / prev_tech - 1
        hr = None if prev_hsi is None else hc / prev_hsi - 1
        spread = None if tr is None or hr is None else tr - hr
        rows.append({
            "date": d,
            tech.label: tc,
            hsi.label: hc,
            f"{tech.label}_return": tr,
            f"{hsi.label}_return": hr,
            "spread_return": spread,  # 多科技空恒生的日收益spread
        })
        prev_tech = tc
        prev_hsi = hc
    return rows


def compute_nday_spread(rows: list[dict], n: int, tech_label: str, hsi_label: str) -> list[dict]:
    """计算 n 日持有期 spread 收益（对数收益近似）。"""
    tech_prices = [row[tech_label] for row in rows]
    hsi_prices = [row[hsi_label] for row in rows]
    dates = [row["date"] for row in rows]
    results = []
    for i in range(len(rows) - n):
        if tech_prices[i] is None or tech_prices[i + n] is None:
            continue
        if hsi_prices[i] is None or hsi_prices[i + n] is None:
            continue
        tech_ret = tech_prices[i + n] / tech_prices[i] - 1
        hsi_ret = hsi_prices[i + n] / hsi_prices[i] - 1
        spread = tech_ret - hsi_ret
        results.append({
            "entry_date": dates[i],
            "exit_date": dates[i + n],
            f"tech_{n}d_return": tech_ret,
            f"hsi_{n}d_return": hsi_ret,
            f"spread_{n}d": spread,
            "win": 1 if spread > 0 else 0,
        })
    return results


def strategy_metrics(spreads: list[float], label: str) -> dict:
    """计算胜率、赔率、期望值等策略统计。"""
    wins = [s for s in spreads if s > 0]
    losses = [s for s in spreads if s < 0]
    total = len(spreads)
    win_rate = len(wins) / total if total else math.nan
    avg_win = fmean(wins) if wins else math.nan
    avg_loss = abs(fmean(losses)) if losses else math.nan
    odds = avg_win / avg_loss if (wins and losses and avg_loss != 0) else math.nan
    ev = win_rate * avg_win - (1 - win_rate) * avg_loss if not math.isnan(odds) else math.nan
    vol = math.sqrt(sample_variance(spreads)) if len(spreads) >= 2 else math.nan
    mean_spread = fmean(spreads) if spreads else math.nan
    sharpe = mean_spread / vol if (not math.isnan(vol) and vol != 0) else math.nan
    return {
        "period": label,
        "observations": total,
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss_abs": avg_loss,
        "odds_ratio": odds,  # avg_win / avg_loss，>1 表示盈大于亏
        "expected_value": ev,
        "mean_spread": mean_spread,
        "spread_volatility": vol,
        "sharpe_ratio_raw": sharpe,  # 未年化，用于相对比较
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plots(
    rows: list[dict],
    metrics_list: list[dict],
    holding_data: dict[int, list[dict]],
    tech_label: str,
    hsi_label: str,
    output_dir: Path,
    rolling_window: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec
    from matplotlib import font_manager

    # Use WenQuanYi Zen Hei if available for CJK character rendering
    wqy_fonts = [f.fname for f in font_manager.fontManager.ttflist if "WenQuanYi" in f.name]
    if wqy_fonts:
        plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"

    dates = [row["date"] for row in rows]
    tech_prices = [float(row[tech_label]) for row in rows]
    hsi_prices = [float(row[hsi_label]) for row in rows]
    tech_rets = [row[f"{tech_label}_return"] for row in rows]
    hsi_rets = [row[f"{hsi_label}_return"] for row in rows]
    spreads = [row["spread_return"] for row in rows]
    corr_series = rolling_corr(tech_rets, hsi_rets, rolling_window)
    spread_rolling_mean = rolling_mean(spreads, rolling_window)

    # 归一化价格（起点=100）
    norm_tech = [v / tech_prices[0] * 100 for v in tech_prices]
    norm_hsi = [v / hsi_prices[0] * 100 for v in hsi_prices]

    # 累积 spread 净值（从1出发）
    cumulative_spread = []
    cum = 1.0
    for s in spreads:
        if s is None:
            cumulative_spread.append(None)
        else:
            cum *= (1 + s)
            cumulative_spread.append(cum)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(14, 18))
    fig.suptitle(
        f"恒生科技(HSTECH) vs 恒生指数(HSI) 分析报告\n"
        f"（{dates[0]} 至 {dates[-1]}）",
        fontsize=14, fontweight="bold", y=0.98
    )
    gs = GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── 图1：价格走势（归一化）
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, norm_tech, label="恒生科技 HSTECH", color="#E6414A", linewidth=1.2)
    ax1.plot(dates, norm_hsi, label="恒生指数 HSI", color="#1F6FEB", linewidth=1.2)
    ax1.set_title("价格走势（起点=100归一化）")
    ax1.set_ylabel("归一化价格")
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()

    # ── 图2：滚动相关性
    ax2 = fig.add_subplot(gs[1, 0])
    corr_dates = [d for d, c in zip(dates, corr_series) if c is not None]
    corr_vals = [c for c in corr_series if c is not None]
    ax2.plot(corr_dates, corr_vals, color="#9B59B6", linewidth=1.0)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.axhline(0.8, color="#E67E22", linewidth=0.8, linestyle=":")
    ax2.set_title(f"{rolling_window}日滚动收益率相关性")
    ax2.set_ylabel("Pearson r")
    ax2.set_ylim(-1.1, 1.1)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate()

    # ── 图3：日收益散点图
    ax3 = fig.add_subplot(gs[1, 1])
    sx, sy = paired_values(hsi_rets, tech_rets)
    ax3.scatter(sx, sy, alpha=0.3, s=8, color="#2ECC71")
    if sx:
        # 回归线
        n = len(sx)
        xm, ym = fmean(sx), fmean(sy)
        b1 = sum((x - xm) * (y - ym) for x, y in zip(sx, sy)) / sum((x - xm) ** 2 for x in sx)
        b0 = ym - b1 * xm
        xl = [min(sx), max(sx)]
        ax3.plot(xl, [b0 + b1 * x for x in xl], color="#E6414A", linewidth=1.2, label=f"β={b1:.2f}")
    ax3.axhline(0, color="gray", linewidth=0.6)
    ax3.axvline(0, color="gray", linewidth=0.6)
    ax3.set_title("日收益率散点图（HSI vs HSTECH）")
    ax3.set_xlabel("HSI 日收益")
    ax3.set_ylabel("HSTECH 日收益")
    ax3.legend(fontsize=9)

    # ── 图4：多空spread累积净值
    ax4 = fig.add_subplot(gs[2, 0])
    cum_dates = [d for d, v in zip(dates, cumulative_spread) if v is not None]
    cum_vals = [v for v in cumulative_spread if v is not None]
    ax4.plot(cum_dates, cum_vals, color="#E6414A", linewidth=1.2)
    ax4.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax4.fill_between(cum_dates, 1.0, cum_vals,
                     where=[v >= 1.0 for v in cum_vals], alpha=0.15, color="#2ECC71")
    ax4.fill_between(cum_dates, 1.0, cum_vals,
                     where=[v < 1.0 for v in cum_vals], alpha=0.15, color="#E6414A")
    ax4.set_title("多恒科空恒生累积净值（日频）")
    ax4.set_ylabel("净值")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate()

    # ── 图5：滚动spread均值
    ax5 = fig.add_subplot(gs[2, 1])
    sm_dates = [d for d, v in zip(dates, spread_rolling_mean) if v is not None]
    sm_vals = [v for v in spread_rolling_mean if v is not None]
    ax5.plot(sm_dates, [v * 100 for v in sm_vals], color="#F39C12", linewidth=1.0)
    ax5.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax5.set_title(f"{rolling_window}日滚动spread均值（%）")
    ax5.set_ylabel("Spread 均值 (%)")
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate()

    # ── 图6：不同持仓周期胜率/赔率柱状图
    ax6 = fig.add_subplot(gs[3, 0])
    periods = [m["period"] for m in metrics_list]
    win_rates = [m["win_rate"] * 100 for m in metrics_list]
    colors = ["#2ECC71" if w >= 50 else "#E6414A" for w in win_rates]
    bars = ax6.bar(periods, win_rates, color=colors, alpha=0.75, edgecolor="white")
    ax6.axhline(50, color="black", linewidth=0.8, linestyle="--", label="50%基准")
    for bar, val in zip(bars, win_rates):
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax6.set_title("各持仓周期胜率")
    ax6.set_ylabel("胜率 (%)")
    ax6.set_ylim(0, 75)
    ax6.legend(fontsize=9)

    # ── 图7：赔率（odds ratio）
    ax7 = fig.add_subplot(gs[3, 1])
    odds = [m["odds_ratio"] for m in metrics_list]
    ev_vals = [m["expected_value"] * 100 for m in metrics_list]
    x = range(len(periods))
    width = 0.35
    b1_bars = ax7.bar([xi - width / 2 for xi in x], odds, width, label="赔率（盈/亏）", color="#3498DB", alpha=0.75)
    ax7_twin = ax7.twinx()
    ax7_twin.bar([xi + width / 2 for xi in x], ev_vals, width, label="期望值(%)", color="#F39C12", alpha=0.75)
    ax7.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax7.set_xticks(list(x))
    ax7.set_xticklabels(periods)
    ax7.set_title("赔率 & 期望收益")
    ax7.set_ylabel("赔率（倍）")
    ax7_twin.set_ylabel("期望值 (%)")
    ax7.legend(loc="upper left", fontsize=9)
    ax7_twin.legend(loc="upper right", fontsize=9)

    fig.savefig(output_dir / "hk_tech_hsi_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图表已保存: {output_dir / 'hk_tech_hsi_analysis.png'}")


def print_report(
    rows: list[dict],
    metrics_list: list[dict],
    tech_label: str,
    hsi_label: str,
    rolling_window: int,
) -> None:
    tech_rets = [row[f"{tech_label}_return"] for row in rows if row[f"{tech_label}_return"] is not None]
    hsi_rets = [row[f"{hsi_label}_return"] for row in rows if row[f"{hsi_label}_return"] is not None]
    spread_rets = [row["spread_return"] for row in rows if row["spread_return"] is not None]
    overall_corr = pearson(
        [row[f"{tech_label}_return"] for row in rows],
        [row[f"{hsi_label}_return"] for row in rows],
    )

    print("\n" + "=" * 65)
    print("  恒生科技指数(HSTECH) vs 恒生指数(HSI) 分析报告")
    print("=" * 65)
    print(f"  数据区间 : {rows[0]['date']} 至 {rows[-1]['date']}")
    print(f"  对齐交易日: {len(rows)} 天")
    print()
    print("── 相关性 ──────────────────────────────────────────────")
    print(f"  日收益率 Pearson 相关系数   : {overall_corr:.4f}")

    tech_px = [row[tech_label] for row in rows]
    hsi_px = [row[hsi_label] for row in rows]
    price_corr = pearson(tech_px, hsi_px)
    print(f"  价格水平相关系数            : {price_corr:.4f}")

    # Beta（HSTECH 对 HSI）
    paired_t, paired_h = paired_values(hsi_rets, tech_rets)
    if len(paired_h) >= 2:
        hsi_var = sample_variance(paired_h)
        from statistics import fmean as _fm
        cov = sum((t - _fm(paired_t)) * (h - _fm(paired_h))
                  for t, h in zip(paired_t, paired_h)) / (len(paired_t) - 1)
        # beta = cov(TECH, HSI) / var(HSI)
        hsi_var2 = sample_variance(paired_h)
        beta = cov / hsi_var2 if hsi_var2 else math.nan
        print(f"  HSTECH 对 HSI 的 Beta       : {beta:.4f}")

    print(f"  HSTECH 年化波动率           : {annualized_volatility(tech_rets):.2%}")
    print(f"  HSI    年化波动率           : {annualized_volatility(hsi_rets):.2%}")
    print(f"  Spread 年化波动率           : {annualized_volatility(spread_rets):.2%}")

    mean_spread_ann = fmean(spread_rets) * TRADING_DAYS_PER_YEAR if spread_rets else math.nan
    print(f"  Spread 年化均值（漂移）     : {mean_spread_ann:.2%}")

    print()
    print("── 多恒科空恒生配对策略表现 ────────────────────────────")
    print(f"  {'持仓周期':<10} {'样本数':>6} {'胜率':>8} {'平均盈利':>10} "
          f"{'平均亏损':>10} {'赔率':>8} {'期望值':>10} {'Sharpe':>8}")
    print("  " + "-" * 70)
    for m in metrics_list:
        print(
            f"  {m['period']:<10} {m['observations']:>6} "
            f"  {m['win_rate']:>7.2%} "
            f"{m['avg_win']:>10.3%} "
            f"{m['avg_loss_abs']:>10.3%} "
            f"{m['odds_ratio']:>8.3f} "
            f"{m['expected_value']:>10.4%} "
            f"{m['sharpe_ratio_raw']:>8.4f}"
        )

    print()
    print("── 策略解读 ────────────────────────────────────────────")
    m1d = next((m for m in metrics_list if "1日" in m["period"]), None)
    m5d = next((m for m in metrics_list if "5日" in m["period"]), None)
    m20d = next((m for m in metrics_list if "20日" in m["period"]), None)
    for m, label in [(m1d, "1日"), (m5d, "5日"), (m20d, "20日")]:
        if m is None:
            continue
        wr = m["win_rate"]
        od = m["odds_ratio"]
        ev = m["expected_value"]
        verdict = "正期望" if ev > 0 else "负期望"
        kelly = max(0, wr - (1 - wr) / od) if od > 0 and not math.isnan(od) else 0
        print(
            f"  [{label}] 胜率={wr:.1%}, 赔率={od:.2f}x → {verdict}"
            f"（期望={ev:.4%}，Kelly比例≈{kelly:.1%}）"
        )

    print()
    print("  注：赔率>1 且胜率>50% 为双优；期望值>0 为可做方向。")
    print("      Kelly 比例仅供参考，实战中需考虑滑点、融券成本、再平衡频率。")
    print("=" * 65)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="分析恒生科技与恒生指数相关性及多空配对策略")
    parser.add_argument("--start", type=parse_date, default=parse_date(DEFAULT_START_DATE))
    parser.add_argument("--end", type=parse_date, default=date.today())
    # 3033.HK = CSOP Hang Seng TECH Index ETF，与 HSTECH 高度相关（直接指数数据 Yahoo 不提供）
    parser.add_argument("--tech-symbol", default="3033.HK", help="恒生科技代理 Yahoo 代码（默认3033.HK ETF）")
    parser.add_argument("--hsi-symbol", default="^HSI", help="恒生指数 Yahoo 代码")
    parser.add_argument("--tech-label", default="HSTECH(3033.HK)")
    parser.add_argument("--hsi-label", default="HSI")
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    if args.end <= args.start:
        parser.error("--end 必须晚于 --start")

    tech = AssetConfig(symbol=args.tech_symbol, label=args.tech_label)
    hsi = AssetConfig(symbol=args.hsi_symbol, label=args.hsi_label)

    print(f"下载 {tech.label}({tech.symbol}) 数据...")
    print(f"下载 {hsi.label}({hsi.symbol}) 数据...")
    rows = build_aligned_data(tech, hsi, args.start, args.end)
    print(f"获取 {len(rows)} 个共同交易日。")

    holding_periods = [1, 5, 10, 20]
    metrics_list = []
    holding_data: dict[int, list[dict]] = {}
    for n in holding_periods:
        nd_data = compute_nday_spread(rows, n, tech.label, hsi.label)
        holding_data[n] = nd_data
        spreads = [d[f"spread_{n}d"] for d in nd_data]
        label = f"{n}日持有"
        metrics_list.append(strategy_metrics(spreads, label))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "hk_tech_hsi_daily.csv", rows)
    write_csv(args.output_dir / "hk_tech_hsi_strategy_metrics.csv", metrics_list)
    for n, nd_data in holding_data.items():
        write_csv(args.output_dir / f"hk_tech_hsi_{n}d_trades.csv", nd_data)

    print_report(rows, metrics_list, tech.label, hsi.label, args.rolling_window)

    if not args.no_plots:
        try:
            save_plots(rows, metrics_list, holding_data, tech.label, hsi.label,
                       args.output_dir, args.rolling_window)
        except ImportError:
            print("  [提示] matplotlib 未安装，跳过图表生成。")

    print(f"\n结果文件已保存到: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
