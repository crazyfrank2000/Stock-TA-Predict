#!/usr/bin/env python3
"""
板块轮动量化分析、未来20个交易日强势板块预测与模拟。

输入数据要求为长表CSV，至少包含以下列：
    date,symbol,close
可选列：
    volume,sector,name

示例：
    python sector_rotation.py --input data/etf_prices.csv --top-n 5 --simulate

说明：
- rotation_score 是基于截面相对强弱、趋势、波动、回撤和成交量的规则评分。
- forecast_20d 使用历史条件分布模拟，不使用未来真实收益。
- backtest_top_n_rotation 使用当天可见分数选股，再持有未来20个交易日，用于历史模拟检验。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_BENCHMARK = "SPY"
DEFAULT_HORIZON = 20

DEFAULT_ETF_SECTORS = {
    "SPY": "US Large Cap",
    "IVV": "US Large Cap",
    "VOO": "US Large Cap",
    "QQQ": "Nasdaq 100",
    "QQQM": "Nasdaq 100",
    "XLK": "Technology",
    "VGT": "Technology",
    "XLE": "Energy",
    "XLF": "Financials",
    "KBE": "Banks",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLC": "Communication Services",
    "SMH": "Semiconductors",
    "SOXX": "Semiconductors",
    "IWM": "Small Cap",
    "VB": "Small Cap",
    "VTV": "Value",
    "VUG": "Growth",
    "QUAL": "Quality",
    "SCHD": "Dividend",
    "JEPI": "Covered Call",
    "JEPQ": "Covered Call",
    "TLT": "Long Treasury",
    "IEF": "Intermediate Treasury",
    "SGOV": "T-Bill",
    "BIL": "T-Bill",
    "GLD": "Gold",
    "IAU": "Gold",
    "SLV": "Silver",
    "GDX": "Gold Miners",
    "IEMG": "Emerging Markets",
    "VWO": "Emerging Markets",
    "EWY": "Korea",
    "EWJ": "Japan",
    "INDA": "India",
    "TUR": "Turkey",
    "EIS": "Israel",
}

FACTOR_COLUMNS = [
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "excess_ret_20d",
    "trend_20",
    "trend_60",
    "ma_ratio",
    "volume_ratio",
    "vol_20",
    "drawdown_60",
    "score_change_5d",
]

SCORE_WEIGHTS = {
    "ret_5d_z": 0.12,
    "ret_20d_z": 0.18,
    "ret_60d_z": 0.12,
    "excess_ret_20d_z": 0.20,
    "trend_20_z": 0.10,
    "trend_60_z": 0.10,
    "ma_ratio_z": 0.05,
    "volume_ratio_z": 0.04,
    "vol_20_z": -0.08,
    "drawdown_60_z": 0.10,
    "score_change_5d_z": 0.07,
}


@dataclass(frozen=True)
class ForecastConfig:
    """20日预测/模拟参数。"""

    horizon: int = DEFAULT_HORIZON
    benchmark: str = DEFAULT_BENCHMARK
    top_n: int = 5
    simulations: int = 5000
    min_history: int = 24
    random_state: int = 120


def load_price_panel(input_path: str | Path) -> pd.DataFrame:
    """读取ETF价格长表，并补齐常用ETF的sector。"""
    df = pd.read_csv(input_path)
    required = {"date", "symbol", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"输入文件缺少必要列: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    if "volume" not in df.columns:
        df["volume"] = np.nan
    else:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    if "sector" not in df.columns:
        df["sector"] = df["symbol"].map(DEFAULT_ETF_SECTORS).fillna("Other")
    else:
        df["sector"] = df["sector"].fillna(df["symbol"].map(DEFAULT_ETF_SECTORS)).fillna("Other")

    if "name" not in df.columns:
        df["name"] = df["symbol"]

    df = df.dropna(subset=["date", "symbol", "close"])
    df = df.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")
    return df.reset_index(drop=True)


def _safe_zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def calculate_rotation_factors(df: pd.DataFrame, benchmark: str = DEFAULT_BENCHMARK) -> pd.DataFrame:
    """计算收益、趋势、相对强弱、波动和回撤等板块轮动因子。"""
    data = df.copy().sort_values(["symbol", "date"])

    for window in [1, 5, 20, 60]:
        data[f"ret_{window}d"] = data.groupby("symbol")["close"].pct_change(window)

    data["ma_20"] = data.groupby("symbol")["close"].transform(lambda s: s.rolling(20).mean())
    data["ma_60"] = data.groupby("symbol")["close"].transform(lambda s: s.rolling(60).mean())
    data["trend_20"] = data["close"] / data["ma_20"] - 1
    data["trend_60"] = data["close"] / data["ma_60"] - 1
    data["ma_ratio"] = data["ma_20"] / data["ma_60"] - 1

    data["vol_20"] = data.groupby("symbol")["ret_1d"].transform(lambda s: s.rolling(20).std())
    data["high_60"] = data.groupby("symbol")["close"].transform(lambda s: s.rolling(60).max())
    data["drawdown_60"] = data["close"] / data["high_60"] - 1

    data["vol_ma_20"] = data.groupby("symbol")["volume"].transform(lambda s: s.rolling(20).mean())
    data["volume_ratio"] = data["volume"] / data["vol_ma_20"]
    data["volume_ratio"] = data["volume_ratio"].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    benchmark = benchmark.upper()
    bench = data[data["symbol"] == benchmark][["date", "ret_5d", "ret_20d", "ret_60d"]].rename(
        columns={
            "ret_5d": "bench_ret_5d",
            "ret_20d": "bench_ret_20d",
            "ret_60d": "bench_ret_60d",
        }
    )
    if bench.empty:
        data["bench_ret_5d"] = 0.0
        data["bench_ret_20d"] = 0.0
        data["bench_ret_60d"] = 0.0
    else:
        data = data.merge(bench, on="date", how="left")

    for window in [5, 20, 60]:
        data[f"excess_ret_{window}d"] = data[f"ret_{window}d"] - data[f"bench_ret_{window}d"].fillna(0.0)

    return data


def add_rotation_score(df: pd.DataFrame) -> pd.DataFrame:
    """对每天的ETF截面做z-score并合成板块轮动分数。"""
    data = df.copy().sort_values(["date", "symbol"])

    preliminary_factors = [col for col in FACTOR_COLUMNS if col != "score_change_5d"]
    for col in preliminary_factors:
        data[f"{col}_z"] = data.groupby("date")[col].transform(_safe_zscore)

    base_score = sum(data[factor_z] * weight for factor_z, weight in SCORE_WEIGHTS.items() if factor_z != "score_change_5d_z")
    data["base_rotation_score"] = base_score
    data["score_change_5d"] = data.groupby("symbol")["base_rotation_score"].diff(5)
    data["score_change_5d_z"] = data.groupby("date")["score_change_5d"].transform(_safe_zscore)

    data["rotation_score"] = sum(data[factor_z] * weight for factor_z, weight in SCORE_WEIGHTS.items())
    data["score_rank"] = data.groupby("date")["rotation_score"].rank(ascending=False, method="min")
    data["score_percentile"] = data.groupby("date")["rotation_score"].rank(pct=True)
    return data


def add_forward_returns(df: pd.DataFrame, horizon: int = DEFAULT_HORIZON, benchmark: str = DEFAULT_BENCHMARK) -> pd.DataFrame:
    """添加未来horizon交易日收益，仅用于训练、回测或历史模拟，不可作为当日特征。"""
    data = df.copy().sort_values(["symbol", "date"])
    data[f"future_ret_{horizon}d"] = data.groupby("symbol")["close"].shift(-horizon) / data["close"] - 1

    benchmark = benchmark.upper()
    bench = data[data["symbol"] == benchmark][["date", f"future_ret_{horizon}d"]].rename(
        columns={f"future_ret_{horizon}d": f"future_bench_ret_{horizon}d"}
    )
    if bench.empty:
        data[f"future_bench_ret_{horizon}d"] = 0.0
    else:
        data = data.merge(bench, on="date", how="left")

    data[f"future_excess_ret_{horizon}d"] = (
        data[f"future_ret_{horizon}d"] - data[f"future_bench_ret_{horizon}d"].fillna(0.0)
    )
    data[f"label_outperform_{horizon}d"] = (data[f"future_excess_ret_{horizon}d"] > 0).astype("Int64")
    return data


def prepare_rotation_dataset(
    df: pd.DataFrame,
    benchmark: str = DEFAULT_BENCHMARK,
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """一站式生成板块轮动数据集。"""
    data = calculate_rotation_factors(df, benchmark=benchmark)
    data = add_rotation_score(data)
    data = add_forward_returns(data, horizon=horizon, benchmark=benchmark)
    return data.replace([np.inf, -np.inf], np.nan)


def latest_rotation_ranking(data: pd.DataFrame, top_n: int = 10, as_of: str | None = None) -> pd.DataFrame:
    """输出某日最新板块轮动排名。"""
    if as_of is None:
        date = data["date"].max()
    else:
        date = pd.to_datetime(as_of)

    daily = data[data["date"] == date].copy()
    if daily.empty:
        raise ValueError(f"没有找到日期 {date.date()} 的数据")

    columns = [
        "date",
        "symbol",
        "name",
        "sector",
        "close",
        "ret_5d",
        "ret_20d",
        "ret_60d",
        "excess_ret_20d",
        "trend_20",
        "vol_20",
        "drawdown_60",
        "score_change_5d",
        "rotation_score",
        "score_rank",
    ]
    return daily.sort_values("rotation_score", ascending=False)[columns].head(top_n)


def forecast_20d_strength(data: pd.DataFrame, config: ForecastConfig, as_of: str | None = None) -> pd.DataFrame:
    """
    预测未来20个交易日强势板块。

    方法：对每个ETF找到历史上“轮动分数不低于当前分数、且有未来20日真实收益”的样本，
    用这些历史条件样本的未来收益分布估计未来20日收益、跑赢概率和分位数。
    """
    rng = np.random.default_rng(config.random_state)
    date = data["date"].max() if as_of is None else pd.to_datetime(as_of)
    latest = data[data["date"] == date].copy()
    history = data[data["date"] < date].dropna(
        subset=["rotation_score", f"future_ret_{config.horizon}d", f"future_excess_ret_{config.horizon}d"]
    )

    rows = []
    for _, row in latest.iterrows():
        symbol_history = history[history["symbol"] == row["symbol"]]
        if len(symbol_history) < config.min_history:
            symbol_history = history

        threshold = row["rotation_score"]
        conditioned = symbol_history[symbol_history["rotation_score"] >= threshold]
        if len(conditioned) < config.min_history:
            cutoff = symbol_history["rotation_score"].quantile(0.70)
            conditioned = symbol_history[symbol_history["rotation_score"] >= cutoff]
        if len(conditioned) < config.min_history:
            conditioned = symbol_history

        if conditioned.empty:
            continue

        sampled = conditioned.sample(
            n=config.simulations,
            replace=True,
            random_state=int(rng.integers(0, 1_000_000_000)),
        )
        future_returns = sampled[f"future_ret_{config.horizon}d"]
        future_excess = sampled[f"future_excess_ret_{config.horizon}d"]

        rows.append(
            {
                "date": date,
                "symbol": row["symbol"],
                "name": row.get("name", row["symbol"]),
                "sector": row.get("sector", "Other"),
                "close": row["close"],
                "rotation_score": row["rotation_score"],
                "score_rank": row["score_rank"],
                "expected_ret_20d": future_returns.mean(),
                "expected_excess_ret_20d": future_excess.mean(),
                "prob_positive_20d": (future_returns > 0).mean(),
                "prob_outperform_20d": (future_excess > 0).mean(),
                "p25_ret_20d": future_returns.quantile(0.25),
                "p50_ret_20d": future_returns.quantile(0.50),
                "p75_ret_20d": future_returns.quantile(0.75),
                "history_samples": len(conditioned),
            }
        )

    forecast = pd.DataFrame(rows)
    if forecast.empty:
        return forecast

    forecast["forecast_score"] = (
        0.45 * forecast["rotation_score"].rank(pct=True)
        + 0.30 * forecast["expected_excess_ret_20d"].rank(pct=True)
        + 0.25 * forecast["prob_outperform_20d"].rank(pct=True)
    )
    return forecast.sort_values("forecast_score", ascending=False).head(config.top_n).reset_index(drop=True)


def backtest_top_n_rotation(
    data: pd.DataFrame,
    top_n: int = 5,
    horizon: int = DEFAULT_HORIZON,
    cost: float = 0.001,
    benchmark: str = DEFAULT_BENCHMARK,
) -> pd.DataFrame:
    """每horizon个交易日调仓一次，买入轮动分数最高的top_n个ETF，生成历史模拟。"""
    available_dates = sorted(data.dropna(subset=["rotation_score", f"future_ret_{horizon}d"])["date"].unique())
    rebalance_dates = available_dates[::horizon]
    previous_picks: set[str] = set()
    rows = []

    for date in rebalance_dates:
        daily = data[data["date"] == date].sort_values("rotation_score", ascending=False)
        picks = daily.head(top_n).copy()
        if picks.empty:
            continue

        current_picks = set(picks["symbol"])
        turnover = 1.0 if not previous_picks else len(current_picks.symmetric_difference(previous_picks)) / (2 * top_n)
        gross_return = picks[f"future_ret_{horizon}d"].mean()
        net_return = gross_return - cost * turnover

        bench_row = data[(data["date"] == date) & (data["symbol"] == benchmark.upper())]
        benchmark_return = bench_row[f"future_ret_{horizon}d"].iloc[0] if not bench_row.empty else np.nan

        rows.append(
            {
                "date": date,
                "picks": ",".join(picks["symbol"].tolist()),
                "gross_return": gross_return,
                "turnover": turnover,
                "cost": cost * turnover,
                "net_return": net_return,
                "benchmark_return": benchmark_return,
                "excess_return": net_return - benchmark_return if pd.notna(benchmark_return) else np.nan,
            }
        )
        previous_picks = current_picks

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["strategy_equity"] = (1 + result["net_return"].fillna(0)).cumprod()
    result["benchmark_equity"] = (1 + result["benchmark_return"].fillna(0)).cumprod()
    return result


def summarize_backtest(backtest: pd.DataFrame) -> dict[str, float]:
    """汇总历史模拟绩效。"""
    if backtest.empty:
        return {}

    returns = backtest["net_return"].dropna()
    bench_returns = backtest["benchmark_return"].dropna()
    periods_per_year = 252 / DEFAULT_HORIZON
    total_return = backtest["strategy_equity"].iloc[-1] - 1
    benchmark_total_return = backtest["benchmark_equity"].iloc[-1] - 1
    annual_return = backtest["strategy_equity"].iloc[-1] ** (periods_per_year / len(backtest)) - 1
    annual_vol = returns.std(ddof=0) * np.sqrt(periods_per_year)
    sharpe = annual_return / annual_vol if annual_vol else np.nan
    running_max = backtest["strategy_equity"].cummax()
    max_drawdown = (backtest["strategy_equity"] / running_max - 1).min()
    win_rate = (returns > bench_returns.reindex(returns.index)).mean()

    return {
        "periods": float(len(backtest)),
        "total_return": float(total_return),
        "benchmark_total_return": float(benchmark_total_return),
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "win_rate_vs_benchmark": float(win_rate),
    }


def classify_rotation_regime(latest: pd.DataFrame) -> str:
    """根据关键ETF相对分数给出当前板块轮动状态描述。"""
    score_map = latest.set_index("symbol")["rotation_score"].to_dict()

    def score(symbol: str) -> float:
        return score_map.get(symbol, np.nan)

    def best_available(*symbols: str) -> float:
        values = [score(symbol) for symbol in symbols]
        values = [value for value in values if pd.notna(value)]
        return max(values) if values else np.nan

    spy = score("SPY")
    xle = score("XLE")
    xlk = score("XLK")
    smh = best_available("SMH", "SOXX")
    iwm = score("IWM")
    tlt = score("TLT")
    sgov = best_available("SGOV", "BIL")
    gld = best_available("GLD", "IAU")
    xlf = score("XLF")

    if pd.notna(xle) and pd.notna(spy) and xle > spy and pd.notna(smh) and smh < spy:
        return "能源强、半导体弱：偏再通胀/风险降温轮动"
    if pd.notna(xlk) and pd.notna(smh) and pd.notna(iwm) and xlk > spy and smh > spy and iwm > spy:
        return "科技、半导体、小盘共振：风险偏好扩张"
    if pd.notna(sgov) and pd.notna(tlt) and pd.notna(iwm) and sgov > spy and tlt < spy and iwm < spy:
        return "短债/现金占优，小盘和长债偏弱：防御型轮动"
    if pd.notna(gld) and pd.notna(tlt) and gld > spy and tlt > spy:
        return "黄金和长债占优：避险型轮动"
    if pd.notna(iwm) and pd.notna(xlf) and iwm > spy and xlf > spy:
        return "小盘和金融转强：经济复苏扩散型轮动"
    return "混合轮动：暂无单一主线"


def format_percent_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """便于CLI展示的百分比格式化。"""
    output = df.copy()
    for col in columns:
        if col in output.columns:
            output[col] = output[col].map(lambda value: "" if pd.isna(value) else f"{value:.2%}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="板块轮动分析、20交易日强势板块预测与历史模拟")
    parser.add_argument("--input", required=True, help="ETF价格长表CSV，至少包含 date,symbol,close")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="相对强弱基准，默认SPY")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="预测/持有交易日数，默认20")
    parser.add_argument("--top-n", type=int, default=5, help="输出前N个强势板块，默认5")
    parser.add_argument("--date", default=None, help="分析日期，默认使用最新日期")
    parser.add_argument("--simulate", action="store_true", help="运行每20个交易日调仓的历史模拟")
    parser.add_argument("--simulations", type=int, default=5000, help="未来20日bootstrap模拟次数")
    parser.add_argument("--cost", type=float, default=0.001, help="单次换手成本，默认0.1%%")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_price_panel(args.input)
    config = ForecastConfig(
        horizon=args.horizon,
        benchmark=args.benchmark,
        top_n=args.top_n,
        simulations=args.simulations,
    )
    data = prepare_rotation_dataset(raw, benchmark=args.benchmark, horizon=args.horizon)

    ranking = latest_rotation_ranking(data, top_n=args.top_n, as_of=args.date)
    forecast = forecast_20d_strength(data, config=config, as_of=args.date)
    latest = data[data["date"] == (data["date"].max() if args.date is None else pd.to_datetime(args.date))]

    percent_cols = [
        "ret_5d",
        "ret_20d",
        "ret_60d",
        "excess_ret_20d",
        "trend_20",
        "vol_20",
        "drawdown_60",
        "expected_ret_20d",
        "expected_excess_ret_20d",
        "prob_positive_20d",
        "prob_outperform_20d",
        "p25_ret_20d",
        "p50_ret_20d",
        "p75_ret_20d",
    ]

    print("\n=== 最新板块轮动排名 ===")
    print(format_percent_columns(ranking, percent_cols).to_string(index=False))
    print("\n=== 未来20个交易日强势板块预测/模拟 ===")
    print(format_percent_columns(forecast, percent_cols).to_string(index=False))
    print("\n=== 当前轮动状态 ===")
    print(classify_rotation_regime(latest))

    if args.simulate:
        backtest = backtest_top_n_rotation(
            data,
            top_n=args.top_n,
            horizon=args.horizon,
            cost=args.cost,
            benchmark=args.benchmark,
        )
        summary = summarize_backtest(backtest)
        print("\n=== 历史模拟汇总 ===")
        if summary:
            for key, value in summary.items():
                if key == "periods":
                    print(f"{key}: {value:.0f}")
                else:
                    print(f"{key}: {value:.2%}" if "sharpe" not in key else f"{key}: {value:.2f}")
            print("\n最近5次调仓:")
            print(format_percent_columns(backtest.tail(5), ["gross_return", "cost", "net_return", "benchmark_return", "excess_return"]).to_string(index=False))
        else:
            print("历史数据不足，无法模拟。")


if __name__ == "__main__":
    main()
