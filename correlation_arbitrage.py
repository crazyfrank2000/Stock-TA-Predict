"""
相关性套利策略实现
Correlation Arbitrage Strategy

核心逻辑：
  - 隐含相关性 (implied correlation) 长期高于实际相关性 (realized correlation) 10-20%
  - 卖出指数波动率 (做空隐含相关性) + 买入个股波动率 → vega 中性组合
  - 当 realized corr < implied corr 时获利

参考: S&P 500 期权市场，2000-2020 年年化超额收益约 5-8%
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm
from scipy.optimize import brentq
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ─────────────────────────────────────────────
# 1. Black-Scholes 工具函数
# ─────────────────────────────────────────────

def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Black-Scholes 期权定价"""
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes Vega"""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def implied_vol(market_price: float, S: float, K: float, T: float, r: float,
                option_type: str = "call", tol: float = 1e-6) -> Optional[float]:
    """
    用 Brent 法求隐含波动率
    返回 None 表示无解（深度实值/虚值或价格无效）
    """
    intrinsic = max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    if market_price <= intrinsic + 1e-8:
        return None

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - market_price

    try:
        return brentq(objective, 1e-6, 10.0, xtol=tol)
    except ValueError:
        return None


# ─────────────────────────────────────────────
# 2. 隐含相关性计算
# ─────────────────────────────────────────────

def implied_correlation(
    index_iv: float,
    stock_ivs: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    根据指数隐含波动率反推隐含相关性

    公式:
        σ_index² = Σ_i Σ_j w_i w_j ρ_ij σ_i σ_j

    简化：假设所有股票对相关性相同 ρ_ij = ρ（flat correlation）

        σ_index² = ρ * (Σ_i w_i σ_i)² + (1-ρ) * Σ_i w_i² σ_i²

    解出 ρ:
        ρ = (σ_index² - Σ_i w_i² σ_i²) / ((Σ_i w_i σ_i)² - Σ_i w_i² σ_i²)

    参数:
        index_iv   : 指数 ATM 隐含波动率 (年化)
        stock_ivs  : 各成分股 ATM 隐含波动率数组
        weights    : 各成分股权重 (加权和=1)
    """
    weights = weights / weights.sum()  # 归一化

    var_index = index_iv**2
    weighted_iv_sum = np.sum(weights * stock_ivs)          # Σ w_i σ_i
    sum_wi2_si2 = np.sum(weights**2 * stock_ivs**2)        # Σ w_i² σ_i²

    numerator = var_index - sum_wi2_si2
    denominator = weighted_iv_sum**2 - sum_wi2_si2

    if abs(denominator) < 1e-10:
        return np.nan

    rho = numerator / denominator
    return float(np.clip(rho, -1.0, 1.0))


def realized_correlation(returns_matrix: np.ndarray, weights: np.ndarray) -> float:
    """
    计算成分股之间的加权平均实际相关性

    参数:
        returns_matrix : shape (T, N)，T 个交易日、N 只股票的日收益率
        weights        : N 维权重向量
    """
    weights = weights / weights.sum()
    n = returns_matrix.shape[1]
    corr_matrix = np.corrcoef(returns_matrix.T)

    # 加权平均非对角线相关系数
    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                w = weights[i] * weights[j]
                numerator += w * corr_matrix[i, j]
                denominator += w

    return numerator / denominator if denominator > 0 else np.nan


# ─────────────────────────────────────────────
# 3. Vega 中性权重计算
# ─────────────────────────────────────────────

def vega_neutral_weights(
    index_vega: float,
    stock_vegas: np.ndarray,
    weights: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    构建 vega 中性组合:
        做空 1 单位指数 straddle (vega = index_vega)
        做多个股 straddle，使总 vega = 0

    返回:
        index_position  : 指数空头名义份数 (固定为 1.0)
        stock_positions : 各股票多头份数
    """
    weights = weights / weights.sum()
    # 按权重分配到各股票的 vega 覆盖量
    target_vega_per_stock = index_vega * weights
    # 各股票需买入份数 = 目标 vega / 该股票 vega
    stock_positions = target_vega_per_stock / (stock_vegas + 1e-10)
    return 1.0, stock_positions


# ─────────────────────────────────────────────
# 4. 数据生成 / 回测引擎
# ─────────────────────────────────────────────

@dataclass
class MarketParams:
    n_stocks: int = 10          # 成分股数量
    T_days: int = 1260          # 回测总交易日 (≈5年)
    true_corr: float = 0.25     # 真实平均相关性
    implied_corr_premium: float = 0.15  # 隐含相关性溢价
    annual_vol_index: float = 0.16      # 指数年化波动率
    stock_vol_mean: float = 0.28        # 个股年化波动率均值
    stock_vol_std: float = 0.06         # 个股波动率标准差
    rf: float = 0.02                    # 无风险利率
    rebalance_days: int = 21            # 再平衡频率（交易日）


def simulate_correlated_returns(
    n: int, T: int, corr: float, vols: np.ndarray, dt: float = 1 / 252
) -> np.ndarray:
    """模拟相关股票日收益率，使用 Cholesky 分解"""
    corr_matrix = corr * np.ones((n, n)) + (1 - corr) * np.eye(n)
    L = np.linalg.cholesky(corr_matrix)
    Z = np.random.randn(T, n)
    correlated_Z = Z @ L.T
    returns = correlated_Z * vols * np.sqrt(dt)
    return returns


def simulate_implied_vols(
    base_vols: np.ndarray,
    T: int,
    vol_of_vol: float = 0.02,
    mean_reversion: float = 0.1,
) -> np.ndarray:
    """模拟个股隐含波动率时间序列（OU 均值回归过程）"""
    dt = 1 / 252
    n = len(base_vols)
    iv_paths = np.zeros((T, n))
    iv_paths[0] = base_vols
    for t in range(1, T):
        noise = np.random.randn(n) * vol_of_vol * np.sqrt(dt)
        iv_paths[t] = (iv_paths[t - 1]
                       + mean_reversion * (base_vols - iv_paths[t - 1]) * dt
                       + noise)
        iv_paths[t] = np.clip(iv_paths[t], 0.05, 1.5)
    return iv_paths


def simulate_implied_corr_ou(
    T: int,
    mu: float,
    theta_speed: float = 2.0,
    sigma_vol: float = 0.05,
) -> np.ndarray:
    """
    用 Ornstein-Uhlenbeck 过程模拟隐含相关性路径

    dρ = θ(μ - ρ)dt + σ dW

    参数:
        mu          : 长期均值（真实相关性 + 溢价）
        theta_speed : 均值回归速度（越大回归越快）
        sigma_vol   : 相关性波动率
    """
    dt = 1 / 252
    path = np.zeros(T)
    path[0] = mu
    for t in range(1, T):
        dW = np.random.randn() * np.sqrt(dt)
        path[t] = path[t - 1] + theta_speed * (mu - path[t - 1]) * dt + sigma_vol * dW
    return np.clip(path, 0.05, 0.95)


def run_backtest(params: MarketParams, seed: int = 42) -> pd.DataFrame:
    """
    相关性套利策略完整回测（三个盈利来源）

    PnL 构成：
      1. Vega PnL     : 隐含波动率变动带来的头寸盈亏（vega × Δσ）
      2. Gamma Carry  : 指数实际方差低于隐含方差的持续收益
                        = 0.5 × Γ × (σ²_implied - σ²_realized) × dt
                        等价于：(ρ_implied - ρ_realized) × 分散化因子
      3. Stock Gamma  : 做多个股 straddle 的 delta 再对冲收益

    归一化：S=1，以单位名义本金计算，收益即为百分比。
    """
    np.random.seed(seed)
    dt = 1 / 252
    T_option = 30 / 252  # 30 天到期期权

    weights = np.random.dirichlet(np.ones(params.n_stocks))
    stock_base_ivs = np.clip(
        np.random.normal(params.stock_vol_mean, params.stock_vol_std, params.n_stocks),
        0.10, 0.80
    )

    # 个股 IV：OU 均值回归到各自基准 IV
    stock_iv_paths = simulate_implied_vols(stock_base_ivs, params.T_days)

    # 隐含相关性：OU 均值回归到 (真实相关性 + 溢价)
    mu_implied = params.true_corr + params.implied_corr_premium
    implied_corr_path = simulate_implied_corr_ou(
        params.T_days, mu=mu_implied, theta_speed=2.0, sigma_vol=0.05
    )

    # 指数 IV = f(implied_corr, stock_ivs)
    index_iv_path = np.array([
        np.sqrt(
            implied_corr_path[t] * np.sum(weights * stock_iv_paths[t]) ** 2
            + (1 - implied_corr_path[t]) * np.sum(weights ** 2 * stock_iv_paths[t] ** 2)
        )
        for t in range(params.T_days)
    ])

    # 实际收益率：使用真实相关性生成，个股 IV 为基准
    realized_returns = simulate_correlated_returns(
        params.n_stocks, params.T_days, params.true_corr, stock_base_ivs
    )
    index_returns = realized_returns @ weights  # 指数日收益

    # ── 辅助函数（归一化 S=1）──────────────────────────────────────
    def unit_vega(iv, T=T_option):
        """ATM straddle 单位 vega"""
        return 2 * bs_vega(1.0, 1.0, T, params.rf, iv)

    def unit_gamma(iv, T=T_option):
        """ATM straddle 单位 gamma = vega / (sigma × T)"""
        d1 = iv * np.sqrt(T) / 2
        return 2 * norm.pdf(d1) / (iv * np.sqrt(T))

    # ── 持仓状态 ──────────────────────────────────────────────────
    pos_idx_vega  = 0.0
    pos_stk_vega  = np.zeros(params.n_stocks)
    pos_idx_gamma = 0.0
    pos_stk_gamma = np.zeros(params.n_stocks)

    records = []
    for t in range(params.T_days):
        idx_iv  = index_iv_path[t]
        stk_ivs = stock_iv_paths[t]

        # 隐含 / 实际相关性
        ic = implied_correlation(idx_iv, stk_ivs, weights)
        window = min(t + 1, 60)
        rc = realized_correlation(
            realized_returns[max(0, t - window + 1):t + 1], weights
        ) if window >= 5 else np.nan
        corr_gap = (ic - rc) if not np.isnan(rc) else np.nan

        pnl_vega = pnl_gamma = 0.0

        if t > 0 and pos_idx_vega != 0:
            r_idx = index_returns[t]
            r_stk = realized_returns[t]

            # ── 1. Vega PnL（IV 变动） ───────────────────────────
            d_idx_iv  = idx_iv - index_iv_path[t - 1]
            d_stk_ivs = stk_ivs - stock_iv_paths[t - 1]
            pnl_vega = (
                -pos_idx_vega * d_idx_iv
                + float(np.sum(pos_stk_vega * d_stk_ivs))
            )

            # ── 2. Gamma-Theta PnL（realized var vs implied var） ─
            # 做空指数：实际方差 < 隐含方差时获利（相关性溢价的来源）
            pnl_gamma_idx = -0.5 * pos_idx_gamma * (r_idx ** 2 - idx_iv ** 2 * dt)

            # 做多个股：实际方差 > 隐含方差时获利（个股 IV 校准准确时约为零）
            pnl_gamma_stk = 0.5 * float(
                np.sum(pos_stk_gamma * (r_stk ** 2 - stk_ivs ** 2 * dt))
            )
            pnl_gamma = pnl_gamma_idx + pnl_gamma_stk

        daily_pnl = pnl_vega + pnl_gamma

        # ── 再平衡 ─────────────────────────────────────────────────
        if t % params.rebalance_days == 0:
            v_idx = unit_vega(idx_iv)
            v_stk = np.array([unit_vega(iv) for iv in stk_ivs])
            g_idx = unit_gamma(idx_iv)
            g_stk = np.array([unit_gamma(iv) for iv in stk_ivs])

            if not np.isnan(corr_gap) and corr_gap > 0.05:
                # vega 中性：各股票 vega 合计 = 指数 vega
                pos_idx_vega  = v_idx
                pos_stk_vega  = weights * v_idx          # 各股 vega 分配

                # gamma 按 vega/sigma/T 关系同步确定
                pos_idx_gamma = g_idx
                pos_stk_gamma = weights * v_idx / (stk_ivs * T_option)
            else:
                pos_idx_vega  = pos_idx_gamma = 0.0
                pos_stk_vega  = np.zeros(params.n_stocks)
                pos_stk_gamma = np.zeros(params.n_stocks)

        records.append({
            "day":           t,
            "index_iv":      idx_iv,
            "implied_corr":  ic,
            "realized_corr": rc,
            "corr_gap":      corr_gap,
            "pnl_vega":      pnl_vega,
            "pnl_gamma":     pnl_gamma,
            "daily_pnl":     daily_pnl,
        })

    df = pd.DataFrame(records)
    df["cum_pnl"]     = df["daily_pnl"].cumsum()
    df["cum_pnl_pct"] = df["cum_pnl"]   # 归一化，直接为百分比收益
    return df


# ─────────────────────────────────────────────
# 5. 绩效分析
# ─────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame) -> dict:
    """计算策略关键绩效指标"""
    daily_ret = df["daily_pnl"]
    annual_ret = daily_ret.mean() * 252
    annual_vol = daily_ret.std() * np.sqrt(252)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else np.nan

    cum = df["cum_pnl_pct"]
    rolling_max = cum.cummax()
    max_dd = (cum - rolling_max).min()
    calmar = annual_ret / abs(max_dd) if max_dd != 0 else np.nan

    active_days = (df["corr_gap"] > 0.05).sum()
    vega_contrib  = df["pnl_vega"].sum()
    gamma_contrib = df["pnl_gamma"].sum()
    total         = df["daily_pnl"].sum()

    return {
        "年化收益率":      f"{annual_ret:.2%}",
        "年化波动率":      f"{annual_vol:.2%}",
        "夏普比率":        f"{sharpe:.2f}",
        "Calmar 比率":    f"{calmar:.2f}",
        "最大回撤":        f"{max_dd:.2%}",
        "平均隐含相关性":  f"{df['implied_corr'].mean():.4f}",
        "平均实际相关性":  f"{df['realized_corr'].dropna().mean():.4f}",
        "平均相关性溢价":  f"{df['corr_gap'].dropna().mean():.4f}",
        "持仓天数占比":    f"{active_days / len(df):.1%}",
        "Vega PnL 占比":  f"{vega_contrib / total:.1%}" if total != 0 else "N/A",
        "Gamma PnL 占比": f"{gamma_contrib / total:.1%}" if total != 0 else "N/A",
    }


# ─────────────────────────────────────────────
# 6. 可视化
# ─────────────────────────────────────────────

def plot_results(df: pd.DataFrame, metrics: dict, save_path: str = "correlation_arb_results.png"):
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    fig.suptitle("相关性套利策略回测结果\nCorrelation Arbitrage Backtest", fontsize=14, fontweight="bold")

    x = df["day"]

    # --- 图1: 隐含相关性 vs 实际相关性 ---
    ax1 = axes[0]
    ax1.plot(x, df["implied_corr"], label="隐含相关性 (Implied Corr)", color="#E74C3C", linewidth=1.2)
    ax1.plot(x, df["realized_corr"], label="实际相关性 (Realized Corr, 60d)", color="#2ECC71", linewidth=1.2)
    ax1.fill_between(x, df["realized_corr"], df["implied_corr"],
                     where=df["implied_corr"] > df["realized_corr"],
                     alpha=0.2, color="#E74C3C", label="套利空间（相关性溢价）")
    ax1.set_ylabel("相关性")
    ax1.set_title("隐含相关性 vs 实际相关性")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- 图2: 累计 PnL 总收益 ---
    ax2 = axes[1]
    cum_pct = df["cum_pnl_pct"] * 100
    ax2.plot(x, cum_pct, color="#3498DB", linewidth=1.8, label="总累计收益 (%)")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.fill_between(x, 0, cum_pct, where=cum_pct >= 0, alpha=0.2, color="#2ECC71")
    ax2.fill_between(x, 0, cum_pct, where=cum_pct < 0,  alpha=0.2, color="#E74C3C")
    ax2.set_ylabel("累计收益 (%)")
    ax2.set_title("策略总累计收益")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 标注关键指标
    metrics_text = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
    ax2.text(0.015, 0.97, metrics_text, transform=ax2.transAxes,
             fontsize=7.5, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    # --- 图3: PnL 来源拆分（累计 Vega vs Gamma）---
    ax3 = axes[2]
    cum_vega  = df["pnl_vega"].cumsum() * 100
    cum_gamma = df["pnl_gamma"].cumsum() * 100
    ax3.plot(x, cum_vega,  label="Vega PnL（IV变动）",  color="#E67E22", linewidth=1.2)
    ax3.plot(x, cum_gamma, label="Gamma PnL（方差溢价）", color="#8E44AD", linewidth=1.2)
    ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax3.set_ylabel("累计收益 (%)")
    ax3.set_title("PnL 来源拆分：Vega vs Gamma Carry")
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # --- 图4: 相关性溢价分布 ---
    ax4 = axes[3]
    gap = df["corr_gap"].dropna()
    ax4.hist(gap, bins=50, color="#9B59B6", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax4.axvline(gap.mean(), color="#E74C3C", linewidth=2, linestyle="--",
                label=f"均值 = {gap.mean():.4f}")
    ax4.axvline(0, color="black", linewidth=1)
    ax4.set_xlabel("相关性溢价 (Implied − Realized)")
    ax4.set_ylabel("频数")
    ax4.set_title("相关性溢价分布（套利空间直方图）")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"图表已保存: {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# 7. 单期隐含相关性演示（静态）
# ─────────────────────────────────────────────

def demo_implied_correlation():
    """演示单期隐含相关性计算"""
    print("\n" + "=" * 55)
    print("  隐含相关性计算演示")
    print("=" * 55)

    # SPY-like 参数
    n = 10
    weights = np.array([0.15, 0.12, 0.10, 0.09, 0.09,
                        0.09, 0.08, 0.08, 0.10, 0.10])
    weights /= weights.sum()

    stock_ivs = np.array([0.28, 0.32, 0.25, 0.30, 0.22,
                          0.35, 0.26, 0.29, 0.31, 0.24])

    # 用两个不同的指数 IV（体现溢价）
    for label, idx_iv in [("市场平静期 (低)", 0.168), ("市场压力期 (高)", 0.210)]:
        ic = implied_correlation(idx_iv, stock_ivs, weights)
        print(f"\n{label}")
        print(f"  指数隐含波动率 : {idx_iv:.3f} ({idx_iv*100:.1f}%)")
        print(f"  个股 IV 均值   : {stock_ivs.mean():.3f}")
        print(f"  隐含相关性     : {ic:.4f} ({ic*100:.2f}%)")

    # Vega 中性示例
    print("\n" + "-" * 55)
    print("  Vega 中性组合构建示例")
    print("-" * 55)
    S_idx, S_stk = 4500.0, 150.0
    T, r, sigma_idx = 30 / 252, 0.05, 0.168
    idx_vega = 2 * bs_vega(S_idx, S_idx, T, r, sigma_idx)
    stk_vegas = np.array([2 * bs_vega(S_stk, S_stk, T, r, iv) for iv in stock_ivs])
    _, stock_lots = vega_neutral_weights(idx_vega, stk_vegas, weights)

    print(f"  指数 Vega (做空 1 张 straddle): {idx_vega:.2f}")
    print(f"  {'股票':>6}  {'权重':>6}  {'IV':>6}  {'Vega':>8}  {'做多份数':>10}")
    for i in range(n):
        print(f"  股票{i+1:02d}  {weights[i]:.4f}  {stock_ivs[i]:.3f}  "
              f"{stk_vegas[i]:8.3f}  {stock_lots[i]:10.4f}")
    print(f"\n  组合总 Vega ≈ {-idx_vega + np.sum(stock_lots * stk_vegas):.6f} (目标=0)")


# ─────────────────────────────────────────────
# 8. 敏感性分析
# ─────────────────────────────────────────────

def sensitivity_analysis():
    """分析不同相关性溢价水平对策略收益的影响"""
    print("\n" + "=" * 55)
    print("  敏感性分析：相关性溢价 vs 年化收益")
    print("=" * 55)

    premiums = [0.05, 0.10, 0.15, 0.20, 0.25]
    results = []
    for premium in premiums:
        p = MarketParams(implied_corr_premium=premium, T_days=1260)
        df = run_backtest(p, seed=0)
        m = compute_metrics(df)
        results.append({
            "隐含溢价": f"{premium:.0%}",
            "年化收益": m["年化收益率"],
            "夏普比率": m["夏普比率"],
            "最大回撤": m["最大回撤"],
        })

    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False))
    return result_df


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # 1. 静态演示
    demo_implied_correlation()

    # 2. 回测
    print("\n" + "=" * 55)
    print("  运行回测 (5年, 10只成分股, 再平衡周期=21天)...")
    print("=" * 55)
    params = MarketParams(
        n_stocks=10,
        T_days=1260,
        true_corr=0.25,
        implied_corr_premium=0.15,
        rebalance_days=21,
    )
    df_result = run_backtest(params, seed=42)
    metrics = compute_metrics(df_result)

    print("\n绩效指标:")
    for k, v in metrics.items():
        print(f"  {k:<18}: {v}")

    # 3. 敏感性分析
    sensitivity_analysis()

    # 4. 可视化
    plot_results(df_result, metrics, save_path="reports/correlation_arb_results.png")
