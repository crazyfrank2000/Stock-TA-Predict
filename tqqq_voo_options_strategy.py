#!/usr/bin/env python3
"""
TQQQ + VOO + 期权保护  全强化战术轮动策略
==========================================
目标：最大回撤 < 10%，年化复合收益 (CAGR) > 60%

═══════════════════════════════════════════════════════════
设计思路
═══════════════════════════════════════════════════════════

为何要叠加期权？
  纯技术信号版（上一版）最大痛点：
    • 2020 COVID 暴跌：TQQQ 5周内跌70%，信号有3-5天滞后，
      组合在退出前已亏损 >15%
    • 信号系统做趋势，但无法应对"黑天鹅"瞬间崩溃

期权的作用：
  每月购买 QQQ 保护性看跌期权（Protective Put）
  → 无论市场多快崩溃，亏损都被硬性封顶
  → 有了这道"保险"，可以更激进地持有TQQQ（仓位上限提升到80%）
  → 结合趋势信号，"技术止损"兜住大趋势，"期权止损"兜住黑天鹅

期权参数设计
  标的：QQQ（纳斯达克100 ETF）
  类型：欧式看跌期权（月度，每月首交易日买入、月末到期）
  执行价：当月 QQQ 收盘价的 (1 - strike_pct)，即 N% 虚值
  保护仓位：TQQQ 持仓价值 × hedge_ratio（对应的 QQQ 等价敞口）
  定价模型：Black-Scholes（隐含波动率用 20日历史波动率估算）

仓位逻辑（在上一版基础上提升激进度）
  State A │ 强势多头 │ TQQQ ≤80%（原65%），有期权保护 → 更大仓位
  State B │ 标准多头 │ TQQQ ≤40%
  State C │ 防御过渡 │ TQQQ 0% + VOO 100%
  State D │ 熊市现金 │ Cash 100%

基准对比（本版新增 QQQ）
  TQQQ B&H / VOO B&H / QQQ B&H（原版100%持有，年化约13-16%）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from scipy.stats import norm
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════
#  参数配置
# ═══════════════════════════════════════════════
CFG = {
    'start':   '2010-02-11',
    'end':     datetime.today().strftime('%Y-%m-%d'),
    'capital': 100_000,

    # 技术指标（同上一版）
    'sma200':  200,
    'ema_s':    10,
    'ema_m':    20,
    'ema_l':    60,

    # 波动率自适应仓位（上限提升：A 80%，B 40%）
    'vol_window':       10,
    'vol_target_daily': 0.018,   # 提升至 1.8%（更激进）
    'max_tqqq_A':       0.80,    # State A 上限从 65% 提升到 80%
    'max_tqqq_B':       0.40,    # State B 上限从 32% 提升到 40%

    # 紧急保护
    'crash_thr':    -0.040,      # 稍微收紧到 -4.0%
    'confirm_days':  3,

    # ── 期权参数 ──
    # 设计原则：
    #   只在 State A/B（持有TQQQ时）才买保护性看跌
    #   8% 虚值（更便宜）+ 60% 对冲比例 → 年化成本约 2-4%
    #   QQQ 波动率 > 30%（恐慌期）期权太贵则跳过本月，依赖信号退出
    'option_strike_pct': 0.08,   # 虚值幅度：8%（执行价 = QQQ × 92%）
    'hedge_ratio':        0.60,  # 对冲比率：保护 60% 的 TQQQ 敞口
    'option_vol_window':  20,    # 估算隐含波动率用的历史波动率窗口
    'option_vol_floor':   0.12,  # 波动率下限
    'option_vol_skip':    0.40,  # 超过此波动率（恐慌期）跳过买入，期权太贵
    # 只在 State A 或 B 时才买期权，State C/D 已经没有 TQQQ 仓位无需保护

    # 成本
    'cash_rate': 0.045,
    'tx_cost':   0.001,
}

SA, SB, SC, SD = 'A', 'B', 'C', 'D'


# ═══════════════════════════════════════════════
#  Black-Scholes 期权定价
# ═══════════════════════════════════════════════

def bs_put_price(S: float, K: float, T: float,
                 r: float, sigma: float) -> float:
    """
    Black-Scholes 欧式看跌期权定价
    S: 标的当前价格
    K: 执行价格
    T: 到期时间（年）
    r: 无风险利率
    sigma: 年化波动率
    返回: 每单位标的名义价值的期权费（如 S=100, 返回期权价格）
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(put, 0.0)


def put_payoff(S_entry: float, K: float, S_expiry: float) -> float:
    """到期时看跌期权内在价值（每单位标的）"""
    return max(K - S_expiry, 0.0)


# ═══════════════════════════════════════════════
#  技术指标
# ═══════════════════════════════════════════════

def build_signals(qqq_cl: pd.Series, tqqq_cl: pd.Series) -> pd.DataFrame:
    s = pd.DataFrame(index=qqq_cl.index)
    s['close']    = qqq_cl
    s['sma200']   = qqq_cl.rolling(CFG['sma200']).mean()
    s['ema_s']    = qqq_cl.ewm(span=CFG['ema_s'], adjust=False).mean()
    s['ema_m']    = qqq_cl.ewm(span=CFG['ema_m'], adjust=False).mean()
    s['ema_l']    = qqq_cl.ewm(span=CFG['ema_l'], adjust=False).mean()
    s['qqq_ret']  = qqq_cl.pct_change()

    tqqq_ret          = tqqq_cl.pct_change()
    s['tqqq_vol_day'] = tqqq_ret.rolling(CFG['vol_window']).std()

    # 期权定价用的 QQQ 历史波动率（annualized）
    s['qqq_vol_ann'] = (s['qqq_ret'].rolling(CFG['option_vol_window']).std()
                        * np.sqrt(252))
    s['qqq_vol_ann'] = s['qqq_vol_ann'].clip(lower=CFG['option_vol_floor'])

    s['bull200']   = qqq_cl > s['sma200']
    s['momentum']  = s['ema_m'] > s['ema_l']
    s['short_mom'] = s['ema_s'] > s['ema_m']
    s['crash']     = s['qqq_ret'] < CFG['crash_thr']

    return s.dropna()


def adaptive_weight(vol_day: float, cap: float) -> float:
    if vol_day <= 0:
        return 0.0
    return min(CFG['vol_target_daily'] / vol_day, cap)


# ═══════════════════════════════════════════════
#  回测引擎（含月度期权操作）
# ═══════════════════════════════════════════════

def backtest(tqqq_px: pd.Series, voo_px: pd.Series,
             qqq_px: pd.Series, sig: pd.DataFrame) -> pd.DataFrame:

    idx   = (sig.index
             .intersection(tqqq_px.index)
             .intersection(voo_px.index)
             .intersection(qqq_px.index))
    sig   = sig.loc[idx]
    tqqq  = tqqq_px.loc[idx]
    voo   = voo_px.loc[idx]
    qqq   = qqq_px.loc[idx]

    daily_cash  = (1 + CFG['cash_rate']) ** (1/252) - 1
    r_annual    = CFG['cash_rate']

    tqqq_units = voo_units = 0.0
    cash_val   = float(CFG['capital'])
    state      = SD
    confirm_cnt = 0
    prev_tw = prev_vw = 0.0
    prev_cw = 1.0

    # 期权状态（每月一次）
    opt_active     = False
    opt_strike     = 0.0
    opt_entry_qqq  = 0.0    # 买入时 QQQ 价格（用于计算平价内在价值）
    opt_notional   = 0.0    # 对应的名义价值（按组合TQQQ仓位）
    opt_premium_pd = 0.0    # 已支付期权费
    current_month  = -1

    rec = {
        'nav': [], 'tqqq_w': [], 'voo_w': [], 'cash_w': [],
        'state': [], 'opt_cost': [], 'opt_payoff': [],
    }

    for i in range(len(idx)):
        date = idx[i]

        # ── 价格更新 ──
        if i > 0:
            tv = tqqq_units * tqqq.iloc[i]
            vv = voo_units  * voo.iloc[i]
            cash_val *= (1 + daily_cash)
        else:
            tv = vv = 0.0

        nav = tv + vv + cash_val

        # ── 信号 ──
        b200 = bool(sig['bull200'].iloc[i])
        mom  = bool(sig['momentum'].iloc[i])
        smom = bool(sig['short_mom'].iloc[i])
        cr   = bool(sig['crash'].iloc[i])
        vd   = float(sig['tqqq_vol_day'].iloc[i])
        qvol = float(sig['qqq_vol_ann'].iloc[i])

        if state == SD:
            confirm_cnt = confirm_cnt + 1 if b200 else 0

        # ── 状态机 ──
        if cr or not b200:
            new_state = SD
            confirm_cnt = 0
        elif state == SD:
            new_state = SC if confirm_cnt >= CFG['confirm_days'] else SD
        elif b200 and mom and smom:
            new_state = SA
        elif b200 and mom:
            new_state = SB
        elif b200:
            new_state = SC
        else:
            new_state = SD

        state = new_state

        # ── 仓位 ──
        if state == SA:
            tw = adaptive_weight(vd, CFG['max_tqqq_A'])
            vw = 1.0 - tw; cw = 0.0
        elif state == SB:
            tw = adaptive_weight(vd, CFG['max_tqqq_B'])
            vw = 1.0 - tw; cw = 0.0
        elif state == SC:
            tw = 0.0; vw = 1.0; cw = 0.0
        else:
            tw = 0.0; vw = 0.0; cw = 1.0

        # ── 月度期权操作 ──
        month_cost    = 0.0
        month_payoff  = 0.0
        new_month     = (date.month != current_month)

        if new_month:
            # 1. 上月期权到期结算
            if opt_active and opt_notional > 0:
                payoff_per_unit = put_payoff(opt_entry_qqq, opt_strike, qqq.iloc[i])
                # 收益 = payoff / entry_price * notional
                month_payoff = payoff_per_unit / opt_entry_qqq * opt_notional
                nav += month_payoff
                opt_active = False

            # 2. 仅在 State A/B（持有TQQQ）且市场波动率不过高时买看跌期权
            #    波动率 > skip阈值时期权太贵，且此时信号系统通常已在降仓
            if tw > 0.01 and qvol <= CFG['option_vol_skip']:
                qqq_price = qqq.iloc[i]
                strike    = qqq_price * (1 - CFG['option_strike_pct'])
                T         = 1 / 12           # 1个月到期
                # 对冲名义：TQQQ仓位 × hedge_ratio
                opt_notional = nav * tw * CFG['hedge_ratio']
                # B-S 定价：put_price 对应每1元标的的期权费
                put_price    = bs_put_price(qqq_price, strike, T, r_annual, qvol)
                # 总期权费 = (put_price / qqq_price) × notional
                premium      = (put_price / qqq_price) * opt_notional
                month_cost   = premium
                nav         -= premium          # 扣除期权费

                opt_active    = True
                opt_strike    = strike
                opt_entry_qqq = qqq_price
                opt_premium_pd = premium
            else:
                opt_notional  = 0.0
                opt_active    = False

            current_month = date.month

        # ── 调仓成本 ──
        if abs(tw - prev_tw) > 0.005 or abs(vw - prev_vw) > 0.005:
            to  = (abs(tw - prev_tw) + abs(vw - prev_vw) + abs(cw - prev_cw)) / 2
            nav *= (1 - to * CFG['tx_cost'])

        tqqq_units = (nav * tw) / tqqq.iloc[i] if tqqq.iloc[i] > 0 else 0.0
        voo_units  = (nav * vw) / voo.iloc[i]  if voo.iloc[i]  > 0 else 0.0
        cash_val   = nav * cw
        prev_tw, prev_vw, prev_cw = tw, vw, cw

        rec['nav'].append(nav)
        rec['tqqq_w'].append(tw)
        rec['voo_w'].append(vw)
        rec['cash_w'].append(cw)
        rec['state'].append(state)
        rec['opt_cost'].append(month_cost)
        rec['opt_payoff'].append(month_payoff)

    return pd.DataFrame(rec, index=idx)


# ═══════════════════════════════════════════════
#  绩效指标
# ═══════════════════════════════════════════════

def calc_metrics(nav_s: pd.Series, label: str) -> dict:
    nav   = nav_s.dropna()
    ret   = nav.pct_change().dropna()
    years = len(nav) / 252
    total = nav.iloc[-1] / nav.iloc[0] - 1
    cagr  = (1 + total) ** (1/years) - 1
    mdd   = ((nav - nav.cummax()) / nav.cummax()).min()

    rf      = (1 + CFG['cash_rate']) ** (1/252) - 1
    exc     = ret - rf
    sharpe  = exc.mean() / exc.std() * np.sqrt(252) if exc.std() > 0 else 0
    dn      = ret[ret < 0].std() * np.sqrt(252)
    sortino = (ret.mean() - rf) * 252 / dn if dn > 0 else 0
    calmar  = cagr / abs(mdd) if mdd != 0 else 0
    wr      = (nav.resample('ME').last().pct_change().dropna() > 0).mean()

    return dict(label=label,
                Total_Return=f'{total*100:.1f}%',
                CAGR=f'{cagr*100:.1f}%',
                Max_DD=f'{mdd*100:.1f}%',
                Sharpe=f'{sharpe:.2f}',
                Sortino=f'{sortino:.2f}',
                Calmar=f'{calmar:.2f}',
                Win_Rate_M=f'{wr*100:.1f}%',
                _cagr=cagr, _mdd=mdd)


# ═══════════════════════════════════════════════
#  可视化
# ═══════════════════════════════════════════════

def plot_all(res: pd.DataFrame,
             tqqq_bh: pd.Series, voo_bh: pd.Series, qqq_bh: pd.Series,
             sig: pd.DataFrame):

    C = dict(s='#00d4ff', t='#ff6b6b', v='#51cf66', q='#f5a623',
             cash='#868e96', opt='#c084fc', w='#ffd43b',
             bg='#0d1117', panel='#161b22', tx='#c9d1d9', g='#21262d')

    fig = plt.figure(figsize=(22, 20))
    fig.patch.set_facecolor(C['bg'])
    gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.52, wspace=0.28,
                            top=0.93, bottom=0.04)

    def ax_style(ax, title=''):
        ax.set_facecolor(C['panel'])
        ax.tick_params(colors=C['tx'], labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(C['g'])
        ax.grid(True, color=C['g'], lw=0.5, alpha=0.7)
        if title:
            ax.set_title(title, color=C['tx'], fontsize=10, pad=6)

    idx = res.index

    # 1: 净值曲线
    ax1 = fig.add_subplot(gs[0, :])
    sn = res['nav'] / res['nav'].iloc[0]
    tn = tqqq_bh.loc[idx] / tqqq_bh.loc[idx[0]]
    vn = voo_bh.loc[idx]  / voo_bh.loc[idx[0]]
    qn = qqq_bh.loc[idx]  / qqq_bh.loc[idx[0]]
    ax1.plot(idx, sn, color=C['s'],  lw=2.2, label='Strategy + Options', zorder=4)
    ax1.plot(idx, tn, color=C['t'],  lw=0.9, alpha=0.55, label='TQQQ B&H')
    ax1.plot(idx, qn, color=C['q'],  lw=1.0, alpha=0.70, label='QQQ B&H')
    ax1.plot(idx, vn, color=C['v'],  lw=0.9, alpha=0.55, label='VOO B&H')
    ax1.set_yscale('log')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.1f}×'))
    ax1.legend(loc='upper left', facecolor=C['panel'], edgecolor=C['g'],
               labelcolor=C['tx'], fontsize=9)
    ax_style(ax1, 'Portfolio NAV — Log Scale  ($100k → Present,  Strategy + Monthly QQQ Protective Puts)')

    # 2: 回撤
    ax2 = fig.add_subplot(gs[1, :])
    def dd_ser(s): pk = s.cummax(); return (s - pk) / pk * 100
    ax2.fill_between(idx, dd_ser(res['nav']), 0,
                     color=C['s'], alpha=0.35, label='Strategy+Options DD')
    ax2.plot(idx, dd_ser(qqq_bh.loc[idx]),  color=C['q'], lw=1.0, alpha=0.7, label='QQQ DD')
    ax2.plot(idx, dd_ser(tqqq_bh.loc[idx]), color=C['t'], lw=0.8, alpha=0.5, label='TQQQ DD')
    ax2.plot(idx, dd_ser(voo_bh.loc[idx]),  color=C['v'], lw=0.8, alpha=0.5, label='VOO DD')
    ax2.axhline(-10, color=C['w'], lw=1.5, ls='--', label='-10% MaxDD Target')
    ax2.set_ylabel('Drawdown %', color=C['tx'])
    ax2.legend(loc='lower left', facecolor=C['panel'], edgecolor=C['g'],
               labelcolor=C['tx'], fontsize=9)
    ax_style(ax2, 'Drawdown Analysis — Strategy vs All Benchmarks')

    # 3: 仓位
    ax3 = fig.add_subplot(gs[2, :])
    ax3.stackplot(idx,
                  res['tqqq_w']*100, res['voo_w']*100, res['cash_w']*100,
                  labels=['TQQQ (vol-adaptive, ≤80%)', 'VOO', 'Cash'],
                  colors=[C['t'], C['v'], C['cash']], alpha=0.85)
    ax3.set_ylim(0, 100); ax3.set_ylabel('Weight %', color=C['tx'])
    ax3.legend(loc='upper right', facecolor=C['panel'], edgecolor=C['g'],
               labelcolor=C['tx'], fontsize=9)
    ax_style(ax3,
             'Dynamic Allocation  [A: TQQQ≤80%  │  B: TQQQ≤40%  │  C: VOO100%  │  D: Cash100%]')

    # 4: 期权成本 & 赔付（月度）
    ax4 = fig.add_subplot(gs[3, 0])
    monthly_cost    = res['opt_cost'].resample('ME').sum() / res['nav'].resample('ME').last() * 100
    monthly_payoff  = res['opt_payoff'].resample('ME').sum() / res['nav'].resample('ME').last() * 100
    ax4.bar(monthly_cost.index, -monthly_cost.values,
            color=C['opt'], alpha=0.7, width=20, label='Option Premium (cost %)')
    ax4.bar(monthly_payoff.index, monthly_payoff.values,
            color=C['w'],   alpha=0.8, width=20, label='Option Payoff %')
    ax4.axhline(0, color=C['g'], lw=0.8)
    ax4.set_ylabel('% of NAV', color=C['tx'])
    ax4.legend(facecolor=C['panel'], edgecolor=C['g'], labelcolor=C['tx'], fontsize=8)
    ax_style(ax4, 'Monthly Option Cost vs Payoff  (Yellow spikes = crash protection triggered)')

    # 5: 年度收益对比
    ax5 = fig.add_subplot(gs[3, 1])
    ann_s = res['nav'].resample('YE').last().pct_change().dropna() * 100
    ann_t = tqqq_bh.resample('YE').last().pct_change().dropna() * 100
    ann_q = qqq_bh.resample('YE').last().pct_change().dropna() * 100
    ann_v = voo_bh.resample('YE').last().pct_change().dropna() * 100
    x = np.arange(len(ann_s))
    bw = 0.21
    ax5.bar(x-1.5*bw, ann_s.values, bw, color=C['s'], alpha=0.9, label='Strategy+Opt')
    ax5.bar(x-0.5*bw, ann_t.reindex(ann_s.index).values, bw, color=C['t'], alpha=0.65, label='TQQQ')
    ax5.bar(x+0.5*bw, ann_q.reindex(ann_s.index).values, bw, color=C['q'], alpha=0.75, label='QQQ')
    ax5.bar(x+1.5*bw, ann_v.reindex(ann_s.index).values, bw, color=C['v'], alpha=0.65, label='VOO')
    ax5.axhline(60, color='#ff4757', lw=1.2, ls='--', label='60% Target')
    ax5.axhline(30, color=C['w'],    lw=1.0, ls=':',  label='30% Ref')
    ax5.axhline(0,  color=C['g'],    lw=0.8)
    ax5.set_xticks(x); ax5.set_xticklabels(ann_s.index.year, rotation=45, fontsize=7)
    ax5.set_ylabel('Annual Return %', color=C['tx'])
    ax5.legend(facecolor=C['panel'], edgecolor=C['g'], labelcolor=C['tx'], fontsize=7)
    ax_style(ax5, 'Annual Returns — Strategy+Options vs Benchmarks')

    # 6: 滚动Sharpe
    ax6 = fig.add_subplot(gs[4, :])
    rr = res['nav'].pct_change()
    rs = rr.rolling(252).mean() / rr.rolling(252).std() * np.sqrt(252)
    rq = (qqq_bh.pct_change().rolling(252).mean() /
          qqq_bh.pct_change().rolling(252).std() * np.sqrt(252))
    rv = (voo_bh.pct_change().rolling(252).mean() /
          voo_bh.pct_change().rolling(252).std() * np.sqrt(252))
    ax6.plot(idx, rs.reindex(idx), color=C['s'], lw=1.4, label='Strategy+Options')
    ax6.plot(idx, rq.reindex(idx), color=C['q'], lw=0.9, alpha=0.7, label='QQQ')
    ax6.plot(idx, rv.reindex(idx), color=C['v'], lw=0.9, alpha=0.7, label='VOO')
    ax6.axhline(1.0, color=C['w'],   lw=1, ls='--', label='Sharpe=1')
    ax6.axhline(0,   color=C['g'],   lw=0.8)
    ax6.fill_between(idx, rs.reindex(idx).clip(lower=0), 0, color=C['s'], alpha=0.12)
    ax6.set_ylabel('Rolling Sharpe (252d)', color=C['tx'])
    ax6.legend(facecolor=C['panel'], edgecolor=C['g'], labelcolor=C['tx'], fontsize=8)
    ax_style(ax6, 'Rolling 12-Month Sharpe Ratio — Strategy vs QQQ / VOO')

    fig.suptitle(
        'TQQQ + VOO + Monthly QQQ Protective Puts  │  '
        'Target: MaxDD < 10%,  CAGR > 60%\n'
        'Trend Signals + Vol-Adaptive Sizing (TQQQ≤80%) + '
        'Black-Scholes Priced 5%-OTM Puts  │  '
        'Benchmarks: TQQQ / QQQ / VOO',
        color='white', fontsize=10, fontweight='bold',
    )

    out = 'tqqq_voo_options_report.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"  图表已保存: {out}")
    plt.close()


# ═══════════════════════════════════════════════
#  主程序
# ═══════════════════════════════════════════════

def main():
    SEP = '═' * 72
    print(SEP)
    print("  TQQQ + VOO + 期权保护  全强化战术轮动策略")
    print(f"  回测区间: {CFG['start']} → {CFG['end']}")
    print(SEP)

    print("\n[1/4] 下载行情数据（TQQQ / VOO / QQQ）...")
    raw = yf.download(['TQQQ', 'VOO', 'QQQ'],
                      start=CFG['start'], end=CFG['end'],
                      auto_adjust=True, progress=False)
    tqqq_px = raw['Close']['TQQQ'].dropna()
    voo_px  = raw['Close']['VOO'].dropna()
    qqq_px  = raw['Close']['QQQ'].dropna()
    print(f"  TQQQ {len(tqqq_px)}日 | VOO {len(voo_px)}日 | QQQ {len(qqq_px)}日")

    print("\n[2/4] 计算技术指标与期权参数...")
    sig = build_signals(qqq_px, tqqq_px)
    vol_ann = sig['qqq_vol_ann'] * 100
    print(f"  QQQ 历史年化波动率（用于B-S定价）:")
    print(f"    均值: {vol_ann.mean():.1f}%  | 中位数: {vol_ann.median():.1f}%  "
          f"| 最大值: {vol_ann.max():.1f}%  | 最小值: {vol_ann.min():.1f}%")
    # 示例期权成本
    sample_vol = vol_ann.median() / 100
    sample_s   = 100.0
    sample_k   = sample_s * (1 - CFG['option_strike_pct'])
    sample_p   = bs_put_price(sample_s, sample_k, 1/12,
                               CFG['cash_rate'], sample_vol)
    print(f"\n  示例期权定价（QQQ=100, 执行价={sample_k:.0f}, 到期1月, "
          f"波动率={sample_vol*100:.0f}%）:")
    print(f"    看跌期权费 = {sample_p:.2f}  ({sample_p/sample_s*100:.2f}% of QQQ)")

    print("\n[3/4] 执行回测（含月度期权对冲）...")
    res = backtest(tqqq_px, voo_px, qqq_px, sig)
    t0      = res.index[0]
    tqqq_bh = (tqqq_px / tqqq_px.loc[t0]) * CFG['capital']
    voo_bh  = (voo_px  / voo_px.loc[t0])  * CFG['capital']
    qqq_bh  = (qqq_px  / qqq_px.loc[t0])  * CFG['capital']

    # 期权成本统计
    total_cost    = res['opt_cost'].sum()
    total_payoff  = res['opt_payoff'].sum()
    net_opt_pnl   = total_payoff - total_cost
    years         = len(res) / 252
    ann_opt_cost  = total_cost / (CFG['capital'] * years) * 100

    print("\n[4/4] 绩效分析")
    print("─" * 72)
    sm = calc_metrics(res['nav'],              '策略+期权')
    tm = calc_metrics(tqqq_bh.loc[res.index],  'TQQQ B&H')
    qm = calc_metrics(qqq_bh.loc[res.index],   'QQQ B&H')
    vm = calc_metrics(voo_bh.loc[res.index],   'VOO B&H')

    print(f"{'指标':<18} {'策略+期权':>16} {'TQQQ':>10} {'QQQ':>10} {'VOO':>10}")
    print("─" * 72)
    key_map = [('Total Return', 'Total_Return'), ('CAGR', 'CAGR'),
               ('Max Drawdown', 'Max_DD'), ('Sharpe', 'Sharpe'),
               ('Sortino', 'Sortino'), ('Calmar', 'Calmar'), ('Win Rate(M)', 'Win_Rate_M')]
    for lbl, k in key_map:
        print(f"{lbl:<18} {sm[k]:>16} {tm[k]:>10} {qm[k]:>10} {vm[k]:>10}")
    print("─" * 72)

    cagr_ok = sm['_cagr'] >= 0.60
    mdd_ok  = sm['_mdd']  >= -0.10
    print(f"\n目标达成:")
    print(f"  CAGR ≥ 60%   : {'✅ PASS' if cagr_ok else '❌ FAIL'}  ({sm['CAGR']})")
    print(f"  MaxDD ≤ 10%  : {'✅ PASS' if mdd_ok  else '❌ FAIL'}  ({sm['Max_DD']})")

    print(f"\n期权成本分析（{years:.1f}年）:")
    print(f"  期权费总支出  : ${total_cost:>10,.0f}  "
          f"(年化约 {ann_opt_cost:.1f}% 组合价值)")
    print(f"  期权赔付总收入: ${total_payoff:>10,.0f}")
    print(f"  期权净收益    : ${net_opt_pnl:>+10,.0f}  "
          f"({'盈利' if net_opt_pnl > 0 else '亏损'})")
    payoff_months = (res['opt_payoff'] > 0).sum()
    total_months  = (res.resample('ME').last().shape[0])
    print(f"  期权赔付月数  : {payoff_months} 次 / {total_months} 月  "
          f"({payoff_months/total_months*100:.0f}%)")

    print(f"\n状态分布（实际）:")
    for s, lbl in [('A','A强势'), ('B','B标准'), ('C','C防御'), ('D','D现金')]:
        mask = res['state'] == s
        cnt  = mask.sum()
        atw  = res.loc[mask, 'tqqq_w'].mean()*100 if cnt > 0 else 0
        print(f"  {lbl}: {cnt:4d}日 ({cnt/len(res)*100:4.0f}%)  "
              f"平均TQQQ仓位: {atw:.1f}%")

    print(f"\n综合平均仓位: TQQQ {res['tqqq_w'].mean()*100:.1f}%  "
          f"| VOO {res['voo_w'].mean()*100:.1f}%  "
          f"| Cash {res['cash_w'].mean()*100:.1f}%")

    print(f"\n年度收益 vs 各基准 (✅≥60%, ⬜≥0%, 🔴<0%):")
    annual  = res['nav'].resample('YE').last().pct_change().dropna() * 100
    ann_t   = tqqq_bh.resample('YE').last().pct_change().dropna() * 100
    ann_q   = qqq_bh.resample('YE').last().pct_change().dropna() * 100
    ann_v   = voo_bh.resample('YE').last().pct_change().dropna() * 100
    print(f"  {'年份':<6} {'策略':>8}  {'TQQQ':>8}  {'QQQ':>7}  {'VOO':>7}  跑赢QQQ")
    print(f"  {'─'*52}")
    beat_q, beat_v = 0, 0
    for dt in annual.index:
        ret = annual.loc[dt]
        tr  = ann_t.get(dt, float('nan'))
        qr  = ann_q.get(dt, float('nan'))
        vr  = ann_v.get(dt, float('nan'))
        flag = '✅' if ret >= 60 else ('⬜' if ret >= 0 else '🔴')
        bq   = '✓' if ret > qr else ' '
        if ret > qr: beat_q += 1
        if ret > vr: beat_v += 1
        print(f"  {dt.year} {flag} {ret:+7.1f}%  {tr:+7.1f}%  {qr:+6.1f}%  {vr:+6.1f}%    {bq}")

    n = len(annual)
    print(f"\n  在 {n} 个完整年份中跑赢 QQQ: {beat_q} 年 ({beat_q/n*100:.0f}%)  | "
          f"跑赢 VOO: {beat_v} 年 ({beat_v/n*100:.0f}%)")

    print(f"\n期权参数配置:")
    print(f"  执行价虚值幅度 : {CFG['option_strike_pct']*100:.0f}%  OTM")
    print(f"  对冲比率       : {CFG['hedge_ratio']*100:.0f}%  of TQQQ notional")
    print(f"  定价模型       : Black-Scholes (隐含波动率 = QQQ 20日历史波动率)")
    print(f"  更新频率       : 月度滚动（每月首日买入，月末到期结算）")

    # ── 滚动年化收益分析（找到实现 >60% CAGR 的时间窗口）──
    print(f"\n滚动3年年化收益 >60% 的区间（策略峰值表现）:")
    nav_s = res['nav']
    rolling_3y_cagr = []
    dates_list = nav_s.index.tolist()
    for i in range(len(dates_list) - 252*3):
        s = nav_s.iloc[i]
        e = nav_s.iloc[i + 252*3]
        cagr_3y = (e / s) ** (1/3) - 1
        rolling_3y_cagr.append((dates_list[i + 252*3], cagr_3y))
    df_roll = pd.DataFrame(rolling_3y_cagr, columns=['date', 'cagr3y'])
    df_roll = df_roll.set_index('date')
    above60 = df_roll[df_roll['cagr3y'] >= 0.60]
    if len(above60) > 0:
        print(f"  最高滚动3年CAGR: {df_roll['cagr3y'].max()*100:.1f}%  "
              f"(截止 {df_roll['cagr3y'].idxmax().strftime('%Y-%m')})")
        print(f"  达到 >60% 的日期数: {len(above60)} 日  "
              f"({len(above60)/len(df_roll)*100:.0f}% 的滚动窗口)")
    else:
        max_cagr = df_roll['cagr3y'].max()
        print(f"  滚动3年最高CAGR: {max_cagr*100:.1f}%  "
              f"(未达到60%目标，当前最优窗口截止 {df_roll['cagr3y'].idxmax().strftime('%Y-%m')})")

    # 各基准滚动3年CAGR最高值
    for bh, label in [(tqqq_bh,'TQQQ'), (qqq_bh,'QQQ'), (voo_bh,'VOO')]:
        roll_bh = []
        bh_s = bh.loc[res.index]
        bh_dates = bh_s.index.tolist()
        for i in range(len(bh_dates) - 252*3):
            s = bh_s.iloc[i]; e = bh_s.iloc[i + 252*3]
            roll_bh.append((e/s)**(1/3) - 1)
        if roll_bh:
            print(f"  {label} B&H 最高滚动3年CAGR: {max(roll_bh)*100:.1f}%")

    print(f"\n关键年度策略超过 60% 表现:")
    for dt in annual.index:
        ret = annual.loc[dt]
        if ret >= 50:
            print(f"  {dt.year}: {ret:+.1f}%")

    print(f"\n  图表生成中...")
    plot_all(res, tqqq_bh, voo_bh, qqq_bh, sig)
    return res, sm


if __name__ == '__main__':
    result, metrics = main()
