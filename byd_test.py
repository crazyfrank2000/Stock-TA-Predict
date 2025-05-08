import tushare as ts
import pandas as pd
import datetime
from trading_system import StockTradingSystem

def test_byd_today_data():
    """
    测试获取比亚迪(BYD)当天收盘数据
    """
    # 初始化交易系统 - 使用比亚迪股票代码
    byd_code = '002594.SZ'  # 比亚迪A股代码
    token = 'db1723bf9e9009f186c134b6813da7730256c87f31759400e170ddab'  # 替换为您的token
    
    print(f"初始化交易系统，股票代码: {byd_code}")
    trading_system = StockTradingSystem(
        stock_code=byd_code,
        token=token,
        initial_capital=100000
    )
    
    # 获取当天数据
    success = trading_system.get_today_data()
    
    if success and trading_system.data is not None and len(trading_system.data) > 0:
        # 获取最新一条数据
        latest_data = trading_system.data.iloc[-1]
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        
        print("\n====== 比亚迪今日数据 ======")
        print(f"日期: {latest_data.name}")
        print(f"开盘价: {latest_data['open']:.2f}")
        print(f"收盘价: {latest_data['close']:.2f}")
        print(f"最高价: {latest_data['high']:.2f}")
        print(f"最低价: {latest_data['low']:.2f}")
        print(f"成交量: {latest_data['vol']/10000:.2f}万")
        
        # 如果有前一日数据，计算涨跌幅
        if len(trading_system.data) > 1:
            prev_data = trading_system.data.iloc[-2]
            change_pct = (latest_data['close'] / prev_data['close'] - 1) * 100
            print(f"涨跌幅: {change_pct:.2f}%")
            
        # 打印技术指标
        if 'RSI' in latest_data:
            rsi = latest_data['RSI']
            rsi_status = "超买" if rsi > 70 else "超卖" if rsi < 30 else "中性"
            print(f"RSI指标: {rsi:.2f} ({rsi_status})")
        
        # 检查MACD指标
        if all(ind in latest_data for ind in ['MACD', 'MACDsignal', 'MACDhist']):
            macd = latest_data['MACD']
            macd_signal = latest_data['MACDsignal']
            macd_hist = latest_data['MACDhist']
            macd_status = "多头" if macd_hist > 0 else "空头"
            print(f"MACD: {macd:.4f}, 信号线: {macd_signal:.4f}, 柱状线: {macd_hist:.4f} ({macd_status})")
        
        # 检查强K日指标
        if 'strong_K' in latest_data:
            print(f"强K日: {'是' if latest_data['strong_K'] == 1 else '否'}")
        
        print("=============================")
        
        # 获取预测
        prediction = trading_system.predict_next_day()
        if prediction:
            print("\n====== 明日预测 ======")
            print(f"预测方向: {prediction['direction']}")
            print(f"预期幅度: {prediction['expected_change']}")
            print(f"置信度: {prediction['confidence']} ({prediction['probability']})")
            print("=======================")
    else:
        print("未能获取到当天数据")

if __name__ == "__main__":
    test_byd_today_data() 