#!/usr/bin/env python3
"""
统一股票预测系统 - 支持单日和多周期预测
从配置文件读取股票代码，自动检查模型完整性
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import tushare as ts
import ta
from use_trained_model import StockPredictor

def print_single_result(result):
    """格式化输出单日预测结果"""
    if result is None:
        print("无预测结果")
        return
    
    print("\n" + "="*50)
    print(f"股票代码: {result['stock_code']}")
    print(f"股票名称: {result['stock_name']}")
    print(f"最新价格: ¥{result['latest_price']:.2f}")
    print(f"预测方向: {'上涨' if result['prediction'] == 1 else '下跌'}")
    print(f"上涨概率: {result['probability'][1]:.2%}")
    print(f"下跌概率: {result['probability'][0]:.2%}")
    print(f"预测置信度: {result['confidence']:.2%}")
    
    # 置信度评级
    confidence = result['confidence']
    if confidence >= 0.8:
        rating = "很高 ⭐⭐⭐"
    elif confidence >= 0.7:
        rating = "较高 ⭐⭐"
    elif confidence >= 0.6:
        rating = "一般 ⭐"
    else:
        rating = "较低 ⚠️"
    
    print(f"置信度评级: {rating}")
    print("="*50)

def get_recent_data(stock_code, config, days=60):
    """获取最近数据"""
    ts.set_token(config['data_config']['api_token'])
    pro = ts.pro_api()
    
    # 获取最近数据
    df = pro.daily(ts_code=stock_code, limit=days)
    df = df.sort_values('trade_date').reset_index(drop=True)
    return df

def calculate_technical_indicators(df):
    """计算技术指标"""
    # 基础衍生变量
    df['O-C'] = df['open'] - df['close']
    df['H-L'] = df['high'] - df['low']
    
    # 移动平均线
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA10'] = df['close'].rolling(10).mean()
    
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    
    # MOM
    df['MOM'] = ta.momentum.awesome_oscillator(df['high'], df['low'])
    
    # EMA
    df['EMA12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
    df['EMA26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
    
    # MACD
    macd_indicator = ta.trend.MACD(df['close'], window_fast=6, window_slow=12, window_sign=9)
    df['MACD'] = macd_indicator.macd()
    df['MACDsignal'] = macd_indicator.macd_signal()
    df['MACDhist'] = macd_indicator.macd_diff()
    
    # 删除空值
    df = df.dropna()
    return df

def load_model_and_predict(model_path, features_path, latest_data):
    """加载模型并预测"""
    try:
        # 加载模型和特征
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(features_path, 'rb') as f:
            features = pickle.load(f)
        
        # 准备特征向量
        X_pred = []
        for feature in features:
            if feature in latest_data:
                X_pred.append(latest_data[feature])
            else:
                X_pred.append(0)
        
        # 转换为pandas DataFrame以保持特征名称
        X_pred_df = pd.DataFrame([X_pred], columns=features)
        
        # 执行预测
        prediction = model.predict(X_pred_df)[0]
        probability = model.predict_proba(X_pred_df)[0]
        
        return {
            'prediction': prediction,
            'probability': probability,
            'confidence': max(probability)
        }
    except Exception as e:
        return None

def format_prediction(result, period_name):
    """格式化预测结果"""
    if result is None:
        return f"📊 {period_name}: ❌ 模型不可用"
    
    direction = "上涨" if result['prediction'] == 1 else "下跌"
    confidence = result['confidence']
    up_prob = result['probability'][1]
    
    # 置信度评级
    if confidence >= 0.8:
        rating = "⭐⭐⭐"
    elif confidence >= 0.7:
        rating = "⭐⭐"
    elif confidence >= 0.6:
        rating = "⭐"
    else:
        rating = "⚠️"
    
    return f"📈 {period_name}: {direction} ({confidence:.1%}) {rating}"

def multi_period_prediction(config, stock_code, stock_name):
    """执行多周期预测"""
    print(f"\n🔮 多周期预测：{stock_name} ({stock_code})")
    print("=" * 50)
    
    # 获取数据
    print("🔄 获取最新数据...")
    try:
        df = get_recent_data(stock_code, config)
        df = calculate_technical_indicators(df)
        
        if len(df) == 0:
            print("❌ 无法获取有效数据")
            return
        
        latest_data = df.iloc[-1]
        latest_price = latest_data['close']
        trade_date = latest_data['trade_date']
        
        print(f"✅ 最新价格: ¥{latest_price:.2f}")
        print(f"📅 交易日期: {trade_date}")
        print()
        
        # 预测各个周期
        print("🎯 多周期预测结果:")
        print("-" * 50)
        
        model_folder = config['output_config']['model_folder']
        results = {}
        
        # 1天预测（单日模型）
        result_1d = load_model_and_predict(
            f'{model_folder}/model_{stock_code}.pkl',
            f'{model_folder}/features_{stock_code}.pkl',
            latest_data
        )
        print(format_prediction(result_1d, "1天预测 "))
        results['1d'] = result_1d
        
        # 5天预测
        result_5d = load_model_and_predict(
            f'{model_folder}/multi-period/model_5d_{stock_code}.pkl',
            f'{model_folder}/multi-period/features_5d_{stock_code}.pkl',
            latest_data
        )
        print(format_prediction(result_5d, "5天预测 "))
        results['5d'] = result_5d
        
        # 20天预测
        result_20d = load_model_and_predict(
            f'{model_folder}/multi-period/model_20d_{stock_code}.pkl',
            f'{model_folder}/multi-period/features_20d_{stock_code}.pkl',
            latest_data
        )
        print(format_prediction(result_20d, "20天预测"))
        results['20d'] = result_20d
        
        print()
        print("=" * 50)
        
        # 综合分析
        print("💡 综合分析:")
        valid_results = [r for r in results.values() if r is not None]
        if len(valid_results) >= 2:
            predictions = []
            if results['1d']: predictions.append(("短期", results['1d']['prediction']))
            if results['5d']: predictions.append(("中期", results['5d']['prediction']))
            if results['20d']: predictions.append(("长期", results['20d']['prediction']))
            
            ups = sum(1 for _, pred in predictions if pred == 1)
            total = len(predictions)
            
            if ups == total:
                analysis = "全面看涨，趋势强劲 📈"
            elif ups >= total * 0.6:
                analysis = "多数看涨，谨慎乐观 📊"
            elif ups >= total * 0.4:
                analysis = "分歧较大，保持观望 ⚖️"
            else:
                analysis = "多数看跌，规避风险 📉"
                
            print(f"   {analysis}")
            
            # 投资建议
            print("\n📋 投资建议:")
            if ups == total:
                print("   • 可考虑逢低布局")
                print("   • 关注技术面突破")
            elif ups >= total * 0.6:
                print("   • 谨慎参与，控制仓位")
                print("   • 注意风险管理")
            elif ups >= total * 0.4:
                print("   • 暂时观望为主")
                print("   • 等待明确信号")
            else:
                print("   • 避免盲目抄底")
                print("   • 重点关注止损")
        else:
            print("   模型数据不足，建议重新训练")
        
    except Exception as e:
        print(f"❌ 多周期预测失败: {e}")

def check_model_completeness(model_folder, stock_code):
    """检查模型完整性"""
    result = {
        'single_day': False,
        'multi_period': {},
        'missing_models': []
    }
    
    # 检查单日模型 - 新的文件名格式
    single_model_path = f"{model_folder}/model_{stock_code}.pkl"
    if os.path.exists(single_model_path):
        result['single_day'] = True
        # 单日模型同时可用作1天预测模型
        result['multi_period']['1d'] = True
    else:
        result['missing_models'].append('单日预测模型')
        result['multi_period']['1d'] = False
    
    # 检查多周期模型（5天和20天）
    multi_period_folder = f"{model_folder}/multi-period"
    periods = ['5d', '20d']  # 移除1d，因为使用单日模型
    period_names = {'5d': '5天(1周)', '20d': '20天(1月)'}
    
    if os.path.exists(multi_period_folder):
        for period in periods:
            # 多周期模型也使用股票代码
            model_path = f"{multi_period_folder}/model_{period}_{stock_code}.pkl"
            if os.path.exists(model_path):
                result['multi_period'][period] = True
            else:
                result['multi_period'][period] = False
                result['missing_models'].append(f'{period_names[period]}预测模型')
    else:
        for period in periods:
            result['multi_period'][period] = False
            result['missing_models'].append(f'{period_names[period]}预测模型')
    
    return result

def load_config(config_path='config.json'):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return None

def main():
    """统一预测主函数"""
    print("🚀 统一股票预测系统")
    print("=" * 40)
    
    # 加载配置文件
    config = load_config()
    if not config:
        print("❌ 无法加载配置文件，程序退出")
        return
    
    # 从配置文件获取股票信息
    stock_code = config['data_config']['stock_code']
    stock_name = config['data_config']['stock_name']
    model_folder = config['output_config']['model_folder']
    
    print(f"📊 目标股票: {stock_name} ({stock_code})")
    
    # 检查模型完整性
    print("\n🔍 检查模型状态...")
    model_status = check_model_completeness(model_folder, stock_code)
    
    # 显示模型状态
    print(f"📈 单日预测模型: {'✅ 可用' if model_status['single_day'] else '❌ 缺失'}")
    
    # 显示多周期模型状态
    period_names = {'1d': '1天', '5d': '5天(1周)', '20d': '20天(1月)'}
    available_periods = []
    multi_period_available = True
    
    for period, available in model_status['multi_period'].items():
        status = '✅ 可用' if available else '❌ 缺失'
        if period == '1d':
            print(f"📊 {period_names[period]}预测模型: {status} (使用单日模型)")
        else:
            print(f"📊 {period_names[period]}预测模型: {status}")
            if not available:
                multi_period_available = False
        if available:
            available_periods.append(period)
    
    # 如果有缺失的模型，给出提示
    if model_status['missing_models']:
        print(f"\n⚠️ 缺失模型: {', '.join(model_status['missing_models'])}")
        print("💡 训练建议:")
        if not model_status['single_day']:
            print("   - 单日模型: python stock_prediction.py")
        if not multi_period_available:
            print("   - 多周期模型: python train_multi_period.py")
        print()
    
    # 检查是否有可用的模型
    if not model_status['single_day']:
        print("❌ 没有可用的预测模型！")
        print("💡 请先训练模型:")
        print("   python stock_prediction.py        # 训练单日模型")
        print("   python train_multi_period.py      # 训练多周期模型")
        return
    
    # 用户选择预测类型
    print("🎯 预测类型选择:")
    print("1. 单日预测 (明日涨跌)")
    if multi_period_available:
        print("2. 多周期预测 (1天/5天/20天)")
    else:
        print("2. 多周期预测 (❌ 需要先训练多周期模型)")
    
    try:
        if multi_period_available:
            choice = input("\n请选择 (1-2, 默认1): ").strip()
        else:
            choice = "1"
            print("\n默认执行单日预测...")
        
        if choice == "2" and multi_period_available:
            # 多周期预测
            multi_period_prediction(config, stock_code, stock_name)
        else:
            # 单日预测
            print("\n📋 执行单日预测...")
            
            try:
                # 创建预测器
                predictor = StockPredictor()
                
                # 执行预测
                print(f"\n🔮 执行单日预测: {stock_name} ({stock_code})")
                result = predictor.predict_single(stock_code)
                
                if result:
                    print_single_result(result)
                else:
                    print("❌ 预测失败")
                    
            except Exception as e:
                print(f"❌ 预测过程中出错: {e}")
                
    except KeyboardInterrupt:
        print("\n\n👋 用户取消操作")
        return
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
    
    print("\n⚠️ 免责声明: 预测结果仅供参考，不构成投资建议")

if __name__ == "__main__":
    main() 