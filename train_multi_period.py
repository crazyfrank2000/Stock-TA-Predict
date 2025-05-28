#!/usr/bin/env python3
"""
多周期股票预测模型训练系统
训练5天、20天预测模型
"""

import json
import os
import numpy as np
import pandas as pd
import tushare as ts
import ta
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

def load_config(config_path='config.json'):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"配置文件加载成功: {config_path}")
        return config
    except Exception as e:
        print(f"配置文件加载失败: {e}")
        return None

def get_stock_data(config):
    """获取股票数据"""
    data_config = config['data_config']
    
    print(f"正在获取 {data_config['stock_name']}({data_config['stock_code']}) 的数据...")
    print(f"时间范围: {data_config['start_date']} 至 {data_config['end_date']}")
    
    # 设置API token
    ts.set_token(data_config['api_token'])
    pro = ts.pro_api()
    
    # 获取股票数据
    df = pro.daily(
        ts_code=data_config['stock_code'], 
        start_date=data_config['start_date'], 
        end_date=data_config['end_date']
    )
    
    # 数据处理
    df = df.sort_values('trade_date')
    df = df.reset_index(drop=True)
    
    print(f"获取到 {len(df)} 条数据")
    return df

def calculate_technical_indicators(df, config):
    """计算技术指标"""
    features_config = config['features_config']
    
    print("正在计算技术指标...")
    
    # 基础衍生变量
    df['O-C'] = df['open'] - df['close']
    df['H-L'] = df['high'] - df['low']
    
    # 移动平均线
    for period in features_config['ma_periods']:
        df[f'MA{period}'] = df['close'].rolling(period).mean()
    
    # 技术指标
    tech_config = features_config['technical_indicators']
    
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=tech_config['rsi_window']).rsi()
    
    # MOM (使用Awesome Oscillator替代)
    df['MOM'] = ta.momentum.awesome_oscillator(df['high'], df['low'])
    
    # EMA
    for ema_window in tech_config['ema_windows']:
        df[f'EMA{ema_window}'] = ta.trend.EMAIndicator(df['close'], window=ema_window).ema_indicator()
    
    # MACD
    macd_params = tech_config['macd_params']
    macd_indicator = ta.trend.MACD(
        df['close'], 
        window_fast=macd_params['window_fast'], 
        window_slow=macd_params['window_slow'], 
        window_sign=macd_params['window_sign']
    )
    df['MACD'] = macd_indicator.macd()
    df['MACDsignal'] = macd_indicator.macd_signal()
    df['MACDhist'] = macd_indicator.macd_diff()
    
    # 删除空值
    df = df.dropna()
    
    print(f"技术指标计算完成，有效数据点: {len(df)}")
    return df

def create_multi_period_labels(df, period_days, threshold):
    """创建多周期预测标签"""
    print(f"\n创建 {period_days} 天预测标签（阈值: {threshold:.1%}）...")
    
    labels = []
    valid_indices = []
    
    for i in range(len(df) - period_days):
        current_price = df.iloc[i]['close']
        future_price = df.iloc[i + period_days]['close']
        
        # 计算涨跌幅
        price_change = (future_price - current_price) / current_price
        
        # 根据阈值判断涨跌
        label = 1 if price_change >= threshold else 0
        labels.append(label)
        valid_indices.append(i)
    
    print(f"有效样本数: {len(labels)}")
    print(f"上涨样本: {sum(labels)} ({sum(labels)/len(labels):.1%})")
    print(f"下跌样本: {len(labels)-sum(labels)} ({(len(labels)-sum(labels))/len(labels):.1%})")
    
    return np.array(labels), valid_indices

def prepare_features(df, valid_indices, config):
    """准备特征数据"""
    features_config = config['features_config']
    
    # 构建特征列表
    feature_columns = []
    feature_columns.extend(features_config['price_features'])
    
    for period in features_config['ma_periods']:
        feature_columns.append(f'MA{period}')
    
    feature_columns.extend(['RSI', 'MOM'])
    
    for ema_window in features_config['technical_indicators']['ema_windows']:
        feature_columns.append(f'EMA{ema_window}')
    
    feature_columns.extend(['MACD', 'MACDsignal', 'MACDhist'])
    
    print(f"使用特征: {feature_columns}")
    
    # 提取有效样本的特征
    X = df.iloc[valid_indices][feature_columns]
    
    return X, feature_columns

def train_model(X, y, feature_columns, config, period_name):
    """训练模型"""
    model_config = config['model_config']
    
    print(f"\n训练 {period_name} 预测模型...")
    
    # 数据集分割
    test_size = model_config['test_size']
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    
    # 网格搜索优化
    print("进行网格搜索优化...")
    grid_params = model_config['grid_search_params']
    model = RandomForestClassifier(random_state=model_config['random_state'])
    grid_search = GridSearchCV(
        model, 
        grid_params, 
        cv=model_config['cv_folds'], 
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    print(f"最优参数: {best_params}")
    
    # 创建最优模型
    best_model = RandomForestClassifier(
        max_depth=best_params['max_depth'],
        n_estimators=best_params['n_estimators'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=model_config['random_state']
    )
    best_model.fit(X_train, y_train)
    
    # 模型评估
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{period_name} 模型准确率: {accuracy:.4f}")
    
    return {
        'model': best_model,
        'accuracy': accuracy,
        'feature_columns': feature_columns,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }

def save_multi_period_model(result, config, period_days, period_name):
    """保存多周期模型"""
    output_config = config['output_config']
    stock_code = config['data_config']['stock_code']
    
    # 确保多周期模型文件夹存在
    multi_period_folder = f"{output_config['model_folder']}/multi-period"
    os.makedirs(multi_period_folder, exist_ok=True)
    
    # 保存模型
    model_path = f"{multi_period_folder}/model_{period_days}d_{stock_code}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(result['model'], f)
    
    # 保存特征列表
    features_path = f"{multi_period_folder}/features_{period_days}d_{stock_code}.pkl"
    with open(features_path, 'wb') as f:
        pickle.dump(result['feature_columns'], f)
    
    # 保存模型信息
    model_info = {
        'model_type': type(result['model']).__name__,
        'period_days': period_days,
        'period_name': period_name,
        'features': result['feature_columns'],
        'feature_count': len(result['feature_columns']),
        'accuracy': result['accuracy'],
        'model_params': {
            'max_depth': result['model'].max_depth,
            'n_estimators': result['model'].n_estimators,
            'min_samples_leaf': result['model'].min_samples_leaf,
            'random_state': result['model'].random_state
        },
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'stock_code': config['data_config']['stock_code'],
        'stock_name': config['data_config']['stock_name']
    }
    
    model_info_path = f"{multi_period_folder}/model_info_{period_days}d_{stock_code}.json"
    with open(model_info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n{period_name} 模型已保存:")
    print(f"  模型文件: {model_path}")
    print(f"  特征文件: {features_path}")
    print(f"  信息文件: {model_info_path}")

def main():
    """主函数"""
    print("=== 多周期股票预测模型训练系统 ===")
    
    # 加载配置
    config = load_config()
    if not config:
        return
    
    # 获取股票数据
    df = get_stock_data(config)
    
    # 计算技术指标
    df = calculate_technical_indicators(df, config)
    
    # 检查数据量是否足够
    min_days_required = 50  # 至少需要50天数据
    if len(df) < min_days_required:
        print(f"❌ 数据不足！当前: {len(df)} 天，需要至少: {min_days_required} 天")
        return
    
    # 获取多周期配置
    unified_config = config['unified_predictor']
    supported_periods = unified_config['supported_periods']
    period_thresholds = unified_config['period_thresholds']
    
    # 训练每个周期的模型（跳过1天，因为已有单日模型）
    for period in supported_periods:
        if period == 1:
            print(f"\n⏭️ 跳过 {period} 天模型（使用现有单日模型）")
            continue
            
        period_name = f"{period}天"
        if period == 5:
            period_name = "5天(1周)"
        elif period == 20:
            period_name = "20天(1月)"
        
        threshold = period_thresholds[str(period)]
        
        print(f"\n{'='*50}")
        print(f"训练 {period_name} 预测模型")
        print(f"{'='*50}")
        
        # 检查数据是否足够
        if len(df) < period + 30:  # 需要额外30天用于测试
            print(f"❌ 数据不足！{period_name} 模型需要至少 {period + 30} 天数据")
            continue
        
        # 创建标签
        y, valid_indices = create_multi_period_labels(df, period, threshold)
        
        # 准备特征
        X, feature_columns = prepare_features(df, valid_indices, config)
        
        # 训练模型
        result = train_model(X, y, feature_columns, config, period_name)
        
        # 保存模型
        save_multi_period_model(result, config, period, period_name)
    
    print("\n" + "="*50)
    print("✅ 多周期模型训练完成!")
    print("="*50)
    print("\n💡 使用建议:")
    print("- 运行 python predict.py 查看所有模型状态")
    print("- 多周期模型需要在 unified_predictor.py 中集成使用")

if __name__ == "__main__":
    main() 