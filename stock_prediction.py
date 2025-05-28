import json
import os
from csv import field_size_limit
from os import closerange
from cmath import sqrt
import tushare as ts
import numpy as np
import pandas as pd
# import talib  # 注释掉talib导入
import ta  # 使用ta库替代
from pandas import DataFrame as DF
from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as pre
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from matplotlib.font_manager import FontProperties
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import classification_report
import pickle
from datetime import datetime

def load_config(config_path='config.json'):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"配置文件加载成功: {config_path}")
        return config
    except FileNotFoundError:
        print(f"配置文件未找到: {config_path}")
        print("使用默认配置...")
        return get_default_config()
    except json.JSONDecodeError as e:
        print(f"配置文件格式错误: {e}")
        print("使用默认配置...")
        return get_default_config()

def get_default_config():
    """获取默认配置"""
    return {
        "data_config": {
            "stock_code": "002594.SZ",
            "stock_name": "比亚迪",
            "start_date": "20160101",
            "end_date": "20230630",
            "api_token": "db1723bf9e9009f186c134b6813da7730256c87f31759400e170ddab"
        },
        "model_config": {
            "test_size": 0.1,
            "random_state": 120,
            "threshold": 0.0025,
            "initial_params": {
                "max_depth": 3,
                "n_estimators": 10,
                "min_samples_leaf": 10
            },
            "grid_search_params": {
                "n_estimators": [5, 10, 20],
                "max_depth": [2, 3, 4, 5, 6],
                "min_samples_leaf": [5, 10, 20, 30]
            },
            "cv_folds": 6
        },
        "features_config": {
            "price_features": ["close", "vol", "O-C", "H-L"],
            "ma_periods": [5, 10],
            "technical_indicators": {
                "rsi_window": 14,
                "ema_windows": [12, 26],
                "macd_params": {
                    "window_fast": 6,
                    "window_slow": 12,
                    "window_sign": 9
                }
            }
        },
        "output_config": {
            "data_folder": "data",
            "reports_folder": "reports",
            "save_plots": True,
            "plot_size": {
                "feature_importance": [10, 6],
                "confusion_matrix": [8, 8],
                "roc_curve": [8, 8]
            }
        }
    }

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
    df1 = df.set_index('trade_date')
    
    # 保存原始数据
    output_config = config['output_config']
    os.makedirs(output_config['data_folder'], exist_ok=True)
    data_file = f"{output_config['data_folder']}/{data_config['stock_code']}_daily.csv"
    df1.to_csv(data_file)
    print(f"原始数据已保存到: {data_file}")
    
    return df, df1

def calculate_features(df, df1, config):
    """计算技术指标和特征"""
    features_config = config['features_config']
    model_config = config['model_config']
    
    print("正在计算技术指标...")
    
    # 计算对数收益率
    df['log_return'] = np.log(df['close'] / df['pre_close'])
    df['up'] = np.where(df.log_return >= model_config['threshold'], 1, 0)
    df = df.sort_values('trade_date')
    df1 = df.set_index('trade_date')

    # 基础衍生变量
    df1['O-C'] = df1['open'] - df1['close']
    df1['H-L'] = df1['high'] - df1['low']
    df1['pre_close'] = df1['close'].shift(1)
    df1['price_change'] = df1['close'] - df1['pre_close']
    df1['p_change'] = (df1['close'] - df1['pre_close']) / df1['pre_close'] * 100

    # 移动平均线
    for period in features_config['ma_periods']:
        df1[f'MA{period}'] = df1['close'].rolling(period).mean()
    
    df1.dropna(inplace=True)

    # 技术指标
    tech_config = features_config['technical_indicators']
    
    # RSI
    df1['RSI'] = ta.momentum.RSIIndicator(df1['close'], window=tech_config['rsi_window']).rsi()
    
    # MOM (使用Awesome Oscillator替代)
    df1['MOM'] = ta.momentum.awesome_oscillator(df1['high'], df1['low'])
    
    # EMA
    for ema_window in tech_config['ema_windows']:
        df1[f'EMA{ema_window}'] = ta.trend.EMAIndicator(df1['close'], window=ema_window).ema_indicator()
    
    # MACD
    macd_params = tech_config['macd_params']
    macd_indicator = ta.trend.MACD(
        df1['close'], 
        window_fast=macd_params['window_fast'], 
        window_slow=macd_params['window_slow'], 
        window_sign=macd_params['window_sign']
    )
    df1['MACD'] = macd_indicator.macd()
    df1['MACDsignal'] = macd_indicator.macd_signal()
    df1['MACDhist'] = macd_indicator.macd_diff()
    
    df1.dropna(inplace=True)
    
    print(f"技术指标计算完成，有效数据点: {len(df1)}")
    return df1

def prepare_features_and_target(df1, config):
    """准备特征和目标变量"""
    features_config = config['features_config']
    model_config = config['model_config']
    
    # 动态构建特征列表
    feature_columns = []
    
    # 添加价格特征
    feature_columns.extend(features_config['price_features'])
    
    # 添加移动平均特征
    for period in features_config['ma_periods']:
        feature_columns.append(f'MA{period}')
    
    # 添加技术指标特征
    feature_columns.append('RSI')
    feature_columns.append('MOM')
    
    # 添加EMA特征
    for ema_window in features_config['technical_indicators']['ema_windows']:
        feature_columns.append(f'EMA{ema_window}')
    
    # 添加MACD特征
    feature_columns.extend(['MACD', 'MACDsignal', 'MACDhist'])
    
    print(f"使用特征: {feature_columns}")
    
    # 准备特征矩阵和目标变量
    X = df1[feature_columns]
    y = np.where(df1.log_return >= model_config['threshold'], 1, 0)
    
    return X, y, feature_columns

def train_and_evaluate_model(X, y, feature_columns, config):
    """训练和评估模型"""
    model_config = config['model_config']
    
    print("正在训练模型...")
    
    # 数据集分割
    test_size = model_config['test_size']
    X_length = X.shape[0]
    split = int(X_length * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    
    # 初始模型训练
    initial_params = model_config['initial_params']
    model = RandomForestClassifier(
        max_depth=initial_params['max_depth'],
        n_estimators=initial_params['n_estimators'],
        min_samples_leaf=initial_params['min_samples_leaf'],
        random_state=model_config['random_state']
    )
    model.fit(X_train, y_train)
    
    # 初始模型评估
    y_pred = model.predict(X_test)
    initial_accuracy = accuracy_score(y_pred, y_test)
    print(f"初始模型准确率: {initial_accuracy:.4f}")
    
    # 特征重要性分析
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "features": feature_columns,
        "importance": importances
    }).sort_values("importance", ascending=False)
    
    print("\n特征重要性排序:")
    print(feature_importance_df)
    
    # 网格搜索优化
    print("\n正在进行网格搜索优化...")
    grid_params = model_config['grid_search_params']
    new_model = RandomForestClassifier(random_state=model_config['random_state'])
    grid_search = GridSearchCV(
        new_model, 
        grid_params, 
        cv=model_config['cv_folds'], 
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    print(f"最优参数: {best_params}")
    
    # 创建优化模型
    optimized_model = RandomForestClassifier(
        max_depth=best_params['max_depth'],
        n_estimators=best_params['n_estimators'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=model_config['random_state']
    )
    optimized_model.fit(X_train, y_train)
    
    # 优化模型评估
    optimized_y_pred = optimized_model.predict(X_test)
    optimized_accuracy = accuracy_score(optimized_y_pred, y_test)
    print(f"优化后模型准确率: {optimized_accuracy:.4f}")
    
    return {
        'model': model,
        'optimized_model': optimized_model,
        'feature_importance_df': feature_importance_df,
        'X_test': X_test,
        'y_test': y_test,
        'optimized_y_pred': optimized_y_pred,
        'feature_columns': feature_columns
    }

def save_visualizations(results, config):
    """保存可视化图表"""
    output_config = config['output_config']
    stock_code = config['data_config']['stock_code']
    
    if not output_config['save_plots']:
        print("跳过图表保存")
        return
    
    print("正在生成可视化图表...")
    
    # 确保输出文件夹存在
    os.makedirs(output_config['reports_folder'], exist_ok=True)
    
    # 1. 特征重要性图 - 添加股票代码到文件名
    plt.figure(figsize=output_config['plot_size']['feature_importance'])
    importances = results['feature_importance_df']['importance'].values
    sorted_idx = np.argsort(importances)
    feature_names = results['feature_importance_df']['features'].values
    
    plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
    plt.title(f'Feature Importance - {stock_code}')
    plt.tight_layout()
    plt.savefig(f"{output_config['reports_folder']}/feature_importance_{stock_code}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 混淆矩阵 - 添加股票代码到文件名
    plt.figure(figsize=output_config['plot_size']['confusion_matrix'])
    cm = confusion_matrix(results['y_test'], results['optimized_y_pred'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix - {stock_code}')
    plt.savefig(f"{output_config['reports_folder']}/confusion_matrix_{stock_code}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ROC曲线 - 添加股票代码到文件名
    plt.figure(figsize=output_config['plot_size']['roc_curve'])
    y_score = results['optimized_model'].predict_proba(results['X_test'])[:, 1]
    fpr, tpr, _ = roc_curve(results['y_test'], y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) - {stock_code}')
    plt.legend(loc="lower right")
    plt.savefig(f"{output_config['reports_folder']}/roc_curve_{stock_code}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化图表已保存到 {output_config['reports_folder']} 文件夹")
    print(f"文件包含股票代码: {stock_code}")

def save_model_and_features(results, config):
    """保存模型和特征"""
    output_config = config['output_config']
    stock_code = config['data_config']['stock_code']
    
    print("正在保存模型和特征...")
    
    # 确保输出文件夹存在 - 使用model_folder而不是reports_folder
    model_folder = output_config.get('model_folder', 'rf-model')  # 默认为rf-model
    os.makedirs(model_folder, exist_ok=True)
    
    # 保存优化后的模型 - 添加股票代码到文件名
    model_path = f"{model_folder}/model_{stock_code}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(results['optimized_model'], f)
    
    # 保存特征列表 - 添加股票代码到文件名
    features_path = f"{model_folder}/features_{stock_code}.pkl"
    with open(features_path, 'wb') as f:
        pickle.dump(results['feature_columns'], f)
    
    # 保存模型信息 - 添加股票代码到文件名
    model_info = {
        'model_type': type(results['optimized_model']).__name__,
        'features': results['feature_columns'],
        'feature_count': len(results['feature_columns']),
        'model_params': {
            'max_depth': results['optimized_model'].max_depth,
            'n_estimators': results['optimized_model'].n_estimators,
            'min_samples_leaf': results['optimized_model'].min_samples_leaf,
            'random_state': results['optimized_model'].random_state
        },
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'stock_code': config['data_config']['stock_code'],
        'stock_name': config['data_config']['stock_name']
    }
    
    model_info_path = f"{model_folder}/model_info_{stock_code}.json"
    with open(model_info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    print(f"模型已保存到: {model_path}")
    print(f"特征列表已保存到: {features_path}")
    print(f"模型信息已保存到: {model_info_path}")

def main():
    """主函数"""
    print("=== 股票预测模型训练系统 ===")
    
    # 加载配置
    config = load_config()
    
    # 获取股票数据
    df, df1 = get_stock_data(config)
    
    # 计算技术指标
    df1 = calculate_features(df, df1, config)
    
    # 准备特征和目标变量
    X, y, feature_columns = prepare_features_and_target(df1, config)
    
    # 训练和评估模型
    results = train_and_evaluate_model(X, y, feature_columns, config)
    
    # 保存可视化图表
    save_visualizations(results, config)
    
    # 保存模型和特征
    save_model_and_features(results, config)
    
    print("\n=== 训练完成 ===")
    print("请查看 reports 文件夹中的结果文件")

if __name__ == "__main__":
    main() 