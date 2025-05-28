import json
import os
import pickle
import numpy as np
import pandas as pd
import tushare as ts
import ta
from datetime import datetime, timedelta

class StockPredictor:
    def __init__(self, stock_code=None, model_folder='rf-model', config_path='config.json'):
        """
        初始化股票预测器
        
        Args:
            stock_code: 股票代码，如果为None则使用配置文件中的代码
            model_folder: 模型文件夹路径
            config_path: 配置文件路径
        """
        self.model_folder = model_folder
        self.config_path = config_path
        
        # 加载配置
        self.config = self.load_config()
        
        # 确定股票代码
        if stock_code is None:
            if self.config and 'data_config' in self.config:
                self.stock_code = self.config['data_config']['stock_code']
            else:
                print("配置文件未加载或缺少股票代码，请手动提供")
                return
        else:
            self.stock_code = stock_code
        
        # 构建模型文件路径
        self.model_path = f"{model_folder}/model_{self.stock_code}.pkl"
        self.features_path = f"{model_folder}/features_{self.stock_code}.pkl"
        
        # 加载模型和特征
        self.model = self.load_model()
        self.features = self.load_features()
        
        # 设置tushare
        if self.config and 'data_config' in self.config:
            ts.set_token(self.config['data_config']['api_token'])
            self.pro = ts.pro_api()
            
            if self.features is not None:
                print(f"模型加载成功，特征数量: {len(self.features)}")
                print(f"使用特征: {self.features}")
        else:
            print("配置文件加载失败，无法设置API")
    
    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"配置文件加载失败: {e}")
            return None
    
    def load_model(self):
        """加载训练好的模型"""
        try:
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"模型加载成功: {self.model_path}")
            return model
        except Exception as e:
            print(f"模型加载失败: {e}")
            return None
    
    def load_features(self):
        """加载特征列表"""
        try:
            with open(self.features_path, 'rb') as f:
                features = pickle.load(f)
            print(f"特征列表加载成功: {self.features_path}")
            return features
        except Exception as e:
            print(f"特征列表加载失败: {e}")
            return None
    
    def get_recent_data(self, stock_code, days=60):
        """
        获取最近的股票数据
        
        Args:
            stock_code: 股票代码
            days: 获取最近几天的数据
        """
        try:
            # 计算开始日期
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
            
            print(f"正在获取 {stock_code} 最近 {days} 天的数据...")
            
            # 获取股票数据
            df = self.pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)
            df = df.sort_values('trade_date')
            df = df.reset_index(drop=True)
            
            if len(df) == 0:
                print("未获取到数据")
                return None
                
            print(f"获取到 {len(df)} 条数据")
            return df
            
        except Exception as e:
            print(f"数据获取失败: {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """
        计算技术指标
        
        Args:
            df: 股票数据DataFrame
        """
        try:
            if self.config is None:
                print("配置文件未加载，使用默认参数")
                # 使用默认参数
                features_config = {
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
                }
            else:
                features_config = self.config['features_config']
            
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
            
        except Exception as e:
            print(f"技术指标计算失败: {e}")
            return None
    
    def predict_single(self, stock_code=None):
        """
        对单只股票进行预测
        
        Args:
            stock_code: 股票代码，如果为None则使用配置文件中的代码
        """
        if self.model is None or self.features is None:
            print("模型或特征未正确加载")
            return None
        
        # 确定股票代码
        if stock_code is None:
            if self.config is not None:
                stock_code = self.config['data_config']['stock_code']
                stock_name = self.config['data_config']['stock_name']
            else:
                print("请提供股票代码")
                return None
        else:
            stock_name = stock_code
        
        print(f"\n=== 预测 {stock_name}({stock_code}) ===")
        
        # 获取最近数据
        df = self.get_recent_data(stock_code)
        if df is None:
            return None
        
        # 计算技术指标
        df = self.calculate_technical_indicators(df)
        if df is None or len(df) == 0:
            print("技术指标计算失败")
            return None
        
        # 准备预测特征
        try:
            # 获取最新的特征数据
            latest_data = df.iloc[-1]
            X_pred = []
            
            for feature in self.features:
                if feature in latest_data:
                    X_pred.append(latest_data[feature])
                else:
                    print(f"警告: 特征 {feature} 不存在，使用0填充")
                    X_pred.append(0)
            
            # 转换为pandas DataFrame以保持特征名称
            X_pred_df = pd.DataFrame([X_pred], columns=self.features)
            
            # 进行预测
            prediction = self.model.predict(X_pred_df)[0]
            probability = self.model.predict_proba(X_pred_df)[0]
            
            # 输出结果
            print(f"\n预测结果:")
            print(f"最新收盘价: {latest_data['close']:.2f}")
            print(f"预测明日: {'上涨' if prediction == 1 else '下跌'}")
            print(f"上涨概率: {probability[1]:.2%}")
            print(f"下跌概率: {probability[0]:.2%}")
            print(f"置信度: {max(probability):.2%}")
            
            # 输出一些关键特征值
            print(f"\n关键特征值:")
            key_features = ['close', 'vol', 'RSI', 'MACD', 'O-C']
            for feature in key_features:
                if feature in latest_data:
                    print(f"{feature}: {latest_data[feature]:.4f}")
            
            return {
                'stock_code': stock_code,
                'stock_name': stock_name,
                'prediction': int(prediction),
                'probability': probability.tolist(),
                'confidence': float(max(probability)),
                'latest_price': float(latest_data['close']),
                'features': {f: float(latest_data[f]) for f in key_features if f in latest_data}
            }
            
        except Exception as e:
            print(f"预测失败: {e}")
            return None
    
    def predict_multiple(self, stock_codes):
        """
        对多只股票进行预测
        
        Args:
            stock_codes: 股票代码列表
        """
        results = []
        for stock_code in stock_codes:
            result = self.predict_single(stock_code)
            if result is not None:
                results.append(result)
        
        return results
    
    def show_model_info(self):
        """显示模型信息"""
        if self.model is not None:
            print(f"\n=== 模型信息 ===")
            print(f"模型类型: {type(self.model).__name__}")
            print(f"特征数量: {len(self.features)}")
            print(f"使用特征: {', '.join(self.features)}")
            
            # 如果是随机森林，显示参数
            if hasattr(self.model, 'n_estimators'):
                print(f"树的数量: {self.model.n_estimators}")
                print(f"最大深度: {self.model.max_depth}")
                print(f"最小叶子样本: {self.model.min_samples_leaf}")

def main():
    """主函数 - 演示使用方法"""
    print("=== 股票预测模型使用演示 ===")
    
    try:
        # 创建预测器
        predictor = StockPredictor()
        
        # 显示模型信息
        predictor.show_model_info()
        
        # 1. 预测配置文件中的股票
        print("\n1. 预测配置文件中的股票:")
        result = predictor.predict_single()
        
        # 2. 预测指定股票
        print("\n2. 预测指定股票:")
        other_stocks = ["000001.SZ", "600519.SH"]  # 平安银行、贵州茅台
        
        for stock in other_stocks:
            try:
                predictor.predict_single(stock)
            except Exception as e:
                print(f"预测 {stock} 失败: {e}")
        
        print("\n=== 预测完成 ===")
        
    except Exception as e:
        print(f"程序运行出错: {e}")

if __name__ == "__main__":
    main() 