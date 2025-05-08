import numpy as np
import pandas as pd
import talib
import matplotlib.pyplot as plt
import seaborn as sns
import tushare as ts
import pickle
import os
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class StockTradingSystem:
    """
    基于机器学习预测的股票交易系统
    
    功能:
    1. 数据更新和处理
    2. 模型预测
    3. 交易信号生成
    4. 资金管理
    5. 风险控制 (止盈/止损)
    6. 交易执行 (模拟/实盘接口)
    """
    
    def __init__(self, stock_code='002594.SZ', token='db1723bf9e9009f186c134b6813da7730256c87f31759400e170ddab', 
                 model_path='optimized_model.pkl', initial_capital=1000000):
        """
        初始化交易系统
        
        参数:
        stock_code : str, 股票代码
        token : str, Tushare API token
        model_path : str, 预训练模型路径
        initial_capital : float, 初始资金
        """
        # 基础配置
        self.stock_code = stock_code
        self.token = token
        self.model_path = model_path
        
        # 资金账户
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position = 0
        self.position_value = 0
        self.trades = []
        self.current_trade = None
        
        # 风险管理参数
        self.position_size = 0.3  # 默认仓位大小 (资金比例)
        self.trail_stop_pct = 0.03  # 追踪止盈 3%
        self.stop_loss_pct = 0.02   # 固定止损 2%
        
        # 交易统计
        self.portfolio_value_history = []
        self.cash_history = []
        self.position_history = []
        
        # 特征列表定义
        self.features_columns = ['close', 'vol', 'O-C', 'MA5', 'MA10', 'H-L', 'RSI', 'MOM', 'MACD', 'MACDsignal', 'MACDhist', 'EMA12']
        
        # 数据和模型预加载
        self._setup_api()
        self._load_model()
        
        # 特征缩放器
        self.scaler = StandardScaler()
        self.scaler_is_fitted = False
    
    def _setup_api(self):
        """设置数据API"""
        ts.set_token(self.token)
        self.pro = ts.pro_api()
        print("API setup completed")
    
    def _load_model(self):
        """加载预训练的模型"""
        if os.path.exists(self.model_path):
            self.model = pickle.load(open(self.model_path, 'rb'))
            print(f"Model loaded from {self.model_path}")
        else:
            print(f"Model file {self.model_path} not found")
            self.model = None
        
        # 尝试加载同名的scaler模型
        scaler_path = self.model_path.replace('.pkl', '_scaler.pkl')
        if os.path.exists(scaler_path):
            try:
                self.scaler = pickle.load(open(scaler_path, 'rb'))
                self.scaler_is_fitted = True
                print(f"Scaler loaded from {scaler_path}")
            except:
                self.scaler = StandardScaler()
                self.scaler_is_fitted = False
                print("Failed to load scaler, will fit a new one")
        else:
            self.scaler = StandardScaler()
            self.scaler_is_fitted = False
        
        # 尝试加载特征列表
        features_path = self.model_path.replace('.pkl', '_features.pkl')
        if os.path.exists(features_path):
            try:
                self.features_columns = pickle.load(open(features_path, 'rb'))
                print(f"Feature list loaded from {features_path}: {self.features_columns}")
            except:
                print(f"Failed to load feature list, using default")
    
    def save_model(self, path=None):
        """保存模型和scaler"""
        save_path = path or self.model_path
        if self.model is not None:
            pickle.dump(self.model, open(save_path, 'wb'))
            print(f"Model saved to {save_path}")
            
            # 同时保存scaler
            if self.scaler_is_fitted:
                scaler_path = save_path.replace('.pkl', '_scaler.pkl')
                pickle.dump(self.scaler, open(scaler_path, 'wb'))
                print(f"Scaler saved to {scaler_path}")
            
            # 保存特征列表
            features_path = save_path.replace('.pkl', '_features.pkl')
            pickle.dump(self.features_columns, open(features_path, 'wb'))
            print(f"Feature list saved to {features_path}")
    
    def update_data(self, start_date=None, end_date=None):
        """
        更新最新的股票数据
        
        参数:
        start_date : str, 开始日期 (默认为前2年)
        end_date : str, 结束日期 (默认为今天)
        """
        # 设置日期范围
        if end_date is None:
            end_date = datetime.datetime.now().strftime('%Y%m%d')
        if start_date is None:
            # 默认获取两年数据
            start_date = (datetime.datetime.now() - datetime.timedelta(days=2*365)).strftime('%Y%m%d')
            
        # 检查是否已有所需数据
        if hasattr(self, 'data') and self.data is not None and len(self.data) > 0:
            existing_dates = self.data.index.astype(str)
            requested_start = pd.Timestamp(start_date)
            requested_end = pd.Timestamp(end_date)
            
            # 检查现有数据的日期范围
            data_start = pd.Timestamp(min(existing_dates))
            data_end = pd.Timestamp(max(existing_dates))
            
            # 如果请求的日期范围完全包含在现有数据中，直接返回
            if data_start <= requested_start and data_end >= requested_end:
                print(f"已有请求日期范围的数据（{start_date}至{end_date}），无需重新获取")
                return True
            
            # 如果只需要更新部分日期，则只获取缺失部分
            if data_start <= requested_start:
                # 只需要获取结束日期之后的数据
                if data_end < requested_end:
                    new_start_date = (data_end + pd.Timedelta(days=1)).strftime('%Y%m%d')
                    print(f"只需更新{new_start_date}至{end_date}的新数据")
                    start_date = new_start_date
            elif data_end >= requested_end:
                # 只需要获取开始日期之前的数据
                new_end_date = (data_start - pd.Timedelta(days=1)).strftime('%Y%m%d')
                print(f"只需更新{start_date}至{new_end_date}的新数据")
                end_date = new_end_date
        
        print(f"获取数据：{start_date}至{end_date}...")
        try:
            # 获取日K数据
            new_data = self.pro.daily(ts_code=self.stock_code, start_date=start_date, end_date=end_date)
            
            # 如果没有获取到新数据，直接返回
            if new_data is None or len(new_data) == 0:
                print("未获取到新数据")
                return False
                
            # 按交易日期排序
            new_data = new_data.sort_values('trade_date')
            new_data.set_index('trade_date', inplace=True)
            
            # 合并新旧数据
            if hasattr(self, 'raw_data') and self.raw_data is not None and len(self.raw_data) > 0:
                combined_data = pd.concat([self.raw_data, new_data])
                # 删除重复项，保留最新的
                combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                # 重新排序
                combined_data = combined_data.sort_index()
                self.raw_data = combined_data
            else:
                self.raw_data = new_data
            
            # 计算衍生特征
            self._calculate_features()
            print(f"数据更新成功，共{len(self.data)}条记录")
            
            # 确保scaler已经拟合
            self._ensure_scaler_fitted()
            
            return True
        except Exception as e:
            print(f"更新数据时出错: {e}")
            return False
    
    def _calculate_features(self):
        """计算技术指标和特征工程"""
        df = self.raw_data.copy()
        
        # 基础特征
        df['O-C'] = df['open'] - df['close']
        df['H-L'] = df['high'] - df['low']
        
        # 确保有pre_close列
        if 'pre_close' not in df.columns:
            df['pre_close'] = df['close'].shift(1)
        
        # 计算日涨跌幅
        df['return'] = (df['close'] / df['pre_close'] - 1) * 100
        df['log_return'] = np.log(df['close'] / df['pre_close'])
        
        # 计算技术指标
        df['RSI'] = talib.RSI(df['close'].values, timeperiod=14)
        df['MOM'] = talib.MOM(df['close'].values, timeperiod=5)
        # 添加EMA12作为特征，确保与原模型特征匹配
        df['EMA12'] = talib.EMA(df['close'].values, timeperiod=12)
        
        macd, signal, df['MACDhist'] = talib.MACD(
            df['close'].values, fastperiod=6, slowperiod=12, signalperiod=9
        )
        df['MACD'] = macd
        df['MACDsignal'] = signal
        
        # 计算移动平均
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA10'] = df['close'].rolling(10).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        
        # 计算成交量相关
        df['vol_MA5'] = df['vol'].rolling(5).mean()
        
        # 计算OBV (On-Balance Volume)
        df['OBV'] = talib.OBV(df['close'].values, df['vol'].values)
        
        # 生成强K日因子
        df['strong_K'] = (
            (df['O-C'] < 0) &  # 阳线时为负值
            (abs(df['O-C']) > 0.02 * df['open']) &  # 阳线实体大于2%
            (abs(df['O-C']) / df['H-L'] > 0.6) &  # 实体占比 > 60%
            (df['vol'] > df['vol_MA5']) &  # 放量
            (df['MACDhist'] > 0)  # 趋势向上
        ).astype(int)
        
        # 删除NaN值
        df.dropna(inplace=True)
        
        self.data = df
    
    def _ensure_scaler_fitted(self):
        """确保StandardScaler已经fit过数据"""
        if not self.scaler_is_fitted and self.data is not None and len(self.data) > 0:
            print("Fitting scaler with available data...")
            # 准备特征数据
            X = self.data[self.features_columns]
            # 拟合scaler
            self.scaler.fit(X)
            self.scaler_is_fitted = True
            print("Scaler fitted successfully")
    
    def train_model(self, test_size=0.1, random_state=120):
        """使用最新数据训练或更新模型"""
        if self.data is None or len(self.data) < 100:
            print("Insufficient data for training")
            return False
        
        # 准备特征和目标变量
        X = self.data[self.features_columns]
        y = np.where(self.data['log_return'] >= 0.0025, 1, 0)  # 上涨0.25%作为阈值
        
        # 数据集分割
        X_length = X.shape[0]
        split = int(X_length * (1 - test_size))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # 特征缩放
        self.scaler.fit(X_train)
        self.scaler_is_fitted = True
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 训练模型
        self.model = RandomForestClassifier(
            max_depth=6,  # 从grid search中获取的最优参数
            n_estimators=10,
            min_samples_leaf=30,
            random_state=random_state
        )
        self.model.fit(X_train_scaled, y_train)
        
        # 评估模型
        train_accuracy = self.model.score(X_train_scaled, y_train)
        test_accuracy = self.model.score(X_test_scaled, y_test)
        print(f"Model trained - Train accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}")
        
        # 保存模型和scaler
        self.save_model()
        return True
    
    def get_prediction(self, data=None):
        """
        获取最新的预测信号
        
        参数:
        data : DataFrame, 可选的输入数据，如果为None则使用最新的数据
        
        返回:
        dict: 包含预测结果的字典
        """
        if self.model is None:
            print("Model not loaded")
            return {'signal': 0, 'probability': 0, 'confidence': 'No model'}
        
        if data is None:
            if self.data is None or len(self.data) == 0:
                print("No data available")
                return {'signal': 0, 'probability': 0, 'confidence': 'No data'}
            data = self.data.iloc[-1:].copy()
        
        # 确保所有需要的特征都存在
        missing_cols = [col for col in self.features_columns if col not in data.columns]
        if missing_cols:
            print(f"Warning: Missing features {missing_cols}. Attempting to calculate...")
            # 尝试计算缺失的EMA12特征
            if 'EMA12' in missing_cols and 'close' in data.columns:
                # 提供假EMA值（使用当前close，实际应用中应使用更多历史数据）
                data['EMA12'] = data['close']
                print("Added estimated EMA12 feature")
        
        # 提取特征 - 使用模型期望的特征列表
        features = data[self.features_columns]
        
        # 确保scaler已fit
        if not self.scaler_is_fitted:
            self._ensure_scaler_fitted()
            if not self.scaler_is_fitted:
                print("WARNING: Scaler not fitted, prediction may be inaccurate")
                # 尝试用当前特征拟合
                self.scaler.fit(features)
                self.scaler_is_fitted = True
        
        # 特征缩放
        features_scaled = self.scaler.transform(features)
        
        # 预测
        signal_prob = self.model.predict_proba(features_scaled)[0]
        signal = self.model.predict(features_scaled)[0]
        
        # 强K日因子作为辅助信号
        strong_k_signal = data['strong_K'].values[0] if 'strong_K' in data.columns else 0
        
        # 信号置信度评估
        confidence = 'Low'
        if signal == 1:
            if signal_prob[1] > 0.7:
                confidence = 'High'
            elif signal_prob[1] > 0.6:
                confidence = 'Medium'
            
            # 如果随机森林和强K信号一致，提高置信度
            if strong_k_signal == 1:
                if confidence == 'Medium':
                    confidence = 'High'
                elif confidence == 'Low':
                    confidence = 'Medium'
        
        return {
            'date': data.index[0],
            'signal': int(signal),
            'probability': float(signal_prob[1]),
            'strong_k': strong_k_signal,
            'confidence': confidence,
            'price': data['close'].values[0],
            'features': features.iloc[0].to_dict()
        }
    
    def generate_trading_signal(self):
        """
        基于预测和强K因子生成交易信号
        
        返回:
        int: 交易信号 (1=买入, -1=卖出, 0=持有/观望)
        """
        if self.data is None or len(self.data) < 20:
            return {'signal': 0, 'reason': 'Insufficient data'}
        
        # 获取最新数据和预测
        latest_data = self.data.iloc[-1].copy()
        prediction = self.get_prediction()
        
        # 当前持仓状态
        current_position = 1 if self.position > 0 else 0
        
        # 生成信号
        signal = 0
        reason = "No action"
        
        # 基于机器学习+强K因子的买入逻辑
        if current_position == 0:  # 当前无持仓
            # 买入信号条件
            ml_signal = prediction['signal'] == 1 and prediction['probability'] > 0.6
            strong_k = prediction['strong_k'] == 1
            
            # 买入决策逻辑
            if ml_signal and strong_k:
                signal = 1
                reason = "ML + Strong K confirmation"
            elif ml_signal and prediction['probability'] > 0.7:
                signal = 1
                reason = "High ML probability"
            elif strong_k and (latest_data['RSI'] < 60):  # 强K但RSI不过高
                signal = 1
                reason = "Strong K + RSI favorable"
        
        # 风险管理：止盈/止损检查
        elif current_position == 1:  # 当前有持仓
            if self.current_trade is not None:
                entry_price = self.current_trade['entry_price']
                current_price = latest_data['close']
                
                # 更新当前交易的最高价
                if current_price > self.current_trade.get('highest_price', entry_price):
                    self.current_trade['highest_price'] = current_price
                highest_price = self.current_trade['highest_price']
                
                # 追踪止盈
                if current_price <= highest_price * (1 - self.trail_stop_pct):
                    signal = -1
                    reason = f"Trail stop {self.trail_stop_pct*100}% triggered"
                
                # 固定止损
                elif current_price <= entry_price * (1 - self.stop_loss_pct):
                    signal = -1
                    reason = f"Stop loss {self.stop_loss_pct*100}% triggered"
                
                # 获利了结：RSI高位 + ML预测下跌
                elif latest_data['RSI'] > 70 and prediction['signal'] == 0:
                    signal = -1
                    reason = "Take profit - RSI high + ML bearish"
        
        return {'signal': signal, 'reason': reason}
    
    def execute_trade(self, signal=None, price=None, quantity=None, simulate=True):
        """
        执行交易操作
        
        参数:
        signal : dict, 交易信号字典 {'signal': 1|-1|0, 'reason': str}
        price : float, 交易价格 (默认使用最新收盘价)
        quantity : int, 交易数量 (默认使用仓位管理计算)
        simulate : bool, 是否模拟交易
        
        返回:
        dict: 交易结果信息
        """
        if signal is None:
            signal = self.generate_trading_signal()
        
        if signal['signal'] == 0:
            return {'status': 'skipped', 'message': 'No trading signal'}
        
        # 获取最新数据
        if price is None and len(self.data) > 0:
            price = self.data['close'].iloc[-1]
        
        # 计算交易量
        if quantity is None and signal['signal'] == 1:  # 买入
            # 使用资金管理计算买入数量
            max_position = self.cash * self.position_size
            quantity = int(max_position / price / 100) * 100  # 取整百股
            if quantity < 100:  # 至少买入1手
                quantity = 0
        elif quantity is None and signal['signal'] == -1:  # 卖出
            quantity = self.position
        
        if quantity == 0:
            return {'status': 'skipped', 'message': 'Insufficient funds for minimum position'}
        
        # 交易执行
        trade_amount = price * quantity
        commission = trade_amount * 0.0003  # 手续费0.03%
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if signal['signal'] == 1:  # 买入
            if simulate:
                # 模拟买入
                if trade_amount + commission > self.cash:
                    return {'status': 'failed', 'message': 'Insufficient funds'}
                
                self.cash -= (trade_amount + commission)
                self.position += quantity
                self.position_value = self.position * price
                
                # 记录当前交易
                self.current_trade = {
                    'type': 'buy',
                    'timestamp': timestamp,
                    'price': price,
                    'quantity': quantity,
                    'amount': trade_amount,
                    'commission': commission,
                    'entry_price': price,
                    'highest_price': price,
                    'reason': signal['reason']
                }
                self.trades.append(self.current_trade)
                
                return {
                    'status': 'success',
                    'type': 'buy',
                    'price': price,
                    'quantity': quantity,
                    'amount': trade_amount,
                    'commission': commission,
                    'cash_left': self.cash,
                    'reason': signal['reason']
                }
            else:
                # 这里可以接入实盘交易API
                print("Real trading API not implemented yet")
                return {'status': 'simulated', 'message': 'Real trading not implemented'}
        
        elif signal['signal'] == -1:  # 卖出
            if simulate:
                # 模拟卖出
                if quantity > self.position:
                    quantity = self.position
                    
                if quantity == 0:
                    return {'status': 'skipped', 'message': 'No position to sell'}
                
                trade_amount = price * quantity
                commission = trade_amount * 0.0003
                
                self.cash += (trade_amount - commission)
                self.position -= quantity
                self.position_value = self.position * price
                
                # 计算收益
                entry_price = self.current_trade['entry_price'] if self.current_trade else price
                profit = (price - entry_price) * quantity - commission
                profit_pct = (price / entry_price - 1) * 100
                
                # 记录交易
                exit_trade = {
                    'type': 'sell',
                    'timestamp': timestamp,
                    'price': price,
                    'quantity': quantity,
                    'amount': trade_amount,
                    'commission': commission,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'reason': signal['reason']
                }
                self.trades.append(exit_trade)
                
                # 重置当前交易
                self.current_trade = None
                
                return {
                    'status': 'success',
                    'type': 'sell',
                    'price': price,
                    'quantity': quantity,
                    'amount': trade_amount,
                    'commission': commission,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'cash_left': self.cash,
                    'reason': signal['reason']
                }
            else:
                # 实盘交易API
                print("Real trading API not implemented yet")
                return {'status': 'simulated', 'message': 'Real trading not implemented'}
    
    def get_portfolio_value(self, current_price=None):
        """计算当前组合价值"""
        if current_price is None and len(self.data) > 0:
            current_price = self.data['close'].iloc[-1]
        elif current_price is None:
            return self.cash
        
        position_value = self.position * current_price
        return self.cash + position_value
    
    def update_portfolio_history(self, current_price=None):
        """更新投资组合历史记录"""
        portfolio_value = self.get_portfolio_value(current_price)
        self.portfolio_value_history.append(portfolio_value)
        self.cash_history.append(self.cash)
        self.position_history.append(self.position)
    
    def backtest(self, start_date=None, end_date=None):
        """
        回测策略
        
        参数:
        start_date : str, 回测开始日期
        end_date : str, 回测结束日期
        
        返回:
        dict: 回测结果统计
        """
        # 确保数据已加载并包含所需日期范围
        if self.data is None or len(self.data) == 0 or not self._check_date_range(start_date, end_date):
            if not self.update_data(start_date, end_date):
                return {'status': 'failed', 'message': '无法加载数据'}
        
        # 确保scaler已经拟合
        self._ensure_scaler_fitted()
        
        # 初始化回测
        self.cash = self.initial_capital
        self.position = 0
        self.position_value = 0
        self.trades = []
        self.current_trade = None
        self.portfolio_value_history = []
        
        # 创建回测日期索引
        if start_date:
            backtest_data = self.data[self.data.index >= start_date]
        else:
            backtest_data = self.data.copy()
        if end_date:
            backtest_data = backtest_data[backtest_data.index <= end_date]
        
        # 每日循环
        for i in range(20, len(backtest_data)):  # 从第20天开始，确保有足够的历史数据
            current_date = backtest_data.index[i]
            current_row = backtest_data.iloc[i]
            
            # 记录每日投资组合价值
            self.update_portfolio_history(current_row['close'])
            
            # 使用当前日期前的数据生成预测
            historical_data = backtest_data.iloc[:i].copy()
            
            # 获取交易信号
            prediction = self.get_prediction(historical_data.iloc[-1:])
            
            # 生成交易信号
            signal_dict = self.generate_trading_signal()
            
            # 执行交易
            if signal_dict['signal'] != 0:
                trade_result = self.execute_trade(
                    signal=signal_dict,
                    price=current_row['close']
                )
                
                # 记录交易结果
                if trade_result['status'] == 'success':
                    trade_result['date'] = current_date
                    print(f"[{current_date}] {trade_result['type'].upper()}: {trade_result['quantity']} shares at {trade_result['price']}, reason: {signal_dict['reason']}")
        
        # 计算回测结果
        final_portfolio_value = self.get_portfolio_value(backtest_data['close'].iloc[-1])
        total_return = (final_portfolio_value / self.initial_capital - 1) * 100
        
        # 统计交易
        buy_trades = [t for t in self.trades if t['type'] == 'buy']
        sell_trades = [t for t in self.trades if t['type'] == 'sell']
        
        # 计算胜率
        profitable_trades = [t for t in sell_trades if t.get('profit', 0) > 0]
        win_rate = len(profitable_trades) / len(sell_trades) if len(sell_trades) > 0 else 0
        
        # 计算平均收益
        avg_profit_pct = np.mean([t.get('profit_pct', 0) for t in sell_trades]) if len(sell_trades) > 0 else 0
        
        # 计算最大回撤
        portfolio_values = np.array(self.portfolio_value_history)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        results = {
            'initial_capital': self.initial_capital,
            'final_portfolio_value': final_portfolio_value,
            'total_return': f"{total_return:.2f}%",
            'total_trades': len(buy_trades),
            'win_rate': f"{win_rate:.2%}",
            'avg_profit_pct': f"{avg_profit_pct:.2f}%",
            'max_drawdown': f"{max_drawdown:.2f}%"
        }
        
        print("\nBacktest Results:")
        for key, value in results.items():
            print(f"{key}: {value}")
        
        return results
    
    def plot_backtest_results(self):
        """可视化回测结果"""
        if not self.portfolio_value_history:
            print("No backtest data to plot")
            return
        
        plt.figure(figsize=(14, 10))
        
        # 绘制投资组合价值
        plt.subplot(2, 1, 1)
        plt.plot(self.portfolio_value_history)
        plt.title('Portfolio Value')
        plt.grid(True)
        
        # 绘制买卖点
        buy_dates = [t['timestamp'] if isinstance(t['timestamp'], datetime.datetime) else 
                    pd.Timestamp(t['timestamp']) for t in self.trades if t['type'] == 'buy']
        buy_prices = [t['price'] for t in self.trades if t['type'] == 'buy']
        
        sell_dates = [t['timestamp'] if isinstance(t['timestamp'], datetime.datetime) else 
                     pd.Timestamp(t['timestamp']) for t in self.trades if t['type'] == 'sell']
        sell_prices = [t['price'] for t in self.trades if t['type'] == 'sell']
        
        if len(buy_dates) > 0:
            plt.plot(buy_dates, buy_prices, '^', markersize=10, color='g', label='Buy')
        
        if len(sell_dates) > 0:
            plt.plot(sell_dates, sell_prices, 'v', markersize=10, color='r', label='Sell')
        
        plt.legend()
        
        # 绘制交易结果分布
        plt.subplot(2, 1, 2)
        profits = [t.get('profit_pct', 0) for t in self.trades if t['type'] == 'sell']
        if profits:
            plt.hist(profits, bins=15, alpha=0.7)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title('Profit Distribution (%)')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=300)
        plt.show()

    def run_live_trading(self, days=1, interval_minutes=30, simulate=True):
        """
        运行实时/模拟交易逻辑
        
        参数:
        days : int, 运行天数
        interval_minutes : int, 检查间隔(分钟)
        simulate : bool, 是否仅模拟交易
        
        返回:
        dict: 交易统计
        """
        if self.model is None:
            return {'status': 'failed', 'message': 'Model not loaded'}
        
        print(f"Starting {'simulated' if simulate else 'real'} trading for {days} days...")
        
        # 设置运行时间
        end_time = datetime.datetime.now() + datetime.timedelta(days=days)
        
        while datetime.datetime.now() < end_time:
            # 当前时间
            now = datetime.datetime.now()
            current_time_str = now.strftime('%Y-%m-%d %H:%M:%S')
            
            # 检查是否在交易时间内
            is_trading_hours = self._check_trading_hours(now)
            
            if is_trading_hours:
                # 更新最新数据
                if self.update_data():
                    # 生成交易信号
                    signal_dict = self.generate_trading_signal()
                    
                    # 执行交易
                    if signal_dict['signal'] != 0:
                        trade_result = self.execute_trade(signal=signal_dict, simulate=simulate)
                        
                        if trade_result['status'] == 'success':
                            print(f"[{current_time_str}] {trade_result['type'].upper()}: {trade_result['quantity']} shares at {trade_result['price']}")
                            print(f"  Reason: {signal_dict['reason']}")
                            print(f"  Cash left: {trade_result['cash_left']:.2f}")
                    else:
                        print(f"[{current_time_str}] No trading signal")
                else:
                    print(f"[{current_time_str}] Failed to update data")
            else:
                print(f"[{current_time_str}] Outside trading hours, waiting...")
            
            # 等待指定的间隔时间
            sleep_time = datetime.timedelta(minutes=interval_minutes)
            print(f"Waiting for {interval_minutes} minutes until next check...")
            
            # 在实际应用中，这里应该使用 time.sleep()
            # 为了避免在此次演示中实际等待，我们显示一条消息
            print(f"Next check scheduled at: {(now + sleep_time).strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 如果是实际运行而非演示，取消下面这行的注释
            # time.sleep(interval_minutes * 60)
            
            # 仅供演示：立即继续下一次迭代或结束
            break
        
        # 计算交易统计
        trade_stats = {
            'total_trades': len(self.trades),
            'buys': len([t for t in self.trades if t['type'] == 'buy']),
            'sells': len([t for t in self.trades if t['type'] == 'sell']),
            'current_position': self.position,
            'cash': self.cash,
            'portfolio_value': self.get_portfolio_value()
        }
        
        return trade_stats
    
    def _check_trading_hours(self, now=None):
        """检查是否在交易时间内"""
        if now is None:
            now = datetime.datetime.now()
        
        # 周末不交易
        if now.weekday() >= 5:  # 5=Saturday, 6=Sunday
            return False
        
        # 设置交易时段 (中国A股: 9:30-11:30, 13:00-15:00)
        morning_start = datetime.time(9, 30)
        morning_end = datetime.time(11, 30)
        afternoon_start = datetime.time(13, 0)
        afternoon_end = datetime.time(15, 0)
        
        current_time = now.time()
        
        return ((current_time >= morning_start and current_time <= morning_end) or
                (current_time >= afternoon_start and current_time <= afternoon_end))

    def predict_next_day(self):
        """
        在收盘后预测明天的涨跌方向和幅度
        
        返回:
        dict: 包含预测方向、概率和预计幅度的字典
        """
        if self.model is None:
            print("模型未加载，无法进行预测")
            return None
            
        if self.data is None or len(self.data) < 5:
            print("数据不足，无法进行预测")
            return None
            
        # 获取最新数据
        latest_data = self.data.iloc[-1:].copy()
        
        # 使用现有的预测方法获取信号
        prediction = self.get_prediction(latest_data)
        
        # 预测明天涨跌方向
        direction = "上涨" if prediction['signal'] == 1 else "下跌"
        
        # 获取预测概率
        probability = prediction['probability'] if prediction['signal'] == 1 else (1 - prediction['probability'])
        
        # 估计涨跌幅度 (基于概率和历史波动性)
        # 使用过去20天的平均波动性作为基准
        if len(self.data) >= 20:
            # 计算过去20天的日涨跌幅绝对值平均值
            avg_daily_change = self.data['return'].tail(20).abs().mean()
            
            # 根据预测概率调整预期幅度
            if probability > 0.8:
                expected_change = avg_daily_change * 1.5  # 高概率，预期更大幅度
            elif probability > 0.6:
                expected_change = avg_daily_change * 1.0  # 中等概率，预期正常幅度
            else:
                expected_change = avg_daily_change * 0.6  # 低概率，预期较小幅度
        else:
            # 数据不足时使用固定值
            expected_change = 1.0  # 默认预期1%的涨跌幅
        
        # 考虑市场整体趋势 (MACD作为趋势指标)
        if 'MACDhist' in latest_data.columns:
            macd_hist = latest_data['MACDhist'].values[0]
            trend_factor = 1 + (0.5 * np.sign(macd_hist) * min(abs(macd_hist) / 2, 0.5))
            expected_change *= trend_factor
        
        # 加入强K日因子的影响
        if prediction['strong_k'] == 1 and prediction['signal'] == 1:
            expected_change *= 1.3  # 强K日预期更大的上涨幅度
        
        # 根据RSI调整预期幅度 (极端RSI值可能导致反转)
        if 'RSI' in latest_data.columns:
            rsi = latest_data['RSI'].values[0]
            if rsi > 80 and prediction['signal'] == 1:  # 超买且预测上涨
                expected_change *= 0.7  # 下调上涨预期
            elif rsi < 20 and prediction['signal'] == 0:  # 超卖且预测下跌
                expected_change *= 0.7  # 下调下跌预期
        
        # 格式化预测结果
        result = {
            'date': latest_data.index[0],
            'prediction_for': pd.Timestamp(latest_data.index[0]) + pd.Timedelta(days=1),
            'direction': direction,
            'probability': f"{probability:.2%}",
            'expected_change': f"{expected_change:.2f}%",
            'confidence': prediction['confidence'],
            'current_price': latest_data['close'].values[0],
            'market_trend': "上升" if (latest_data.get('MACDhist', [0])[0] > 0) else "下降",
            'rsi_level': latest_data.get('RSI', [None])[0]
        }
        
        return result

    def get_today_data(self):
        """
        获取当天的股票交易数据
        
        返回:
        bool: 数据获取是否成功
        """
        # 获取当天日期
        today = datetime.datetime.now().strftime('%Y%m%d')
        
        print(f"正在获取{self.stock_code}当天({today})交易数据...")
        try:
            # 方法1：使用trade_date参数直接获取当天数据
            print(f"尝试使用trade_date参数获取{self.stock_code}在{today}的数据...")
            today_data = self.pro.daily(ts_code=self.stock_code, trade_date=today)
            
            # 如果方法1未获取到数据，尝试方法2：使用日期范围
            if today_data is None or len(today_data) == 0:
                print("未获取到数据，尝试使用日期范围方法...")
                today_data = self.pro.daily(ts_code=self.stock_code, start_date=today, end_date=today)
            
            # 检查是否最终获取到数据
            if today_data is None or len(today_data) == 0:
                print(f"未获取到{today}的交易数据，可能的原因:")
                print("1. 今天不是交易日")
                print("2. 交易尚未结束")
                print("3. 数据尚未更新")
                print("4. API调用限制或网络问题")
                return False
            
            # 数据获取成功，打印基本信息
            print(f"\n成功获取{self.stock_code}当天数据:")
            print(f"开盘价: {today_data['open'].values[0]}")
            print(f"收盘价: {today_data['close'].values[0]}")
            print(f"最高价: {today_data['high'].values[0]}")
            print(f"最低价: {today_data['low'].values[0]}")
            print(f"成交量: {today_data['vol'].values[0]/10000:.2f}万")
            print(f"涨跌幅: {today_data['pct_chg'].values[0]:.2f}%")
                
            # 设置索引并排序
            today_data = today_data.sort_values('trade_date')
            today_data.set_index('trade_date', inplace=True)
            
            # 合并到现有数据中
            if hasattr(self, 'raw_data') and self.raw_data is not None and len(self.raw_data) > 0:
                # 检查是否已经有当天数据
                if today in self.raw_data.index:
                    print(f"已有{today}的数据，将使用最新数据更新...")
                    
                # 合并数据
                combined_data = pd.concat([self.raw_data, today_data])
                # 删除重复项，保留最新的
                combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                # 重新排序
                combined_data = combined_data.sort_index()
                self.raw_data = combined_data
            else:
                self.raw_data = today_data
            
            # 计算衍生特征
            self._calculate_features()
            print(f"当天数据处理成功，当前数据共{len(self.data)}条记录")
            
            # 确保scaler已经拟合
            self._ensure_scaler_fitted()
            
            return True
        except Exception as e:
            print(f"获取当天数据时出错: {e}")
            
            # 如果出错，尝试获取前一天的数据作为参考
            yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y%m%d')
            print(f"\n尝试获取昨日({yesterday})数据作为参考...")
            try:
                yesterday_data = self.pro.daily(ts_code=self.stock_code, trade_date=yesterday)
                if yesterday_data is not None and not yesterday_data.empty:
                    print(f"获取到昨日数据，收盘价: {yesterday_data['close'].values[0]}")
                else:
                    print("未能获取昨日数据")
            except:
                print("获取昨日数据也失败")
                
            return False

    def run_daily_prediction(self, market_close_time=None):
        """
        运行每日收盘后预测，可自动调度或手动触发
        
        参数:
        market_close_time: datetime.time, 市场收盘时间，默认为15:00(中国A股)
        
        返回:
        dict: 预测结果
        """
        if market_close_time is None:
            market_close_time = datetime.time(15, 0)  # 默认15:00收盘
            
        now = datetime.datetime.now()
        today = now.date()
        
        # 获取今天的收盘时间
        close_datetime = datetime.datetime.combine(today, market_close_time)
        
        # 检查是否是交易日
        if now.weekday() >= 5:  # 周末不交易
            print(f"今天是{['周一','周二','周三','周四','周五','周六','周日'][now.weekday()]}，非交易日")
            return None
            
        # 检查是否已过收盘时间
        if now.time() < market_close_time:
            time_to_close = (close_datetime - now).total_seconds() / 60
            print(f"交易尚未结束，距离收盘还有 {time_to_close:.0f} 分钟")
            return None
            
        # 更新当天数据
        print("正在获取今日收盘数据...")
        if self.get_today_data():
            print("当天数据更新成功，准备预测明日行情")
        else:
            # 如果当天数据获取失败，尝试获取最近数据
            one_week_ago = (today - datetime.timedelta(days=7)).strftime('%Y%m%d')
            today_str = today.strftime('%Y%m%d')
            print(f"当天数据获取失败，尝试获取近期数据 ({one_week_ago} 至 {today_str})...")
            if not self.update_data(start_date=one_week_ago, end_date=today_str):
                print("数据获取失败，无法进行预测")
                return None
        
        # 打印今日市场情况
        if self.data is not None and len(self.data) > 0:
            latest_data = self.data.iloc[-1]
            prev_data = self.data.iloc[-2] if len(self.data) > 1 else None
            
            print("\n====== 今日市场回顾 ======")
            print(f"日期: {latest_data.name}")
            print(f"开盘价: {latest_data['open']:.2f}")
            print(f"收盘价: {latest_data['close']:.2f}")
            print(f"最高价: {latest_data['high']:.2f}")
            print(f"最低价: {latest_data['low']:.2f}")
            print(f"成交量: {latest_data['vol']/10000:.2f}万")
            
            # 计算今日涨跌幅
            if 'return' in latest_data:
                print(f"涨跌幅: {latest_data['return']:.2f}%")
            elif prev_data is not None:
                change_pct = (latest_data['close'] / prev_data['close'] - 1) * 100
                print(f"涨跌幅: {change_pct:.2f}%")
            
            # 打印技术指标
            if 'RSI' in latest_data:
                rsi = latest_data['RSI']
                rsi_status = "超买" if rsi > 70 else "超卖" if rsi < 30 else "中性"
                print(f"RSI指标: {rsi:.2f} ({rsi_status})")
                
            if 'MACD' in latest_data and 'MACDsignal' in latest_data and 'MACDhist' in latest_data:
                macd = latest_data['MACD']
                macd_signal = latest_data['MACDsignal']
                macd_hist = latest_data['MACDhist']
                macd_status = "多头" if macd_hist > 0 else "空头"
                print(f"MACD: {macd:.4f}, 信号线: {macd_signal:.4f}, 柱状线: {macd_hist:.4f} ({macd_status})")
                
            # 移动平均线
            if 'MA5' in latest_data and 'MA10' in latest_data and 'MA20' in latest_data:
                ma5 = latest_data['MA5']
                ma10 = latest_data['MA10']
                ma20 = latest_data['MA20']
                
                ma_trend = ""
                if ma5 > ma10 > ma20:
                    ma_trend = "强势上涨"
                elif ma5 < ma10 < ma20:
                    ma_trend = "强势下跌"
                elif ma5 > ma10 and ma10 < ma20:
                    ma_trend = "可能反弹"
                elif ma5 < ma10 and ma10 > ma20:
                    ma_trend = "可能回调"
                
                print(f"MA5: {ma5:.2f}, MA10: {ma10:.2f}, MA20: {ma20:.2f} ({ma_trend})")
            
            # 强K日指标
            if 'strong_K' in latest_data:
                print(f"强K日: {'是' if latest_data['strong_K'] == 1 else '否'}")
            
            print("==========================\n")
        
        # 进行明日预测
        prediction = self.predict_next_day()
        
        if prediction:
            # 输出预测结果
            print("\n====== 明日市场预测 ======")
            print(f"预测日期: {prediction['prediction_for'].strftime('%Y-%m-%d')}")
            print(f"预测方向: {prediction['direction']}")
            print(f"预期幅度: {prediction['expected_change']}")
            print(f"置信度: {prediction['confidence']} ({prediction['probability']})")
            print(f"当前价格: {prediction['current_price']:.2f}")
            print(f"市场趋势: {prediction['market_trend']}")
            if prediction['rsi_level'] is not None:
                print(f"RSI水平: {prediction['rsi_level']:.1f}")
            print("==========================")
            
            # 可以将预测结果保存到文件或数据库
            self._save_prediction_to_history(prediction)
            
        return prediction
    
    def _save_prediction_to_history(self, prediction):
        """保存预测结果到历史记录"""
        # 创建预测历史文件
        history_file = 'prediction_history.csv'
        
        # 准备要保存的数据
        data = {
            'date': [prediction['date']],
            'prediction_for': [prediction['prediction_for']],
            'direction': [prediction['direction']],
            'probability': [prediction['probability']],
            'expected_change': [prediction['expected_change']],
            'confidence': [prediction['confidence']],
            'current_price': [prediction['current_price']]
        }
        df = pd.DataFrame(data)
        
        # 检查文件是否存在
        if os.path.exists(history_file):
            # 加载现有历史记录并追加
            history = pd.read_csv(history_file)
            history = pd.concat([history, df])
            history.to_csv(history_file, index=False)
        else:
            # 创建新文件
            df.to_csv(history_file, index=False)
            
        print(f"预测结果已保存到 {history_file}")

    def _check_date_range(self, start_date=None, end_date=None):
        """检查现有数据是否包含请求的日期范围"""
        if self.data is None or len(self.data) == 0:
            return False
            
        # 如果未指定日期范围，默认为需要获取新数据
        if start_date is None and end_date is None:
            return True
            
        # 转换日期格式
        if start_date:
            requested_start = pd.Timestamp(start_date)
        else:
            requested_start = pd.Timestamp('19000101')  # 远古日期作为最小值
            
        if end_date:
            requested_end = pd.Timestamp(end_date)
        else:
            requested_end = pd.Timestamp('21000101')  # 未来日期作为最大值
        
        # 获取现有数据的日期范围
        existing_dates = self.data.index.astype(str)
        data_start = pd.Timestamp(min(existing_dates))
        data_end = pd.Timestamp(max(existing_dates))
        
        # 检查请求的日期范围是否在现有数据范围内
        return data_start <= requested_start and data_end >= requested_end


def main():
    # 保存优化后的模型
    try:
        import stock_prediction
        print("正在从stock_prediction.py保存优化模型...")
        with open('optimized_model.pkl', 'wb') as f:
            pickle.dump(stock_prediction.optimized_model, f)
        
        # 同时保存特征列表，以匹配原模型
        features = ['close', 'vol', 'O-C', 'MA5', 'MA10', 'H-L', 'RSI', 'MOM', 'EMA12', 'MACD', 'MACDsignal', 'MACDhist']
        with open('optimized_model_features.pkl', 'wb') as f:
            pickle.dump(features, f)
            
        print("模型和特征保存成功")
    except:
        print("从stock_prediction.py保存模型失败，请检查文件是否存在并包含已训练的模型")
    
    # 创建交易系统实例
    trading_system = StockTradingSystem(
        stock_code='002594.SZ',
        token='db1723bf9e9009f186c134b6813da7730256c87f31759400e170ddab',
        initial_capital=1000000
    )
    
    # 更新数据
    print("更新股票数据...")
    # 设置回测日期范围
    start_date = '20220101'
    end_date = datetime.datetime.now().strftime('%Y%m%d')
    trading_system.update_data(start_date=start_date, end_date=end_date)
    
    # 运行回测
    print("\n运行回测...")
    trading_system.backtest(start_date=start_date, end_date=end_date)
    
    # 绘制回测结果
    trading_system.plot_backtest_results()
    
    # 模拟实时交易示例
    print("\n模拟实时交易示例...")
    trading_system.run_live_trading(days=1, interval_minutes=30, simulate=True)
    
    # 添加每日收盘预测功能
    print("\n运行收盘后预测...")
    trading_system.run_daily_prediction()
    
    print("\n交易系统演示完成!")

if __name__ == "__main__":
    main() 