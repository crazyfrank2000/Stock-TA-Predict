# 📈 股票技术指标预测系统

基于机器学习的股票技术分析预测系统，支持单日和多周期预测。

## 🎯 系统概述

本系统使用随机森林算法，基于13个技术指标对股票进行涨跌预测，支持：
- **单日预测**：预测明日涨跌
- **多周期预测**：支持1天、5天、20天预测
- **多股票支持**：文件名包含股票代码，支持同时训练多只股票

## 📊 数据获取阶段

### 数据源
- **API**: Tushare Pro API
- **频率**: 日线数据（开、高、低、收、成交量）
- **范围**: 建议3年以上历史数据

### 数据获取流程
```python
# 1. 配置API
ts.set_token('your_token')
pro = ts.pro_api()

# 2. 获取历史数据
df = pro.daily(ts_code='000063.SZ', start_date='20210101', end_date='20250101')

# 3. 数据预处理
df = df.sort_values('trade_date').reset_index(drop=True)
```

### ⚠️ 时间泄漏问题
**重要**: 为避免过拟合，训练数据应截止到预测日前至少60天：

```python
当前日期: 2025-05-24
训练数据: 2021-01-01 → 2024-12-31  ✅ 正确
训练数据: 2021-01-01 → 2025-05-23  ❌ 时间泄漏
```

## 🔧 技术指标计算

系统计算13个技术指标作为模型特征：

### 基础指标
- **close**: 收盘价
- **vol**: 成交量
- **O-C**: 开盘价-收盘价差
- **H-L**: 最高价-最低价差

### 移动平均类
- **MA5**: 5日移动平均
- **MA10**: 10日移动平均
- **EMA12**: 12日指数移动平均
- **EMA26**: 26日指数移动平均

### 技术分析指标
- **RSI**: 相对强弱指数（14日）
- **MOM**: 动量指标
- **MACD**: MACD快线
- **MACDsignal**: MACD信号线
- **MACDhist**: MACD柱状线

### 特征重要性排序
根据中兴通讯(000063.SZ)的训练结果：
```
1. O-C (开收差): 46.23%     ← 最重要
2. RSI: 14.35%
3. vol: 8.04%
4. MACDsignal: 7.20%
5. close: 4.42%
... 其他指标
```

## 🤖 模型训练

### 单日模型训练
```bash
# 训练单日预测模型
python stock_prediction.py
```

**配置参数**：
- **阈值**: 0.25%（涨幅超过0.25%算上涨）
- **算法**: 随机森林
- **优化参数**: 网格搜索（深度、树数量、叶子样本数）

### 多周期模型训练
```bash
# 训练5天、20天预测模型
python train_multi_period.py
```

**多周期配置**：
```python
"period_thresholds": {
    "1": 0.0025,    # 1天: 0.25% (过滤日内噪音)
    "5": 0.01,      # 5天: 1.0%  (确认短期趋势)
    "20": 0.03      # 20天: 3.0% (识别中期行情)
}
```

### 阈值设计原理
不同时间周期使用不同涨跌判断标准：

```python
# 同一只股票，当前价格31.20元
1天后31.28元: +0.26% >= 0.25% → 上涨 ✅ (日内有效波动)
5天后31.52元: +1.03% >= 1.0%  → 上涨 ✅ (短期趋势确立)
20天后32.14元: +3.01% >= 3.0% → 上涨 ✅ (中期行情确认)

# 这样设计避免"虚假信号"
20天才涨0.3%不算有意义的上涨
```

### 时间序列分割
**正确的训练集分割方式**：
```python
# ❌ 错误：随机分割
X_train, X_test = train_test_split(X, y, test_size=0.1, random_state=42)

# ✅ 正确：时间序列分割  
split = int(len(X) * 0.9)
X_train, X_test = X[:split], X[split:]  # 按时间顺序分割
```

## 📁 项目文件结构

### 🐍 核心Python文件

| 文件名 | 主要功能 | 使用场景 |
|--------|----------|----------|
| **stock_prediction.py** | 单日预测模型训练 | 训练1天涨跌预测模型 |
| **train_multi_period.py** | 多周期模型训练 | 训练5天、20天预测模型 |
| **predict.py** | 统一预测系统 | 执行单日和多周期预测 |
| **use_trained_model.py** | 模型使用类库 | 提供StockPredictor类 |
| **model_manager.py** | 模型管理工具 | 查看、验证、备份模型 |

### 📊 核心文件功能详解

#### 🎯 stock_prediction.py - 单日模型训练器
```python
# 主要功能
- 获取股票历史数据（Tushare API）
- 计算13个技术指标特征
- 随机森林模型训练和优化
- 生成可视化报告（特征重要性、混淆矩阵、ROC曲线）
- 保存训练好的模型文件

# 输出文件
rf-model/model_{股票代码}.pkl          # 训练好的模型
rf-model/features_{股票代码}.pkl       # 特征列表
rf-model/model_info_{股票代码}.json    # 模型信息
reports/feature_importance_{股票代码}.png  # 特征重要性图
reports/confusion_matrix_{股票代码}.png    # 混淆矩阵
reports/roc_curve_{股票代码}.png          # ROC曲线
```

#### 📈 train_multi_period.py - 多周期训练器
```python
# 主要功能
- 基于相同技术指标训练多个时间周期模型
- 支持5天、20天预测（可配置）
- 不同周期使用不同涨跌阈值
- 自动保存到multi-period子目录

# 输出文件
rf-model/multi-period/model_5d_{股票代码}.pkl    # 5天预测模型
rf-model/multi-period/model_20d_{股票代码}.pkl   # 20天预测模型
rf-model/multi-period/features_5d_{股票代码}.pkl # 5天模型特征
rf-model/multi-period/features_20d_{股票代码}.pkl # 20天模型特征
```

#### 🔮 predict.py - 统一预测系统
```python
# 主要功能
- 自动检测可用模型（1天、5天、20天）
- 获取最新股票数据并计算技术指标
- 执行多周期预测并格式化输出
- 提供置信度评级和综合分析

# 预测流程
1. 读取config.json获取股票代码
2. 检查模型文件完整性
3. 获取最近60天数据
4. 计算技术指标
5. 执行1天、5天、20天预测
6. 输出格式化结果
```

#### 🛠️ use_trained_model.py - 模型使用类库
```python
# StockPredictor类功能
- 封装模型加载和预测逻辑
- 支持单只股票和批量预测
- 自动处理数据获取和技术指标计算
- 提供详细的预测结果

# 使用示例
predictor = StockPredictor('000063.SZ')
result = predictor.predict_single()
```

#### 🔧 model_manager.py - 模型管理工具
```python
# 主要功能
- 列出所有可用模型文件
- 显示模型详细信息
- 验证模型完整性
- 模型备份和清理

# 使用示例
manager = ModelManager('rf-model', '000063.SZ')
manager.list_models()
manager.show_model_info()
manager.validate_model()
```

### 📂 目录结构

```
Stock-TA-Predict/
├── 📄 config.json                    # 系统配置文件
├── 📄 requirements.txt               # Python依赖包
├── 📄 README.md                      # 项目文档
├── 🐍 stock_prediction.py            # 单日模型训练
├── 🐍 train_multi_period.py          # 多周期模型训练
├── 🐍 predict.py                     # 统一预测系统
├── 🐍 use_trained_model.py           # 模型使用类库
├── 🐍 model_manager.py               # 模型管理工具
├── 📁 rf-model/                      # 模型文件目录
│   ├── model_{股票代码}.pkl          # 单日预测模型
│   ├── features_{股票代码}.pkl       # 特征列表
│   ├── model_info_{股票代码}.json    # 模型信息
│   └── 📁 multi-period/              # 多周期模型
│       ├── model_5d_{股票代码}.pkl   # 5天预测模型
│       ├── model_20d_{股票代码}.pkl  # 20天预测模型
│       ├── features_5d_{股票代码}.pkl # 5天模型特征
│       └── features_20d_{股票代码}.pkl # 20天模型特征
├── 📁 data/                          # 原始数据目录
│   └── {股票代码}_daily.csv          # 股票日线数据
├── 📁 reports/                       # 可视化报告
│   ├── feature_importance_{股票代码}.png # 特征重要性图
│   ├── confusion_matrix_{股票代码}.png   # 混淆矩阵
│   └── roc_curve_{股票代码}.png         # ROC曲线
└── 📁 paper/                         # 研究文档
```

### ⚙️ 配置文件说明

#### config.json 核心配置
```json
{
    "data_config": {
        "stock_code": "000063.SZ",      // 目标股票代码
        "stock_name": "中兴通讯",        // 股票名称
        "start_date": "20210101",       // 训练数据开始日期
        "end_date": "20250101",         // 训练数据结束日期
        "api_token": "your_token"       // Tushare API密钥
    },
    "model_config": {
        "threshold": 0.0025,            // 单日涨跌判断阈值(0.25%)
        "test_size": 0.1,               // 测试集比例
        "grid_search_params": {...}     // 网格搜索参数
    },
    "unified_predictor": {
        "period_thresholds": {
            "1": 0.0025,                // 1天: 0.25%阈值
            "5": 0.01,                  // 5天: 1.0%阈值  
            "20": 0.03                  // 20天: 3.0%阈值
        }
    }
}
```

## 🔮 模型预测使用

### 预测流程详解

#### 第1步：数据准备
```python
# 获取最近60天数据（约40个交易日）
df = get_recent_data(stock_code='000063.SZ', days=60)
```

#### 第2步：技术指标计算
```python
# 计算13个技术指标
df = calculate_technical_indicators(df)
# 结果：7行有效数据（扣除指标计算需要的历史期）
```

#### 第3步：特征向量构建
```python
# 提取最新交易日的特征
latest_data = df.iloc[-1]  # 最新一行
X_pred = [31.20, 481139.91, 0.45, 0.60, 30.95, 30.78, 
          38.125, -0.1523, 31.07, 30.89, -0.2308, -0.1876, -0.0432]
# shape: (1, 13) - 1行13列的特征向量
```

#### 第4步：模型预测
```python
# 随机森林投票机制（以10棵树为例）
Tree1.predict(X_pred) = 0  # 下跌
Tree2.predict(X_pred) = 0  # 下跌  
Tree3.predict(X_pred) = 1  # 上涨
Tree4.predict(X_pred) = 0  # 下跌
...
Tree10.predict(X_pred) = 0 # 下跌

# 统计投票：1票上涨，9票下跌
prediction = 0  # 下跌
probability = [0.882, 0.118]  # [下跌概率, 上涨概率]
```

#### 第5步：置信度评级
```python
confidence = max(probability) = 0.882 = 88.2%

# 评级系统
if confidence >= 0.8:   rating = "很高 ⭐⭐⭐"    # 88.2%
elif confidence >= 0.7: rating = "较高 ⭐⭐"
elif confidence >= 0.6: rating = "中等 ⭐"
else:                   rating = "较低"
```

#### 第6步：结果输出
```python
==================================================
股票代码: 000063.SZ
股票名称: 中兴通讯
最新价格: ¥31.20
预测方向: 下跌
上涨概率: 11.80%
下跌概率: 88.20%
预测置信度: 88.20%
置信度评级: 很高 ⭐⭐⭐
==================================================
```

### 快速预测命令
```bash
# 单日预测
python predict.py

# 交互式预测
python quick_predict.py

# 使用训练好的模型
python use_trained_model.py
```

## 🔄 多周期预测对比

### 数据使用方式
**当前实现**：日线数据 + 多天标签
```python
# 相同的数据源和技术指标
data_source = 日线数据(daily)
indicators = 基于日线计算的RSI, MACD等

# 不同的预测标签  
1天标签 = (1天后价格 - 当前价格) / 当前价格 >= 0.25%
5天标签 = (5天后价格 - 当前价格) / 当前价格 >= 1.0%
20天标签 = (20天后价格 - 当前价格) / 当前价格 >= 3.0%
```

**理论最优**：周线数据 + 周线指标
```python
# 不同时间框架的数据和指标
1天预测 → 日线数据 + 日线指标
5天预测 → 周线数据 + 周线指标  
20天预测 → 月线数据 + 月线指标
```

### 预测结果示例
```python
多周期预测：中兴通讯 (000063.SZ)
==================================================
📈 1天预测:  下跌 (88.2%) ⭐⭐⭐ 很高信心
📊 5天预测:  下跌 (65.0%) ⭐    中等信心
📈 20天预测: 上涨 (55.0%) ⭐    中等信心

💡 综合判断: 短期调整，长期可能反弹
==================================================
```

## 📊 模型性能

### 训练结果示例（中兴通讯）
```
训练数据: 2022-01-04 至 2025-01-15
有效样本: 776条 (扣除技术指标计算期)
数据分割: 90%训练 + 10%测试
最优参数: 深度6, 树数10, 叶子样本20
模型准确率: 85.9%
```

### 特征重要性
- **O-C差值**最关键，占46%权重
- **RSI技术指标**次之，占14%权重
- **成交量信息**排第三，占8%权重

## 🚀 快速开始

### 1. 环境配置
```bash
# 安装依赖包
pip install -r requirements.txt
```

### 2. 配置API
编辑 `config.json` 文件：
```json
{
    "data_config": {
        "stock_code": "000063.SZ",      // 修改为目标股票代码
        "stock_name": "中兴通讯",        // 修改为股票名称
        "api_token": "your_tushare_token"  // 替换为你的Tushare API密钥
    }
}
```

### 3. 训练模型
```bash
# 第一步：训练单日预测模型
python stock_prediction.py

# 第二步：训练多周期预测模型
python train_multi_period.py
```

### 4. 执行预测
```bash
# 统一预测系统（推荐）
python predict.py

# 或使用模型类库
python use_trained_model.py
```

### 5. 模型管理
```bash
# 查看模型信息
python model_manager.py
```

## 📋 完整使用流程

### 🔄 从零开始的完整流程

#### Step 1: 项目初始化
```bash
# 克隆项目
git clone <repository_url>
cd Stock-TA-Predict

# 安装依赖
pip install -r requirements.txt
```

#### Step 2: 配置股票和API
```bash
# 编辑配置文件
notepad config.json  # Windows
# 或
vim config.json      # Linux/Mac

# 修改以下字段：
# - stock_code: 目标股票代码（如 "000001.SZ"）
# - stock_name: 股票名称（如 "平安银行"）
# - api_token: 你的Tushare Pro API密钥
```

#### Step 3: 数据获取和模型训练
```bash
# 训练单日预测模型（约2-5分钟）
python stock_prediction.py

# 输出文件：
# ✅ rf-model/model_{股票代码}.pkl
# ✅ rf-model/features_{股票代码}.pkl  
# ✅ rf-model/model_info_{股票代码}.json
# ✅ reports/feature_importance_{股票代码}.png
# ✅ reports/confusion_matrix_{股票代码}.png
# ✅ reports/roc_curve_{股票代码}.png
```

```bash
# 训练多周期预测模型（约3-8分钟）
python train_multi_period.py

# 输出文件：
# ✅ rf-model/multi-period/model_5d_{股票代码}.pkl
# ✅ rf-model/multi-period/model_20d_{股票代码}.pkl
# ✅ rf-model/multi-period/features_5d_{股票代码}.pkl
# ✅ rf-model/multi-period/features_20d_{股票代码}.pkl
```

#### Step 4: 验证模型完整性
```bash
# 检查模型文件
python model_manager.py

# 预期输出：
# ✅ 模型文件 (3个): model_*.pkl, features_*.pkl, model_info_*.json
# ✅ 多周期模型 (4个): model_5d_*.pkl, model_20d_*.pkl, features_5d_*.pkl, features_20d_*.pkl
```

#### Step 5: 执行预测
```bash
# 多周期预测（推荐）
python predict.py

# 预期输出：
# 📈 1天预测:  下跌 (88.2%) ⭐⭐⭐ 很高信心
# 📊 5天预测:  下跌 (65.0%) ⭐    中等信心  
# 📈 20天预测: 上涨 (55.0%) ⭐    中等信心
```

### 🔧 日常使用流程

#### 每日预测流程（推荐时间：16:30-17:30）
```bash
# 1. 进入项目目录
cd Stock-TA-Predict

# 2. 执行预测
python predict.py

# 3. 查看结果并记录
# 建议保存预测结果用于后续验证准确率
```

#### 模型更新流程（建议每月一次）
```bash
# 1. 更新训练数据时间范围
# 编辑config.json中的end_date为当前日期

# 2. 重新训练模型
python stock_prediction.py
python train_multi_period.py

# 3. 验证新模型
python model_manager.py
```

#### 切换股票流程
```bash
# 1. 修改配置文件
# 编辑config.json中的stock_code和stock_name

# 2. 训练新股票模型
python stock_prediction.py
python train_multi_period.py

# 3. 执行预测
python predict.py
```

### 🛠️ 故障排除

#### 常见问题解决
```bash
# 问题1: API token无效
# 解决: 检查config.json中的api_token是否正确

# 问题2: 模型文件不存在
# 解决: 重新运行训练脚本
python stock_prediction.py
python train_multi_period.py

# 问题3: 数据获取失败
# 解决: 检查网络连接和股票代码格式

# 问题4: 预测结果异常
# 解决: 验证模型完整性
python model_manager.py
```

## ⚠️ 重要注意事项

### 时间泄漏风险
- **问题**：训练数据包含过于接近预测日期的数据
- **症状**：模型准确率虚高，实际效果差
- **解决**：确保训练数据截止到预测日前足够时间

### 模型局限性
- 仅基于技术指标，未考虑基本面
- 市场突发事件无法预测
- 历史规律可能在未来失效

### 投资风险
- **预测结果仅供参考，不构成投资建议**
- **股市有风险，投资需谨慎**
- **建议结合其他分析方法**

## 📞 技术支持

如有问题，请检查：
1. Tushare API token是否有效
2. 网络连接是否正常
3. 数据时间范围是否合理
4. 模型文件是否存在

## ⏰ 最佳运行时间

### 🎯 推荐运行时间
**每个交易日下午 16:30-17:30** 是黄金运行时间：

```
15:00 ▶️ A股收盘
15:30 ▶️ 数据开始更新  
16:00 ▶️ 当日数据基本可用
16:30 ▶️ 🏆 黄金运行时间
17:30 ▶️ 数据完全稳定
```

### 📅 时间安排表

| 时间段 | 状态 | 建议用途 |
|--------|------|----------|
| **16:30-17:30** | 🏆 最佳 | 当日数据已更新，预测明日走势 |
| **19:00-21:00** | ✅ 推荐 | 数据最稳定，适合深度分析 |
| **08:00-09:00** | ⚠️ 可选 | 基于前日数据，开盘前参考 |
| **交易时间内** | ❌ 避免 | 数据不完整，预测不准确 |

### 🗓️ 周期性运行计划
```python
工作日: 16:30 - 预测明日走势
周五晚: 19:00 - 周末深度分析  
周日晚: 20:00 - 预测下周一开盘
节假日前: 提前一天运行
```

## 🔍 收盘后预测机制详解

### 数据完整性原理
```
今日 15:00 ✅ A股收盘 - 完整OHLCV数据
今日 16:00 ✅ 当日数据可用 - 技术指标可计算
今日 16:30 🔮 运行预测 - 基于完整数据
明日 09:30 📈 验证结果 - 预测准确性检验
```

### 预测步骤分解

#### Step 1: 今日数据完整性
```
收盘后获得今日完整数据：
- 开盘价: 30.75
- 最高价: 31.45  
- 最低价: 30.60
- 收盘价: 31.20 ✅ 关键预测基础
- 成交量: 481139 手
```

#### Step 2: 实时技术指标计算
```
基于今日收盘数据计算最新指标：
- RSI(14) = 38.1  ← 包含今日价格
- MACD = -0.23    ← 最新趋势信号
- MA5 = 30.8      ← 含今日均价
- O-C = 0.45      ← 今日多空力量
```

#### Step 3: 模型预测逻辑
```python
# 模型的"经验判断"
训练阶段学到的历史模式:
if RSI < 40 and MACD < 0 and O-C > 0:
    次日下跌概率 = 82%  # 基于776个历史样本

今日指标完全符合：
RSI=38.1 ✅, MACD=-0.23 ✅, O-C=0.45 ✅
→ 预测明日下跌概率 82% ⭐⭐⭐
```

### 🧠 为什么收盘后能预测明日？

#### 技术分析核心假设
1. **历史重演性**: 相似的技术形态会产生相似结果
2. **市场情绪延续**: 今日的技术状态反映明日开盘前的市场预期
3. **数据完整性**: 收盘后拥有当日最完整的市场信息

#### 预测有效时间窗口
```
16:30 预测 → 次日 09:30 开盘
有效窗口: 17小时
涵盖影响:
- 隔夜外盘变化
- 重要消息面
- 技术形态延续
- 投资者情绪
```

#### 实际预测案例
```
今日(2025-05-23)收盘后预测：

输入（今日技术状态）:
✓ 收盘价: 31.20元
✓ RSI: 38.1 (偏空区域)
✓ MACD: -0.23 (负值下行)
✓ 开收差: +0.45 (上影线)

输出（明日概率）:
📉 上涨概率: 17.96%
📈 下跌概率: 82.04% ⭐⭐⭐

模型判断: 明日(2025-05-24)看跌
理由: 当前技术指标组合在历史中82%情况下次日下跌
```

### 💡 预测本质理解

**不是预知未来，而是概率统计**：
- 📊 **数据基础**: 当前完整的技术状态
- 🧮 **计算方法**: 与历史相似情况对比
- 📈 **输出结果**: 统计学意义上的概率分布
- 🎯 **准确率**: 类似天气预报，70-85%准确度

**类比天气预报**：
```
气象预报: 看今天气象数据 → 预测明天降雨概率
股票预测: 看今日技术指标 → 预测明日涨跌概率
```

这就是为什么**收盘后运行是最佳选择** - 数据最完整，预测最可靠！

---

*本系统仅用于技术研究和学习目的* 