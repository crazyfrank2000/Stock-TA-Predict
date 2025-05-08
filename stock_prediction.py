from csv import field_size_limit
from os import closerange
from cmath import sqrt
import tushare as ts
import numpy as np
import pandas as pd
import talib
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

# 1. Stock basic data acquisition
# BYD's stock code is 002594.SZ. Data is collected and stored in a local csv file
ts.set_token('db1723bf9e9009f186c134b6813da7730256c87f31759400e170ddab')
pro = ts.pro_api()
df = pro.daily(ts_code='002594.SZ', start_date='20160101', end_date='20230630')
# Take the transaction date, open, high, low, close, pre_close pct_chg,
# trading volume and other data of BYD in the past six years
df = df.sort_values('trade_date')
df1 = df.set_index('trade_date')
df1.to_csv('002594.SZ_daily.csv')

# Calculating log returns
df['log_return'] = np.log(df['close'] / df['pre_close'])
df['up'] = np.where(df.log_return >= 0.0025, 1, 0)
df = df.sort_values('trade_date')
df1 = df.set_index('trade_date')

# 2. Simple derived variable data construction
df1['O-C'] = df1['open'] - df1['close']
df1['H-L'] = df1['high'] - df1['low']
df1['pre_close'] = df1['close'].shift(1)
df1['price_change'] = df1['close'] - df1['pre_close']
df1['p_change'] = (df1['close'] - df1['pre_close']) / df1['pre_close'] * 100

# 3. Moving average related data construction
df1['MA5'] = df1['close'].rolling(5).mean()
df1['MA10'] = df1['close'].rolling(10).mean()
df1.dropna(inplace=True)

# 4. Construct derived variable data through the TA-Lib library
df1['RSI'] = talib.RSI(df1.close.values, timeperiod=14)
df1['MOM'] = talib.MOM(df1.close.values, timeperiod=5)
df1['EMA12'] = talib.EMA(df1.close.values, timeperiod=12)  # 12-day moving average
df1['EMA26'] = talib.EMA(df1.close.values, timeperiod=26)  # 26-day moving average
df1['MACD'], df1['MACDsignal'], df1['MACDhist'] = talib.MACD(df1.close.values, fastperiod=6, slowperiod=12, signalperiod=9)
df1.dropna(inplace=True)

# Feature selection
X = df1[['close', 'vol', 'O-C', 'MA5', 'MA10', 'H-L', 'RSI', 'MOM', 'EMA12', 'MACD', 'MACDsignal', 'MACDhist']]
y = np.where(df1.log_return >= 0.0025, 1, 0)

# Dividing the overall data into training and testing sets,
# with training sets accounting for 90% and testing sets accounting for 10%.
X_length = X.shape[0]
split = int(X_length * 0.9)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Model building
# Set model parameters: max_depth of the decision tree is set to 3, that is, each decision tree has only 3 layers at most.
# The number of weak learners (i.e., decision tree model) n_estimators is set to 10, that is,
# there are 10 decision trees in the random forest. The minimum sample number of leaf nodes is set to 10, that is,
# if the sample number of leaf nodes is less than 10, the splitting stops.
# parameter random_state is to make the results consistent.
model = RandomForestClassifier(max_depth=3, n_estimators=10, min_samples_leaf=10, random_state=120)
model.fit(X_train, y_train)

# Model evaluation and use, the model is evaluated and used to predict the rise and fall of the stock price the next day
y_pred = model.predict(X_test)
a = pd.DataFrame()
a["prediction"] = list(y_pred)
a["actual"] = list(y_test)

# Model accuracy evaluation
accuracy = accuracy_score(y_pred, y_test)
print(f"Accuracy: {accuracy}")
score = model.score(X_test, y_test)
print(f"Model score: {score}")

# Analyze the characteristic importance of characteristic variables
importances = model.feature_importances_
a = pd.DataFrame()
a["features"] = X.columns
a["importance of features"] = importances
a = a.sort_values("importance of features", ascending=False)
print("\nFeature importance:")
print(a)
print("\nIt is found that the feature variables such as O-C, vol, MACD_hist, RSI, H-L, MOM, MA10, EMA12, MACD, MA5 indicators have a great")
print("influence on the prediction accuracy of the rise and fall of the stock price in the next day")

# Model parameter tuning
parameters = {'n_estimators': [5, 10, 20], 'max_depth': [2, 3, 4, 5, 6], 'min_samples_leaf': [5, 10, 20, 30]}
new_model = RandomForestClassifier(random_state=120)
grid_search = GridSearchCV(new_model, parameters, cv=6, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("\nBest parameters after grid search:")
print(best_params)

# Create the optimized model with the best parameters
optimized_model = RandomForestClassifier(
    max_depth=best_params['max_depth'],
    n_estimators=best_params['n_estimators'],
    min_samples_leaf=best_params['min_samples_leaf'],
    random_state=120
)
optimized_model.fit(X_train, y_train)
optimized_y_pred = optimized_model.predict(X_test)
optimized_accuracy = accuracy_score(optimized_y_pred, y_test)
print(f"\nOptimized model accuracy: {optimized_accuracy}")

# Plot feature importance
plt.figure(figsize=(10, 6))
sorted_idx = np.argsort(importances)
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx])
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')

# Plot confusion matrix
y_pred_optimized = optimized_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_optimized)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
plt.figure(figsize=(8, 8))
disp.plot()
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

# Plot ROC curve
plt.figure(figsize=(8, 8))
y_score = optimized_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')

print("\nAnalysis complete. Check the generated visualization files.") 