# Stock Price Prediction Model

This project implements a machine learning model to predict stock price movements (rise or fall) for BYD (Stock Code: 002594.SZ) using historical data and technical indicators.

## Requirements

- Python 3.8 or higher
- Dependencies listed in requirements.txt

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Note: TA-Lib may require additional steps for installation depending on your operating system:
   - Windows: You might need to download and install a pre-built wheel file
   - Linux: You might need to install the development library: `sudo apt-get install ta-lib`
   - MacOS: You might need to install via Homebrew: `brew install ta-lib`

## Project Overview

This project predicts whether a stock price will rise or fall using a Random Forest classifier. The implementation includes:

1. **Data Acquisition**: Historical stock data for BYD (002594.SZ) is retrieved using the Tushare API
2. **Feature Engineering**: Creation of derived variables and technical indicators
3. **Model Training**: Implementation of a Random Forest classifier
4. **Model Optimization**: Hyperparameter tuning using GridSearchCV
5. **Model Evaluation**: Assessment of model performance and feature importance analysis
6. **Visualization**: Generation of performance charts and feature importance plots

## Implementation Process

### 1. Data Collection

- Historical stock data (2016-2023) is retrieved using Tushare API
- Data includes trading dates, opening/closing prices, high/low prices, trading volume, etc.

### 2. Feature Engineering

The following features are created:
- Basic price differences (Open-Close, High-Low)
- Price changes and percentage changes
- Moving averages (5-day, 10-day)
- Technical indicators using TA-Lib:
  - RSI (Relative Strength Index)
  - Momentum
  - EMA (Exponential Moving Average)
  - MACD (Moving Average Convergence Divergence)

### 3. Model Training

- Data is split into training (90%) and testing (10%) sets
- A Random Forest Classifier is implemented with the following initial parameters:
  - max_depth = 3
  - n_estimators = 10
  - min_samples_leaf = 10

### 4. Hyperparameter Tuning

GridSearchCV is used to find optimal parameters among:
- n_estimators: [5, 10, 20]
- max_depth: [2, 3, 4, 5, 6]
- min_samples_leaf: [5, 10, 20, 30]

### 5. Model Evaluation

- Accuracy assessment on test data
- Feature importance analysis
- Confusion matrix visualization
- ROC curve analysis

## Usage

To run the prediction model:

```bash
python stock_prediction.py
```

## Results

The model achieves good prediction performance, with the following outcomes:
- Feature importance analysis reveals that O-C, volume, MACD histogram, RSI, and other technical indicators are significant predictors
- Visualization outputs include:
  - Feature importance chart
  - Confusion matrix
  - ROC curve

## Notes

- The Tushare API requires a token for authentication, which is included in the code
- The prediction target is whether the stock price will rise by at least 0.25% the next day
- The model's random state is fixed to ensure reproducibility

## Future Improvements

Potential enhancements to the model include:
- Incorporating additional features like sentiment analysis from news
- Testing different machine learning algorithms
- Implementing a trading strategy based on predictions
- Expanding to multiple stocks for comparison 