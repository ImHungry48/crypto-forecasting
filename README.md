<h1 align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=28&pause=1000&center=true&vCenter=true&width=600&lines=Crypto+Price+Forecasting+with+LSTM" alt="Typing SVG" />
</h1>
<p align="center">Predict the future of Bitcoin prices using deep learning and public market data from Kraken </p>

<p align="center">
  <img src="bitcoin-bitcoin-coaster.gif" width="600"/>
</p>

<p align="center">
  <a href="https://img.shields.io/badge/Python-3.12-blue"><img src="https://img.shields.io/badge/Python-3.12-blue" /></a>
  <a href="#"><img src="https://img.shields.io/badge/LSTM-DeepLearning-orange" /></a>
  <a href="#"><img src="https://img.shields.io/badge/DataSource-Kraken-blueviolet" /></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-green" /></a>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Data Sources](#data-sources)
- [Setup](#setup)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Version Log](#version-log)
- [Built With](#built-with)
- [Future Work](#future-work)

---

## Overview

This project explores how to use sequential models (LSTM) to forecast future **Bitcoin (BTC/USD)** prices based on weekly OHLC (Open, High, Low, Close) data pulled from the **Kraken API**. The model learns temporal patterns to predict the next week's closing price.

---

## Dataset

- **Source**: [Kraken Public OHLC API](https://docs.kraken.com/rest/#operation/getOHLCData)
- **Pair**: `XBTUSD`
- **Interval**: Weekly candles (`10080` minutes)
- **Samples**: 720 entries = ~13.8 years of data

Each row includes:
[time, open, high, low, close, vwap, volume, count]

---

## Model

<table align="center">
  <tr>
    <td align="center" width="300">
      <img src="images/model-icon.png" width="80" /><br />
      <strong>Model</strong><br />
      2-layer LSTM with Dropout<br />
      Forecasting weekly BTC prices
    </td>
    <td align="center" width="300">
      <img src="images/data-icon.png" width="80" /><br />
      <strong>Input</strong><br />
      Last 12 weeks of OHLCV + macro data<br />
      (e.g., interest rate, CPI, EPU)
    </td>
  </tr>
</table>

- **Type**: LSTM Neural Network
- **Input Shape**: (12, 12) - 12 time steps (weeks), 12 features (OHLCV + macro)
- **Target**: Next week's normalized closing price
- **Loss**: MSE (Mean Squared Error)
- **Optimizer**: Adam
- **Regularization**: Dropout(0.2) between stacked LSTMs
- **Callbacks**: EarlyStopping(patience=10)

```python
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(window_size, X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
```

---
## Results
- **Sample output below (actual vs predicted closing prices)**
<p align="center"> <img src="images/actual_vs_predicted.png" width="600"/> </p>


### Observations
- The model captures **broad macro trends** in Bitcoin’s price, especially during stable or modest growth periods.
- However, it tends to **underpredict during sharp upward trends**, likely due to the limitations of U.S.-only macro features.
- Volatility and breakout behavior remain challenging to capture with LSTM alone and without more reactive data (e.g., sentiment or global context).

---

## Setup

1. Clone the repo

```bash
git clone https://github.com/yourusername/crypto-forecasting.git
cd crypto-forecasting
```
2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Fetch data from Kraken
```bash
python fetch_data.py
```

4. Train the model
```bash
python train_model.py
```

---

## Model Evaluation
Initial experiments began with raw closing price prediction using a simple LSTM. In v2, the input was expanded to include OHLCV features and percent returns, introduced dropout regularization, and stacked LSTM layers with early stopping. While RMSE increased slightly, the model became better suited for learning temporal dependencies and market structure.

In v3, U.S. macroeconomic indicators (interest rates, inflation, recession signals, and policy uncertainty) were incorporated. This helped the model capture broad economic trends, but it still struggled with sharp price surges and volatility — suggesting that Bitcoin’s behavior is influenced by **global signals beyond U.S. metrics alone**.

These iterations have shown that:
- LSTM captures trend but not high-volatility breakout behavior
- U.S.-only macro inputs provide some signal, but are **not sufficient**
- Incorporating global data (EU, sentiment, events) and alternative targets (e.g., price direction, % return) may improve predictive accuracy

| Version | Change | RMSE | Learned |
|---------|--------|------|---------|
| v1 | Raw closing price prediction | 5,452.00 | Baseline model. Simple LSTM with just price failed to capture patterns well, underfit most of the trend. |
| v2 | + OHLCV features, + percent return, + stacked LSTMs (128→64) + Dropout(0.2), + EarlyStopping, fixed inverse scaling | 7,348.37 | More features improved structure, but the model still struggled with volatility and sharp directional shifts. More complexity doesn't always mean better performance. |
| v3 | + U.S. macro data (interest rates, inflation, recession, uncertainty index) | 12,446.00 | U.S.-only macro context isn't enough — Bitcoin reacts to global signals. Model learned general trends but underperformed during rapid growth phases. |

**Next Steps (v4):**
- Incorporate **Eurozone macro indicators** (e.g., ECB rates, EU CPI)
- Add sentiment data and event-driven features (e.g., GDELT, ACLED)
- Explore alternative targets: return %, directional classification
- Try advanced architectures: **CNN-LSTM**, **attention**, or **transformers**
---
## 🛠️ Built With

- Python 3.12
- [TensorFlow/Keras](https://www.tensorflow.org/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [fredapi](https://github.com/mortada/fredapi)
- [krakenohlc](https://docs.kraken.com/api/docs/rest-api/get-ohlc-data/)
- Jupyter + VS Code
---
## Future Work

Planned future improvements to enhance accuracy and signal diversity include:

###  Macro & Policy Signals
- European Central Bank rates and EU inflation
- Global government regulations and sanctions
- Real-time ETF approval tracking (e.g., SEC)

### Market & On-Chain Indicators
- Exchange inflows/outflows
- Miner hash rate and network health
- Whale wallet movements and transaction volume
- Blockchain forks and protocol upgrades

### Sentiment & Media Signals
- Social media sentiment (Twitter, Reddit, Google Trends)
- News event frequency from GDELT or ACLED
- Corporate adoption and major partnership announcements

### Modeling Enhancements
- Predictive targets beyond price: % return, direction classification
- Multi-head models for trend + volatility
- Hybrid CNN-LSTM or Transformer-based architectures
- SHAP-based feature attribution and confidence estimation

