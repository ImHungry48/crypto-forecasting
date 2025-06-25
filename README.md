<h1 align="center">Crypto Price Forecasting with LSTM</h1>
<p align="center">Predict the future of Bitcoin prices using deep learning and public market data from Kraken </p>

<p align="center">
  <img src="images/crypto-banner.png" width="600"/>
</p>

<p align="center">
  <a href="https://img.shields.io/badge/Python-3.12-blue"><img src="https://img.shields.io/badge/Python-3.12-blue" /></a>
  <a href="#"><img src="https://img.shields.io/badge/LSTM-DeepLearning-orange" /></a>
  <a href="#"><img src="https://img.shields.io/badge/DataSource-Kraken-blueviolet" /></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-green" /></a>
</p>

---

## 📚 Table of Contents

- [🚀 Overview](#-overview)
- [📦 Dataset](#-dataset)
- [🧠 Model](#-model)
- [📈 Results](#-results)
- [🛠 Setup](#-setup)
- [🧰 Built With](#-built-with)
- [✅ Future Work](#-future-work)
- [🙌 Acknowledgements](#-acknowledgements)
- [📜 License](#-license)

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

- **Type**: LSTM Neural Network
- **Input**: Last 12 weeks of closing prices
- **Target**: Next week’s closing price
- **Loss**: MSE (Mean Squared Error)
- **Optimizer**: Adam

```python
model = Sequential()
model.add(LSTM(64, input_shape=(12, 1)))
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

  ### Next Improvements
- Incorporate **Eurozone macro indicators** (e.g., ECB rates, EU CPI)
- Add sentiment data and event-driven features (e.g., GDELT, ACLED)
- Explore alternative targets: return %, directional classification
- Try advanced architectures: **CNN-LSTM**, **attention**, or **transformers**

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
- Add EU-based macro indicators (ECB rates, Eurozone inflation)
- Experiment with alternative targets (return, direction classification)
- Add sentiment or geopolitical event data (e.g. GDELT, Twitter)
- Try more expressive architectures (e.g., CNN-LSTM, attention)
