import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

df = pd.read_csv(r"data\weekly_btc_ohlc.csv")
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# Calculate percent returns
df['percent_return'] = df['close'].pct_change() * 100

# Drop rows with NaN values (due to pct_change)
df.dropna(inplace=True)

# Use OHLCV features and percent returns as inputs
data = df[['open', 'high', 'low', 'close', 'volume', 'percent_return']].copy()

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Make the sequences
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size, 3]) # index 3 is 'close'
    return np.array(X), np.array(y)

window_size = 12  # 12 weeks (3 months) to predict the next week
X, y = create_sequences(scaled_data, window_size)

# Reshape X for LSTM: (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

# Train/Test Split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Create the LSTM Model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(window_size, X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=10)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stop])

# Create output folder if it doesn't exist
os.makedirs("images", exist_ok=True)

# Save training loss plot
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Training Loss Over Time")
plt.savefig("images/training_loss.png", dpi=300)
plt.close()

# Make predictions and invert scaling
y_pred = model.predict(X_test)

# Get the same close prices used in y_train/y_test
raw_close = df['close'].values[window_size:]  # because you dropped `window_size` rows making sequences

target_scaler = MinMaxScaler()
target_scaler.fit(raw_close[:split].reshape(-1, 1))  # fit only on training portion

# In create_sequences: return X, y (scaled close only)
# In model: use same X, predict scaled close
# Then inverse-transform:
y_pred_inv = target_scaler.inverse_transform(y_pred)
y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1))

print("Predicted:", y_pred_inv[:5].flatten())
print("Actual:   ", y_test_inv[:5].flatten())

# Save Predictions vs Actual
plt.figure()
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.legend()
plt.title("Actual vs Predicted Closing Prices")
plt.savefig("images/actual_vs_predicted.png", dpi=300)
plt.close()

# Evalutation
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import json

mse = mean_squared_error(y_test_inv, y_pred_inv)
mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mse)

# Save metrics to a JSON file
metrics = {
    "MSE": round(float(mse), 4),
    "RMSE": round(float(rmse), 4),
    "MAE": round(float(mae), 4)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)