import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import os

df = pd.read_csv(r"data\weekly_btc_ohlc.csv")
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# Use only 'close' price for simplicity
data = df[['close']].copy()

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Make the sequences
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

window_size = 12  # 12 weeks (3 months) to predict the next week
X, y = create_sequences(scaled_data, window_size)

# Reshape X for LSTM: (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train/Test Split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Create the LSTM Model
model = Sequential()
model.add(LSTM(64, return_sequences=False, input_shape=(window_size, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

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
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

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