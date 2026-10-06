import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv('Dataset.csv')

# Convert date
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

# Use these columns as features
features = ['Open', 'High', 'Close', 'Volume']
target_column = 'High'  # target column

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

SEQ_LEN = 60
X = []
y = []

for i in range(SEQ_LEN, len(scaled_data)):
    X.append(scaled_data[i-SEQ_LEN:i])
    y.append(scaled_data[i, features.index(target_column)])

X, y = np.array(X), np.array(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=40, batch_size=64, validation_data=(X_test, y_test))

predicted_scaled = model.predict(X_test)

predicted_highs = []
real_highs = []

for i in range(len(predicted_scaled)):
    temp_input = X_test[i, -1].copy()

    # predicted
    temp_input[features.index(target_column)] = predicted_scaled[i].item()
    predicted_high = scaler.inverse_transform([temp_input])[0][features.index(target_column)]
    predicted_highs.append(predicted_high)

    # actual
    temp_input[features.index(target_column)] = y_test[i].item()
    actual_high = scaler.inverse_transform([temp_input])[0][features.index(target_column)]
    real_highs.append(actual_high)

plt.figure(figsize=(12, 6))
plt.plot(real_highs, label='Actual High Price', color='green')
plt.plot(predicted_highs, label='Predicted High Price', color='orange')
plt.title('Predicted vs Actual High Price', fontsize=16)
plt.xlabel('Days')
plt.ylabel('High Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

error = np.array(real_highs) - np.array(predicted_highs)

plt.figure(figsize=(8, 5))
plt.hist(error, bins=30, color='purple', edgecolor='black')
plt.title('Distribution of Prediction Errors', fontsize=14)
plt.xlabel('Error ($)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(real_highs, predicted_highs)
rmse = np.sqrt(mse)
mae = mean_absolute_error(real_highs, predicted_highs)
r2 = r2_score(real_highs, predicted_highs)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# Baseline: naive forecast (tomorrow's High = today's High) on the same test set
naive_highs = []
for i in range(len(X_test)):
    temp_input = X_test[i, -1].copy()
    naive_highs.append(scaler.inverse_transform([temp_input])[0][features.index(target_column)])

n_mae = mean_absolute_error(real_highs, naive_highs)
n_rmse = np.sqrt(mean_squared_error(real_highs, naive_highs))
n_r2 = r2_score(real_highs, naive_highs)

print("Naive baseline (previous day's High)")
print(f"MAE  : {n_mae:.2f}")
print(f"RMSE : {n_rmse:.2f}")
print(f"R2   : {n_r2:.4f}")
