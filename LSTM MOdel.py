import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

# Define the ticker symbol and date range
ticker = 'AAPL'
start = '2020-01-01'
end = '2023-12-31'

# Fetch the data using yfinance
df = yf.download(ticker, start=start, end=end)

# Reset index to get a numerical index
df = df.reset_index()

# Split the data into training and testing sets
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

# Scale the training data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Prepare the training data
X_train = []
y_train = []
for i in range(100, len(data_training_array)):
    X_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Prepare the testing data
# Append the last 100 days of the training data to the testing data
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)

# Scale the combined data
input_data = scaler.transform(final_df)

# Prepare test sequences
X_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    X_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

# Make predictions
y_predicted = model.predict(X_test)

# Inverse scaling for predicted values
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor

# Inverse scaling for y_test
y_test = y_test * scale_factor

print("y_test shape:", y_test.shape)
print("y_predicted shape:", y_predicted.shape)

# result
plt.figure(figsize=(12, 6))
plt.plot(y_test, color='blue', label='Actual Price')
plt.plot(y_predicted, color='red', label='Predicted Price')
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
