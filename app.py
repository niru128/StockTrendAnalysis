import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from keras.models import load_model
import streamlit as st
import pandas_datareader as data

start = '2020-01-01'
end = '2023-12-31'
st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start=start, end=end)

# Describing data
st.subheader('Data from 2020-2024')
st.write(df.describe())

# VISUALISATION

st.subheader('Closing Price vs Time Chart')
fig1 = plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
st.pyplot(fig1)

st.subheader('Closing Price vs Time Chart with 100 Moving Average')
ma100 = df['Close'].rolling(100).mean()
fig2 = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Close Price')
plt.plot(ma100, label='100-day MA', color='orange')
plt.legend()
st.pyplot(fig2)

st.subheader('Closing Price vs Time Chart with 100 Moving Average and 200 Moving Average')
ma200 = df['Close'].rolling(200).mean()
fig3 = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Close Price')
plt.plot(ma100, label='100-day MA', color='red')
plt.plot(ma200, label='200-day MA', color='green')
plt.legend()
st.pyplot(fig3)
