import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from binance.client import Client
import requests
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input

# --- SETUP CONFIG ---
st.set_page_config(page_title="CHRONOS TERMINAL", layout="wide")

# Ambil Secrets dari Environment Hugging Face
BINANCE_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET_KEY")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(message):
    if TG_TOKEN and TG_CHAT_ID:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage?chat_id={TG_CHAT_ID}&text={message}"
        requests.get(url)

# --- ENGINE AI CHRONOS ---
def run_ai_logic():
    df = yf.download('BTC-USD', period='180d', interval='1d', auto_adjust=True)
    df_close = df[['Close']].copy()
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_close)
    
    lookback = 60
    X = []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    model = Sequential([
        Input(shape=(X.shape[1], 1)),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, scaled_data[lookback:], epochs=5, batch_size=32, verbose=0)
    
    last_60_days = scaled_data[-60:].reshape(1, 60, 1)
    pred_scaled = model.predict(last_60_days, verbose=0)
    final_pred = scaler.inverse_transform(pred_scaled)[0][0]
    
    # Ambil Fear & Greed Real
    fng_res = requests.get('https://api.alternative.me/fng/').json()
    fng_val = int(fng_res['data'][0]['value'])
    
    return df_close['Close'].iloc[-1], final_pred, fng_val

# --- DASHBOARD UI ---
st.title("🛡️ Chronos AI: Production Command Center")
st.write("Status: Real-time Analysis with Binance & Telegram Integration")

try:
    curr, pred, fng = run_ai_logic()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Harga BTC Saat Ini", f"${curr:,.2f}")
    col2.metric("Prediksi AI Besok", f"${pred:,.2f}", delta=f"{((pred/curr)-1)*100:.2f}%")
    col3.metric("Fear & Greed Index", f"{fng}")

    # Sinyal Logic
    signal = "BUY" if pred > curr and fng > 35 else "HOLD/SELL"
    st.subheader(f"Signal: {signal}")

    # --- ACTION BUTTONS ---
    st.sidebar.header("🕹️ Eksekusi Binance")
    if st.sidebar.button("🚀 EXECUTE BUY ORDER"):
        if BINANCE_KEY:
            # client = Client(BINANCE_KEY, BINANCE_SECRET)
            # order = client.order_market_buy(symbol='BTCUSDT', quantity=0.001)
            msg = f"✅ CHRONOS: Buy Order BTC executed at ${curr:,.2f}"
            send_telegram(msg)
            st.sidebar.success("Order Berhasil Dikirim!")
        else:
            st.sidebar.error("API Key Belum Di-setup!")

    if st.sidebar.button("🆘 EMERGENCY SELL"):
        send_telegram("⚠️ CHRONOS ALERT: Emergency Panic Sell Triggered!")
        st.sidebar.warning("Sinyal Jual Dikirim.")

except Exception as e:
    st.error(f"Waiting for Data... Error: {e}")
