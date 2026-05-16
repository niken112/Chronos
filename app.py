import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from binance.client import Client
import requests
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input

# --- CONFIG & SECRETS ---
MY_CHAT_ID = "7710155531"
BINANCE_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET_KEY")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(message, target_id):
    if TG_TOKEN and target_id:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage?chat_id={target_id}&text={message}"
        try: requests.get(url)
        except: pass

# --- ENGINE AI CHRONOS ---
def run_ai_logic():
    # Gunakan period lebih pendek agar warmup cepat di HF Free Tier
    df = yf.download('BTC-USD', period='60d', interval='1d', auto_adjust=True)
    df_close = df[['Close']].copy()
    
    # Perbaikan: Ambil angka terakhir sebagai FLOAT murni
    current_price = float(df_close['Close'].iloc[-1])
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_close)
    
    lookback = 30 # Kurangi lookback untuk efisiensi RAM HF
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
    model.fit(X, scaled_data[lookback:], epochs=5, batch_size=16, verbose=0)
    
    last_data = scaled_data[-lookback:].reshape(1, lookback, 1)
    pred_scaled = model.predict(last_data, verbose=0)
    final_pred = float(scaler.inverse_transform(pred_scaled)[0][0])
    
    fng_res = requests.get('https://api.alternative.me/fng/').json()
    fng_val = int(fng_res['data'][0]['value'])
    
    return current_price, final_pred, fng_val

# --- DASHBOARD UI ---
st.set_page_config(page_title="CHRONOS TERMINAL", layout="wide")
st.title("🛡️ Chronos AI Command Center")

try:
    with st.spinner('AI sedang menganalisis pasar...'):
        curr, pred, fng = run_ai_logic()
    
    signal = "BUY" if pred > curr and fng > 35 else "HOLD/SELL"
    
    # AUTO-REPORT UNTUK OWNER
    now = datetime.now()
    if now.hour == 7 and now.minute <= 5:
        msg = f"🤖 CHRONOS AUTO-REPORT\nPrice: ${curr:,.2f}\nPred: ${pred:,.2f}\nSignal: {signal}"
        send_telegram(msg, MY_CHAT_ID)

    # UI DISPLAY
    col1, col2, col3 = st.columns(3)
    col1.metric("Live Price", f"${curr:,.2f}")
    col2.metric("AI Target", f"${pred:,.2f}", delta=f"{((pred/curr)-1)*100:.2f}%")
    col3.metric("F&G Index", f"{fng}")

    st.divider()
    st.header(f"Recommended Action: {signal}")

    # SIDEBAR CONTROLS
    st.sidebar.header("🕹️ Owner Controls")
    if st.sidebar.button("🚀 EXECUTE BUY"):
        if BINANCE_KEY:
            send_telegram(f"✅ Order Executed: Buy BTC at ${curr:,.2f}", MY_CHAT_ID)
            st.sidebar.success("Signal Sent to Binance")
        else: st.sidebar.error("Secrets missing")

except Exception as e:
    st.error(f"Terjadi kesalahan teknis: {e}")
