from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
from binance.client import Client
import requests
import os
import time

app = FastAPI()

# --- CONFIG & SECRETS ---
SUPA_URL = "https://nzpzddyjcthhzkyfrjpb.supabase.co"
SUPA_KEY = os.getenv("SUPABASE_KEY")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BIN_KEY = os.getenv("BINANCE_API_KEY")
BIN_SEC = os.getenv("BINANCE_SECRET_KEY")

def send_telegram(msg):
    if TG_TOKEN and TG_CHAT_ID:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        try: requests.post(url, json={"chat_id": TG_CHAT_ID, "text": msg})
        except: pass

def get_binance_data_and_balance():
    """Ganti Yahoo Finance dengan Binance API yang lebih stabil"""
    if not BIN_KEY or not BIN_SEC:
        return None, None, "API Key Missing"
    try:
        client = Client(BIN_KEY, BIN_SEC)
        
        # Ambil data kline (candlestick) BTCUSDT - 60 hari terakhir, interval 1 hari
        klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1DAY, "60 days ago UTC")
        
        # Format ke DataFrame
        df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'vol', 'close_time', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore'])
        df['close'] = df['close'].astype(float)
        
        # Ambil Saldo
        acc = client.get_account()
        balances = {item['asset']: item['free'] for item in acc['balances'] if float(item['free']) > 0}
        wallet = f"💰 Wallet: {balances.get('USDT', '0')} USDT | {balances.get('BTC', '0')} BTC"
        
        return df[['close']], wallet, None
    except Exception as e:
        return None, None, str(e)

def save_to_supabase(payload):
    if not SUPA_KEY: return
    headers = {"apikey": SUPA_KEY, "Authorization": f"Bearer {SUPA_KEY}", "Content-Type": "application/json"}
    try: requests.post(f"{SUPA_URL}/rest/v1/signals", json=payload, headers=headers, timeout=5)
    except: pass

# --- AI LOGIC ---
def run_ai_logic():
    df, wallet, err = get_binance_data_and_balance()
    if err: raise ValueError(f"Binance Error: {err}")
    
    current_price = float(df['close'].iloc[-1])
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    lookback = 15
    X = np.array([scaled_data[i-lookback:i, 0] for i in range(lookback, len(scaled_data))])
    X = X.reshape(-1, lookback, 1)
    
    model = Sequential([Input(shape=(lookback, 1)), LSTM(16), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, scaled_data[lookback:], epochs=2, verbose=0)
    
    pred_scaled = model.predict(scaled_data[-lookback:].reshape(1, lookback, 1), verbose=0)
    final_pred = float(scaler.inverse_transform(pred_scaled).flatten()[0])
    
    return current_price, final_pred, wallet

@app.get("/predict")
def get_prediction():
    try:
        curr, pred, wallet = run_ai_logic()
        signal = "BUY 🟢" if pred > curr else "HOLD/SELL 🔴"
        
        payload = {"price": curr, "prediction": pred, "fng_index": 50, "signal_type": signal}
        save_to_supabase(payload)
        
        msg = f"🛡️ CHRONOS REPORT (Via Binance)\n\nPrice: ${curr:,.2f}\nPred: ${pred:,.2f}\nSignal: {signal}\n\n{wallet}"
        send_telegram(msg)
        
        return {**payload, "wallet": wallet}
    except Exception as e:
        return {"error": "System Busy", "detail": str(e)}

@app.get("/")
def health():
    return {"status": "Chronos Online", "engine": "Binance Data Stream"}
