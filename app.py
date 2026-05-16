from fastapi import FastAPI
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
from binance.client import Client
import requests
import os

app = FastAPI()

# --- CONFIG & SECRETS ---
SUPA_URL = "https://nzpzddyjcthhzkyfrjpb.supabase.co"
SUPA_KEY = os.getenv("SUPABASE_KEY")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BIN_KEY = os.getenv("BINANCE_API_KEY")
BIN_SEC = os.getenv("BINANCE_SECRET_KEY")

# --- TOOLS ---
def send_telegram(msg):
    if TG_TOKEN and TG_CHAT_ID:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        try: requests.post(url, json={"chat_id": TG_CHAT_ID, "text": msg})
        except: pass

def get_binance_balance():
    if not BIN_KEY or not BIN_SEC:
        return "Key Missing"
    try:
        client = Client(BIN_KEY, BIN_SEC)
        acc = client.get_account()
        balances = {item['asset']: item['free'] for item in acc['balances'] if float(item['free']) > 0}
        # Fokus ke USDT dan BTC
        usdt = balances.get('USDT', '0')
        btc = balances.get('BTC', '0')
        return f"💰 Wallet: {usdt} USDT | {btc} BTC"
    except:
        return "Binance Auth Failed"

def save_to_supabase(payload):
    if not SUPA_KEY: return
    headers = {"apikey": SUPA_KEY, "Authorization": f"Bearer {SUPA_KEY}", "Content-Type": "application/json"}
    try: requests.post(f"{SUPA_URL}/rest/v1/signals", json=payload, headers=headers, timeout=5)
    except: pass

# --- AI LOGIC ---
def run_ai_logic():
    df = yf.download('BTC-USD', period='60d', interval='1d', auto_adjust=True, progress=False)
    if df.empty: raise ValueError("Yahoo Finance Limit")
    
    close_series = df['Close'].iloc[:, 0] if isinstance(df.columns, pd.MultiIndex) else df['Close']
    current_price = float(close_series.iloc[-1])
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_series.to_frame())
    
    lookback = 15
    X = np.array([scaled_data[i-lookback:i, 0] for i in range(lookback, len(scaled_data))])
    X = X.reshape(-1, lookback, 1)
    
    model = Sequential([Input(shape=(lookback, 1)), LSTM(16), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, scaled_data[lookback:], epochs=2, verbose=0)
    
    pred_scaled = model.predict(scaled_data[-lookback:].reshape(1, lookback, 1), verbose=0)
    final_pred = float(scaler.inverse_transform(pred_scaled).flatten()[0])
    
    return current_price, final_pred

# --- ENDPOINTS ---
@app.get("/predict")
def get_prediction():
    try:
        curr, pred = run_ai_logic()
        signal = "BUY 🟢" if pred > curr else "HOLD/SELL 🔴"
        wallet = get_binance_balance()
        
        payload = {"price": curr, "prediction": pred, "fng_index": 50, "signal_type": signal}
        save_to_supabase(payload)
        
        # LOG TO TELEGRAM
        msg = f"🛡️ CHRONOS REPORT\n\nPrice: ${curr:,.2f}\nPred: ${pred:,.2f}\nSignal: {signal}\n\n{wallet}"
        send_telegram(msg)
        
        return {**payload, "wallet": wallet}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def health():
    return {"status": "Chronos Online"}
