from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
import requests
import os

app = FastAPI()

# --- CONFIG & SECRETS ---
SUPA_URL = "https://nzpzddyjcthhzkyfrjpb.supabase.co"
SUPA_KEY = os.getenv("SUPABASE_KEY")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(msg):
    if TG_TOKEN and TG_CHAT_ID:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        try: requests.post(url, json={"chat_id": TG_CHAT_ID, "text": msg}, timeout=5)
        except: pass

def get_crypto_data_global():
    """Menggunakan Binance Vision API (Jalur Tol Global, Bebas Blokir Wilayah & Anti-Timeout)"""
    try:
        # Menggunakan endpoint api.binance.vision yang ramah buat server US
        url = "https://api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=60"
        res = requests.get(url, timeout=10).json()
        
        # Format data Binance: [[openTime, open, high, low, close, volume, ...], ...]
        if isinstance(res, dict) and "code" in res:
            return None, f"Binance Vision Error: {res.get('msg')}"
            
        # Ambil harga close (index ke-4)
        closes = [float(item[4]) for item in res]
        
        df = pd.DataFrame(closes, columns=['close'])
        return df, None
    except Exception as e:
        return None, str(e)

def save_to_supabase(payload):
    if not SUPA_KEY: return
    headers = {"apikey": SUPA_KEY, "Authorization": f"Bearer {SUPA_KEY}", "Content-Type": "application/json"}
    try: requests.post(f"{SUPA_URL}/rest/v1/signals", json=payload, headers=headers, timeout=5)
    except: pass

@app.get("/predict")
def get_prediction():
    try:
        df, err = get_crypto_data_global()
        if err: 
            return {"error": "Gagal mengambil data market via Jalur Tol Global", "detail": err}
            
        current_price = float(df['close'].iloc[-1])
        
        # Scaling data untuk LSTM
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
        
        lookback = 15
        X = np.array([scaled_data[i-lookback:i, 0] for i in range(lookback, len(scaled_data))])
        X = X.reshape(-1, lookback, 1)
        
        # AI Modeling kilat
        model = Sequential([Input(shape=(lookback, 1)), LSTM(16), Dense(1)])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, scaled_data[lookback:], epochs=2, verbose=0)
        
        pred_scaled = model.predict(scaled_data[-lookback:].reshape(1, lookback, 1), verbose=0)
        final_pred = float(scaler.inverse_transform(pred_scaled).flatten()[0])
        
        signal = "BUY 🟢" if final_pred > current_price else "HOLD/SELL 🔴"
        
        # Catat ke Database
        payload = {"price": current_price, "prediction": final_pred, "fng_index": 50, "signal_type": signal}
        save_to_supabase(payload)
        
        # Kirim Laporan ke Telegram
        msg = f"🛡️ CHRONOS AI REPORT\n\nPrice: ${current_price:,.2f}\nPred: ${final_pred:,.2f}\nSignal: {signal}\n\n🤖 Server Status: Super Stable"
        send_telegram(msg)
        
        return {**payload, "telegram_status": "Pesan dikirim"}
    except Exception as e:
        return {"error": "System Busy", "detail": str(e)}

@app.get("/")
def health():
    return {"status": "Chronos Online", "engine": "Binance Vision Data Stream"}
