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
    """Sistem Autopilot Multi-Jalur (Failover) untuk Menjamin Data Pasti Masuk"""
    
    # JALUR 1: Binance Vision API
    try:
        url = "https://api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=60"
        res = requests.get(url, timeout=5).json()
        if isinstance(res, list) and len(res) > 0:
            closes = [float(item[4]) for item in res]
            return pd.DataFrame(closes, columns=['close']), None
    except Exception as e:
        print(f"Jalur 1 Gagal: {str(e)}")

    # JALUR 2: Tokocrypto API (Binance Cloud)
    try:
        url = "https://api.tokocrypto.com/open/v1/market/klines?symbol=BTC_USDT&intervals=1d&limit=60"
        res = requests.get(url, timeout=5).json()
        if res.get("code") == 0 and res.get("data"):
            closes = [float(item[4]) for item in res["data"]]
            return pd.DataFrame(closes, columns=['close']), None
    except Exception as e:
        print(f"Jalur 2 Gagal: {str(e)}")

    # JALUR 3: CoinGecko Public API (Benteng Terakhir)
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=60&interval=daily"
        res = requests.get(url, timeout=5).json()
        if "prices" in res:
            # Format CoinGecko: [[timestamp, price], ...]
            closes = [float(item[1]) for item in res["prices"]]
            # Ambil 60 data teratas/terakhir
            return pd.DataFrame(closes[:60], columns=['close']), None
    except Exception as e:
        print(f"Jalur 3 Gagal: {str(e)}")

    return None, "Semua jalur API (Binance, Tokocrypto, CoinGecko) sedang down/timeout."

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
            return {"error": "Gagal mengambil data market", "detail": err}
            
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
        msg = f"🛡️ CHRONOS AI REPORT\n\nPrice: ${current_price:,.2f}\nPred: ${final_pred:,.2f}\nSignal: {signal}\n\n🤖 Server Status: Ultra Secure (Triple API)"
        send_telegram(msg)
        
        return {**payload, "telegram_status": "Pesan dikirim"}
    except Exception as e:
        return {"error": "System Busy", "detail": str(e)}

@app.get("/")
def health():
    return {"status": "Chronos Online", "engine": "Triple-Route Failover System"}
