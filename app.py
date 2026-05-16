from fastapi import FastAPI
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
import requests
import os
import time

app = FastAPI()

# Kredensial Supabase
SUPA_URL = "https://nzpzddyjcthhzkyfrjpb.supabase.co"
SUPA_KEY = os.getenv("SUPABASE_KEY")

def save_to_supabase(payload):
    if not SUPA_KEY: return
    headers = {"apikey": SUPA_KEY, "Authorization": f"Bearer {SUPA_KEY}", "Content-Type": "application/json"}
    try: requests.post(f"{SUPA_URL}/rest/v1/signals", json=payload, headers=headers, timeout=5)
    except: pass

def get_crypto_data(retries=3):
    """Fungsi ambil data dengan mekanisme retry agar tahan banting"""
    for i in range(retries):
        try:
            df = yf.download('BTC-USD', period='60d', interval='1d', auto_adjust=True, progress=False)
            if not df.empty and len(df) > 30:
                return df
        except Exception as e:
            print(f"Retry {i+1} failed: {e}")
            time.sleep(2)
    return None

def run_ai_logic():
    df = get_crypto_data()
    if df is None:
        raise ValueError("Data pasar tidak tersedia (Rate Limit Yahoo Finance)")

    # Handle multi-index columns dari yfinance baru
    if isinstance(df.columns, pd.MultiIndex):
        close_series = df['Close'].iloc[:, 0]
    else:
        close_series = df['Close']
    
    current_price = float(close_series.iloc[-1])
    
    # Preprocessing
    data_df = close_series.to_frame()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_df)
    
    lookback = 15
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)
    
    # Model Super Fast
    model = Sequential([
        Input(shape=(lookback, 1)),
        LSTM(16),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=2, batch_size=32, verbose=0)
    
    # Predict
    last_input = scaled_data[-lookback:].reshape(1, lookback, 1)
    pred_scaled = model.predict(last_input, verbose=0)
    final_pred = float(scaler.inverse_transform(pred_scaled).flatten()[0])
    
    return current_price, final_pred

@app.get("/")
def home():
    return {"status": "Chronos Online", "info": "Use /predict to run AI"}

@app.get("/predict")
def get_prediction():
    try:
        curr, pred = run_ai_logic()
        signal = "BUY" if pred > curr else "HOLD/SELL"
        
        payload = {
            "price": curr,
            "prediction": pred,
            "fng_index": 50, # Static if API fails
            "signal_type": signal
        }
        
        save_to_supabase(payload)
        return payload
    except Exception as e:
        return {"error": "Backend Busy or Rate Limited", "detail": str(e)}
