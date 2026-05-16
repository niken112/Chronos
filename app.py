from fastapi import FastAPI
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
import requests
import os

app = FastAPI()

# Kredensial Supabase
SUPA_URL = "https://nzpzddyjcthhzkyfrjpb.supabase.co"
SUPA_KEY = os.getenv("SUPABASE_KEY") # Masukkan Anon Key tadi di Secrets HF

def save_to_supabase(data):
    url = f"{SUPA_URL}/rest/v1/signals"
    headers = {
        "apikey": SUPA_KEY,
        "Authorization": f"Bearer {SUPA_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }
    requests.post(url, json=data, headers=headers)

def run_ai_logic():
    df = yf.download('BTC-USD', period='60d', interval='1d', auto_adjust=True)
    df_close = df[['Close']].copy()
    current_price = float(df_close['Close'].iloc[-1])
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_close)
    
    lookback = 30
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

@app.get("/predict")
def get_prediction():
    curr, pred, fng = run_ai_logic()
    signal = "BUY" if pred > curr and fng > 35 else "HOLD/SELL"
    
    payload = {
        "price": curr,
        "prediction": pred,
        "fng_index": fng,
        "signal_type": signal
    }
    
    # Kirim data ke Supabase agar Vercel bisa baca history-nya
    save_to_supabase(payload)
    
    return payload
