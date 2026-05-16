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
SUPA_KEY = os.getenv("SUPABASE_KEY")

def save_to_supabase(data):
    if not SUPA_KEY:
        return
    url = f"{SUPA_URL}/rest/v1/signals"
    headers = {
        "apikey": SUPA_KEY,
        "Authorization": f"Bearer {SUPA_KEY}",
        "Content-Type": "application/json"
    }
    try:
        requests.post(url, json=data, headers=headers)
    except:
        pass

def run_ai_logic():
    # Ambil data
    df = yf.download('BTC-USD', period='60d', interval='1d', auto_adjust=True)
    
    # PERBAIKAN KRUSIAL: Pastikan kita dapat angka tunggal meskipun datanya multi-index
    if isinstance(df['Close'], pd.DataFrame):
        close_series = df['Close'].iloc[:, 0]
    else:
        close_series = df['Close']
        
    current_price = float(close_series.iloc[-1])
    
    # Preprocessing
    data_df = close_series.to_frame()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_df)
    
    lookback = 30
    X = []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Model Lite
    model = Sequential([
        Input(shape=(X.shape[1], 1)),
        LSTM(50, return_sequences=True),
        Dropout(0.1),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, scaled_data[lookback:], epochs=3, batch_size=32, verbose=0)
    
    # Predict
    last_data = scaled_data[-lookback:].reshape(1, lookback, 1)
    pred_scaled = model.predict(last_data, verbose=0)
    final_pred = float(scaler.inverse_transform(pred_scaled).flatten()[0])
    
    # Fear and Greed
    fng_val = 50 # Default jika API gagal
    try:
        fng_res = requests.get('https://api.alternative.me/fng/').json()
        fng_val = int(fng_res['data'][0]['value'])
    except:
        pass
    
    return current_price, final_pred, fng_val

@app.get("/")
def home():
    return {"status": "Chronos AI Engine is Running", "endpoint": "/predict"}

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
    
    save_to_supabase(payload)
    return payload
