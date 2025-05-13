[9:32 pm, 12/5/2025] Jashan Brar: from fastapi import FastAPI
import pandas as pd
import numpy as np
from app.preprocessing import load_and_preprocess, scale_features
from app.model import train_model, predict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Optional: Allow CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict/{nutrient}")
def get_prediction(nutrient: str):
    try:
        # Load and split the data
        df, exog_features = load_and_preprocess(target=nutrient)
        split_idx = int(len(df) * 0.8)
        train, test = df.iloc[:split_idx], df.iloc[split_idx:]
        train_scaled, test_scaled, scaler = scale_features(train, test, exog_fea…
[9:33 pm, 12/5/2025] Jashan Brar: from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_model(train, target, exog):
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 7)
    model = SARIMAX(
        endog=train[target],
        exog=train[exog],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model.fit(disp=False)

def predict(model, exog, start_date, end_date, dynamic=False):
    return model.get_prediction(start=start_date, end=end_date, exog=exog, dynamic=dynamic)
