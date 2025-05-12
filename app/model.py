import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

def forecast_series(series: pd.Series, steps: int):
    if series.isnull().any():
        print(f"[DEBUG] Missing values in series. Applying forward fill.")
        series = series.fillna(method='ffill')
    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=steps)
    return forecast.tolist()

def predict_npk(steps: int = 5):
    data_path = os.path.join("data", "wheat_data.csv")
    print(f"[DEBUG] Checking for file: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[ERROR] File not found: {data_path}")
    
    print(f"[INFO] Loading data from {data_path}")
    df = pd.read_csv(data_path)

    required_cols = ['N', 'P', 'K']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"[ERROR] Missing required column '{col}' in CSV.")

    print(f"[INFO] Forecasting {steps} future steps for N, P, K...")
    predictions = {
        "N_prediction": forecast_series(df['N'], steps),
        "P_prediction": forecast_series(df['P'], steps),
        "K_prediction": forecast_series(df['K'], steps),
    }

    print(f"[INFO] Prediction successful for {steps} steps.")
    return predictions
