import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

def predict_npk(steps: int = 5):
    data_path = os.path.join("data", "wheat_data.csv")
    print(f"[INFO] Loading data from {data_path}")
    df = pd.read_csv(data_path)

    if 'N' not in df.columns:
        raise ValueError("Input CSV must contain 'N' column for nitrogen values")

    ts = df['N']
    print(f"[INFO] Fitting SARIMA model...")
    model = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,12))
    model_fit = model.fit(disp=False)

    forecast = model_fit.forecast(steps=steps)
    print(f"[INFO] Forecast generated: {forecast.tolist()}")
    return forecast.tolist()
