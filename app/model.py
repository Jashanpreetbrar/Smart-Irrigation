import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

def forecast_series(series: pd.Series, steps: int):
    model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=steps)
    return forecast.tolist()

def predict_npk(steps: int = 5):
    data_path = os.path.join("data", "wheat_data.csv")
    print(f"[INFO] Loading data from {data_path}")
    df = pd.read_csv(data_path)

    for col in ['N', 'P', 'K']:
        if col not in df.columns:
            raise ValueError(f"Missing '{col}' column in CSV")

    predictions = {
        "N_prediction": forecast_series(df['N'], steps),
        "P_prediction": forecast_series(df['P'], steps),
        "K_prediction": forecast_series(df['K'], steps),
    }

    print(f"[INFO] Predictions: {predictions}")
    return predictions
