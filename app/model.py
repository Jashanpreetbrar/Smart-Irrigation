import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

def forecast_series(series: pd.Series, steps: int):
    if series.isnull().any():
        print(f"[DEBUG] Filling missing values with forward fill.")
        series = series.fillna(method='ffill')
    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    return model_fit.forecast(steps=steps).tolist()

def predict_npk(steps: int = 5):
    path = os.path.join("data", "wheat_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("CSV file missing in data/")
    df = pd.read_csv(path)
    for col in ['N', 'P', 'K']:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in CSV.")
    return {
        "N_prediction": forecast_series(df['N'], steps),
        "P_prediction": forecast_series(df['P'], steps),
        "K_prediction": forecast_series(df['K'], steps)
    }
