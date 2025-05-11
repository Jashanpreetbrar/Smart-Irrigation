import pandas as pd
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv("wheat_data.csv")
model = SARIMAX(df['N'], order=(1,1,1), seasonal_order=(1,1,1,12))
model_fit = model.fit(disp=False)

def predict_npk(steps: int):
    forecast = model_fit.forecast(steps=steps)
    return forecast.tolist()