from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to the Fertilizer Prediction API"}

@app.get("/predict")
def predict_fertilizer():
    try:
        # Load and preprocess data
        df = pd.read_csv("output.csv")
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Fit SARIMAX model (You can tune the order as needed)
        model = SARIMAX(df['N'], order=(1,1,1), seasonal_order=(1,0,0,12), enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False)

        # Forecast next 6 months
        forecast = results.get_forecast(steps=6)
        pred_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()

        response = [
            {
                "month": f"Month {i+1}",
                "predicted_value": round(float(pred_mean[i]), 2),
                "lower_ci": round(float(conf_int.iloc[i, 0]), 2),
                "upper_ci": round(float(conf_int.iloc[i, 1]), 2)
            }
            for i in range(6)
        ]

        return JSONResponse(content=response)

    except Exception as e:
        return {"error": str(e)}
