from fastapi import FastAPI
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

@app.get("/")
def read_root():
    return {"message": "Welcome to the NPK Fertilizer Predictor API"}

@app.get("/predict/{nutrient}")
def get_prediction(nutrient: str):
    try:
        df, exog_features = load_and_preprocess(target=nutrient)
        split_idx = int(len(df) * 0.8)
        train, test = df.iloc[:split_idx], df.iloc[split_idx:]
        train_scaled, test_scaled, scaler = scale_features(train, test, exog_features)

        model = train_model(train_scaled, nutrient, exog_features)

        # Future dates
        future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
        future_exog = pd.DataFrame(index=future_dates)

        for feature in exog_features:
            future_exog[feature] = test_scaled[feature].iloc[-1]

        # Forecast next 30 days
        pred_mean = model.forecast(steps=30, exog=future_exog)

        # Dummy confidence intervals (±5 kg/ha) - can be improved
        conf_int = pd.DataFrame({
            f'lower {nutrient}': pred_mean - 5,
            f'upper {nutrient}': pred_mean + 5
        }, index=future_dates)

        response = {
            "nutrient": nutrient,
            "predictions": [
                {
                    "date": str(date.date()),
                    "predicted_value": round(float(pred_mean[date]), 2),
                    "lower_ci": round(float(conf_int.loc[date][f'lower {nutrient}']), 2),
                    "upper_ci": round(float(conf_int.loc[date][f'upper {nutrient}']), 2),
                }
                for date in future_dates
            ]
        }

        return response

    except Exception as e:
        return {"error": str(e)}
