from fastapi import FastAPI, HTTPException
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
        # Step 1: Load and preprocess data
        df, exog_features = load_and_preprocess(target=nutrient)

        # Step 2: Train/test split
        split_idx = int(len(df) * 0.8)
        train, test = df.iloc[:split_idx], df.iloc[split_idx:]

        # Step 3: Scale data
        train_scaled, test_scaled, scaler = scale_features(train, test, exog_features)

        # Step 4: Train model
        model = train_model(train_scaled, nutrient, exog_features)

        # Step 5: Prepare future exogenous features (copy last known values)
        future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
        future_exog = pd.DataFrame(index=future_dates)

        for feature in exog_features:
            future_exog[feature] = test_scaled[feature].iloc[-1]

        # Step 6: Forecast
        forecast = model.get_forecast(steps=30, exog=future_exog)
        pred_mean_scaled = forecast.predicted_mean
        conf_int_scaled = forecast.conf_int()

        # Step 7: Inverse transform the predictions
        inverse_df = pd.DataFrame({nutrient: pred_mean_scaled})
        inverse_df[exog_features] = test_scaled[exog_features].iloc[-1].values
        inverse_transformed = scaler.inverse_transform(inverse_df)
        pred_mean = inverse_transformed[:, 0]

        lower_ci_scaled = conf_int_scaled[f'lower {nutrient}']
        upper_ci_scaled = conf_int_scaled[f'upper {nutrient}']

        lower_df = pd.DataFrame({nutrient: lower_ci_scaled})
        lower_df[exog_features] = test_scaled[exog_features].iloc[-1].values
        lower_transformed = scaler.inverse_transform(lower_df)[:, 0]

        upper_df = pd.DataFrame({nutrient: upper_ci_scaled})
        upper_df[exog_features] = test_scaled[exog_features].iloc[-1].values
        upper_transformed = scaler.inverse_transform(upper_df)[:, 0]

        # Step 8: Construct response
        response = {
            "nutrient": nutrient,
            "predictions": [
                {
                    "date": str(date.date()),
                    "predicted_value": float(pred),
                    "lower_ci": float(lower),
                    "upper_ci": float(upper),
                }
                for date, pred, lower, upper in zip(future_dates, pred_mean, lower_transformed, upper_transformed)
            ]
        }
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
