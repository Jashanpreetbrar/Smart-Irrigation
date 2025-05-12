from fastapi import FastAPI
from app.preprocessing import load_and_preprocess, scale_features
from app.model import train_model, predict
import pandas as pd

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the NPK Fertilizer Predictor API"}

@app.get("/predict/{nutrient}")
def get_prediction(nutrient: str):
    df, exog_features = load_and_preprocess(target=nutrient)
    split_idx = int(len(df) * 0.8)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]
    train_scaled, test_scaled, _ = scale_features(train, test, exog_features)

    model = train_model(train_scaled, nutrient, exog_features)
    future_dates = pd.date_range(df.index[-1], periods=30, freq='D')
    future_exog = pd.DataFrame(index=future_dates)

    for feature in exog_features:
        future_exog[feature] = test_scaled[feature].iloc[-1]

    forecast = model.get_prediction(start=future_dates[0],end=future_dates[-1],exog=future_exog,dynamic=False)
    pred_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    response = {
        "nutrient": nutrient,
        "predictions": [
            {
                "date": str(date.date()),
                "predicted_value": float(pred_mean[date]),
                "lower_ci": float(conf_int.loc[date][f'lower {nutrient}']),
                "upper_ci": float(conf_int.loc[date][f'upper {nutrient}']),
            }
            for date in future_dates
        ]
    }

    return response
