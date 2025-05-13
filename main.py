from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = FastAPI()

# Allow CORS (to enable frontend to access our API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/forecast")
def get_forecast():
    # Load the CSV file
    df = pd.read_csv("output.csv")

    # Convert 'Date' to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Focus on Nitrogen (N) values
    # Ensure data is monthly - resample to monthly frequency
    monthly_df = df['N'].resample('MS').mean()
    
    # Handle any missing values
    monthly_df = monthly_df.fillna(monthly_df.bfill())

    # Fit SARIMAX model (parameters can be tuned based on data analysis)
    model = SARIMAX(monthly_df, 
                   order=(1, 1, 1),  # (p,d,q) parameters
                   seasonal_order=(1, 0, 1, 12))  # (P,D,Q,s) seasonal parameters
    model_fit = model.fit(disp=False)

    # Forecast next 6 months
    forecast = model_fit.get_forecast(steps=6)
    pred_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    # Format the result
    results = [
        {
            "month": f"Month {i+1}",  # Just use Month 1, Month 2, etc.
            "predicted_value": round(pred_mean.iloc[i], 2),
            "lower_ci": round(conf_int.iloc[i, 0], 2),
            "upper_ci": round(conf_int.iloc[i, 1], 2),
        }
        for i in range(6)
    ]

    return {"forecast": results}
