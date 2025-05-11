# NPK Fertilizer Prediction System

Predicts optimal NPK levels using time series forecasting. Offers a REST API and integrates with ESP32 display.

## Features
- SARIMA time series model
- FastAPI REST API
- MongoDB logging
- ESP32 integration

## Usage
- Run API: `uvicorn app.api:app --reload`
- ESP32 fetches from `/predict?steps=5`