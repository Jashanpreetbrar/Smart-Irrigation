# NPK Fertilizer Prediction API

This project provides a simple web-based interface to forecast Nitrogen (N), Phosphorus (P), and Potassium (K) values using SARIMA.

## Features
- Predict NPK levels for up to 36 months
- Interactive frontend
- FastAPI backend
- Deployable to Render

## Setup
```bash
pip install -r requirements.txt
uvicorn app.api:app --reload
