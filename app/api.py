from fastapi import FastAPI
from app.model import predict_npk
from app.db import store_prediction

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "NPK Fertilizer Prediction API is live!"}

@app.get("/predict")
def get_prediction(steps: int = 5):
    prediction = predict_npk(steps)
    store_prediction(prediction)
    return {"NPK_Prediction": prediction}
