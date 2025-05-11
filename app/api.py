from fastapi import FastAPI
from app.model import predict_npk
from app.db import store_prediction

app = FastAPI()

@app.get("/predict")
def get_prediction(steps: int = 5):
    prediction = predict_npk(steps)
    store_prediction(prediction)
    return {"NPK_Prediction": prediction}
