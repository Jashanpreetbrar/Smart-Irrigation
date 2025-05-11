from fastapi import FastAPI
from app.model import predict_npk

app = FastAPI()

@app.get("/")
def root():
    return {"message": "NPK Fertilizer Prediction API"}

@app.get("/predict")
def get_prediction(steps: int = 5):
    print(f"[INFO] Predicting next {steps} NPK values...")
    prediction = predict_npk(steps)
    print(f"[INFO] Prediction complete: {prediction}")
    return {"NPK_Prediction": prediction}
