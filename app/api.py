from fastapi import FastAPI, Query
from app.model import predict_npk

app = FastAPI()

@app.get("/")
def root():
    return {"message": "NPK Fertilizer Prediction API"}

@app.get("/predict")
def get_prediction(steps: int = Query(5, gt=0, le=36)):
    """
    Predict the next `steps` values for N, P, and K.
    - steps: number of months to forecast (1–36)
    """
    print(f"[INFO] Predicting next {steps} months expected values NPK values...")
    prediction = predict_npk(steps)
    print(f"[INFO] Prediction complete.")
    return prediction
