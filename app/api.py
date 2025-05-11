from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from app.model import predict_npk

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "NPK Fertilizer Prediction API"}

@app.get("/predict")
def get_prediction(steps: int = Query(5, gt=0, le=36)):
    print(f"[INFO] Received prediction request for {steps} steps")
    try:
        prediction = predict_npk(steps)
        return prediction
    except Exception as e:
        print(f"[ERROR] {e}")
        return {"error": str(e)}
