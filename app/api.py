from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from app.model import predict_npk
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_path = Path("app/index.html")
    if not html_path.exists():
        return HTMLResponse("<h2>UI file not found</h2>", status_code=404)
    return html_path.read_text()

@app.get("/predict")
def get_prediction(steps: int = Query(5, gt=0, le=36)):
    try:
        prediction = predict_npk(steps)
        return prediction
    except Exception as e:
        return {"error": str(e)}
