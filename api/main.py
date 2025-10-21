from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from utils import config, io_utils
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Demand Forecast API")


class ForecastRequest(BaseModel):
    weeks: int


@app.post("/forecast")
def forecast(req: ForecastRequest):
    try:
        model, model_name = io_utils.load_latest_model(config.MODEL_DIR)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No trained models found.")
    X = pd.DataFrame([[req.weeks]], columns=["weeks"])
    prediction = float(model.predict(X)[0])
    return {"week": req.weeks, "prediction": prediction, "model_used": model_name}


@app.get("/healthz")
def health():
    return {"status": "ok"}
