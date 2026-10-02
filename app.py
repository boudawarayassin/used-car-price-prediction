from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from pydantic import BaseModel

import joblib
import pandas as pd


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Used Car Price Predictor",
    description="Machine Learning powered used car price prediction",
    version="1.0.0"
)


# ============================================================
# Static files
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# ============================================================
# Templates
# ============================================================

templates = Jinja2Templates(
    directory="templates"
)


# ============================================================
# Load model
# ============================================================

model = joblib.load(
    "models/car_price_model_final.pkl"
)

model_frequency = joblib.load(
    "models/model_frequency.pkl"
)


# ============================================================
# Input schema
# ============================================================

class CarData(BaseModel):

    year: float = Field(..., ge=1950, le=2026)

    region: str
    manufacturer: str
    model: str

    condition: str | None = None
    cylinders: str | None = None
    fuel: str | None = None

    odometer: float = Field(..., ge=0)

    title_status: str | None = None
    transmission: str | None = None
    drive: str | None = None
    type: str | None = None
    paint_color: str | None = None

    state: str

    lat: float = Field(..., ge=-90, le=90)
    long: float = Field(..., ge=-180, le=180)

# ============================================================
# Home
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# ============================================================
# Health check
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "OK",
        "model_loaded": True,
        "frequency_loaded": True
    }


# ============================================================
# Prediction
# ============================================================

@app.post("/predict")
def predict(car: CarData):

    # Convert input to dictionary
    data = car.model_dump()

    # Calculate model frequency
    data["model_frequency"] = (
        model_frequency
        .get(data["model"], 0)
    )

    # Create DataFrame
    input_data = pd.DataFrame([data])

    # Keep exactly the columns expected by the model
    input_data = input_data[
    [
        "region",
        "manufacturer",
        "model",
        "condition",
        "cylinders",
        "fuel",
        "odometer",
        "title_status",
        "transmission",
        "drive",
        "type",
        "paint_color",
        "state",
        "lat",
        "long",
        "year",
        "model_frequency"
    ]
]

    # Prediction
    prediction = model.predict(input_data)[0]

    return {
        "predicted_price": round(float(prediction), 2)
    }