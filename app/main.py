"""FastAPI demo for the Hinglish sentiment model."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from src.predict import load_model, predict

DESCRIPTION = (
    "Demo of the Hinglish Sentiment Lab model (TF-IDF + logistic regression, trained on YouTube "
    "creator and review comments). It was not validated on brand-campaign comments, and the "
    "probabilities are raw model scores, not calibrated confidence."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
        app.state.model_ready = True
    except FileNotFoundError:
        app.state.model_ready = False
    yield


app = FastAPI(title="Hinglish Sentiment API", description=DESCRIPTION, version="0.1.0",
              lifespan=lifespan)


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100,
                             description="Comments to classify (1 to 100).")

    @field_validator("texts")
    @classmethod
    def check_length(cls, v):
        if any(len(t) > 2000 for t in v):
            raise ValueError("each text must be at most 2000 characters")
        return v


class Prediction(BaseModel):
    text: str
    label: str
    probabilities: dict[str, float]


class PredictResponse(BaseModel):
    model: str
    predictions: list[Prediction]


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_loaded": request.app.state.model_ready}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(body: PredictRequest, request: Request):
    if not request.app.state.model_ready:
        raise HTTPException(status_code=503,
                            detail="Model not found. Train it first: python -m src.tfidf_model")
    return {"model": "tfidf-logreg", "predictions": predict(body.texts)}
