"""
FastAPI serving-сервис для ML-модели.
"""

from __future__ import annotations

import os
import time
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "churn-lgbm")
MODEL_VERSION = os.getenv("MODEL_VERSION", "Production")
CHURN_THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.4"))

# Prometheus метрики
REQUEST_COUNT = Counter(
    "churn_api_requests_total", "Total API requests", ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "churn_api_latency_seconds", "Request latency", ["endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)
PREDICTION_SCORE = Histogram(
    "churn_prediction_score", "Distribution of churn probability scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Глобальный model стейт
model_state: dict = {}


# Lifespan: загрузка модели при старте
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружаем модель из MLflow Registry при старте приложения."""
    logger.info(f"Загрузка модели {MODEL_NAME} (stage: {MODEL_VERSION}) из MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        model_state["model"] = mlflow.sklearn.load_model(model_uri)
        model_state["version"] = MODEL_VERSION
        model_state["loaded_at"] = time.time()
        logger.info(f"✅ Модель загружена: {model_uri}")
    except Exception as e:
        logger.error(f"❌ Не удалось загрузить модель: {e}")
        raise RuntimeError(f"Model load failed: {e}") from e

    yield  # Приложение работает

    logger.info("Shutting down — clearing model state")
    model_state.clear()


# FastAPI
app = FastAPI(
    title="Churn Prediction API",
    description="ML-сервис прогнозирования оттока клиентов. MLOps Level 2.",
    version="1.0.0",
    lifespan=lifespan,
)


# Pydantic схемы
class CustomerFeatures(BaseModel):
    """Признаки одного клиента."""
    gender: int = Field(..., ge=0, le=1, description="0=Female, 1=Male")
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    Dependents: int = Field(..., ge=0, le=1)
    tenure: int = Field(..., ge=0, le=72, description="Месяцев с компанией")
    PhoneService: int = Field(..., ge=0, le=1)
    MultipleLines: int = Field(..., ge=0, le=1)
    InternetService: int = Field(..., ge=0, le=2, description="0=DSL, 1=Fiber optic, 2=No")
    OnlineSecurity: int = Field(..., ge=0, le=1)
    OnlineBackup: int = Field(..., ge=0, le=1)
    DeviceProtection: int = Field(..., ge=0, le=1)
    TechSupport: int = Field(..., ge=0, le=1)
    StreamingTV: int = Field(..., ge=0, le=1)
    StreamingMovies: int = Field(..., ge=0, le=1)
    Contract: int = Field(..., ge=0, le=2, description="0=Month-to-month, 1=One year, 2=Two year")
    PaperlessBilling: int = Field(..., ge=0, le=1)
    PaymentMethod: int = Field(..., ge=0, le=3)
    MonthlyCharges: float = Field(..., ge=0, le=200)
    TotalCharges: float = Field(..., ge=0)
    avg_monthly_charges: float = Field(..., ge=0)
    charges_delta: float
    num_services: int = Field(..., ge=0, le=8)
    is_long_contract: int = Field(..., ge=0, le=1)


class PredictionRequest(BaseModel):
    customers: List[CustomerFeatures]
    return_explanations: bool = Field(default=True, description="Возвращать топ-3 SHAP объяснения")


class CustomerPrediction(BaseModel):
    churn_probability: float
    is_at_risk: bool
    risk_level: str
    top_reasons: Optional[List[dict]] = None


class PredictionResponse(BaseModel):
    predictions: List[CustomerPrediction]
    model_version: str
    inference_time_ms: float
    batch_size: int


# Вспомогательные функции
def get_risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "HIGH"
    elif prob >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"


FEATURE_NAMES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "avg_monthly_charges",
    "charges_delta", "num_services", "is_long_contract"
]

FEATURE_LABELS = {
    "tenure": "Срок абонентства",
    "Contract": "Тип контракта",
    "MonthlyCharges": "Ежемесячный платёж",
    "TotalCharges": "Суммарные расходы",
    "InternetService": "Тип интернета",
    "num_services": "Кол-во сервисов",
    "OnlineSecurity": "Онлайн-безопасность",
    "TechSupport": "Тех. поддержка",
    "charges_delta": "Изменение платежа",
    "is_long_contract": "Длительный контракт",
}


def compute_shap_top3(X: pd.DataFrame) -> List[List[dict]]:
    """Вычисляет топ-3 факторов риска для каждого клиента через SHAP."""
    model = model_state["model"]
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        # Для бинарной классификации берём класс 1 (churn)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        results = []
        for row_shap in shap_vals:
            # Топ-3 по абсолютному влиянию
            top_idx = np.argsort(np.abs(row_shap))[-3:][::-1]
            reasons = []
            for idx in top_idx:
                feat = FEATURE_NAMES[idx]
                reasons.append({
                    "feature": FEATURE_LABELS.get(feat, feat),
                    "shap_value": float(row_shap[idx]),
                    "direction": "увеличивает риск" if row_shap[idx] > 0 else "снижает риск"
                })
            results.append(reasons)
        return results
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return [[] for _ in range(len(X))]


# Эндпоинты API

@app.get("/health", summary="Health check downloading model")
async def health():
    """
    Возвращает 200 OK если модель загружена.
    """
    if "model" not in model_state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_name": MODEL_NAME,
        "model_version": model_state.get("version"),
        "loaded_at": model_state.get("loaded_at"),
    }


@app.post("/predict", response_model=PredictionResponse, summary="Batch инференс")
async def predict(request: PredictionRequest):
    """
    Принимает батч клиентов, возвращает вероятность оттока и топ-3 причины.
    SLO: P95 latency < 500 мс.
    """
    start_time = time.time()

    if "model" not in model_state:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        raise HTTPException(status_code=503, detail="Model not ready")

    if len(request.customers) == 0:
        raise HTTPException(status_code=400, detail="Empty customer list")

    if len(request.customers) > 10000:
        raise HTTPException(status_code=400, detail="Batch size exceeds limit (10000)")

    try:
        # Преобразуем в DataFrame
        rows = [c.dict() for c in request.customers]
        X = pd.DataFrame(rows)[FEATURE_NAMES]

        # Предсказание
        model = model_state["model"]
        probas = model.predict_proba(X)[:, 1]

        # SHAP объяснения (если запрошены)
        explanations = compute_shap_top3(X) if request.return_explanations else [None] * len(X)

        # Формируем ответ
        predictions = []
        for i, prob in enumerate(probas):
            PREDICTION_SCORE.observe(float(prob))
            predictions.append(CustomerPrediction(
                churn_probability=round(float(prob), 4),
                is_at_risk=bool(prob >= CHURN_THRESHOLD),
                risk_level=get_risk_level(prob),
                top_reasons=explanations[i] if request.return_explanations else None,
            ))

        elapsed_ms = (time.time() - start_time) * 1000
        REQUEST_LATENCY.labels(endpoint="/predict").observe(elapsed_ms / 1000)
        REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()

        return PredictionResponse(
            predictions=predictions,
            model_version=model_state.get("version", "unknown"),
            inference_time_ms=round(elapsed_ms, 2),
            batch_size=len(predictions),
        )

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        logger.exception(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/metrics", summary="Prometheus метрики")
async def metrics():
    """Эндпоинт для Prometheus"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info", summary="Информация о текущей модели")
async def model_info():
    """Возвращает данные о загруженной модели из MLflow."""
    if "model" not in model_state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        version_info = versions[0] if versions else {}
        run_id = version_info.run_id if versions else None
        metrics = {}
        if run_id:
            run = client.get_run(run_id)
            metrics = run.data.metrics
    except Exception as e:
        metrics = {}
        version_info = {}

    return {
        "model_name": MODEL_NAME,
        "stage": MODEL_VERSION,
        "churn_threshold": CHURN_THRESHOLD,
        "performance_metrics": {k: round(v, 4) for k, v in metrics.items()},
    }
