# ============================================
# fastapi_service.py (с метриками в /predict + логирование)
# ============================================

import mlflow
import numpy as np
import pandas as pd
import pickle
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from tensorflow.keras.preprocessing.sequence import pad_sequences
import uvicorn
from loguru import logger
import sys

# ---------- Настройка логирования ----------
# Удаляем стандартный вывод loguru
logger.remove()

# Добавляем вывод в консоль (для отладки)
logger.add(sys.stdout, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

# Создаём папку logs если её нет
os.makedirs("logs", exist_ok=True)

# Добавляем вывод в файл (ротация 1 МБ, хранение 7 дней)
logger.add(
    "logs/app.log", 
    rotation="1 MB", 
    retention="7 days", 
    level="INFO", 
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

logger.info("=" * 60)
logger.info("Запуск сервиса классификации ОКПД2")
logger.info("=" * 60)

# ---------- Конфигурация ----------
MODEL_URI = "models:/okpd2_classifier@PRD"
MAX_LEN = 1500
THRESHOLD = 0.5

# ---------- Загрузка модели и токенизатора ----------
mlflow.set_tracking_uri("http://127.0.0.1:5000")

logger.info("Загрузка модели Bi-GRU...")

try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    logger.info(f" Модель загружена из {MODEL_URI}")
except Exception as e:
    logger.error(f" Ошибка загрузки модели: {e}")
    raise

logger.info("Загрузка токенизатора...")
try:
    with open("bigru_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    logger.info(" Токенизатор загружен")
except Exception as e:
    logger.error(f" Ошибка загрузки токенизатора: {e}")
    raise

# ---------- Загрузка метрик модели ----------
def load_model_metrics() -> dict:
    """Загружает метрики из baseline_comparison.csv или возвращает значения по умолчанию"""
    metrics_file = "baseline_comparison.csv"
    
    if os.path.exists(metrics_file):
        try:
            df = pd.read_csv(metrics_file)
            bigru_row = df[df['Model'].str.contains('Bi-GRU', case=False, na=False)]
            if not bigru_row.empty:
                return {
                    "subset_accuracy": float(bigru_row['Subset Accuracy'].values[0]),
                    "f1_micro": float(bigru_row['F1-micro'].values[0]),
                    "f1_macro": float(bigru_row['F1-macro'].values[0]),
                    "hamming_loss": float(bigru_row['Hamming Loss'].values[0]),
                }
        except Exception as e:
            logger.warning(f"Ошибка чтения {metrics_file}: {e}")
    
    # Значения по умолчанию из чекпойнта 7
    return {
        "subset_accuracy": 0.7764,
        "f1_micro": 0.8177,
        "f1_macro": 0.5080,
        "hamming_loss": 0.0043,
    }

logger.info("Загрузка метрик модели...")
MODEL_METRICS = load_model_metrics()
logger.info(f" Метрики загружены: F1-micro={MODEL_METRICS['f1_micro']}, F1-macro={MODEL_METRICS['f1_macro']}")

logger.info("=" * 60)
logger.info("Сервис готов к работе")
logger.info("Swagger UI: http://127.0.0.1:8000/docs")
logger.info("=" * 60)


# ---------- Pydantic модели ----------
class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    predicted_codes: List[str]
    confidence: List[float]
    status: str = "success"
    model_metrics: dict


# ---------- Вспомогательные функции ----------
def preprocess(text: str) -> np.ndarray:
    """Токенизация и паддинг текста"""
    if not text or not isinstance(text, str):
        text = ""
    seq = tokenizer.texts_to_sequences([text.lower()])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    return padded


# ---------- FastAPI эндпоинты ----------
app = FastAPI(
    title="OKPD2 Classifier",
    description="Классификация текстов договоров по кодам ОКПД2 (второй уровень)",
    version="1.0.0"
)


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Классифицирует текст договора и возвращает:
    - коды ОКПД2 с уверенностью
    - метрики качества модели (F1-micro, F1-macro, Hamming Loss, Subset Accuracy)
    """
    start_time = time.time()
    
    try:
        # Предобработка
        X = preprocess(request.text)
        
        # Инференс
        probs = model.predict(X)[0]
        
        # Выбор кодов с порогом
        predicted_indices = np.where(probs > THRESHOLD)[0]
        predicted_codes = [str(i) for i in predicted_indices]
        confidence = probs[predicted_indices].tolist()
        
        # Время обработки
        latency = time.time() - start_time
        
        # Логируем запрос
        logger.info(f"Input: {request.text[:100]}... | Prediction: {predicted_codes} | Confidence: {confidence} | Latency: {latency:.3f}s")
        
        # Возвращаем предсказания + метрики модели
        return PredictResponse(
            predicted_codes=predicted_codes,
            confidence=confidence,
            status="success",
            model_metrics=MODEL_METRICS
        )
        
    except Exception as e:
        latency = time.time() - start_time
        logger.error(f"Ошибка: {str(e)} | Input: {request.text[:100]}... | Latency: {latency:.3f}s")
        raise HTTPException(status_code=500, detail=f"Ошибка при предсказании: {str(e)}")


@app.get("/health", tags=["Health"])
def health_check():
    """Проверка работоспособности сервиса"""
    logger.info("Health check requested")
    return {
        "status": "healthy", 
        "model_loaded": True,
        "tokenizer_loaded": tokenizer is not None
    }


@app.get("/metrics", tags=["Monitoring"])
def get_metrics():
    """Возвращает метрики качества модели (отдельный эндпоинт)"""
    baseline_acc = 0.6771
    improvement = MODEL_METRICS["subset_accuracy"] - baseline_acc
    
    return {
        "model": "Bi-GRU (PRD)",
        "subset_accuracy": MODEL_METRICS["subset_accuracy"],
        "f1_micro": MODEL_METRICS["f1_micro"],
        "f1_macro": MODEL_METRICS["f1_macro"],
        "hamming_loss": MODEL_METRICS["hamming_loss"],
        "improvement_vs_baseline": round(improvement, 4),
        "note": "Метрики из baseline_comparison.csv (ЧП 7)"
    }


# ---------- Запуск ----------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )