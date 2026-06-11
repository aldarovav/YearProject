import mlflow
import mlflow.tensorflow
from tensorflow.keras.models import load_model
import joblib
import json
from pathlib import Path

# Настройка подключения к MLflow серверу
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("okpd2_classification")

# Пути к файлам (в текущей папке)
MODEL_PATH = Path("bigru_prd.keras")
TOKENIZER_PATH = Path("bigru_tokenizer.pkl")
CONFIG_PATH = Path("bigru_config.json")

# Проверка наличия файлов
for p in [MODEL_PATH, TOKENIZER_PATH, CONFIG_PATH]:
    if not p.exists():
        print(f" Файл не найден: {p}")
        exit(1)

# Загрузка
print("Загрузка модели...")
model = load_model(MODEL_PATH)
print("Загрузка токенизатора...")
tokenizer = joblib.load(TOKENIZER_PATH)

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# Логирование
print("Логирование в MLflow...")
with mlflow.start_run(run_name="bigru_prd_final") as run:
    mlflow.log_params(config)
    mlflow.log_metric("test_accuracy", config.get("test_accuracy", 0.7764))
    mlflow.log_artifact(str(MODEL_PATH))
    mlflow.log_artifact(str(TOKENIZER_PATH))
    mlflow.log_artifact(str(CONFIG_PATH))
    
    mlflow.tensorflow.log_model(
        model, 
        "model",
        registered_model_name="okpd2_classifier"
    )
    
    print(f"\n Модель залогирована!")
    print(f"   Run ID: {run.info.run_id}")
    print(f"   UI: http://127.0.0.1:5000")