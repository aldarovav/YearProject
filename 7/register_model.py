import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

# ID вашего run (из предыдущего вывода)
run_id = "aef22499175c4bd6a38792ec8ccdb04f"

# Регистрируем модель
model_uri = f"runs:/{run_id}/model"
registered_model = mlflow.register_model(model_uri, "okpd2_classifier")

print(f" Модель зарегистрирована")
print(f"   Name: {registered_model.name}")
print(f"   Version: {registered_model.version}")

# Добавляем тег PRD
client.set_registered_model_alias("okpd2_classifier", "PRD", registered_model.version)
print(f" Тег PRD добавлен version {registered_model.version}")