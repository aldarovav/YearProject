from mlflow.tracking import MlflowClient

client = MlflowClient()

# Указываем модель и версию
model_name = "okpd2_classifier"
version = "1"  # или номер версии, которая есть

# Добавляем тег PRD
client.set_registered_model_alias(model_name, "PRD", version)
print(f" Тег PRD добавлен модели {model_name} version {version}")