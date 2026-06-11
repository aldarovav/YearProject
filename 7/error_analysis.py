import mlflow
import joblib
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Настройка MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# 2. Загрузка модели по тегу PRD
model_uri = "models:/okpd2_classifier@PRD"
model = mlflow.pyfunc.load_model(model_uri)

# 3. Загрузка токенизатора (из артефактов или локально)
# Способ 1: если токенизатор есть локально
tokenizer = joblib.load("bigru_tokenizer.pkl")

# 4. Загрузка тестовых данных
with open("checkpoint6_data.pkl", 'rb') as f:
    data = pickle.load(f)

# Для Bi-GRU нужны исходные тексты (не BoW)
# Если в data есть тексты:
if 'X_test_texts' in data:
    X_test_texts = data['X_test_texts']
else:
    # Альтернатива: загрузить из исходного датасета
    print("Нужны исходные тексты для токенизации")
    exit(1)

y_test = data['y_test_bin']
classes = data['classes']

# 5. Токенизация тестовых текстов
MAX_LEN = 3000
X_test_seq = tokenizer.texts_to_sequences(X_test_texts)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

# 6. Предсказания
print("Предсказание...")
predictions = model.predict(X_test_pad)
y_pred_bin = (predictions > 0.5).astype(int)

# 7. Поиск ошибок
errors = []
for i in range(min(len(y_test), 1000)):  # первые 1000
    true_set = set(np.where(y_test[i] == 1)[0])
    pred_set = set(np.where(y_pred_bin[i] == 1)[0])
    
    if true_set != pred_set:
        errors.append({
            'text': X_test_texts[i][:500],
            'true_codes': [classes[j] for j in true_set],
            'pred_codes': [classes[j] for j in pred_set],
            'missing': [classes[j] for j in (true_set - pred_set)],
            'extra': [classes[j] for j in (pred_set - true_set)],
            'confidence': float(np.max(predictions[i]))
        })

print(f" Найдено ошибок: {len(errors)}")

# 8. Сохраняем 20 примеров
df_errors = pd.DataFrame(errors[:20])
df_errors.to_csv('error_analysis.csv', index=False)
print(" error_analysis.csv сохранён")

# 9. Вывод статистики
error_types = {
    'missing_only': 0,
    'extra_only': 0,
    'both': 0
}

for e in errors[:20]:
    has_missing = len(e['missing']) > 0
    has_extra = len(e['extra']) > 0
    if has_missing and not has_extra:
        error_types['missing_only'] += 1
    elif not has_missing and has_extra:
        error_types['extra_only'] += 1
    else:
        error_types['both'] += 1

print("\n📊 Типы ошибок (первые 20):")
print(f"   Только пропуск (FN): {error_types['missing_only']}")
print(f"   Только ложное срабатывание (FP): {error_types['extra_only']}")
print(f"   И то, и другое: {error_types['both']}")