import mlflow
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
from collections import Counter

# Настройка MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# ============================================
# 1. ЗАГРУЗКА МОДЕЛИ ПО ТЕГУ PRD
# ============================================
print("1. Загрузка модели из MLflow...")
model_uri = "models:/okpd2_classifier@PRD"
model = mlflow.pyfunc.load_model(model_uri)
print("    Модель загружена")

# ============================================
# 2. ЗАГРУЗКА ТЕСТОВЫХ ДАННЫХ
# ============================================
print("\n2. Загрузка тестовых данных...")
data_dir = Path("./data_for_error_analysis")

X_test_pad = np.load(data_dir / "X_test_pad.npy")
print(f"   X_test_pad shape: {X_test_pad.shape}")

with open(data_dir / "y_test.pkl", 'rb') as f:
    y_test = pickle.load(f)
print(f"   y_test shape: {y_test.shape}")

with open(data_dir / "classes.pkl", 'rb') as f:
    classes = pickle.load(f)
print(f"   Классов: {len(classes)}")

with open(data_dir / "X_test_texts.pkl", 'rb') as f:
    X_test_texts = pickle.load(f)
print(f"   Текстов загружено: {len(X_test_texts)}")

# ============================================
# 3. ПРЕДСКАЗАНИЯ
# ============================================
print("\n3. Предсказание...")
predictions = model.predict(X_test_pad)
y_pred_bin = (predictions > 0.5).astype(int)
print(f"    Предсказания сделаны")

# ============================================
# 4. ПОИСК ОШИБОК (первые 500 для скорости)
# ============================================
print("\n4. Поиск ошибок...")
errors = []
max_samples = min(500, X_test_pad.shape[0])

for i in range(max_samples):
    true_set = set(np.where(y_test[i] == 1)[0])
    pred_set = set(np.where(y_pred_bin[i] == 1)[0])
    
    if true_set != pred_set:
        error = {
            'index': i,
            'text': X_test_texts[i][:400] + "..." if len(X_test_texts[i]) > 400 else X_test_texts[i],
            'true_codes': [classes[j] for j in true_set],
            'pred_codes': [classes[j] for j in pred_set],
            'missing': [classes[j] for j in (true_set - pred_set)],
            'extra': [classes[j] for j in (pred_set - true_set)],
            'confidence': float(np.max(predictions[i]))
        }
        errors.append(error)

print(f"   Найдено ошибок: {len(errors)} из {max_samples} ({len(errors)/max_samples*100:.1f}%)")

# ============================================
# 5. КЛАССИФИКАЦИЯ ОШИБОК ПО ТИПАМ
# ============================================
print("\n5. Классификация ошибок...")

error_types = {
    'missing_only': 0,      # только пропуск (FN)
    'extra_only': 0,        # только лишнее (FP)
    'both': 0,              # и пропуск, и лишнее
    'wrong_confidence': 0   # низкая уверенность (уверенность < 0.6)
}

for e in errors:
    has_missing = len(e['missing']) > 0
    has_extra = len(e['extra']) > 0
    low_conf = e['confidence'] < 0.6
    
    if low_conf:
        error_types['wrong_confidence'] += 1
    elif has_missing and not has_extra:
        error_types['missing_only'] += 1
    elif not has_missing and has_extra:
        error_types['extra_only'] += 1
    else:
        error_types['both'] += 1

print(f"   Только пропуск (False Negative): {error_types['missing_only']}")
print(f"   Только лишнее (False Positive): {error_types['extra_only']}")
print(f"   И пропуск, и лишнее: {error_types['both']}")
print(f"   Низкая уверенность (<0.6): {error_types['wrong_confidence']}")

# ============================================
# 6. КАКИЕ КЛАССЫ ЧАЩЕ ВСЕГО ПРОПУЩЕНЫ
# ============================================
print("\n6. Самые частые пропущенные классы...")
missing_counter = Counter()
for e in errors:
    for code in e['missing']:
        missing_counter[code] += 1

print("   Топ-10 пропущенных кодов ОКПД2:")
for code, count in missing_counter.most_common(10):
    print(f"      {code}: {count} раз")

# ============================================
# 7. КАКИЕ КЛАССЫ ЧАЩЕ ВСЕГО ЛИШНИЕ
# ============================================
print("\n7. Самые частые лишние классы...")
extra_counter = Counter()
for e in errors:
    for code in e['extra']:
        extra_counter[code] += 1

print("   Топ-10 лишних кодов ОКПД2:")
for code, count in extra_counter.most_common(10):
    print(f"      {code}: {count} раз")

# ============================================
# 8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================
print("\n8. Сохранение результатов...")

# Сохраняем все ошибки
df_errors = pd.DataFrame(errors)
df_errors.to_csv('error_analysis_full.csv', index=False)
print(f"    error_analysis_full.csv ({len(errors)} записей)")

# Сохраняем топ-20 ошибок с текстами
df_errors.head(20).to_csv('error_analysis_top20.csv', index=False)
print(f"    error_analysis_top20.csv (20 записей)")

# Сохраняем статистику
stats = pd.DataFrame([error_types])
stats.to_csv('error_analysis_stats.csv', index=False)
print(f"    error_analysis_stats.csv")

# ============================================
# 9. ВЫВОД ПРИМЕРОВ ОШИБОК
# ============================================

for i, e in enumerate(errors[:10]):
    print(f"\n--- Ошибка #{i+1} ---")
    print(f"Текст: {e['text'][:200]}...")
    print(f"Верные коды: {e['true_codes']}")
    print(f"Предсказанные коды: {e['pred_codes']}")
    print(f"Пропущенные: {e['missing']}")
    print(f"Лишние: {e['extra']}")
    print(f"Уверенность модели: {e['confidence']:.3f}")
