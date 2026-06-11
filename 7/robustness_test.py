import mlflow
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score

# Настройка MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

print("="*80)
print("ROBUSTNESS ТЕСТЫ")
print("="*80)

# ============================================
# 1. ЗАГРУЗКА МОДЕЛИ И ДАННЫХ
# ============================================
print("\n1. Загрузка модели и данных...")
model_uri = "models:/okpd2_classifier/1"
model = mlflow.pyfunc.load_model(model_uri)

data_dir = Path("./data_for_error_analysis")
X_test_pad = np.load(data_dir / "X_test_pad.npy")
with open(data_dir / "y_test.pkl", 'rb') as f:
    y_test = pickle.load(f)
with open(data_dir / "X_test_texts.pkl", 'rb') as f:
    X_test_texts = pickle.load(f)

print(f"   Тестовых примеров: {X_test_pad.shape[0]}")
print(f"   Классов: {y_test.shape[1]}")

# ============================================
# 2. БАЗОВЫЕ ПРЕДСКАЗАНИЯ
# ============================================
print("\n2. Базовые предсказания (без изменений)...")
y_pred_base = model.predict(X_test_pad)
y_pred_base_bin = (y_pred_base > 0.5).astype(int)
base_accuracy = accuracy_score(y_test[:len(y_pred_base_bin)], y_pred_base_bin)
print(f"   Базовый Accuracy: {base_accuracy:.4f}")

# ============================================
# 3. ТЕСТ 1: ШУМ В ТОКЕНАХ
# ============================================
print("\n3. Тест 1: Добавление шума в токены...")
# Случайно заменяем 5% токенов на случайные
noise_mask = np.random.random(X_test_pad.shape) < 0.05
X_test_noise = X_test_pad.copy()
random_tokens = np.random.randint(1, 100000, X_test_pad.shape)
X_test_noise[noise_mask] = random_tokens[noise_mask]

y_pred_noise = model.predict(X_test_noise)
y_pred_noise_bin = (y_pred_noise > 0.5).astype(int)
noise_accuracy = accuracy_score(y_test[:len(y_pred_noise_bin)], y_pred_noise_bin)
print(f"   Accuracy после шума: {noise_accuracy:.4f}")
print(f"   Падение: {base_accuracy - noise_accuracy:.4f} ({(base_accuracy - noise_accuracy)/base_accuracy*100:.1f}%)")

# ============================================
# 4. ТЕСТ 2: УДАЛЕНИЕ КАЖДОГО 10-ГО ТОКЕНА
# ============================================
print("\n4. Тест 2: Удаление каждого 10-го токена...")
X_test_dropped = X_test_pad.copy()
# Удаляем каждый 10-й токен (заменяем на 0 - padding)
for i in range(0, X_test_dropped.shape[1], 10):
    X_test_dropped[:, i] = 0

y_pred_dropped = model.predict(X_test_dropped)
y_pred_dropped_bin = (y_pred_dropped > 0.5).astype(int)
dropped_accuracy = accuracy_score(y_test[:len(y_pred_dropped_bin)], y_pred_dropped_bin)
print(f"   Accuracy после удаления: {dropped_accuracy:.4f}")
print(f"   Падение: {base_accuracy - dropped_accuracy:.4f} ({(base_accuracy - dropped_accuracy)/base_accuracy*100:.1f}%)")

# ============================================
# 5. ТЕСТ 3: ОБРЕЗАНИЕ ТЕКСТА (50% длины)
# ============================================
print("\n5. Тест 3: Обрезание текста до 50% длины...")
half_len = X_test_pad.shape[1] // 2
X_test_half = X_test_pad[:, :half_len]
# Паддинг до исходной длины
X_test_half_padded = np.zeros_like(X_test_pad)
X_test_half_padded[:, :half_len] = X_test_half

y_pred_half = model.predict(X_test_half_padded)
y_pred_half_bin = (y_pred_half > 0.5).astype(int)
half_accuracy = accuracy_score(y_test[:len(y_pred_half_bin)], y_pred_half_bin)
print(f"   Accuracy после обрезания: {half_accuracy:.4f}")
print(f"   Падение: {base_accuracy - half_accuracy:.4f} ({(base_accuracy - half_accuracy)/base_accuracy*100:.1f}%)")

# ============================================
# 6. ТЕСТ 4: ЗАМЕНА СЛОВ НА СИНОНИМЫ (эмуляция)
# ============================================
print("\n6. Тест 4: Синонимические замены (эмуляция)...")
# В реальности нужен словарь синонимов. Эмулируем сдвигом токенов
X_test_syn = X_test_pad.copy()
# Сдвигаем некоторые токены на 1
syn_mask = np.random.random(X_test_syn.shape) < 0.03
X_test_syn[syn_mask] = X_test_syn[syn_mask] + 1

y_pred_syn = model.predict(X_test_syn)
y_pred_syn_bin = (y_pred_syn > 0.5).astype(int)
syn_accuracy = accuracy_score(y_test[:len(y_pred_syn_bin)], y_pred_syn_bin)
print(f"   Accuracy после синонимов: {syn_accuracy:.4f}")
print(f"   Падение: {base_accuracy - syn_accuracy:.4f} ({(base_accuracy - syn_accuracy)/base_accuracy*100:.1f}%)")

# ============================================
# 7. СВОДНАЯ ТАБЛИЦА ROBUSTNESS
# ============================================

robustness_results = pd.DataFrame({
    'Тест': ['Базовый', 'Шум (5% токенов)', 'Удаление каждого 10-го токена', 'Обрезание до 50%', 'Синонимические замены'],
    'Accuracy': [
        f"{base_accuracy:.4f}",
        f"{noise_accuracy:.4f}",
        f"{dropped_accuracy:.4f}",
        f"{half_accuracy:.4f}",
        f"{syn_accuracy:.4f}"
    ],
    'Падение': [
        "-",
        f"{base_accuracy - noise_accuracy:.4f}",
        f"{base_accuracy - dropped_accuracy:.4f}",
        f"{base_accuracy - half_accuracy:.4f}",
        f"{base_accuracy - syn_accuracy:.4f}"
    ],
    'Относительное падение': [
        "-",
        f"{(base_accuracy - noise_accuracy)/base_accuracy*100:.1f}%",
        f"{(base_accuracy - dropped_accuracy)/base_accuracy*100:.1f}%",
        f"{(base_accuracy - half_accuracy)/base_accuracy*100:.1f}%",
        f"{(base_accuracy - syn_accuracy)/base_accuracy*100:.1f}%"
    ]
})

print(robustness_results.to_string(index=False))

# ============================================
# 8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================
print("\n7. Сохранение результатов...")
robustness_results.to_csv('robustness_test_results.csv', index=False)
print("    robustness_test_results.csv сохранён")

# Логируем в MLflow
with mlflow.start_run(run_name="robustness_tests", nested=True):
    mlflow.log_metrics({
        'base_accuracy': base_accuracy,
        'noise_accuracy': noise_accuracy,
        'dropped_accuracy': dropped_accuracy,
        'half_accuracy': half_accuracy,
        'syn_accuracy': syn_accuracy
    })
    print("    Результаты залогированы в MLflow")

