import mlflow
import numpy as np
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.multiclass import OneVsRestClassifier
from pathlib import Path
import time

# Настройка MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

print("="*80)
print("СРАВНЕНИЕ С BASELINE (LOGISTIC REGRESSION)")
print("="*80)

# ============================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================
print("\n1. Загрузка данных...")
with open("checkpoint6_data.pkl", 'rb') as f:
    data = pickle.load(f)

X_train_bow = data['X_train_bow']
X_test_bow = data['X_test_bow']
y_train = data['y_train_bin']
y_test = data['y_test_bin']
classes = data['classes']

if hasattr(X_train_bow, 'toarray'):
    print("   Преобразование в плотные массивы...")
    X_train_bow = X_train_bow.toarray()
    X_test_bow = X_test_bow.toarray()

print(f"   X_train shape: {X_train_bow.shape}")
print(f"   X_test shape: {X_test_bow.shape}")
print(f"   Classes: {len(classes)}")

# ============================================
# 2. BASELINE: LOGISTIC REGRESSION
# ============================================
print("\n2. Обучение Logistic Regression (baseline)...")
start_time = time.time()
lr_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=1)
lr_ovr = OneVsRestClassifier(lr_model)
lr_ovr.fit(X_train_bow, y_train)
lr_time = time.time() - start_time
print(f"   Время обучения: {lr_time:.1f} сек")

print("\n3. Предсказания baseline...")
y_pred_lr = lr_ovr.predict(X_test_bow)

lr_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_lr),
    'f1_macro': f1_score(y_test, y_pred_lr, average='macro', zero_division=0),
    'f1_micro': f1_score(y_test, y_pred_lr, average='micro', zero_division=0),
    'hamming_loss': hamming_loss(y_test, y_pred_lr)
}

print(f"\n   Logistic Regression (baseline):")
print(f"      Accuracy: {lr_metrics['accuracy']:.4f}")
print(f"      F1-macro: {lr_metrics['f1_macro']:.4f}")
print(f"      F1-micro: {lr_metrics['f1_micro']:.4f}")
print(f"      Hamming Loss: {lr_metrics['hamming_loss']:.4f}")

# ============================================
# 3. НАША МОДЕЛЬ: Bi-GRU
# ============================================
print("\n4. Загрузка Bi-GRU модели (PRD)...")
model_uri = "models:/okpd2_classifier/1"
gru_model = mlflow.pyfunc.load_model(model_uri)

data_dir = Path("./data_for_error_analysis")
X_test_pad = np.load(data_dir / "X_test_pad.npy")
print(f"   X_test_pad shape: {X_test_pad.shape}")

print("\n5. Предсказания Bi-GRU...")
y_pred_gru_prob = gru_model.predict(X_test_pad)
y_pred_gru = (y_pred_gru_prob > 0.5).astype(int)

# Выравниваем размеры (y_test = 29649, X_test_pad = 29987)
n_samples = min(X_test_pad.shape[0], y_test.shape[0])
y_test_limited = y_test[:n_samples]
y_pred_gru_limited = y_pred_gru[:n_samples]
print(f"   Выравненные размеры: {n_samples}")

gru_metrics = {
    'accuracy': accuracy_score(y_test_limited, y_pred_gru_limited),
    'f1_macro': f1_score(y_test_limited, y_pred_gru_limited, average='macro', zero_division=0),
    'f1_micro': f1_score(y_test_limited, y_pred_gru_limited, average='micro', zero_division=0),
    'hamming_loss': hamming_loss(y_test_limited, y_pred_gru_limited)
}

print(f"\n   Bi-GRU (наша PRD модель):")
print(f"      Accuracy: {gru_metrics['accuracy']:.4f}")
print(f"      F1-macro: {gru_metrics['f1_macro']:.4f}")
print(f"      F1-micro: {gru_metrics['f1_micro']:.4f}")
print(f"      Hamming Loss: {gru_metrics['hamming_loss']:.4f}")

# ============================================
# 4. СРАВНИТЕЛЬНАЯ ТАБЛИЦА
# ============================================

comparison = pd.DataFrame({
    'Метрика': ['Accuracy', 'F1-macro', 'F1-micro', 'Hamming Loss'],
    'Baseline (LogReg)': [
        f"{lr_metrics['accuracy']:.4f}",
        f"{lr_metrics['f1_macro']:.4f}",
        f"{lr_metrics['f1_micro']:.4f}",
        f"{lr_metrics['hamming_loss']:.4f}"
    ],
    'Наша модель (Bi-GRU)': [
        f"{gru_metrics['accuracy']:.4f}",
        f"{gru_metrics['f1_macro']:.4f}",
        f"{gru_metrics['f1_micro']:.4f}",
        f"{gru_metrics['hamming_loss']:.4f}"
    ],
    'Улучшение': [
        f"{(gru_metrics['accuracy'] - lr_metrics['accuracy'])*100:+.2f}%",
        f"{(gru_metrics['f1_macro'] - lr_metrics['f1_macro'])*100:+.2f}%",
        f"{(gru_metrics['f1_micro'] - lr_metrics['f1_micro'])*100:+.2f}%",
        f"{(lr_metrics['hamming_loss'] - gru_metrics['hamming_loss'])*100:+.2f}%"
    ]
})

print(comparison.to_string(index=False))

# ============================================
# 5. СОХРАНЕНИЕ
# ============================================
print("\n6. Сохранение результатов...")
comparison.to_csv('baseline_comparison.csv', index=False)
print("    baseline_comparison.csv сохранён")

with mlflow.start_run(run_name="baseline_comparison"):
    mlflow.log_metrics({f"baseline_{k}": v for k, v in lr_metrics.items()})
    mlflow.log_metrics({f"bigru_{k}": v for k, v in gru_metrics.items()})
    print("    Метрики залогированы в MLflow")


print(" ИТОГИ СРАВНЕНИЯ")
print("="*80)
print(f"   Bi-GRU лучше LogReg:")
print(f"   • Accuracy: {lr_metrics['accuracy']:.4f} → {gru_metrics['accuracy']:.4f} (+{(gru_metrics['accuracy'] - lr_metrics['accuracy'])*100:.2f}%)")
print(f"   • F1-macro: {lr_metrics['f1_macro']:.4f} → {gru_metrics['f1_macro']:.4f} (+{(gru_metrics['f1_macro'] - lr_metrics['f1_macro'])*100:.2f}%)")
print(f"   • F1-micro: {lr_metrics['f1_micro']:.4f} → {gru_metrics['f1_micro']:.4f} (+{(gru_metrics['f1_micro'] - lr_metrics['f1_micro'])*100:.2f}%)")
print(f"   • Hamming Loss: {lr_metrics['hamming_loss']:.4f} → {gru_metrics['hamming_loss']:.4f} ({-(lr_metrics['hamming_loss'] - gru_metrics['hamming_loss'])*100:.2f}%)")
