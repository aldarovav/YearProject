import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:5000")

print("="*80)
print("СРАВНЕНИЕ С BASELINE (на основе результатов чекпойнта 6)")
print("="*80)

# Метрики из чекпойнта 6 (уже известны)
baseline_metrics = {
    'model': 'Logistic Regression',
    'accuracy': 0.6771,
    'f1_macro': 0.5247,
    'f1_micro': 0.7662,
    'hamming_loss': 0.0053
}

bigru_metrics = {
    'model': 'Bi-GRU (PRD)',
    'accuracy': 0.7764,
    'f1_macro': 0.5080,
    'f1_micro': 0.8177,
    'hamming_loss': 0.0043
}

print("\n Метрики Logistic Regression (baseline):")
print(f"   Accuracy: {baseline_metrics['accuracy']:.4f}")
print(f"   F1-macro: {baseline_metrics['f1_macro']:.4f}")
print(f"   F1-micro: {baseline_metrics['f1_micro']:.4f}")
print(f"   Hamming Loss: {baseline_metrics['hamming_loss']:.4f}")

print("\n Метрики Bi-GRU (наша PRD модель):")
print(f"   Accuracy: {bigru_metrics['accuracy']:.4f}")
print(f"   F1-macro: {bigru_metrics['f1_macro']:.4f}")
print(f"   F1-micro: {bigru_metrics['f1_micro']:.4f}")
print(f"   Hamming Loss: {bigru_metrics['hamming_loss']:.4f}")

# ============================================
# СРАВНИТЕЛЬНАЯ ТАБЛИЦА
# ============================================
print("\n" + "="*80)
print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
print("="*80)

comparison = pd.DataFrame([
    baseline_metrics,
    bigru_metrics
])

print(comparison.to_string(index=False))

# Вычисляем улучшения
acc_improvement = (bigru_metrics['accuracy'] - baseline_metrics['accuracy']) * 100
f1_micro_improvement = (bigru_metrics['f1_micro'] - baseline_metrics['f1_micro']) * 100
f1_macro_change = (bigru_metrics['f1_macro'] - baseline_metrics['f1_macro']) * 100
hamming_improvement = (baseline_metrics['hamming_loss'] - bigru_metrics['hamming_loss']) * 100

print("\n" + "="*80)
print(" УЛУЧШЕНИЕ Bi-GRU ОТНОСИТЕЛЬНО BASELINE")
print("="*80)
print(f"   Accuracy:    {baseline_metrics['accuracy']:.4f} → {bigru_metrics['accuracy']:.4f}  (+{acc_improvement:.2f}%)")
print(f"   F1-macro:    {baseline_metrics['f1_macro']:.4f} → {bigru_metrics['f1_macro']:.4f}  ({f1_macro_change:+.2f}%)")
print(f"   F1-micro:    {baseline_metrics['f1_micro']:.4f} → {bigru_metrics['f1_micro']:.4f}  (+{f1_micro_improvement:.2f}%)")
print(f"   Hamming Loss: {baseline_metrics['hamming_loss']:.4f} → {bigru_metrics['hamming_loss']:.4f}  (-{hamming_improvement:.2f}%)")

print("\n ИНТЕРПРЕТАЦИЯ:")
print("   • Bi-GRU лучше по общей точности (Accuracy, F1-micro, Hamming Loss)")
print("   • Logistic Regression лучше работает с редкими классами (F1-macro выше)")
print("   → Bi-GRU выбран как PRD из-за более высокой общей точности, ")
print("     что критично для задачи классификации ОКПД2.")

# ============================================
# СОХРАНЕНИЕ
# ============================================
print("\n Сохранение результатов...")
comparison.to_csv('baseline_comparison.csv', index=False)
print("    baseline_comparison.csv сохранён")

# Логируем в MLflow
with mlflow.start_run(run_name="baseline_comparison"):
    mlflow.log_metrics({f"baseline_{k}": v for k, v in baseline_metrics.items() if k != 'model'})
    mlflow.log_metrics({f"bigru_{k}": v for k, v in bigru_metrics.items() if k != 'model'})
    print("    Метрики залогированы в MLflow")

print("\n Готово!")