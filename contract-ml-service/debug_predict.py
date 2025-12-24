# debug_predict.py
import pickle
import re

# Загружаем модель
with open('contract_model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    vectorizer = data['vectorizer']
    mlb = data['mlb']

print("Классы модели:", mlb.classes_)
print("Количество классов:", len(mlb.classes_))

# Тестовые тексты
test_texts = [
    "строительство дома из кирпича",
    "разработка мобильного приложения",
    "обучение программированию python",
    "поставка компьютеров в офис",
    "ремонт автомобилей"
]

# Функция очистки
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^а-я0-9\\s]', ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

print("\n" + "="*50 + "\n")

for text in test_texts:
    clean = clean_text(text)
    print(f"Текст: {text}")
    print(f"Очищенный: {clean}")
    
    # Векторизация
    X = vectorizer.transform([clean])
    print(f"Размер вектора: {X.shape}")
    print(f"Ненулевые элементы: {X.nnz}")
    
    # Предсказание
    pred = model.predict(X)
    print(f"Предсказание (бинарный формат): {pred}")
    print(f"Формат предсказания: {type(pred)}, shape: {pred.shape}")
    
    # Вероятности
    try:
        proba = model.predict_proba(X)
        print(f"Вероятности: {proba}")
        print(f"Максимальная вероятность: {proba.max():.4f}")
    except:
        pass
    
    # Преобразуем в метки
    labels = mlb.inverse_transform(pred)
    print(f"Метки: {labels}")
    
    print("-" * 50)