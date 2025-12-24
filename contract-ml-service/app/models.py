import pickle
import re
import numpy as np

class ContractModel:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.mlb = None
        
    def load_model(self, model_path='contract_model.pkl'):
        """Загрузка модели из файла"""
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.vectorizer = data['vectorizer']
                self.mlb = data['mlb']
                print(f"Модель загружена. Классов: {len(self.mlb.classes_)}")
                return True
        except FileNotFoundError:
            print(f"Файл модели не найден: {model_path}")
            return False
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False
    
    def clean_text(self, text):
        """Очистка текста договора"""
        text = str(text).lower()
        # Удаляем специальные символы, оставляем только буквы, цифры, пробелы
        text = re.sub(r'[^а-я0-9\\s]', ' ', text)
        text = re.sub(r'\\s+', ' ', text)
        return text.strip()
    
    def predict(self, text, threshold=0.2):
        """
        Предсказание кодов ОКПД2 с настраиваемым порогом
        
        Args:
            text (str): Текст договора для классификации
            threshold (float): Порог вероятности (0.0-1.0)
            
        Returns:
            list: Список предсказанных кодов с вероятностями
        """
        if self.model is None:
            return [{"error": "Модель не загружена"}]
        
        if not text or len(text.strip()) == 0:
            return [{"error": "Пустой текст"}]
        
        # Очистка
        clean = self.clean_text(text)
        
        if len(clean) < 5:
            return [{"error": "Текст слишком короткий после очистки"}]
        
        try:
            # Векторизация
            X = self.vectorizer.transform([clean])
            
            # Вероятности
            probabilities = self.model.predict_proba(X)[0]
            
            # Собираем предсказания выше порога
            predictions = []
            for i, prob in enumerate(probabilities):
                if prob >= threshold:
                    predictions.append({
                        'okpd_code': str(self.mlb.classes_[i]),
                        'probability': float(prob)
                    })
            
            # Если ничего не предсказано, возвращаем топ-1
            if not predictions:
                max_idx = np.argmax(probabilities)
                max_prob = probabilities[max_idx]
                predictions.append({
                    'okpd_code': str(self.mlb.classes_[max_idx]),
                    'probability': float(max_prob)
                })
            
            return predictions
            
        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            return [{"error": f"Ошибка предсказания: {str(e)}"}]

# Инициализация модели
model = ContractModel()
model.load_model()