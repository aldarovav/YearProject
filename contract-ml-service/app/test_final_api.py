# test_final_api.py
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_api():
    print("=== Тестирование ML Service API ===\n")
    
    # Тест 1: Проверка доступности
    print("1. Проверка доступности...")
    try:
        response = requests.get(BASE_URL, timeout=5)
        print(f"   ✓ Сервер работает: {response.json()}")
    except:
        print("   ✗ Сервер не доступен")
        return
    
    print("\n2. Тестирование классификации:")
    
    test_cases = [
        "выполнение работ по строительству жилого дома согласно проектной документации",
        "поставка компьютерного оборудования для оснащения рабочих мест",
        "предоставление в аренду офисных помещений на срок 12 месяцев",
        "оказание услуг по проведению обучающих курсов повышения квалификации"
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n   Тест {i}: {text[:50]}...")
        
        try:
            start = time.time()
            response = requests.post(
                f"{BASE_URL}/forward",
                json={"text": text},
                timeout=10
            )
            
            elapsed = time.time() - start
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✓ Успех ({elapsed:.3f} сек)")
                print(f"   Предсказания: {json.dumps(result['predictions'], ensure_ascii=False)}")
            else:
                print(f"   ✗ Ошибка {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"   ✗ Исключение: {e}")
    
    print("\n3. Получение истории запросов...")
    try:
        response = requests.get(f"{BASE_URL}/history", timeout=5)
        if response.status_code == 200:
            history = response.json()
            print(f"   ✓ Записей в истории: {len(history)}")
        else:
            print(f"   ✗ Ошибка: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Исключение: {e}")
    
    print("\n4. Получение статистики...")
    try:
        response = requests.get(f"{BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print(f"   ✓ Статистика получена")
            print(f"   Всего запросов: {stats.get('total_requests', 0)}")
        else:
            print(f"   ✗ Ошибка: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Исключение: {e}")
    
    print("\n=== Тестирование завершено ===")

if __name__ == "__main__":
    test_api()