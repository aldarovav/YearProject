import requests
import json

BASE_URL = "http://localhost:8000"

def main():
    print("=== Тестирование ML Service API ===\n")
    
    # Проверка доступности
    print("1. Проверка доступности сервера...")
    try:
        response = requests.get(BASE_URL, timeout=5)
        print(f"   Статус: {response.status_code}")
        print(f"   Ответ: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("   ❌ Сервер не доступен! Запустите сервер:")
        print("   python -m uvicorn app.main:app --reload")
        return
    except Exception as e:
        print(f"   Ошибка: {e}")
        return
    
    print("\n" + "="*60)
    
    # Тест POST /forward
    print("2. Тестируем POST /forward...")
    tests = [
        {"text": "строительство дома из кирпича с отделкой фасада"},
        {"text": "разработка мобильного приложения для бизнеса"},
        {"text": "проведение обучающих курсов по программированию"}
    ]
    
    for i, test_data in enumerate(tests, 1):
        print(f"\n   Тест {i}: {test_data['text'][:40]}...")
        try:
            response = requests.post(
                f"{BASE_URL}/forward",
                json=test_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            print(f"   Статус: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"   Результат: {json.dumps(result, indent=4, ensure_ascii=False)}")
            else:
                print(f"   Текст ошибки: {response.text}")
        except Exception as e:
            print(f"   Исключение: {e}")
    
    print("\n" + "="*60)
    
    # Тест GET /history
    print("3. Тестируем GET /history...")
    try:
        response = requests.get(f"{BASE_URL}/history", timeout=5)
        print(f"   Статус: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Количество записей в истории: {len(data)}")
            if data:
                print(f"   Пример записи: {json.dumps(data[0], indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"   Ошибка: {e}")
    
    print("\n" + "="*60)
    
    # Тест GET /stats
    print("4. Тестируем GET /stats...")
    try:
        response = requests.get(f"{BASE_URL}/stats", timeout=5)
        print(f"   Статус: {response.status_code}")
        if response.status_code == 200:
            stats = response.json()
            print(f"   Статистика: {json.dumps(stats, indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"   Ошибка: {e}")
    
    print("\n" + "="*60)
    
    # Тест DELETE /history
    print("5. Тестируем DELETE /history...")
    try:
        # С подтверждением
        response = requests.delete(f"{BASE_URL}/history?confirmation=yes", timeout=5)
        print(f"   Статус: {response.status_code}")
        print(f"   Ответ: {response.json()}")
    except Exception as e:
        print(f"   Ошибка: {e}")

if __name__ == "__main__":
    main()