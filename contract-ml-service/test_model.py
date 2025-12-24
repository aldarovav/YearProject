# test_all_features.py
import requests
import json
import base64
from io import BytesIO
from PIL import Image

BASE_URL = "http://localhost:8000"

def test_root():
    """Тестирование корневого эндпоинта"""
    print("1. Тестирование корневого эндпоинта...")
    try:
        response = requests.get(BASE_URL, timeout=5)
        print(f"   Статус: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✓ Сервер работает")
            data = response.json()
            print(f"   Сервис: {data.get('service')}")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")

def test_text_classification():
    """Тестирование классификации текста"""
    print("\n2. Тестирование классификации текста...")
    
    test_cases = [
        {
            "text": "выполнение работ по строительству жилого дома согласно проектной документации и смете",
            "description": "Строительные работы"
        },
        {
            "text": "поставка компьютерного оборудования системных блоков мониторов для оснащения рабочих мест",
            "description": "Компьютерное оборудование"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n   Тест {i}: {test['description']}")
        print(f"   Текст: {test['text'][:50]}...")
        
        try:
            headers = {'threshold': '0.2'}
            response = requests.post(
                f"{BASE_URL}/forward",
                json={"text": test["text"]},
                headers=headers,
                timeout=10
            )
            
            print(f"   Статус: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"   ✓ Успех")
                print(f"   Тип: {result.get('type')}")
                print(f"   Порог: {result.get('threshold')}")
                print(f"   Предсказания: {json.dumps(result['predictions'], ensure_ascii=False)}")
            else:
                print(f"   ✗ Ошибка: {response.text}")
                
        except Exception as e:
            print(f"   ✗ Исключение: {e}")

def test_image_upload():
    """Тестирование загрузки изображения"""
    print("\n3. Тестирование загрузки изображения...")
    
    # Создаем тестовое изображение
    try:
        img = Image.new('RGB', (50, 50), color='blue')
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='JPEG')
        image_data = img_byte_arr.getvalue()
        
        files = {'image': ('test_image.jpg', image_data, 'image/jpeg')}
        headers = {'threshold': '0.3'}
        
        response = requests.post(
            f"{BASE_URL}/forward",
            files=files,
            headers=headers,
            timeout=10
        )
        
        print(f"   Статус: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Успех")
            print(f"   Тип: {result.get('type')}")
            print(f"   Размер: {result.get('image_size')} байт")
            print(f"   Формат: {result.get('image_format')}")
            print(f"   Порог: {result.get('threshold')}")
        else:
            print(f"   ✗ Ошибка: {response.text}")
            
    except Exception as e:
        print(f"   ✗ Исключение: {e}")

def test_error_cases():
    """Тестирование обработки ошибок"""
    print("\n4. Тестирование обработки ошибок...")
    
    # Тест 1: Неверный формат JSON
    print("\n   Тест 1: Неверный формат JSON")
    try:
        response = requests.post(
            f"{BASE_URL}/forward",
            data="invalid json",
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        print(f"   Статус: {response.status_code} (ожидается 400)")
        if response.status_code == 400:
            print(f"   ✓ Корректная ошибка: {response.text}")
    except Exception as e:
        print(f"   ✗ Исключение: {e}")
    
    # Тест 2: Короткий текст
    print("\n   Тест 2: Короткий текст")
    try:
        response = requests.post(
            f"{BASE_URL}/forward",
            json={"text": "а"},
            timeout=5
        )
        print(f"   Статус: {response.status_code} (ожидается 400)")
        if response.status_code == 400:
            print(f"   ✓ Корректная ошибка: {response.text}")
    except Exception as e:
        print(f"   ✗ Исключение: {e}")
    
    # Тест 3: Нет поля text
    print("\n   Тест 3: Нет поля text в JSON")
    try:
        response = requests.post(
            f"{BASE_URL}/forward",
            json={"wrong_field": "текст"},
            timeout=5
        )
        print(f"   Статус: {response.status_code} (ожидается 400)")
        if response.status_code == 400:
            print(f"   ✓ Корректная ошибка: {response.text}")
    except Exception as e:
        print(f"   ✗ Исключение: {e}")

def test_history_and_stats():
    """Тестирование истории и статистики"""
    print("\n5. Тестирование истории и статистики...")
    
    try:
        # История
        response = requests.get(f"{BASE_URL}/history", timeout=5)
        if response.status_code == 200:
            history = response.json()
            print(f"   ✓ История: {len(history)} записей")
            
            # Показываем последние записи
            for item in history[-2:]:
                print(f"     - {item.get('request', {}).get('type', 'unknown')}: "
                      f"{item.get('processing_time', 0):.3f} сек")
        
        # Статистика
        response = requests.get(f"{BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print(f"   ✓ Статистика:")
            print(f"     Всего запросов: {stats.get('total_requests', 0)}")
            print(f"     Типы запросов: {stats.get('request_types', {})}")
            print(f"     Среднее время: {stats.get('processing_time', {}).get('avg', 0):.3f} сек")
            
    except Exception as e:
        print(f"   ✗ Исключение: {e}")

def test_delete_history():
    """Тестирование удаления истории"""
    print("\n6. Тестирование удаления истории...")
    
    # Тест 1: Без подтверждения
    print("\n   Тест 1: Без подтверждения")
    try:
        response = requests.delete(f"{BASE_URL}/history", timeout=5)
        print(f"   Статус: {response.status_code} (ожидается ошибка)")
        if response.status_code != 200:
            print(f"   ✓ Корректная ошибка: {response.json().get('error', '')}")
    except Exception as e:
        print(f"   ✗ Исключение: {e}")
    
    # Тест 2: С подтверждением
    print("\n   Тест 2: С подтверждением")
    try:
        response = requests.delete(f"{BASE_URL}/history?confirmation=yes", timeout=5)
        print(f"   Статус: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✓ История удалена: {response.json().get('message', '')}")
    except Exception as e:
        print(f"   ✗ Исключение: {e}")

if __name__ == "__main__":
    print("=" * 80)
    print("ПОЛНОЕ ТЕСТИРОВАНИЕ ML СЕРВИСА")
    print("=" * 80)
    
    test_root()
    test_text_classification()
    test_image_upload()
    test_error_cases()
    test_history_and_stats()
    test_delete_history()
    
    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 80)