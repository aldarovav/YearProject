import kagglehub
import pandas as pd
import numpy as np
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pathlib import Path

# ============================================
# 1. ЗАГРУЗКА СЫРЫХ ДАННЫХ
# ============================================
print("1. Загрузка сырых данных...")
file_path = "contracts_dataset_unique.json"
dataset_path = kagglehub.dataset_download("aldarovalexander/contract")
full_file_path = f"{dataset_path}/{file_path}"
df = pd.read_json(full_file_path, dtype={'regNum': str})
print(f"   Загружено {len(df)} записей")

# ============================================
# 2. ПОДГОТОВКА ТЕКСТОВ И МЕТОК
# ============================================
print("2. Подготовка текстов и меток...")
all_codes = []
for codes in df['OKPD2_codes']:
    if codes:
        all_codes.extend(codes)

unique_codes = sorted(set(all_codes))
print(f"   Всего уникальных кодов: {len(unique_codes)}")

# Фильтрация записей с кодами
mask = df['OKPD2_codes'].apply(lambda x: x is not None and len(x) > 0)
df_filtered = df[mask].copy()
df_filtered['target'] = df_filtered['OKPD2_codes']

print(f"   Размер выборки после фильтрации: {len(df_filtered)}")

# One-hot encoding меток
mlb = MultiLabelBinarizer(classes=unique_codes)
y_bin = mlb.fit_transform(df_filtered['target'])
print(f"   Размер меток: {y_bin.shape}")

# Минимальная очистка текстов
def minimal_clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b\w{1,2}\b', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

texts = df_filtered['contractSubjectFull'].apply(minimal_clean_text).values
print(f"   Пример текста: {texts[0][:200]}...")

# ============================================
# 3. РАЗДЕЛЕНИЕ НА ВЫБОРКИ
# ============================================
print("3. Разделение на выборки...")
X_train, X_temp, y_train, y_temp = train_test_split(
    texts, y_bin, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"   Train: {len(X_train)}")
print(f"   Val: {len(X_val)}")
print(f"   Test: {len(X_test)}")

# ============================================
# 4. ТОКЕНИЗАЦИЯ (как в чекпойнте 6)
# ============================================
print("4. Токенизация...")
MAX_FEATURES = 100000
MAX_LEN = 1500

tokenizer = Tokenizer(num_words=MAX_FEATURES, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_val_pad = pad_sequences(X_val_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

print(f"   X_train_pad shape: {X_train_pad.shape}")
print(f"   Размер словаря: {len(tokenizer.word_index)}")

# ============================================
# 5. СОХРАНЕНИЕ ВСЕГО НЕОБХОДИМОГО
# ============================================
print("5. Сохранение данных...")
output_dir = Path("./data_for_error_analysis")
output_dir.mkdir(exist_ok=True)

# Сохраняем тексты (нужны для error analysis)
with open(output_dir / "X_test_texts.pkl", 'wb') as f:
    pickle.dump(X_test, f)

# Сохраняем метки
with open(output_dir / "y_test.pkl", 'wb') as f:
    pickle.dump(y_test, f)

# Сохраняем классы
with open(output_dir / "classes.pkl", 'wb') as f:
    pickle.dump(unique_codes, f)

# Сохраняем токенизатор
import joblib
joblib.dump(tokenizer, output_dir / "tokenizer.pkl")

# Сохраняем паддинги (на всякий случай)
np.save(output_dir / "X_test_pad.npy", X_test_pad)

print(f"✅ Все данные сохранены в {output_dir}")
print(f"\nСодержимое папки:")
for f in output_dir.iterdir():
    print(f"   {f.name}")

# ============================================
# 6. ИНФОРМАЦИЯ ДЛЯ СЛЕДУЮЩЕГО ШАГА
# ============================================
print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ГОТОВО! Теперь у вас есть:                                  ║
║                                                              ║
║   📁 data_for_error_analysis/                               ║
║      ├── X_test_texts.pkl   - исходные тексты (очищенные)   ║
║      ├── y_test.pkl         - бинарные метки                ║
║      ├── classes.pkl        - список классов (84 кода)      ║
║      ├── tokenizer.pkl      - токенизатор для GRU           ║
║      └── X_test_pad.npy     - паддинги (готовые входы)      ║
║                                                              ║
║  ДЛЯ ERROR ANALYSIS:                                        ║
║  1. Загрузите модель из MLflow по тегу PRD                  ║
║  2. Загрузите X_test_pad.npy и y_test.pkl                  ║
║  3. Сделайте предсказания и найдите ошибки                  ║
╚══════════════════════════════════════════════════════════════╝
""")