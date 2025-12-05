import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Фиксированные признаки
FEATURES = ['year', 'km_driven', 'mileage', 'engine', 'max_power']

@st.cache_resource
def load_model():
    with open('ML/models/churn_model.pkl', 'rb') as f:
        return pickle.load(f)

def clear(a):
    if pd.isna(a):
        return None
    t = str(a).strip()
    if t:
        parts = t.split(' ')
        try:
            return float(parts[0])
        except:
            return None
    return None

model = load_model()

st.title("Предсказание модели")

# Переменная для хранения данных
df_global = None

# Вкладки для разных режимов
tab1, tab2, tab3 = st.tabs(["📁 Загрузить файл", "⌨️ Ручной ввод", "📊 Визуализации"])

# Вкладка 1: Загрузка файла
with tab1:
    uploaded_file = st.file_uploader("Загрузите CSV", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df_global = df.copy()  # Сохраняем для EDA
        df = df[[col for col in FEATURES if col in df.columns]]
        
        # Очистка данных
        for col in ['mileage', 'engine', 'max_power']:
            if col in df.columns:
                df[col] = df[col].apply(clear)
        
        df = df.dropna()
        
        if len(df) > 0:
            st.dataframe(df.head())
            
            if st.button("Предсказать"):
                try:
                    predictions = model.predict(df)
                    df['prediction'] = predictions
                    st.write("Результаты:")
                    st.dataframe(df[['prediction'] + FEATURES].head())
                except Exception as e:
                    st.error(f"Ошибка: {e}")

# Вкладка 2: Ручной ввод
with tab2:
    st.write("Введите значения:")
    
    year = st.number_input("Год", 1990, 2023, 2015, key="year_input")
    km_driven = st.number_input("Пробег", 0, 1000000, 50000, key="km_input")
    mileage = st.number_input("Расход", 0.0, 50.0, 15.0, key="mileage_input")
    engine = st.number_input("Двигатель", 500, 5000, 1200, key="engine_input")
    max_power = st.number_input("Мощность", 30, 1000, 80, key="power_input")
    
    if st.button("Предсказать", key="predict_btn"):
        input_data = pd.DataFrame([[year, km_driven, mileage, engine, max_power]], 
                                  columns=FEATURES)
        
        try:
            prediction = model.predict(input_data)[0]
            st.success(f"Результат: {prediction}")
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_data)[0]
                for i, p in enumerate(proba):
                    st.write(f"Класс {i}: {p:.1%}")
        except Exception as e:
            st.error(f"Ошибка: {e}")

# Вкладка 3: Визуализации
with tab3:
    # 1. Веса модели
    st.subheader("Веса модели")
    try:
        if hasattr(model, 'coef_'):
            coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(FEATURES, coef)
            ax.set_xlabel("Признаки")
            ax.set_ylabel("Вес")
            ax.set_title("Коэффициенты модели")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        elif hasattr(model, 'feature_importances_'):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(FEATURES, model.feature_importances_)
            ax.set_xlabel("Признаки")
            ax.set_ylabel("Важность")
            ax.set_title("Важность признаков")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.write("Модель не имеет атрибутов для визуализации весов")
    except Exception as e:
        st.write(f"Невозможно отобразить веса модели: {e}")
    
    # 2. EDA графики если есть данные
    if df_global is not None:
        st.subheader("Анализ данных")
        
        df_for_eda = df_global.copy()
        df_for_eda = df_for_eda[[col for col in FEATURES if col in df_for_eda.columns]]
        
        # Очистка для EDA
        for col in ['mileage', 'engine', 'max_power']:
            if col in df_for_eda.columns:
                df_for_eda[col] = df_for_eda[col].apply(clear)
        
        df_for_eda = df_for_eda.dropna()
        
        if len(df_for_eda) > 0:
            # Выбор признака для гистограммы
            numeric_cols = [c for c in df_for_eda.columns if pd.api.types.is_numeric_dtype(df_for_eda[c])]
            
            if numeric_cols:
                col = st.selectbox("Выберите признак для гистограммы", numeric_cols)
                
                if col:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.hist(df_for_eda[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
                    ax.set_xlabel(col)
                    ax.set_ylabel("Частота")
                    ax.set_title(f"Распределение признака: {col}")
                    st.pyplot(fig)
                
                # Основные статистики
                st.write("Основные статистики:")
                st.dataframe(df_for_eda.describe())
            else:
                st.write("Нет числовых признаков для анализа")
        else:
            st.write("Нет данных для анализа после очистки")
    else:
        st.write("Загрузите файл во вкладке 'Загрузить файл' для анализа данных")