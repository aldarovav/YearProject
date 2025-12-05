import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

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
tab1, tab2, tab3 = st.tabs(["Загрузить файл", "Ручной ввод", "Визуализации"])

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
            st.success(f"Результат: {prediction:.2f}")
            
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
    
    # 2. Ключевые графики EDA если есть данные
    if df_global is not None:
        st.subheader("Ключевые графики EDA")
        
        df_for_eda = df_global.copy()
        df_for_eda = df_for_eda[[col for col in FEATURES if col in df_for_eda.columns]]
        
        # Очистка для EDA
        for col in ['mileage', 'engine', 'max_power']:
            if col in df_for_eda.columns:
                df_for_eda[col] = df_for_eda[col].apply(clear)
        
        df_for_eda = df_for_eda.dropna()
        
        if len(df_for_eda) > 0:
            # Основные статистики
            st.write("### Основные статистики:")
            st.dataframe(df_for_eda.describe())
            
            # Гистограммы распределения для всех числовых признаков
            st.write("### Распределение числовых признаков")
            numeric_cols = [c for c in df_for_eda.columns if pd.api.types.is_numeric_dtype(df_for_eda[c])]
            
            if numeric_cols:
                # Определяем количество строк и колонок для сетки графиков
                n_cols = 2
                n_rows = (len(numeric_cols) + 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
                axes = axes.flatten()
                
                for idx, col in enumerate(numeric_cols):
                    if idx < len(axes):
                        axes[idx].hist(df_for_eda[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
                        axes[idx].set_xlabel(col)
                        axes[idx].set_ylabel("Частота")
                        axes[idx].set_title(f"Распределение: {col}")
                
                # Скрываем пустые subplot
                for idx in range(len(numeric_cols), len(axes)):
                    fig.delaxes(axes[idx])
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Матрица корреляций (без автокорреляций)
                st.write("### Корреляции между признаками")
                corr_matrix = df_for_eda.corr()

                # Создаем маску для скрытия диагонали (автокорреляций)
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                        center=0, square=True, ax=ax, linewidths=1)
                ax.set_title("Корреляции между признаками (без автокорреляций)")
                st.pyplot(fig)
                
                # Box plot для выявления выбросов
                st.write("### Box plot для выявления выбросов")
                fig, axes = plt.subplots(1, len(numeric_cols), figsize=(20, 6))
                
                if len(numeric_cols) == 1:
                    axes = [axes]
                
                for idx, col in enumerate(numeric_cols):
                    if idx < len(axes):
                        axes[idx].boxplot(df_for_eda[col].dropna())
                        axes[idx].set_title(f"Box plot: {col}")
                        axes[idx].set_ylabel(col)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Диаграмма рассеяния для пар признаков
                if len(numeric_cols) >= 2:
                    st.write("### Диаграммы рассеяния")
                    
                    # Выбираем несколько комбинаций признаков
                    pairs_to_plot = []
                    for i in range(len(numeric_cols)):
                        for j in range(i+1, len(numeric_cols)):
                            if len(pairs_to_plot) < 4:  # Ограничиваем 4 графиками
                                pairs_to_plot.append((numeric_cols[i], numeric_cols[j]))
                    
                    n_pairs = len(pairs_to_plot)
                    if n_pairs > 0:
                        n_cols_scatter = 2
                        n_rows_scatter = (n_pairs + 1) // n_cols_scatter
                        
                        fig, axes = plt.subplots(n_rows_scatter, n_cols_scatter, figsize=(15, 5*n_rows_scatter))
                        
                        if n_pairs == 1:
                            axes = [axes]
                        else:
                            axes = axes.flatten()
                        
                        for idx, (x_col, y_col) in enumerate(pairs_to_plot):
                            if idx < len(axes):
                                axes[idx].scatter(df_for_eda[x_col], df_for_eda[y_col], alpha=0.5)
                                axes[idx].set_xlabel(x_col)
                                axes[idx].set_ylabel(y_col)
                                axes[idx].set_title(f"{x_col} vs {y_col}")
                        
                        # Скрываем пустые subplot
                        for idx in range(n_pairs, len(axes)):
                            fig.delaxes(axes[idx])
                        
                        plt.tight_layout()
                        st.pyplot(fig)
            else:
                st.write("Нет числовых признаков для анализа")
        else:
            st.write("Нет данных для анализа после очистки")
    else:
        st.write("Загрузите файл во вкладке 'Загрузить файл' для анализа данных")