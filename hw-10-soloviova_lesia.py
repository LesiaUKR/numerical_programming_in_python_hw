# -*- coding: utf-8 -*-
"""goit-numericalpy-hw-10-soloviova_lesia.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1UVGXimMDtWvGyQzLCJcpgp08N32klBCm
"""

import numpy as np
import os
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix

# Крок 1: Завантаження даних
print("Крок 1: Завантаження даних")
if not os.path.exists("WorldHappinessReport.zip"):  # Перевірка, чи файл вже завантажено
    !wget -O WorldHappinessReport.zip https://github.com/goitacademy/NUMERICAL-PROGRAMMING-IN-PYTHON/blob/main/WorldHappinessReport.zip?raw=true
    print("Файл WorldHappinessReport.zip успішно завантажено.")
else:
    print("Файл WorldHappinessReport.zip вже існує. Пропускаємо завантаження.")

# Крок 2: Перевірка наявності та розпакування
print("Крок 2: Перевірка наявності та розпакування")
required_files = ["2015.csv", "2016.csv", "2017.csv", "2018.csv"]
all_files_exist = all(os.path.exists(file) for file in required_files)  # Перевірка наявності всіх файлів

if not all_files_exist:  # Якщо файли відсутні, розпакувати архів
    try:
        with zipfile.ZipFile("WorldHappinessReport.zip", 'r') as zip_ref:
            zip_ref.extractall()
        print("Файли успішно розпаковано.")
    except zipfile.BadZipFile:
        print("Помилка: файл WorldHappinessReport.zip пошкоджений або не є ZIP-архівом.")
    except FileNotFoundError:
        print("Помилка: файл WorldHappinessReport.zip не знайдено.")
else:
    print("Всі файли вже існують. Пропускаємо розпакування.")

# Крок 3: Завантаження конкретного файлу
print("Крок 3: Завантаження конкретного файлу")
if os.path.exists("2017.csv"):  # Перевірка наявності файлу
    data = pd.read_csv("2017.csv")
    print("Файл 2017.csv успішно завантажено.")
else:
    print("Помилка: файл 2017.csv не знайдено.")

# Крок 4: Інформація про дані
print("Крок 4: Інформація про дані")
if 'data' in locals():  # Перевірка, чи дані були завантажені
    print("Інформація про дані:")
    print(data.info())
    print("\nСтатистики даних:")
    print(data.describe())
else:
    print("Дані не завантажено. Перевірте наявність файлів.")

# Крок 4: Побудова діаграм розподілу числових ознак
print("Крок 5: Побудова діаграм розподілу числових ознак")
numeric_columns = data.select_dtypes(include=[np.number]).columns
for column in numeric_columns:
    sns.histplot(data[column], kde=True)
    plt.title(f'Розподіл {column}')
    plt.show()

# Крок 5: Відображення кореляційної матриці
print("Крок 6: Відображення кореляційної матриці")
corr_matrix = data[numeric_columns].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Кореляційна матриця')
plt.show()

# Крок 6: Висновок про лінійний зв'язок між ознаками
print("Крок 7: Висновок про лінійний зв'язок між ознаками")
print("Висновок: Сильний позитивний зв'язок спостерігається між Happiness.Score та Economy..GDP.per.Capita., а також між Happiness.Score та Family. Інші ознаки мають слабший зв'язок.")

# Крок 7: Відображення розподілу цільової ознаки за країнами
print("Крок 8: Відображення розподілу цільової ознаки за країнами")
fig = px.choropleth(data,
                    locations="Country",
                    locationmode="country names",
                    color="Happiness.Score",
                    title="Індекс щастя за країнами (2017)")
fig.show()

# Крок 8: Стандартизація даних
print("Крок 9: Стандартизація даних")
def data_scale(data, scaler_type='minmax'):
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'std':
        scaler = StandardScaler()
    elif scaler_type == 'norm':
        scaler = Normalizer()

    scaler.fit(data)
    res = scaler.transform(data)
    return res

data_scaled = data_scale(data[numeric_columns])
df_scaled = pd.DataFrame(data_scaled, columns=numeric_columns)
print("Стандартизовані дані (перші 5 рядків):")
print(df_scaled.head())

# Крок 9: Відображення статистик стандартизованого набору даних
print("Крок 10: Відображення статистик стандартизованого набору даних")
print("Статистики стандартизованих даних:")
print(df_scaled.describe())

# Крок 10: Побудова моделі кластеризації
print("Крок 11: Побудова моделі кластеризації")
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(df_scaled)
labels = gmm.predict(df_scaled)
data['Cluster'] = labels
print("Кластеризація завершена. Додано стовпець 'Cluster' до даних.")

# Крок 11: Побудова теплової мапи для кластерів
print("Крок 12: Побудова теплової мапи для кластерів")
fig = px.choropleth(data,
                    locations="Country",
                    locationmode="country names",
                    color="Cluster",
                    title="Кластеризація країн за індексом щастя")
fig.show()

# Крок 12: Дослідження впливу різного набору ознак на результат кластеризації
print("Крок 13: Дослідження впливу різного набору ознак на результат кластеризації")
selected_features = ['Happiness.Score', 'Economy..GDP.per.Capita.', 'Family']
data_scaled_selected = data_scale(data[selected_features])
gmm_selected = GaussianMixture(n_components=3, random_state=42)
gmm_selected.fit(data_scaled_selected)
labels_selected = gmm_selected.predict(data_scaled_selected)
data['Cluster_Selected'] = labels_selected

fig = px.choropleth(data,
                    locations="Country",
                    locationmode="country names",
                    color="Cluster_Selected",
                    title="Кластеризація країн за вибраними ознаками")
fig.show()

# Крок 13: Загальний висновок
print("Крок 14: Загальний висновок")
print("""
Висновки:
1. Дані показують сильний зв'язок між індексом щастя та економічними показниками (ВВП на душу населення) та соціальними факторами (сім'я).
2. Кластеризація дозволила розділити країни на 3 групи, що відображають різні рівні щастя.
3. Використання різних наборів ознак впливає на результати кластеризації, що підтверджує важливість вибору ознак для аналізу.
4. Візуалізація за допомогою теплових карт дозволяє легко інтерпретувати розподіл країн за кластерами.
""")