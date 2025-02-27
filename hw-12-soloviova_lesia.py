# -*- coding: utf-8 -*-
"""goit-numericalpy-hw-12-soloviova_lesia.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1fHS-b_gJljG7Fp98sNcT9zPDYxABLsf6
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
import pygad

# Завантаження даних
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

print("Кількість зразків у наборі даних:", X.shape[0])
print("Кількість ознак у наборі даних:", X.shape[1])

# Візуалізація попарних точкових діаграм для перших 5 ознак
sns.pairplot(pd.DataFrame(X[:, :5], columns=feature_names[:5]).assign(Target=y), hue="Target", palette='coolwarm')
plt.show()

# Кластеризація
sc = SpectralClustering(n_clusters=2, random_state=42).fit_predict(X)
kmeans = KMeans(n_clusters=2, random_state=42).fit_predict(X)
gmm = GaussianMixture(n_components=2, random_state=42).fit_predict(X)

print("Співвідношення кластерів зі справжніми класами (Спектральна кластеризація):", np.mean(sc == y))
print("Співвідношення кластерів зі справжніми класами (KMeans):", np.mean(kmeans == y))
print("Співвідношення кластерів зі справжніми класами (GMM):", np.mean(gmm == y))

# PCA - зменшення розмірності
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm')
plt.title("Проекція PCA")
plt.xlabel("Головний компонент 1")
plt.ylabel("Головний компонент 2")
plt.colorbar(label="Клас")
plt.show()

# Розбиття на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, stratify=y, random_state=42)

# Логістична регресія
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

# Оцінка якості класифікації
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel("Прогнозовані")
plt.ylabel("Реальні")
plt.title("Матриця плутанини")
plt.show()
print("F1-метрика (Logistic Regression):", f1_score(y_test, y_pred))

# Генетичний алгоритм для оптимізації ваг

def fitness_function(ga_instance, solution, solution_idx):
    predictions = (1 / (1 + np.exp(-np.dot(X_train, solution))) >= 0.5).astype(int)
    return f1_score(y_train, predictions)

num_features = X_train.shape[1]
ga_instance = pygad.GA(num_generations=100,
                        num_parents_mating=10,
                        fitness_func=fitness_function,
                        sol_per_pop=50,
                        num_genes=num_features)

ga_instance.run()
best_solution, best_solution_fitness, _ = ga_instance.best_solution()

def predict_genetic(X_test, weights):
    return (1 / (1 + np.exp(-np.dot(X_test, weights))) >= 0.5).astype(int)

y_pred_genetic = predict_genetic(X_test, best_solution)
cm_genetic = confusion_matrix(y_test, y_pred_genetic)

plt.figure(figsize=(5, 5))
sns.heatmap(cm_genetic, annot=True, cmap='Blues', fmt='d')
plt.xlabel("Прогнозовані")
plt.ylabel("Реальні")
plt.title("Матриця плутанини (Genetic Optimization)")
plt.show()
print("F1-метрика (Genetic Optimization):", f1_score(y_test, y_pred_genetic))

"""
Висновки:
1. PCA допомогла ефективно зменшити розмірність, дозволяючи зберегти значущу інформацію.
2. Логістична регресія показала гарні результати з F1-метрикою ~0.95.
3. Генетичний алгоритм дав схожий результат, але потребує значно більше обчислювальних ресурсів.
4. Кластеризаційні методи (Spectral, KMeans, GMM) показали слабку відповідність реальним класам.
5. Використання методу PCA суттєво покращило наочність та ефективність моделювання.
"""