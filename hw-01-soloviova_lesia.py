
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from colorama import Fore, Style
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix, accuracy_score

os.environ["LOKY_MAX_CPU_COUNT"] = "1"

print(Fore.CYAN + "Крок 1: Завантаження даних Iris" + Style.RESET_ALL)
# Завантаження даних Iris
data = load_iris()
iris_df = pd.DataFrame(data.data, columns=data.feature_names)
iris_df['target'] = data.target

print(Fore.GREEN + "Дані завантажено успішно!" + Style.RESET_ALL)

print(Fore.CYAN + "Крок 2: Отримання базових статистичних характеристик" + Style.RESET_ALL)
# Отримання базових статистичних характеристик
print("\nБазові статистичні характеристики:\n")
print(iris_df.describe())

print(Fore.GREEN + "Статистичні характеристики отримано!" + Style.RESET_ALL)

print(Fore.CYAN + "Крок 3: Візуалізація розподілу спостережень за класами" + Style.RESET_ALL)
# Візуалізація розподілу спостережень за класами
sns.pairplot(iris_df, hue='target', palette='Set2')
plt.title('Розподіл даних Iris')
plt.show()

print(Fore.GREEN + "Розподіл візуалізовано!" + Style.RESET_ALL)

print(Fore.CYAN + "Крок 4: Стандартизація даних" + Style.RESET_ALL)
# Стандартизація даних
scaler = StandardScaler()
scaled_data = scaler.fit_transform(iris_df.iloc[:, :-1])

print(Fore.GREEN + "Дані стандартизовано!" + Style.RESET_ALL)

print(Fore.CYAN + "Крок 5: Виконання спектральної кластеризації" + Style.RESET_ALL)
# Виконання спектральної кластеризації
n_clusters = len(np.unique(data.target))
spectral_model = SpectralClustering(n_clusters=n_clusters, affinity='rbf', random_state=42)
predicted_clusters = spectral_model.fit_predict(scaled_data)

print(Fore.GREEN + "Спектральну кластеризацію виконано!" + Style.RESET_ALL)

print(Fore.CYAN + "Крок 6: Порівняння спрогнозованих кластерів та дійсних класів" + Style.RESET_ALL)
# Додавання прогнозованих кластерів у DataFrame
iris_df['predicted_cluster'] = predicted_clusters

# Порівняння спрогнозованих кластерів та дійсних класів
conf_matrix = confusion_matrix(iris_df['target'], iris_df['predicted_cluster'])
accuracy = accuracy_score(iris_df['target'], iris_df['predicted_cluster'])

print("\nМатриця плутанини:\n", conf_matrix)
print("\nТочність кластеризації: {:.2f}%".format(accuracy * 100))

print(Fore.GREEN + "Порівняння завершено!" + Style.RESET_ALL)

print(Fore.CYAN + "Крок 7: Візуалізація результатів кластеризації" + Style.RESET_ALL)
# Візуалізація результатів кластеризації
sns.scatterplot(x=scaled_data[:, 0], y=scaled_data[:, 1], hue=predicted_clusters, palette='viridis', style=iris_df['target'])
plt.title('Результати спектральної кластеризації')
plt.xlabel('Scaled Feature 1')
plt.ylabel('Scaled Feature 2')
plt.legend(title='Cluster')
plt.show()

print(Fore.GREEN + "Результати кластеризації візуалізовано!" + Style.RESET_ALL)

print(Fore.CYAN + "Крок 8: Висновок" + Style.RESET_ALL)
# Висновок
print("\nВисновок:")
print("""Результати кластеризації показують, що метод спектральної кластеризації
дозволяє успішно розділити дані на кластери. Однак існують деякі невідповідності
між спрогнозованими кластерами та дійсними класами, що може бути пов'язано з 
обраними параметрами моделі або схожістю між класами.""")

print(Fore.GREEN + "Завдання виконано!" + Style.RESET_ALL)
