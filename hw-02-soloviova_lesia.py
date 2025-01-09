import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error

# Крок 1: Завантаження зображення
print("Крок 1: Завантаження зображення у градаціях сірого...")
image_path = 'assets/grey_image.jpg'
image = io.imread(image_path, as_gray=True)  # Завантаження у градаціях сірого

# Крок 2: Відображення оригінального зображення
print("Крок 2: Відображення оригінального зображення...")
plt.imshow(image, cmap='gray')
plt.title("Оригінальне зображення")
plt.axis('off')
plt.show()

# Крок 3: Визначення розмірів зображення
print("Крок 3: Визначення розмірів зображення...")
height, width = image.shape
print(f"Розміри зображення: {height}x{width}")

# Крок 4: Перетворення зображення у 2D матрицю
print("Крок 4: Перетворення зображення у 2D матрицю...")
flat_image = image.reshape(height, -1)

# Крок 5: Застосування SVD-декомпозиції
print("Крок 5: Застосування SVD-декомпозиції...")
U, S, VT = np.linalg.svd(flat_image, full_matrices=False)

# Крок 6: Візуалізація сингулярних значень
print("Крок 6: Візуалізація перших 100 сингулярних значень...")
k = 100  # Кількість сингулярних значень для візуалізації
plt.plot(np.arange(k), S[:k])
plt.title("Сингулярні значення")
plt.xlabel("Індекс")
plt.ylabel("Значення")
plt.grid()
plt.show()

# Крок 7: Використання TruncatedSVD для компресії
n_components = 50  # Експеримент із 50 сингулярними значеннями
print(f"Крок 7: Стиснення зображення з використанням {n_components} сингулярних значень...")
svd = TruncatedSVD(n_components=n_components)
truncated_image = svd.fit_transform(flat_image)

# Крок 8: Реконструкція зображення
print("Крок 8: Реконструкція зображення із стиснених даних...")
reconstructed_image = svd.inverse_transform(truncated_image)

# Обчислення помилки реконструкції
print("Крок 8.1: Обчислення помилки реконструкції...")
reconstruction_error = mean_squared_error(flat_image, reconstructed_image)
print(f"Помилка реконструкції (MSE): {reconstruction_error}")

# Зміна форми та обрізання реконструйованого зображення для візуалізації
reconstructed_image = reconstructed_image.reshape(height, width)
reconstructed_image = np.clip(reconstructed_image, 0, 1)

# Відображення реконструйованого зображення
print("Крок 8.2: Відображення реконструйованого зображення...")
plt.imshow(reconstructed_image, cmap='gray')
plt.title(f"Реконструйоване зображення з {n_components} сингулярними значеннями")
plt.axis('off')
plt.show()

# Крок 9: Експерименти з різними значеннями k та візуалізація помилок реконструкції
print("Крок 9: Експерименти з різними значеннями k та побудова графіків помилок...")
errors = []
components_range = [10, 20, 30, 40, 50]

for n in components_range:
    print(f"Обробка з {n} компонентами...")
    svd = TruncatedSVD(n_components=n)
    truncated_image = svd.fit_transform(flat_image)
    reconstructed = svd.inverse_transform(truncated_image)
    error = mean_squared_error(flat_image, reconstructed)
    errors.append(error)

    # Відображення зображення для кожного k
    reconstructed = reconstructed.reshape(height, width)
    reconstructed = np.clip(reconstructed, 0, 1)
    plt.imshow(reconstructed, cmap='gray')
    plt.title(f"Реконструйоване зображення (k={n})")
    plt.axis('off')
    plt.show()

# Побудова графіка помилок vs компонент
plt.plot(components_range, errors, marker='o')
plt.title("Помилка реконструкції vs Кількість сингулярних значень")
plt.xlabel("Кількість сингулярних значень")
plt.ylabel("Помилка реконструкції (MSE)")
plt.grid()
plt.show()

# Висновок
print(
    """
1. Вплив кількості сингулярних значень (k) на якість реконструкції:

- Якість реконструкції покращується зі збільшенням кількості використаних сингулярних значень. 
- Після досягнення певного значення k покращення якості стає менш помітним.

2. Баланс між стисненням і якістю:

- Менші значення k суттєво зменшують обсяг даних, але якість значно падає.
- Великі значення k забезпечують якісну реконструкцію, але зменшення даних стає менш ефективним.

3. Рекомендації:

- Оптимальне значення k залежить від конкретного застосування та допустимої втрати якості.
"""
)

print("Всі кроки успішно виконані!")
