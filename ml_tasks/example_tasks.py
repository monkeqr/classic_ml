# ml_tasks/example_tasks.py
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
import numpy as np

"""
ML Tasks Examples

Покрывает:
1. Обучение с учителем (Supervised Learning) — классификация и регрессия
2. Обучение без учителя (Unsupervised Learning) — кластеризация
3. Обучение с подкреплением (Reinforcement Learning)

Краткие понятия:
- Supervised Learning: Модель обучается на размеченных данных (X, y). Примеры: классификация, регрессия.
- Unsupervised Learning: Модель ищет структуру в неразмеченных данных. Примеры: кластеризация, PCA.
- Reinforcement Learning: Агент учится через взаимодействие с окружением, получая вознаграждение за "правильные" действия.

PCA:
Стандартизация данных для приведения всех переменных к одному масштабу (приведение к одной размерности).
Вычисление ковариационной матрицы для определения взаимосвязей между признаками.
Нахождение собственных векторов и собственных значений для определения главных компонент.

Когда точно использовать
Когда число признаков очень большое (например, текстовые данные, геномика).
Когда хочется визуализировать данные.
Когда нужно ускорить обучение без существенной потери информации.
!!! Методы типа PCA линейны, поэтому могут теряться нелинейные зависимости.
В таких случаях используют t-SNE, UMAP, Autoencoders.!!!

Когда признаки сильно коррелированы, и можно объединить их в новые “главные компоненты”.
"""

# 1. Обучение с учителем
print("=== Supervised Learning ===")

# Классификация
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)
print("Classification accuracy:", clf.score(X_test, y_test))

# Регрессия
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.2)
reg = LinearRegression()
reg.fit(X_train, y_train)
print("Regression R2 score:", reg.score(X_test, y_test))

# 2. Обучение без учителя
print("\n=== Unsupervised Learning ===")
X = iris.data
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
print("KMeans cluster centers:\n", kmeans.cluster_centers_)

X = iris.data
y = iris.target

# Создаем PCA с 2 главными компонентами
print("\nPCA")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Shape of original data:", X.shape)
print("Shape after PCA:", X_pca.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)
wasted_data = 1
for i in pca.explained_variance_ratio_:
    wasted_data -= i
print("Wasted data after compression:", wasted_data)

# Визуализация первых двух компонент
plt.figure(figsize=(6,4))
for label in set(y):
    plt.scatter(
        X_pca[y==label, 0],
        X_pca[y==label, 1],
        label=iris.target_names[label]
    )
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Iris Dataset")
plt.legend()
plt.show()

# 3. Обучение с подкреплением 
print("\n=== Reinforcement Learning ===")
# Reinforcement Learning example with Gymnasium
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode=None)
observation, info = env.reset()

for _ in range(3):
    action = env.action_space.sample()  # случайное действие
    observation, reward, terminated, truncated, info = env.step(action)
    print("Step reward:", reward)
    if terminated or truncated:
        observation, info = env.reset()

env.close()

