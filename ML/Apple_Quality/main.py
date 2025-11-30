import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b)**2))

    def predict_one(self, x):
        distances = np.array([self._euclidean_distance(x, x_train) for x_train in self.X_train])
        k_idx = distances.argsort()[:self.k]
        k_labels = self.y_train[k_idx]
        values, counts = np.unique(k_labels, return_counts=True)
        return values[counts.argmax()]

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred) * 100


def main():

    file_path = "apple_quality.csv"
    
    dataset = pd.read_csv(file_path)

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    selector = SelectKBest(score_func=f_classif, k=5)
    X_new = selector.fit_transform(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_new, y, test_size=0.2, random_state=42
    )

    model = KNN(k=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = model.accuracy(y_test, y_pred)
    print(f"Accuracy del modelo KNN: {acc:.2f}%")

    plt.figure(figsize=(10,6))
    sns.heatmap(np.corrcoef(X.T), cmap="viridis")
    plt.title("Matriz de Correlación")
    plt.show()

    plt.figure(figsize=(6,4))
    sns.countplot(x=y)
    plt.title("Distribución de clases")
    plt.show()


if __name__ == "__main__":
    main()
