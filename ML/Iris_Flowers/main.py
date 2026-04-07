import numpy as np
import matplotlib.pyplot as plt
import sklearn

numpy_numbers = np.genfromtxt('IRIS.csv', skip_header=1, delimiter=',')
numpy_strings = np.genfromtxt('IRIS.csv', skip_header=1, delimiter=',', dtype=str)

numpy_numbers = np.delete(numpy_numbers, -1, axis=1)
numpy_strings = np.delete(numpy_strings, np.s_[:-1], axis=1)


xpoints = np.array([1, 8])
ypoints = np.array([3, 10])

plt.plot(xpoints, ypoints)
plt.show()


def concat(arr1, arr2):
    """Concatenate a numeric array and a string array column-wise."""
    return np.hstack([arr1.astype(str), arr2])
FEATURE_NAMES = ['sepal length (cm)', 'sepal width (cm)',
                 'petal length (cm)', 'petal width (cm)']
CLASS_NAMES   = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

X = numpy_numbers
label_map = {name: i for i, name in enumerate(CLASS_NAMES)}
y = np.array([label_map[s[0]] for s in numpy_strings])

print(f"Features shape : {X.shape}")
print(f"Labels shape   : {y.shape}")
print(f"Classes        : {CLASS_NAMES}")
print(f"Sample (concat): {concat(numpy_numbers[:1], numpy_strings[:1])}")

def visualize_features(X, y, feature1_idx, feature2_idx):
    """Scatter plot of two features coloured by class."""
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(CLASS_NAMES):
        mask = y == i
        plt.scatter(X[mask, feature1_idx], X[mask, feature2_idx],
                    label=name, alpha=0.7, s=50)
    plt.xlabel(FEATURE_NAMES[feature1_idx])
    plt.ylabel(FEATURE_NAMES[feature2_idx])
    plt.title(f'{FEATURE_NAMES[feature1_idx]} vs {FEATURE_NAMES[feature2_idx]}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

print("\nAll 6 feature combinations:")
for f1, f2 in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
    visualize_features(X, y, f1, f2)


def plot_decision_boundary(model, X, y, title, feature_idx=[2, 3]):
    """Re-trains model on two chosen features and plots decision regions."""
    from sklearn.base import clone
    X2 = X[:, feature_idx]
    m2 = clone(model)
    m2.fit(X2, y)

    h = 0.02
    x_min, x_max = X2[:, 0].min() - .5, X2[:, 0].max() + .5
    y_min, y_max = X2[:, 1].min() - .5, X2[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = m2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    sc = plt.scatter(X2[:, 0], X2[:, 1], c=y,
                     cmap='viridis', edgecolors='black', s=50, alpha=0.8)
    plt.xlabel(FEATURE_NAMES[feature_idx[0]], fontsize=12)
    plt.ylabel(FEATURE_NAMES[feature_idx[1]], fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.colorbar(sc, label='Class')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def data_splitter(X, y, train_ratio):
    """Random shuffle split into train/test."""
    n = len(X)
    idx = np.random.permutation(n)
    n_train = int(n * train_ratio)
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    print(f"Split {train_ratio*100:.0f}%/{(1-train_ratio)*100:.0f}%  "
          f"-> train={len(train_idx)}  test={len(test_idx)}")
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


X_train, y_train, X_test, y_test = data_splitter(X, y, 0.7)

from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier, plot_tree
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm           import SVC
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.naive_bayes   import GaussianNB
from sklearn.metrics       import ConfusionMatrixDisplay

accuracies = {}

lr_model = LogisticRegression(max_iter=200, random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
accuracies['Logistic Regression'] = lr_model.score(X_test, y_test)
print(f"\nLogistic Regression Accuracy: {accuracies['Logistic Regression']:.4f}")

plot_decision_boundary(lr_model, X, y, 'Logistic Regression - Decision Boundary')

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_lr, display_labels=CLASS_NAMES, cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
accuracies['Decision Tree'] = dt_model.score(X_test, y_test)
print(f"Decision Tree Accuracy: {accuracies['Decision Tree']:.4f}")

plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=FEATURE_NAMES, class_names=CLASS_NAMES,
          filled=True, rounded=True)
plt.title('Decision Tree Visualization')
plt.show()

plot_decision_boundary(dt_model, X, y,
                       'Decision Tree - Decision Boundary (Rectangular Regions)')

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_dt, display_labels=CLASS_NAMES, cmap='Greens')
plt.title('Confusion Matrix - Decision Tree')
plt.show()

rf_model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracies['Random Forest'] = rf_model.score(X_test, y_test)
print(f"Random Forest Accuracy: {accuracies['Random Forest']:.4f}")
print("Feature importances:", dict(zip(FEATURE_NAMES, rf_model.feature_importances_.round(3))))

plot_decision_boundary(rf_model, X, y,
                       'Random Forest - Decision Boundary (Smooth Ensemble)')

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_rf, display_labels=CLASS_NAMES, cmap='Oranges')
plt.title('Confusion Matrix - Random Forest')
plt.show()

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
accuracies['SVM'] = svm_model.score(X_test, y_test)
print(f"SVM Accuracy: {accuracies['SVM']:.4f}  "
      f"Support vectors: {svm_model.n_support_}")

plot_decision_boundary(svm_model, X, y,
                       'SVM (RBF Kernel) - Decision Boundary')

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_svm, display_labels=CLASS_NAMES, cmap='Purples')
plt.title('Confusion Matrix - SVM')
plt.show()

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
accuracies['KNN'] = knn_model.score(X_test, y_test)
print(f"KNN Accuracy: {accuracies['KNN']:.4f}")

plot_decision_boundary(knn_model, X, y,
                       'K-Nearest Neighbors (k=5) - Decision Boundary')

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_knn, display_labels=CLASS_NAMES, cmap='Reds')
plt.title('Confusion Matrix - KNN')
plt.show()

gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                      max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
accuracies['Gradient Boosting'] = gb_model.score(X_test, y_test)
print(f"Gradient Boosting Accuracy: {accuracies['Gradient Boosting']:.4f}")

plot_decision_boundary(gb_model, X, y,
                       'Gradient Boosting - Decision Boundary')

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_gb, display_labels=CLASS_NAMES, cmap='YlOrBr')
plt.title('Confusion Matrix - Gradient Boosting')
plt.show()

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
accuracies['Naive Bayes'] = nb_model.score(X_test, y_test)
print(f"Naive Bayes Accuracy: {accuracies['Naive Bayes']:.4f}")

plot_decision_boundary(nb_model, X, y,
                       'Naive Bayes - Decision Boundary (Probabilistic)')

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_nb, display_labels=CLASS_NAMES, cmap='Greys')
plt.title('Confusion Matrix - Naive Bayes')
plt.show()

print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)
sorted_acc = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
for name, acc in sorted_acc:
    print(f"  {name:<22} {acc:.4f}")

names  = [n for n, _ in sorted_acc]
values = [v for _, v in sorted_acc]

plt.figure(figsize=(10, 6))
bars = plt.barh(names, values, color='skyblue')
plt.xlabel('Accuracy')
plt.title('Model Performance Comparison — IRIS Dataset')
plt.xlim([min(values) - 0.05, 1.01])
for bar, v in zip(bars, values):
    plt.text(v + 0.002, bar.get_y() + bar.get_height()/2,
             f'{v:.4f}', va='center')
plt.tight_layout()
plt.show()
