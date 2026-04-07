import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------

wines = load_wine()
X = wines.data
y = wines.target

wine_df = pd.DataFrame(data=X, columns=wines.feature_names)
wine_df['target'] = y

print(f"Samples: {X.shape[0]}  |  Features: {X.shape[1]}")
print(f"Classes: {wines.target_names}")
print(wine_df.head())

# ---------------------------------------------------------------------------
# Pairplot — 4 key features coloured by class
# ---------------------------------------------------------------------------

sns.pairplot(wine_df, hue='target',
             palette='viridis',
             diag_kind='kde',
             x_vars=['alcohol', 'magnesium', 'flavanoids', 'color_intensity'],
             y_vars=['alcohol', 'magnesium', 'flavanoids', 'color_intensity'],
             height=1.5)
plt.suptitle('Pairplot of Wine Dataset Features', y=1.02)
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
clf.fit(X_train, y_train)

plt.figure(figsize=(12, 6))
tree.plot_tree(clf, proportion=True,
               feature_names=wines.feature_names,
               class_names=wines.target_names,
               filled=True)
plt.title("Decision Tree (max_leaf_nodes=3)")
plt.tight_layout()
plt.show()

acc_dt = clf.score(X_test, y_test)
print(f"\nDecision Tree Accuracy: {acc_dt:.4f}")

ConfusionMatrixDisplay.from_predictions(
    y_test, clf.predict(X_test),
    display_labels=wines.target_names, cmap='Blues')
plt.title('Confusion Matrix - Decision Tree')
plt.show()

rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

acc_rf = rf.score(X_test, y_test)
print(f"Random Forest Accuracy: {acc_rf:.4f}")

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_rf,
    display_labels=wines.target_names, cmap='Greens')
plt.title('Confusion Matrix - Random Forest')
plt.show()
importances = pd.Series(rf.feature_importances_, index=wines.feature_names).sort_values(ascending=True)
plt.figure(figsize=(8, 6))
importances.plot(kind='barh', color='steelblue')
plt.title('Random Forest — Feature Importances')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# SelectKBest — reduce to 6 features
# ---------------------------------------------------------------------------

N_BEST = 6
feature_selector = SelectKBest(score_func=f_classif, k=N_BEST)
X_new = feature_selector.fit_transform(X, y)

selected_feature_names = np.array(wines.feature_names)[feature_selector.get_support()]
print(f"\nSelectKBest top-{N_BEST} features: {selected_feature_names.tolist()}")

# Visualise F-scores for all features
scores = pd.Series(feature_selector.scores_, index=wines.feature_names).sort_values(ascending=True)
plt.figure(figsize=(8, 6))
scores.plot(kind='barh', color='coral')
plt.title(f'SelectKBest F-scores  (selected top {N_BEST} highlighted)')
for i, (name, score) in enumerate(scores.items()):
    if name in selected_feature_names:
        plt.barh(i, score, color='tomato')
plt.xlabel('F-score')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Random Forest trained on selected features
# ---------------------------------------------------------------------------

X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(
    X_new, y, test_size=0.3, random_state=0)

rf_sel = RandomForestClassifier(n_estimators=100, random_state=0)
rf_sel.fit(X_train_sel, y_train_sel)
y_pred_sel = rf_sel.predict(X_test_sel)

acc_sel = rf_sel.score(X_test_sel, y_test_sel)
print(f"Random Forest ({N_BEST} features) Accuracy: {acc_sel:.4f}")

ConfusionMatrixDisplay.from_predictions(
    y_test_sel, y_pred_sel,
    display_labels=wines.target_names, cmap='Purples')
plt.title(f'Confusion Matrix - Random Forest ({N_BEST} selected features)')
plt.show()

# Pairplot of selected features
sel_df = pd.DataFrame(X_new, columns=selected_feature_names)
sel_df['target'] = y
sns.pairplot(sel_df, hue='target', palette='viridis', diag_kind='kde', height=1.5)
plt.suptitle(f'Pairplot — SelectKBest top {N_BEST} features', y=1.02)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "="*45)
print("MODEL COMPARISON")
print("="*45)
print(f"  Decision Tree  (all features)    {acc_dt:.4f}")
print(f"  Random Forest  (all features)    {acc_rf:.4f}")
print(f"  Random Forest  ({N_BEST} features)       {acc_sel:.4f}")
