import numpy as np
from matplotlib import pyplot as plt

from sklearn import tree
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier

numpy_numbers = np.genfromtxt('apple_quality.csv', skip_header=1, skip_footer=1, delimiter=',')
numpy_strings = np.genfromtxt('apple_quality.csv', skip_header=1, skip_footer=1, delimiter=',', dtype=str)

X = np.delete(numpy_numbers, -1, axis=1)
y = np.delete(numpy_strings, np.s_[:-1], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print(X_train,y_train)

'''apple_quality = data
X = apple_quality.data
y = apple_quality.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
clf.fit(X_train, y_train)

tree.plot_tree(clf, proportion=True)
plt.show()'''