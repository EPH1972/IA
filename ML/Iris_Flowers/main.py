# Import necessary libraries
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
    return ""