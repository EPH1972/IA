import idx2numpy
from classes import layer
from matplotlib import pyplot as plt


# Reading

y_train = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
X_train = idx2numpy.convert_from_file('train-images.idx3-ubyte')


images = X_train/256

count = 0
acc = 0




plt.figure(figsize=(10, 5))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
