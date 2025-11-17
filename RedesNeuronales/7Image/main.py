import idx2numpy
import numpy as np
from classes import Densa
from classes import ReLU
from classes import model
from matplotlib import pyplot as plt


''' def train_step(model, x,y,lr):
    ypred = model.forward(x)
    loss = (ypred - yreal)
    model.basckward(loss)
    model.update(lr)
    acc = acc(ypred, yreal)
    print(acc,loss)

def train_loop(model,data,lr):
    for i in range(epochs):
        for x,y in data:
            train_step(model,x,y,lr)'''


# Reading
y_test = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')
X_test = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
y_train = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
X_train = idx2numpy.convert_from_file('train-images.idx3-ubyte')


images = X_train/256

X_train = np.reshape(X_train, (60, 1000, 784))
y_train = np.reshape(y_train, )

def train_step():
    pass

def init_params():
    w1=np.random.rand(10,784)-0.5
    b1=np.random.rand(10,1)-0.5
    w2=np.random.rand(10,10)-0.5
    b2=np.random.rand(10,1)-0.5
    return w1, b1, w2, b2





def main():
    D1 =  Densa()
    D1Act = ReLU()
    D2 = Densa()
    D2Act = ReLU()

    Model = model(D1,D1Act,D2,D2Act)
    for i in len(Model.epoch):
        print("In")

    '''plt.figure(figsize=(10, 5))
    for i in range():
        plt.subplot(1, 1000, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Label: {y_train[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()'''

if __name__=='__main__':
    main()
