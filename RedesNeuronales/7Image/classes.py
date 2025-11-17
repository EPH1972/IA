from abc import abstractmethod
import numpy as np

#Classes declaration 
class layer():
    
    def __init__(self, X, W,B):
        self.X = X
        self.W = W
        self.B = B
        self.dW = 0
        self.dB = 0

    @abstractmethod
    def forward(X):
        pass

    @abstractmethod
    def backward(ypred, yreal):
        pass

    @abstractmethod
    def update(W,B,dW,dB,alfa):
        pass


class Densa(layer):
    
    def forward(X):
        y = X @ super().W + super().B
        return y
    
    def backward(ypred, yreal):
        return super().backward(yreal)
    
    def update(W, B, dW, dB, alfa):
        return super().update(B, dW, dB, alfa)

class ReLU(layer):
    
    def forward(X):
        return np.maximum(X,0)
    
    def backward(ypred, yreal):
        return super().backward(yreal)
    
    def update(W, B, dW, dB, alfa):
        return super().update(B, dW, dB, alfa)

class model():
    def __init__(self, d1, dact, l1, lact):
        self.epoch = [d1,dact,l1,lact]


    def loss():
        pass

    def acc():
        pass 