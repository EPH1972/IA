from abc import abstractmethod

#declaration abs 
class layer():
    @abstractmethod
    def __init__(self,W,B):
        pass
    
    @abstractmethod
    def forward(X):
        pass

    @abstractmethod
    def backward(ypred, yreal):
        pass

    @abstractmethod
    def update(W,B,dW,dB,alfa):
        pass

