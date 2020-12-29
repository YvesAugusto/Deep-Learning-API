import numpy as np
class Layer:

    def __int__(self, trainable = True, dropout = 0.0, activation_function='relu'):
        self.W = []
        self.b = []
        self.trainable = trainable
        self.dropout = dropout
        self.activation_function = activation_function

    def f(self, x, derivative=False):
        if self.activation_function == 'sigmoid':
            return self.sigmoid(x, derivative)
        elif self.activation_function == 'softmax':
            return self.softmax(x, derivative)
        elif self.activation_function == 'relu':
            return self.relu(x, derivative)
        elif self.activation_function == 'tanh':
            return self.tanh(x, derivative)

    def softmax(self, x, derivative=False):
        if derivative:
            soft = self.softmax(x)
            return 1 - soft**2
        return np.array(
            np.exp(x)/np.exp(x).sum()
        )

    def sigmoid(self, x, derivative=False):
        if derivative:
            sig = self.sigmoid(x)
            return sig * (1 - sig)
        return np.array(
            1/(1 + np.exp(-x))
        )

    def tanh(self, x, derivative=False):
        return np.array(
            (1 + np.exp(x)) / (1 - np.exp(x))
        )

    def relu(self, x, derivative = False):
        if derivative:
            return (x > 0) * 1
        return np.maximum(x, 0)