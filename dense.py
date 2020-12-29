import numpy as np
from layer import Layer

class Dense(Layer):

    def __init__(self, units, input_shape=None, activation_function='relu'):
        self.__int__(activation_function=activation_function)
        self.input_shape = input_shape
        self.units = units

    def forward(self, X):
        if self.activation_function == 'sigmoid':
            I = X.dot(self.W) + self.b
            return I, self.sigmoid(I)
        elif self.activation_function == 'softmax':
            I = X.dot(self.W) + self.b
            return I, self.softmax(I)
        elif self.activation_function == 'tanh':
            I = X.dot(self.W) + self.b
            return I, self.tanh(I)
        elif self.activation_function == 'relu':
            I = X.dot(self.W) + self.b
            return I, self.relu(I)

    def backward(self, gradient_w, gradient_b, alfa):
        self.W += alfa * gradient_w
        self.b += alfa * gradient_b

    def init_weights(self):
        self.W = np.random.random((self.input_shape, self.units))
        self.b = np.random.random((self.units))