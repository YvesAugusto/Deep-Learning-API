import numpy as np
from Layers.layer import Layer

class Dense(Layer):

    def __init__(self, units=1, input_shape=None, activation_function='relu'):
        self.__int__(activation_function=activation_function)
        self.input_shape = input_shape
        self.units = units
        self.first = 1

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
        # gradient_w -= self.W * 0.1
        # gradient_b -= self.b * 0.1
        if self.first:
            self.W += alfa * gradient_w + 0.99 * (self.W - self.last_W)
            self.b += alfa * gradient_b
            self.first = 0
        else:
            self.W += alfa * gradient_w + 0.99 * (self.W - self.last_W)
            self.last_W = self.W
            self.b += alfa * gradient_b
            self.last_b = self.b
            self.first = 1

        self.W += alfa * gradient_w
        self.b += alfa * gradient_b


    def init_weights(self):
        self.W = np.random.randn(self.input_shape, self.units)
        self.b = np.random.randn(self.units)
        self.last_W = self.W
        self.last_b = self.b