import numpy as np
from Layers.layer import Layer
class Conv2D(Layer):

    def __init__(self, units=1, input_shape=(1,1,3), kernel_size=[3,3], strides=[2,2]):
        # input_shape may be of type (H, W, DIM).
        # kernel_size may be of type 3, 5, 7, ...; or (3,1), (3,5): (height, width)
        # the strides will define the convolutional step in y and x directions respectively
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.strides = strides
        self.units = units
        self.init_filters()

    def convolve2D(self, input, kernel, padding=0):
        pass

    def forward(self, X):
        X = self.convolve2D(X, self.kernels[0])
        return None, X

    def backward(self, X):
        pass

    def init_filters(self):
        self.kernels = np.random.random((self.units, self.kernel_size[0], self.kernel_size[1]))