import numpy as np
class SGD:

    def __init__(self, momentum=0.999, nesterov=False):
        self.momentum = momentum
        self.nesterov = nesterov


    def gradient(self, X, T, forward_function, backward_function, alfa):
        for idx, x in enumerate(X):
            I, Y, original_x = forward_function(x)
            backward_function(T[idx], Y, I, original_x, alfa)
        print()
