import numpy as np
class BGD:

    def __init__(self, momentum=0.999, nesterov=False):
        self.momentum = momentum
        self.nesterov = nesterov

    def backward(self, T, Y, I, layers, deltas, original_x, alfa, error=[]):
        # update the last weight matrix
        if len(error) == 0:
            deltas[-1] = (T - Y[-1]) * layers[-1].f(I[-1], derivative=True)
        elif len(error) == layers[-1].units:
            deltas[-1] = error * layers[-1].f(I[-1], derivative=True)
        else:
            print(f'error shape {error.shape} does not check with layer units')

        self.grad_w[-1] += np.outer(Y[-2], deltas[-1])
        self.grad_b[-1] += deltas[-1].sum(axis=0)
        # layers[-1].backward(grad_w, grad_b, alfa)

        # update the other weight matrices
        for i in range(1, len(layers) - 1):
            deltas[-1 - i] = layers[-i].W.dot(deltas[-i]) * layers[-1 - i].f(I[-1 - i],
                                                                                                 derivative=True)
            self.grad_w[-1 - i] += np.outer(Y[-2 - i], deltas[-1 - i])
            self.grad_b[-1 - i] += deltas[-1 - i].sum(axis=0)
            # layers[-1 - i].backward(grad_w, grad_b, alfa)

        deltas[0] = layers[1].W.dot(deltas[1]) * layers[1].f(I[0], derivative=True)
        self.grad_w[0] += np.outer(original_x, deltas[0])
        self.grad_b[0] += deltas[0].sum(axis=0)
        # layers[0].backward(grad_w, grad_b, alfa)

    def batch_forward(self, X, T, forward, backward, layers, deltas, alfa):
        Y = []
        Y_ = []
        I = []
        origo = []
        batch = 32

        for i in range(batch):
            self.grad_w = np.array([np.zeros(layer.W.shape) for layer in layers])
            self.grad_b = np.array([np.zeros(layer.b.shape) for layer in layers])
            tam = len(X[i*batch : (i+1)*batch])
            for idx, x in enumerate(X[i*batch : (i+1)*batch]):
                I, Y, original_x = forward(x)
                self.backward(T[i*batch : (i+1)*batch][idx], Y, I, layers, deltas, original_x, alfa)

            self.grad_w /= tam
            self.grad_b /= tam
            for idl, layer in enumerate(layers):
                layer.backward(self.grad_w[idl], self.grad_b[idl], alfa)

        return Y[-1]