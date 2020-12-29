import numpy as np
import matplotlib.pyplot as plt
class Model:
    def __init__(self):
        self.layers = []
        self.losses = []
        self.deltas = []

    def add_layer(self, layer):
        if layer.input_shape == None:
            layer.input_shape = self.layers[-1].units
        layer.init_weights()
        self.layers.append(layer)

    def forward(self, X, i=None):
        original_X = X.copy()
        Y = []
        I = []
        if i == None:
            i = len(self.layers)
        for k in range(i):
            I_aux, X = self.layers[k].forward(X)
            Y.append(X)
            I.append(I_aux)
        return I, Y, original_X

    def predict(self, X):
        for k in range(len(self.layers)):
            _, X = self.layers[k].forward(X)
        return X.argmax()

    def evaluate(self, T, X):
        acc = 0
        for idx, x in enumerate(X):
            p = self.predict(x)
            if T[idx][p] == 1:
                acc+=1

        return acc / len(X)


    def backward(self, T, Y, I, original_x, alfa):
        # update the last weight matrix
        self.deltas[-1] = (T - Y[-1]) * self.layers[-1].f(I[-1], derivative=True)
        grad_w = np.outer(Y[-2], self.deltas[-1])
        grad_b = self.deltas[-1].sum(axis=0)
        self.layers[-1].backward(grad_w, grad_b, alfa)

        # update the other weight matrices
        for i in range(1, len(self.layers) - 1):
            self.deltas[-1 - i] = self.layers[-i].W.dot(self.deltas[-i]) * self.layers[-1].f(I[-1-i], derivative=True)
            grad_w = np.outer(Y[-2 - i], self.deltas[-1 - i])
            grad_b = self.deltas[-1 - i].sum(axis=0)
            self.layers[-1-i].backward(grad_w, grad_b, alfa)

        self.deltas[0] = self.layers[1].W.dot(self.deltas[1]) * self.layers[1].f(I[0], derivative=True)
        grad_w = np.outer(original_x, self.deltas[1])
        grad_b = self.deltas[1].sum(axis=0)
        self.layers[0].backward(grad_w, grad_b, alfa)

        # TODO
        pass

    def compile(self):
        self.deltas = [np.zeros(layer.units) for layer in self.layers]

    def fit(self, X, T, batch_size=1, epochs=1000, alfa=1e-4):
        epochs_ = []
        for e in range(epochs):
            loss = 0
            for idx, x in enumerate(X):
                I, Y, original_x = self.forward(x)
                self.backward(T[idx], Y, I, original_x, alfa)
            if (e) % 10 == 0:
                eval = self.evaluate(T, X)
                loss = -(T[idx] * np.log(Y[-1])).sum()
                self.losses.append(loss)
                epochs_.append(e)
                print(f'epoch {e} loss: {loss}, eval: {eval}')
                if eval > 0.95:
                    print(f'early stopping cause reached good acc: {eval}')
                    plt.plot(epochs_, self.losses)
                    plt.ylabel('loss')
                    plt.xlabel('epoch')
                    plt.show()
                    return

        # TODO
        plt.plot(epochs_, self.losses)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()
        return epochs, self.losses