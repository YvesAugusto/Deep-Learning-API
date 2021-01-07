import numpy as np
import matplotlib.pyplot as plt
from Layers.conv import Conv2D
from Layers.dense import Dense
class Model:
    def __init__(self):
        self.layers = []
        self.losses = []
        self.deltas = []

    def pair_shuffle(x, y):
        c = list(zip(x, y))
        np.random.shuffle(c)
        x, y = zip(*c)
        return np.array(x), np.array(y)

    def add_layer(self, layer):
        if layer.input_shape == None:
            layer.input_shape = self.layers[-1].units
        if type(layer) == type(Conv2D()):
            layer.init_filters()
        elif type(layer) == type(Dense()):
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

    def backward(self, T, Y, I, original_x, alfa, error=[]):
        # update the last weight matrix
        if len(error) == 0:
            self.deltas[-1] = (T - Y[-1]) * self.layers[-1].f(I[-1], derivative=True)
        elif len(error) == self.layers[-1].units:
            self.deltas[-1] = error * self.layers[-1].f(I[-1], derivative=True)
        else:
            print(f'error shape {error.shape} does not check with layer units')

        grad_w = np.outer(Y[-2], self.deltas[-1])
        grad_b = self.deltas[-1].sum(axis=0)
        self.layers[-1].backward(grad_w, grad_b, alfa)

        # update the other weight matrices
        for i in range(1, len(self.layers) - 1):
            self.deltas[-1 - i] = self.layers[-i].W.dot(self.deltas[-i]) * self.layers[-1 - i].f(I[-1-i], derivative=True)
            grad_w = np.outer(Y[-2 - i], self.deltas[-1 - i])
            grad_b = self.deltas[-1 - i].sum(axis=0)
            self.layers[-1-i].backward(grad_w, grad_b, alfa)

        self.deltas[0] = self.layers[1].W.dot(self.deltas[1]) * self.layers[1].f(I[0], derivative=True)
        grad_w = np.outer(original_x, self.deltas[0])
        grad_b = self.deltas[0].sum(axis=0)
        self.layers[0].backward(grad_w, grad_b, alfa)

        # TODO
        pass

    def compile(self, loss, optimizer = None):
        self.deltas = [np.zeros(layer.units) for layer in self.layers]
        self.loss = loss
        self.optimizer = optimizer

    def loss_function(self, T, Y):
        if self.loss == 'categorical_crossentropy':
            return -(T * np.log(Y[-1])).sum()

        elif self.loss == 'binary_crossentropy':
            loss = 0
            for n in range(len(T)):
                if T[n] == 1:
                    loss += np.log(Y[-1][n])
                else:
                    loss += np.log(1 - Y[-1][n])
            return loss

    def test_eval(self, validation_data):
        T = validation_data[1]
        X = validation_data[0]
        return self.evaluate(T, X)

    def plot_loss(self, epochs):
        plt.plot(epochs, self.losses)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()

    def plot_acc(self, epochs_, evals, tests_eval):
        plt.plot(epochs_, evals)
        plt.plot(epochs_, tests_eval)
        plt.title('model accuracy')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    def fit(self, X, T, validation_data=None, batch_size=1, epochs=10000, alfa=1e-4):
        epochs_ = []
        lr_0 = 5*1e-4
        lr = 5*1e-4
        if validation_data:
            tests_eval = []
        evals = []
        for e in range(epochs):
            lr -= ((lr_0 - alfa) / epochs)
            Y = self.optimizer.batch_forward(X, T, self.forward, self.backward, self.layers, self.deltas, alfa)
            # print(self.layers[-1].W, self.layers[-1].b)
            # for idx, x in enumerate(X):
            #     I, Y, original_x = self.forward(x)
            #     self.backward(T[idx], Y, I, original_x, alfa)
            if (e) % 10 == 0:
                eval = self.evaluate(T, X)
                evals.append(eval)
                if validation_data:
                    eval_ = self.test_eval(validation_data)
                    tests_eval.append(eval_)
                loss_ = self.loss_function(T[-1], Y)
                self.losses.append(loss_)
                epochs_.append(e)
                if e % 10 == 0:
                    print(f'epoch {e} lr: {lr}, loss: {loss_}, eval: {eval}, val: {eval_}')
                if eval > 0.95:
                    print(f'early stopping cause reached good acc: {eval}')
                    self.plot_loss(epochs_, self.losses)
                    self.plot_acc(epochs_, evals, tests_eval)
                    return

                if eval < 0.2:
                    print(f'Failed')
                    return

        # TODO
        plt.plot(epochs_, self.losses)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()
        plt.plot(epochs_, evals)
        plt.plot(epochs_, tests_eval)
        plt.title('model accuracy')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        return epochs, self.losses