from input import Input
from dense import Dense
from model import Model
import numpy as np

def pair_shuffle(x,y):
    c=list(zip(x,y))
    np.random.shuffle(c)
    x,y=zip(*c)
    return np.array(x), np.array(y)

if __name__ == '__main__':
    x1 = np.random.random((500, 2)) + [2,0]
    x2 = np.random.random((500, 2)) + [0,-2]
    x3 = np.random.random((500, 2)) + [-2, 0]


    y1 = np.zeros((500,3))
    y1[:, 0] = 1
    y2 = np.zeros((500,3))
    y2[:, 1] = 1
    y3 = np.zeros((500,3))
    y3[:, 2] = 1

    X=np.concatenate((x1,x2,x3))
    X[:,0] = (X[:,0] - X[:,0].min()) / (X[:,0].max() - X[:,0].min())
    X[:, 1] = (X[:, 1] - X[:, 1].min()) / (X[:, 1].max() - X[:, 1].min())

    Y=np.concatenate((y1,y2,y3))
    X, Y = pair_shuffle(X, Y)

    model = Model()
    model.add_layer(Input(input_shape=2, units=4, activation_function='relu'))
    model.add_layer(Dense(units=4, activation_function='relu'))
    model.add_layer(Dense(units=3, activation_function='softmax'))
    model.compile()
    model.fit(X, Y)