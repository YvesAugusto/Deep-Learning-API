import numpy as np
import matplotlib.pyplot as plt
def pair_shuffle(x,y):
    c=list(zip(x,y))
    np.random.shuffle(c)
    x,y=zip(*c)
    return np.array(x), np.array(y)
def donut_data(N, r_i, r_o):
    q = int(N/3)
    r = np.random.randn(q) + r_i
    R = np.random.randn(q) + r_o
    R_ = np.random.randn(q) + r_i*2 + r_o
    theta = 2*np.pi*np.random.randn(q)
    X_i = np.concatenate([[r * np.cos(theta)], [r * np.sin(theta)]]).T
    X_o = np.concatenate([[R * np.cos(theta)], [R * np.sin(theta)]]).T
    X_o_ = np.concatenate([[R_ * np.cos(theta)], [R_ * np.sin(theta)]]).T
    print(len(X_i), len(X_o), len(X_o_))
    X = np.concatenate([X_i, X_o, X_o_])
    Y = np.concatenate((np.zeros((q, 3)) + [0,0,1], np.zeros((q,3)) + [0,1,0], np.zeros((q,3)) + [0,0,1]))
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    return X, Y

def centroids():
    x1 = np.random.randn(500, 2) + np.array([3, 0])
    x2 = np.random.randn(500, 2) + np.array([0, -3])
    x3 = np.random.randn(500, 2) + np.array([-3, 0])

    y1 = np.zeros((500, 3))
    y1[:, 0] = 1
    y2 = np.zeros((500, 3))
    y2[:, 1] = 1
    y3 = np.zeros((500, 3))
    y3[:, 2] = 1
    Y = np.concatenate((y1, y2, y3))

    X = np.concatenate((x1, x2, x3))
    X[:, 0] = (X[:, 0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min())
    X[:, 1] = (X[:, 1] - X[:, 1].min()) / (X[:, 1].max() - X[:, 1].min())
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.show()

    X, Y = pair_shuffle(X, Y)

    x1 = np.random.randn(125, 2) + np.array([3, 0])
    x2 = np.random.randn(125, 2) + np.array([0, -3])
    x3 = np.random.randn(125, 2) + np.array([-3, 0])
    test_X = np.concatenate((x1, x2, x3))
    test_X[:, 0] = (test_X[:, 0] - test_X[:, 0].min()) / (test_X[:, 0].max() - test_X[:, 0].min())
    test_X[:, 1] = (test_X[:, 1] - test_X[:, 1].min()) / (test_X[:, 1].max() - test_X[:, 1].min())

    y1 = np.zeros((125, 3))
    y1[:, 0] = 1
    y2 = np.zeros((125, 3))
    y2[:, 1] = 1
    y3 = np.zeros((125, 3))
    y3[:, 2] = 1

    test_Y = np.concatenate((y1, y2, y3))

    test_X, test_Y = pair_shuffle(test_X, test_Y)

    plt.scatter(test_X[:, 0], test_X[:, 1], c=test_Y)
    plt.show()

    return X, Y, test_X, test_Y
