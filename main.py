from Layers.input import Input
from Layers.dense import Dense
from Models.model import Model
from Datasets.datasets import *
from Optimizers.sgd import SGD
from Optimizers.bgd import BGD

if __name__ == '__main__':
    X, Y, test_X, test_Y = centroids()

    model = Model()
    model.add_layer(Input(input_shape=2, units=6, activation_function='relu'))
    model.add_layer(Dense(units=8, activation_function='sigmoid'))
    model.add_layer(Dense(units=3, activation_function='softmax'))
    bgd = BGD(momentum=0.999, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=bgd)
    model.fit(X, Y, validation_data=(test_X, test_Y), batch_size=32)