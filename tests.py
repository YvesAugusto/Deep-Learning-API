from Layers.conv import Conv2D
from Models.model import Model
import cv2 as cv

model = Model()
model.add_layer(Conv2D(units=32, input_shape=(359,638,3)))
# X = np.random.random((28,28,3))
X = cv.imread('./images/image.jpeg')
# plt.imshow(X)
# plt.show()
_, X, _ = model.forward(X)
cv.imwrite('./images/convolved_image.jpeg', X)