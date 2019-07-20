from tensorflow.python import keras
from PIL import Image


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    for i in range(50):
        im = Image.fromarray(x_train[i])
        im.save("images_show/" + str(i) + ".png")
