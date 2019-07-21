from tensorflow.python import keras
import numpy as np
from create_methods import *

INPUT_NEURONS = 784
OUTPUT_NEURONS = 10


def fit_model(train, test, model, name, epochs=1500, batch_size=32):
    tb_callback = keras.callbacks.TensorBoard('./logs/' + name)
    model.fit(*train,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[tb_callback],
              validation_data=test
              )

    keras.utils.plot_model(model, './images/' + name + '.png')
    model.save('./models/' + name + ".h5")


def data():
    # Tuple(Tuple(Array, Array), Tuple(Array, Array))
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Normalize models_project
    x_train = np.reshape(x_train, (-1, INPUT_NEURONS)) / 255.0
    x_test = np.reshape(x_test, (-1, INPUT_NEURONS)) / 255.0

    # Categorize labels
    # eg : 5 to [0, 0, 0, 0, 1, 0, 0, 0, 0]
    y_train = keras.utils.to_categorical(y_train, OUTPUT_NEURONS)
    y_test = keras.utils.to_categorical(y_test, OUTPUT_NEURONS)

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    train_data, test_data = data()

    # Linear Regression model
    fit_model(
        train=train_data,
        test=test_data,
        model=create_reg_lin_model(INPUT_NEURONS, OUTPUT_NEURONS),
        name="linear_regression"
    )

    # Deep learning models
    for layer in range(1, 4):
        for neron in range(2, 5):
            fit_model(
                train=train_data,
                test=test_data,
                model=create_mlp_model(layer, 2**neron, INPUT_NEURONS, OUTPUT_NEURONS),
                name="DNN_" + str(layer) + '_' + str(neron)
            )

            fit_model(
                train=train_data,
                test=test_data,
                model=create_mlp_dropout_model(layer, 2**neron, INPUT_NEURONS, OUTPUT_NEURONS),
                name="DNN_dropout_" + str(layer) + '_' + str(neron)
            )

    # CNN model
    fit_model(
        train=train_data,
        test=test_data,
        model=create_conv_net_model1(INPUT_NEURONS, OUTPUT_NEURONS),
        name='CNN_1'
    )

    fit_model(
        train=train_data,
        test=test_data,
        model=create_conv_net_model2(INPUT_NEURONS, OUTPUT_NEURONS),
        name='CNN_2'
    )

    fit_model(
        train=train_data,
        test=test_data,
        model=create_conv_net_model1(
            INPUT_NEURONS, OUTPUT_NEURONS, optimizer=keras.optimizers.adam(lr=0.0002, beta_1=0.5)
        ),
        name='CNN_1_Adam'
    )
