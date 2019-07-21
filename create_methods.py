from tensorflow.python import keras


def create_reg_lin_model(input_size: int, output_size: int):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(
        output_size,
        input_dim=input_size
    ))

    model.compile(
        loss=keras.losses.mse,
        optimizer=keras.optimizers.sgd(),
        metrics=[keras.metrics.categorical_accuracy]
    )

    return model


def create_mlp_model(layers: int, neurons_per_hidden_layer: int, input_size: int, output_size: int):
    model = keras.models.Sequential()

    for i in range(0, layers):
        if i == 0:
            model.add(keras.layers.Dense(
                neurons_per_hidden_layer,
                activation=keras.activations.tanh,
                input_shape=(input_size,)
            ))
        else:
            model.add(keras.layers.Dense(
                neurons_per_hidden_layer,
                activation=keras.activations.tanh,
            ))

    model.add(keras.layers.Dense(
        output_size,
        activation=keras.activations.sigmoid
    ))

    model.compile(
        loss=keras.losses.mse,
        optimizer=keras.optimizers.sgd(),
        metrics=[keras.metrics.categorical_accuracy]
    )

    return model


def create_mlp_dropout_model(layers: int, neurons_per_hidden_layer: int, input_size: int, output_size: int):
    model = keras.models.Sequential()

    for i in range(0, layers):
        if i == 0:
            model.add(keras.layers.Dropout(0.1, input_shape=(input_size,)))
            model.add(keras.layers.Dense(
                neurons_per_hidden_layer,
                activation=keras.activations.tanh,
                input_shape=(input_size,)
            ))
            model.add(keras.layers.Dropout(0.2))
        else:
            model.add(keras.layers.Dense(
                neurons_per_hidden_layer,
                activation=keras.activations.tanh,
            ))

    model.add(keras.layers.Dense(
        output_size,
        activation=keras.activations.sigmoid
    ))

    model.compile(
        loss=keras.losses.mse,
        optimizer=keras.optimizers.sgd(),
        metrics=[keras.metrics.categorical_accuracy]
    )

    return model


def create_conv_net_model1(input_size: int, output_size: int, optimizer=keras.optimizers.sgd()):
    model = keras.models.Sequential()

    model.add(keras.layers.Reshape((28, 28, 1), input_shape=(input_size,)))
    model.add(keras.layers.Conv2D(16, 3, padding='same', activation=keras.activations.relu))
    model.add(keras.layers.AveragePooling2D())
    model.add(keras.layers.Conv2D(32, 3, padding='same', activation=keras.activations.relu))
    model.add(keras.layers.AveragePooling2D())
    model.add(keras.layers.Conv2D(64, 3, padding='same', activation=keras.activations.relu))
    model.add(keras.layers.AveragePooling2D())

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(output_size, activation=keras.activations.sigmoid))
    model.compile(
        loss=keras.losses.mse,
        optimizer=optimizer,
        metrics=[keras.metrics.categorical_accuracy]
    )

    return model


def create_conv_net_model2(input_size: int, output_size: int):
    model = keras.models.Sequential()

    model.add(keras.layers.Reshape((28, 28, 1), input_shape=(input_size,)))
    model.add(keras.layers.Conv2D(32, 3, padding='same', activation=keras.activations.relu))
    model.add(keras.layers.Conv2D(32, 3, padding='same', activation=keras.activations.relu))
    model.add(keras.layers.Conv2D(32, 3, padding='same', activation=keras.activations.relu))
    model.add(keras.layers.Conv2D(32, 3, padding='same', activation=keras.activations.relu))
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(output_size, activation=keras.activations.sigmoid))
    model.compile(
        loss=keras.losses.mse,
        optimizer=keras.optimizers.sgd(),
        metrics=[keras.metrics.categorical_accuracy]
    )

    return model
