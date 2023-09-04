import tensorflow as tf
# The residual block
def residual_block(inputs, out_channel, activation, same_shape=False):
    strides = (2, 2) if same_shape else (1, 1)

    # Convolution layers
    x = tf.keras.layers.Conv2D(filters=out_channel, kernel_size=(3, 3), strides=strides, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation=activation)(x)

    x = tf.keras.layers.Conv2D(filters=out_channel, kernel_size=(3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Downsample the input if needed
    if same_shape:
        inputs = tf.keras.layers.Conv2D(filters=out_channel, kernel_size=(1, 1), strides=(2, 2), padding="same")(inputs)

    # Add the residual connection
    x = tf.keras.layers.Add()([x, inputs])
    x = tf.keras.layers.Activation(activation=activation)(x)

    return x


# ResNet 18
def ResNet(input_shape, num_classes, activation):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation=activation)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

    x = residual_block(inputs=x, out_channel=64, activation=activation)
    x = residual_block(inputs=x, out_channel=64, activation=activation)

    x = residual_block(inputs=x, out_channel=128, activation=activation, same_shape=True)
    x = residual_block(inputs=x, out_channel=128, activation=activation)

    x = residual_block(inputs=x, out_channel=256, activation=activation, same_shape=True)
    x = residual_block(inputs=x, out_channel=256, activation=activation)

    x = residual_block(inputs=x, out_channel=512, activation=activation, same_shape=True)
    x = residual_block(inputs=x, out_channel=512, activation=activation)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=num_classes, activation="softmax")(x)

    res_net = tf.keras.Model(inputs=inputs, outputs=x)
    return res_net


# Initialize
def utilize(lr, activation, batch_size):
    input_shape = (28, 28, 1)
    num_classes = 62
    model = ResNet(input_shape, num_classes, activation)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr), loss="categorical_crossentropy",
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    print(model.summary())
    result = model.fit(x=train_data, y=train_label, batch_size=batch_size, epochs=1, validation_split=0.2, shuffle=True)
    return result


def resnet_grid_search():
    learning_rate = [0.01, 0.001, 0.0001]
    activations = ["relu", "tanh"]
    batch_sizes = [32, 64, 128]
    for lr in learning_rate:
        for activ in activations:
            for size in batch_sizes:
                result = utilize(lr=lr, activation=activ, batch_size=size)
                print(lr, activ, size, result)


resnet_grid_search()
result = utilize(best_lr, best_activation, best_batch_size)

import pandas as pd
import matplotlib.pyplot as plt

# Convert the history result dictionary to a Pandas dataframe and extract the accuracies
accuracies = pd.DataFrame(result.history)[['accuracy', 'val_accuracy']]

# Plot the accuracies
accuracies.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.xlabel('Epoch')
plt.show()
