import tensorflow as tf


# The residual block
def residual_block(inputs, out_channel, strides=(1, 1)):
    shortcut = inputs

    # First block
    x = tf.keras.layers.Conv2D(out_channel, kernel_size=(1, 1), strides=strides, padding="valid")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # Second block
    x = tf.keras.layers.Conv2D(out_channel, kernel_size=(3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # Third block
    x = tf.keras.layers.Conv2D(out_channel * 4, kernel_size=(1, 1), padding="valid")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Adjust the shortcut connection if needed
    if strides != (1, 1) or inputs.shape[-1] != out_channel * 4:
        shortcut = tf.keras.layers.Conv2D(out_channel * 4, kernel_size=(1, 1), strides=strides, padding="valid")(inputs)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    # Add the residual connection
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation("relu")(x)

    return x


# ResNet 152
def ResNet(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

    x = residual_block(x, 64, strides=(1, 1))
    for _ in range(2):
        x = residual_block(x, 64)

    x = residual_block(x, 128, strides=(2, 2))
    for _ in range(3):
        x = residual_block(x, 128)

    x = residual_block(x, 256, strides=(2, 2))
    for _ in range(35):
        x = residual_block(x, 256)

    x = residual_block(x, 512, strides=(2, 2))
    for _ in range(7):
        x = residual_block(x, 512)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    res_net = tf.keras.Model(inputs=inputs, outputs=x)
    return res_net


# Initialize
input_shape = (28, 28, 1)
num_classes = 62
model = ResNet(input_shape, num_classes)
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001), loss="categorical_crossentropy",
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
model.summary()

import pickle
import numpy as np

f = open('EMNIST_Byclass_Small/emnist_train.pkl', 'rb')
dict = pickle.load(f)
train_data = np.array(dict["data"])
train_label = np.array(dict["labels"])
train_label = tf.keras.utils.to_categorical(train_label, num_classes)
model.fit(x=train_data, y=train_label, batch_size=128, epochs=3, validation_split=0.2, shuffle=True)
