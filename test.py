from tensorflow.keras.layers import *


def cnapd(model, n_kernals, convo_kernal, pool_kernal=2, input_shape=input_shape,
          first_layer=False, normalize=True, activation='relu', drop_rate=.25, zpad=False):

    if zpad:
        model.add(ZeroPadding2D((1, 1)))

    if first_layer:
        model.add(Conv2D(n_kernals, (convo_kernal, convo_kernal), input_shape=input_shape, padding='same'))
    else:
        model.add(Conv2D(n_kernals, (convo_kernal, convo_kernal), padding='same'))

    if normalize:
        model.add(BatchNormalization())

    model.add(Activation(activation))
    model.add(MaxPooling2D((pool_kernal, pool_kernal)))
    model.add(Dropout(drop_rate))


def dnd(model, neurons, first_layer=False, input_shape=0, normalize=True, activation='relu', drop_rate=.25):
    if first_layer:
        model.add(Flatten(input_shape=input_shape))
    model.add(Dense(neurons))
    if normalize:
        model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(drop_rate))
    return model


def mcpd(model, n_conv, n_filters, convo_kernal=3, pool_kernal=(2, 2), activation='relu', drop_rate=0):
    for i in n_conv:
        model.add(Conv2D(n_filters, (convo_kernal, convo_kernal), activation=activation, padding='same'))
    model.add(MaxPooling2D(pool_kernal))
    model.add(Dropout(drop_rate))

