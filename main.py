# Import section
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Flatten, Dense, Dropout, Conv2D
from tensorflow.keras.layers import MaxPooling2D, ZeroPadding2D, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam


def cnapd(model, n_kernals, convo_kernal, pool_kernal=2, input_shape=0,
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


def dense(model, neurons, first_layer=False, input_shape=0, normalize=True, activation='relu', drop_rate=.25):
    if first_layer:
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(neurons))

    if normalize:
        model.add(BatchNormalization())

    model.add(Activation(activation))
    model.add(Dropout(drop_rate))


def dense_tweak(top_dense):
    return [int(top_dense / pow(2, i)) for i in range(n_dense)]


# Loading date
(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()

# Declaring initial constants
numclasses = 10
ytest = to_categorical(ytest, numclasses)
ytrain = to_categorical(ytrain, numclasses)
l2reg = 0
rows = 32
columns = 32
color = 3
input_shape = (rows, columns, color)

# Hyper Parameters
n_convo = 4
n_filter = [32*pow(2, i) for i in range(n_convo)]
convo_krn = [int(input_shape[1]/3) - 2*i for i in range(n_convo)]
pool_krn = 2
top_dense_layer = 512
n_dense = 4
neu_fstd = dense_tweak(top_dense_layer)
lr = 0.001
bs = 128

# start of model
model = Sequential()

for i in range(1, n_convo):
    first_layer = True if i == 1 else False
    cnapd(model, n_filter[i], convo_krn[i], pool_krn, input_shape, first_layer)

model.add(Flatten())

for i in range(1, n_dense):
    dense(model, neu_fstd)

model.add(Dense(10, activation='softmax'))

model.summary()

checkpoint = ModelCheckpoint("best_model.pk1",
                             monitor="val_loss",
                             mode="min",
                             save_best_only=True,)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=5,
                          restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              min_delta=0.00001)

tensorboard = TensorBoard()

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr), metrics=['accuracy'])

history = model.fit(xtrain, ytrain, batch_size=bs, epochs=30, validation_data=(xtest, ytest), shuffle=True,
                    callbacks=[tensorboard, earlystop, reduce_lr, checkpoint])

print(history.history)
