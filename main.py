# Import section
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Flatten, Dense, Dropout, Conv2D
from tensorflow.keras.layers import MaxPooling2D, ZeroPadding2D, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from json import load, dump
from time import time
import argparse


# Defines a group of Convolution, Batch Normalisation, Activation, Maxpooling and Dropout layers
# Optionall added first layer functionality which will add input shape to first Convolution layer
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


# Defines a group of Dense, Batch Normalization, Activation and Dropout
# Optionally added first layer functionality which will add Flattening layer before Dense layer and add input shape to first Dense layer
def dense(model, neurons, first_layer=False, input_shape=0, normalize=True, activation='relu', drop_rate=.25):
    if first_layer:
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(neurons))

    if normalize:
        model.add(BatchNormalization())

    model.add(Activation(activation))
    model.add(Dropout(drop_rate))


# For Argument Parsing in this case model number from JSON file
parser = argparse.ArgumentParser(description="Specify model configuration")
parser.add_argument('-c', '--conf', type=int, metavar=' ', help='Model Configuration')
args = parser.parse_args()

# To load model configrations from JSON file
with open('model_hparam.json', 'r') as f:
    data = load(f)

conf = data['hyper_params'][args.conf]

# Loading data
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
n_convo = conf['n_convo']
n_filter = conf['n_filter']
convo_krn = conf['convo_krn']
pool_krn = conf['pool_krn']
top_dense_layer = conf['top_dense_layer']
n_dense = conf['n_dense']
neu_fstd = conf['neu_fstd']
lr = conf['lr']
bs = conf['bs']
current_time = int(time())
NAME = "Model-{}-{}".format(args.conf, current_time)

# Starting of model
model = Sequential()

# For adding Convolution layers groups
for i in range(0, n_convo):
    first_layer = True if i == 0 else False
    cnapd(model, n_filter[i], convo_krn[i], pool_krn, input_shape, first_layer)

# For Flatten layer
model.add(Flatten())

# For adding Demse layers group
for i in range(1, n_dense):
    dense(model, neu_fstd)

# Output layer
model.add(Dense(10, activation='softmax'))

# Shows Model Architecture
model.summary()

# Creating callbacks
# Saves best model in .pk1 format
checkpoint = ModelCheckpoint("best_model.pk1",
                             monitor="val_loss",
                             mode="min",
                             save_best_only=True,)

# For early stopping if model is not learning further
earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0.00001,
                          patience=5,
                          restore_best_weights=True)

# Reduces learning rate for better learning
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              min_delta=0.00001)

# Creates logfiles which can be used for data visualization in tensorboard by running "tensorboard --logdir logs" on command line
tensorboard = TensorBoard(log_dir=f'''logs/{NAME}''')

# For Compiling model 
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr), metrics=['accuracy'])

# To start Learning Process
history = model.fit(xtrain, ytrain, 
                    batch_size=bs, 
                    epochs=50, 
                    validation_data=(xtest, ytest), 
                    shuffle=True,
                    callbacks=[tensorboard, earlystop, reduce_lr, checkpoint])

print(history.history)

# Creates dictonary object which holds all the data of current model
results = {'model': args.conf,
           'n_convo': n_convo,
           'n_dense': n_dense,
           'convo_krn': convo_krn,
           'n_filter': n_filter,
           'pool_krn': pool_krn,
           'top_dense_layer': top_dense_layer,
           'neu_fstd': neu_fstd,
           'lr': lr,
           'bs': bs,
           'loss': history.history['loss'],
           'val_loss': history.history['val_loss'],
           'accuracy': history.history['accuracy'],
           'val_accuracy': history.history['val_accuracy']}


# Writes model data in JSON file
with open('results.json', 'rw') as f:
    data = load('results.json')
    temp = data['result']
    temp.append(results)
    dump(data, f, indent=2)
    
