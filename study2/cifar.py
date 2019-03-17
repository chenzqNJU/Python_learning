file='e:\\tf\\cifar-10-python\\cifar-10-batches-py\\data_batch_1'
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
a=unpickle(file)

# import the modules we need
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras import metrics
from keras.optimizers import SGD,RMSprop,Adam
from keras.callbacks import EarlyStopping
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt
from imp import reload
reload(plt)
reload(matplotlib)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.09
set_session(tf.Session(config=config))

#define the Sequential model
input_shapes=(32,32,3);nb_class='NB_CLASSES'
class CNNNet:

    @staticmethod
    def createNet(input_shapes,nb_class):

        feature_layers = [
        BatchNormalization(input_shape=input_shapes),
        Conv2D(64,3,3,border_mode="same"),
        Activation("relu"),
        BatchNormalization(),
        Conv2D(64,3,3,border_mode="same"),
        Activation("relu"),
        MaxPooling2D(pool_size=(2,2),strides=(2,2)),
        BatchNormalization(),
        Conv2D(128,3,3,border_mode="same"),
        Activation("relu"),
        BatchNormalization(),
        Dropout(0.5),
        Conv2D(128,3,3,border_mode="same"),
        Activation("relu"),
        MaxPooling2D(pool_size=(2,2),strides=(2,2)),
        BatchNormalization(),
        Dropout(0.5),
        Conv2D(128,3,3,border_mode="same"),
        Activation("relu"),
        Dropout(0.5),
        Conv2D(128,3,3,border_mode="same"),
        Activation("relu"),
        MaxPooling2D(pool_size=(2,2),strides=(2,2)),
        BatchNormalization()
        ]

        classification_layer=[
        Flatten(),
        Dense(512),
        Activation("relu"),
        Dropout(0.5),
        Dense(nb_class),
        Activation("softmax")
        ]

        model = Sequential(feature_layers+classification_layer)
        return model

#parameters
NB_EPOCH = 40
BATCH_SIZE = 128
VERBOSE = 1
VALIDATION_SPLIT = 0.2
IMG_ROWS=32
IMG_COLS = 32
NB_CLASSES = 10
INPUT_SHAPE =(IMG_ROWS,IMG_COLS,3)

import numpy as np
import os
from keras.datasets.cifar import load_batch
from keras import backend as K

import keras
keras.__version__
num_train_samples = 50000
x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
y_train = np.empty((num_train_samples,), dtype='uint8')

path=r'E:\tf\cifar-10-python\cifar-10-batches-py'
i=1
for i in range(1, 6):
    fpath = os.path.join(path, 'data_batch_' + str(i))
    (x_train[(i - 1) * 10000: i * 10000, :, :, :],
     y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

fpath = os.path.join(path, 'test_batch')
x_test, y_test = load_batch(fpath)

y_train = np.reshape(y_train, (len(y_train), 1))
y_test = np.reshape(y_test, (len(y_test), 1))

if K.image_data_format() == 'channels_last':
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

(X_train,Y_train),(X_test,Y_test)= (x_train, y_train), (x_test, y_test)
#load cifar-10 dataset
# (X_train,Y_train),(X_test,Y_test) = cifar10.load_data()
X_train.shape;Y_train.shape

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train = X_train.reshape(X_train.shape[0],IMG_ROWS,IMG_COLS,3)
X_test = X_test.reshape(X_test.shape[0],IMG_ROWS,IMG_COLS,3)

print(X_train.shape[0],"train samples")
print(Y_test.shape[0],"test samples")

#convert class vectors to binary class matrices
Y_train = to_categorical(Y_train,NB_CLASSES)
Y_test = to_categorical(Y_test,NB_CLASSES)

# init the optimizer and model
model = CNNNet.createNet(input_shapes=(32,32,3),nb_class=NB_CLASSES)
model.summary()
model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['acc'])

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

NB_EPOCH=4
history = model.fit(X_train,Y_train,
                batch_size = BATCH_SIZE,
                nb_epoch = NB_EPOCH,
                verbose=VERBOSE,
                validation_split=VALIDATION_SPLIT,
                callbacks=[early_stopping]
                )

score = model.evaluate(X_test,Y_test,verbose = VERBOSE)
print("")
print("====================================")
print("====================================")
print(score[0])
print(score[1])
print("====================================")
print("====================================")

#save model
model.save("my_model"+str(score[1])+".h5")

#show the data in history
print(history.history.keys())

#summarize history for accuracy
plt.close('all')
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("Model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train","test"],loc="upper left")

#summarize history for loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train","test"],loc="upper left")
plt.savefig("Performance:"+str(score[1])+".jpg")

plt.savefig("Performance:"+str(score[1])+".png")


plt.show()

os.getcwd()


import pickle
with open('history.pkl', 'wb') as f:
    pickle.dump(history, f)

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)  # 创建图表1
plt.figure(2)  # 创建图表2
ax1 = plt.subplot(211)  # 在图表2中创建子图1
ax2 = plt.subplot(212)  # 在图表2中创建子图2

x = np.linspace(0, 3, 100)
for i in range(5):
    plt.figure(1)  # 选择图表1
    plt.plot(x, np.exp(i * x / 3))
    plt.sca(ax1)  # 选择图表2的子图1
    plt.plot(x, np.sin(i * x))
    plt.sca(ax2)  # 选择图表2的子图2
    plt.plot(x, np.cos(i * x))

plt.show()
plt.savefig('table.png')