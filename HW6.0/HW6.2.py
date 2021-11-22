

#SOURCE: MODIFIED FROM https://blog.keras.io/building-autoencoders-in-keras.html

import keras
from keras import layers
import matplotlib.pyplot as plt

from keras.datasets import mnist,cifar10
import numpy as np

import logging
import os
import pandas as pd
import seaborn as sns

from sklearn import metrics

#USER PARAM
INJECT_NOISE    =   False
EPOCHS          =   35
NKEEP           =   2500        #DOWNSIZE DATASET
BATCH_SIZE      =   128
DATA            =   "MNIST"

#GET DATA
if(DATA=="MNIST"):
    (x_train, _), (x_test, _) = mnist.load_data()
    N_channels=1; PIX=28

if(DATA=="CIFAR"):
    (x_train, _), (x_test, _) = cifar10.load_data()
    N_channels=3; PIX=32
    EPOCHS=100 #OVERWRITE

#NORMALIZE AND RESHAPE
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

#DOWNSIZE TO RUN FASTER AND DEBUG
print("BEFORE",x_train.shape)
x_train=x_train[0:NKEEP]
x_test=x_test[0:NKEEP]
print("AFTER",x_train.shape)

#ADD NOISE IF DENOISING
if(INJECT_NOISE):
    EPOCHS=2*EPOCHS
    #GENERATE NOISE
    noise_factor = 0.5
    noise= noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
    x_train=x_train+noise
    noise= noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
    x_test=x_test+noise

    #CLIP ANY PIXELS OUTSIDE 0-1 RANGE
    x_train = np.clip(x_train, 0., 1.)
    x_test = np.clip(x_test, 0., 1.)

#BUILD CNN-AE MODEL


if(DATA=="MNIST"):
    input_img = keras.Input(shape=(PIX, PIX, N_channels))

    # #ENCODER
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)

    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    # # AT THIS POINT THE REPRESENTATION IS (4, 4, 8) I.E. 128-DIMENSIONAL
 
    # #DECODER
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(N_channels, (3, 3), activation='sigmoid', padding='same')(x)


if(DATA=="CIFAR"):
    input_img = keras.Input(shape=(PIX, PIX, N_channels))

    #ENCODER
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    #DECODER
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(N_channels, (3, 3), activation='sigmoid', padding='same')(x)



#COMPILE
autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy');
autoencoder.summary()

#TRAIN
history = autoencoder.fit(x_train, x_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                validation_data=(x_test, x_test),
                )

#HISTORY PLOT
epochs = range(1, len(history.history['loss']) + 1)
plt.figure()
plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
plt.legend()

#MAKE PREDICTIONS FOR TEST DATA
decoded_imgs = autoencoder.predict(x_test)

#VISUALIZE THE RESULTS
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(PIX, PIX,N_channels))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(PIX, PIX,N_channels))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


from keras.datasets import mnist,fashion_mnist
(x_train_fashion, y_train_fashion), (x_test_fashion, y_test_fashion) = fashion_mnist.load_data()
x_train_fashion=x_train_fashion[0:NKEEP]

decode_x_train = autoencoder.predict(x_train,batch_size=BATCH_SIZE)
decode_x_test = autoencoder.predict(x_test,batch_size=BATCH_SIZE)
decode_x_train_fashion = autoencoder.predict(x_train_fashion,batch_size=BATCH_SIZE)
mse_threshold = metrics.mean_squared_error(x_train, decode_x_train)

from keras.datasets import mnist,fashion_mnist
(x_train_fashion, y_train_fashion), (x_test_fashion, y_test_fashion) = fashion_mnist.load_data()
x_train_fashion=x_train_fashion[0:NKEEP]

anomaly_indices=[]
for i in range(x_test.shape[0]):
	mse = metrics.mean_squared_error(x_test[i], decode_x_test[i])
	if mse_threshold<=mse:
		anomaly_indices.append(i)

print("the anomaly fraction for minist is: %f" % (len(anomaly_indices)/x_test.shape[0]))


anomaly_indices=[]
for i in range(x_train_fashion.shape[0]):
	mse = metrics.mean_squared_error(x_train_fashion[i], decode_x_train_fashion[i])
	if mse_threshold<=mse:
		anomaly_indices.append(i)
print("the anomaly fraction for fashion minist is: %f" % (len(anomaly_indices)/x_train_fashion.shape[0]))


n = 5
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(PIX,PIX,))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decode_x_test[i].reshape(PIX,PIX,))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('minist.png')

n = 5
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_train_fashion[anomaly_indices[i]].reshape(PIX,PIX,))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decode_x_train_fashion[anomaly_indices[i]].reshape(PIX,PIX,))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('fashion_image_6_1.png')