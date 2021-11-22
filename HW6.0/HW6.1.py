import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from keras import models
from keras import layers
from sklearn import metrics
from keras.callbacks import CSVLogger
import logging
import os

csv_logger = CSVLogger('1.log', separator=',', append=False)
run_model=1
n_bottleneck=50
NKEEP=10000  #DOWNSIZE DATASET
NH=100

#GET DATASET
from keras.datasets import mnist,fashion_mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
(X_train_fashion, Y_train_fashion), (X_test_fashion, Y_test_fashion) = fashion_mnist.load_data()


X_train=X_train[0:NKEEP]
X_test=X_test[0:NKEEP]
X_train_fashion=X_train_fashion[0:NKEEP]

#NORMALIZE AND RESHAPE
X_train=X_train/np.max(X_train) 
X_train=X_train.reshape(NKEEP,28*28); 

X_test=X_test/np.max(X_test) 
X_test=X_test.reshape(NKEEP,28*28); 

X_train_fashion=X_train_fashion/np.max(X_train_fashion) 
X_train_fashion=X_train_fashion.reshape(NKEEP,28*28); 


EPOCHS          =   5
BATCH_SIZE      =   512
PIX=28

if run_model==1:

	input_img = keras.Input(shape=(PIX*PIX))
	encode=layers.Dense(NH, activation='relu')(input_img)
	bottleneck=layers.Dense(n_bottleneck, activation='relu')(encode)
	decode=layers.Dense(NH, activation='relu')(bottleneck)
	decode=layers.Dense(28*28,  activation='sigmoid')(decode)

	autoencoder = keras.Model(input_img, decode)
	autoencoder.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy']);

	#TRAIN
	history = autoencoder.fit(X_train, X_train,
	                epochs=EPOCHS,
	                batch_size=BATCH_SIZE,
	                validation_split=0.2,callbacks=[csv_logger])
	autoencoder.save('hw6_1_model.h5')
	#HISTORY PLOT
	epochs = range(1, len(history.history['loss']) + 1)
	plt.figure()
	plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
	plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.savefig('1.png')

	plt.figure()
	plt.plot(epochs, history.history['accuracy'], 'bo', label='Training acc')
	plt.plot(epochs, history.history['val_accuracy'], 'b', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.savefig('2.png')

else:
	autoencoder=keras.models.load_model("hw6_1_model.h5")

decode_X_train = autoencoder.predict(X_train,batch_size=BATCH_SIZE)
decode_X_test = autoencoder.predict(X_test,batch_size=BATCH_SIZE)
decode_X_train_fashion = autoencoder.predict(X_train_fashion,batch_size=BATCH_SIZE)
mse_threshold = metrics.mean_squared_error(X_train, decode_X_train)



anomaly_indices=[]
for i in range(X_test.shape[0]):
	mse = metrics.mean_squared_error(X_test[i], decode_X_test[i])
	if mse_threshold<=mse:
		anomaly_indices.append(i)

print("the anomaly fraction for minist is: %f" % (len(anomaly_indices)/X_test.shape[0]))


anomaly_indices=[]
for i in range(X_train_fashion.shape[0]):
	mse = metrics.mean_squared_error(X_train_fashion[i], decode_X_train_fashion[i])
	if mse_threshold<=mse:
		anomaly_indices.append(i)
print("the anomaly fraction for fashion minist is: %f" % (len(anomaly_indices)/X_train_fashion.shape[0]))


n = 5
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(X_test[i].reshape(PIX,PIX,))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decode_X_test[i].reshape(PIX,PIX,))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('minist.png')

n = 5
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(X_train_fashion[anomaly_indices[i]].reshape(PIX,PIX,))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decode_X_train_fashion[anomaly_indices[i]].reshape(PIX,PIX,))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('fashion_image_6_1.png')