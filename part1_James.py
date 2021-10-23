import numpy as np
import matplotlib.pyplot as plt
import random
import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.datasets import mnist, fashion_mnist,cifar10
from keras import layers
from keras import models
from keras.utils import to_categorical

dataset="mnist"

#set hyper-parameters
epoch=3,
Loss = 'categorical_crossentropy'
Metric = ['accuracy']
optimizer='rmsprop'
model_type = 'CNN'
batch_size=64
model_visualization = True
data_augument = False
show_image=True

if dataset=='mnist':
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
	train_shape=(28, 28, 1)
elif dataset=='fashion_mnist':
	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
	train_shape=(28, 28, 1)	
else:
	(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
	train_shape=(32, 32, 3)

train_images, valid_images, train_labels, valid_labels = train_test_split(train_images, train_labels, test_size=0.2)
train_shape=(train_images.shape[0], train_shape[0], train_shape[1], train_shape[2])
valid_shape=(valid_images.shape[0], train_shape[1], train_shape[2], train_shape[3])

test_shape=(10000,train_shape[1], train_shape[2], train_shape[3])


train_images = train_images.reshape(train_shape)
train_images = train_images.astype('float32') / 255

valid_images = valid_images.reshape(valid_shape)
valid_images = valid_images.astype('float32') / 255

test_images = test_images.reshape(test_shape)
test_images = test_images.astype('float32') / 255	

train_labels = to_categorical(train_labels)
valid_labels = to_categorical(valid_labels)
test_labels = to_categorical(test_labels)



if dataset == 'mnist' or dataset == 'fashion_mnist':
    inp_shape = (28, 28, 1)
elif dataset == 'cifar10':
    inp_shape = (32, 32, 3)
if model_type == 'CNN':
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=inp_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()
    model.compile(optimizer=optimizer, loss=Loss, metrics=Metric)

elif model_type == 'ANN':
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=inp_shape))
    model.add(layers.Dense(32, activation='relu',
              kernel_regularizer=keras.regularizers.l2(l=0.02)))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer, loss=Loss, metrics=Metric)

if show_image:
	index=random.randint(0, train_images.shape[0]-1)
	image=train_images[index]
	image=np.array(image,dtype='float')
	plt.imshow(image)
	plt.show()

if data_augument:
	if dataset=='mnist' or dataset=='fashion_mnist':
		trainshape = (48000, 28, 28, 1)
		testshape = (10000, 28, 28, 1)

	elif dataset=='cifar10':
		trainshape = (40000, 32, 32, 3)
		testshape = (10000, 32, 32, 3)

	datagen = ImageDataGenerator(rotation_range=40,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest')

	train_images = np.array(train_images)
	train_images = train_images.reshape(train_shape)
	datagen.fit(train_images)
	train_generator = datagen.flow(train_images, train_labels,batch_size=batch_size, subset='training')
	validation_generator = datagen.flow(valid_images, valid_labels,batch_size=batch_size, subset='validation')
	history = model.fit_generator(generator=train_generator,
	                              validation_data=validation_generator,
	                              use_multiprocessing=True,
	                              steps_per_epoch=len(train_generator)/60,
	                              validation_steps=len(validation_generator)/60,
	                              epochs=epoch, workers=-1)
else:
	history = model.fit(train_images, train_labels, epochs=epoch[0],
	                    batch_size=batch_size[0], validation_data=(valid_images, valid_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels)

#Include a method to save model and hyper parameter
if model_save:
    model.save("saved_model.h5")


#Include a training,validation history plot at the end and print (train/test/val) metrics
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
print("The train accuracy is:", acc)
print("The valid accuracy is:", val_acc)
print("The test accuracy is:", test_acc)
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'ro', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'ro', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


if model_visualization:
    num_to_show = random.randint(0, train_images.shape[0]-1)
    img_tensor = train_images[num_to_show]
    img_tensor = np.array(img_tensor)
    img_tensor = img_tensor.reshape((1,) + train_shape[1:])

    layer_outputs = [layer.output for layer in model.layers[:3]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)


    activations = activation_model.predict(img_tensor)

    layer_names = []
    for layer in model.layers[:3]:
        layer_names.append(layer.name)

    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]

        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,
                             row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()
