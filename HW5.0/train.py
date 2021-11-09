import pandas as pd
import numpy as np 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from keras import regularizers
from keras.optimizers import RMSprop 
from sklearn.preprocessing import label_binarize
from keras.callbacks import CSVLogger

csv_logger = CSVLogger('training.log', separator=',', append=False)

glove_type=True
df=pd.read_csv("texts_new.csv")

class_num=4
model_type='LSTM'
texts=df['text'].tolist()
labels=df['label'].tolist()
max_features = 1000
max_len = 100
training_samples = 1200
validation_samples = 400
max_words = 1000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=max_len)
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
labels=label_binarize(labels, classes=[0, 1, 2, 3])
print(data.shape)

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]






# Train same model 


if model_type=='CNN':
	model = Sequential()
	model.add(layers.Embedding(max_features, 32, input_length=max_len))
	model.add(layers.Conv1D(32, 8, activation='relu',kernel_regularizer=regularizers.l2(0.02)))
	model.add(layers.Dropout(0.3))
	model.add(layers.MaxPooling1D(6))
	model.add(layers.Conv1D(32, 8, activation='relu'))
	model.add(layers.Dropout(0.3))
	model.add(layers.GlobalMaxPooling1D())
	model.add(layers.Dense(4,activation='softmax'))
	model.compile(optimizer=RMSprop(lr=1e-3),
	loss='categorical_crossentropy',
	metrics=['accuracy'])
elif model_type=='simple_RNN':
	model = Sequential()
	model.add(layers.Embedding(max_features, 32))
	model.add(layers.SimpleRNN(32))
	model.add(layers.Dense(4, activation='softmax'))
	model.compile(optimizer=RMSprop(lr=1e-3),
	loss='categorical_crossentropy',
	metrics=['accuracy'])
elif model_type=="LSTM":
	model = Sequential()
	model.add(layers.Embedding(max_features, 32))
	model.add(layers.LSTM(32))
	model.add(layers.Dense(4, activation='softmax'))
	model.compile(optimizer=RMSprop(lr=1e-3),
	loss='categorical_crossentropy',
	metrics=['accuracy'])

model.save('training_model.hw5')
history = model.fit(x_train, y_train,
epochs=30,
batch_size=64,
validation_split=0.2,callbacks=[csv_logger])

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('accuracy.png')
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Loss of Training and Validation')
plt.legend()
plt.savefig('train_val_loss.png')


y_prediction= model.predict(x_val)

#ROC and AUC
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc


# Compute ROC curve and ROC area
false_pos = dict()
true_pos = dict()
roc_auc = dict()
novel_names={0:"Friars and Filipinos", 1:"Sister Carrie", 2:"Monday and Tuesday", 3:"The Letters of Jane Austen"}
for i in range(4):
    false_pos[i], true_pos[i], _ = roc_curve(y_val[:, i], y_prediction[:, i])
    roc_auc[i] = auc(false_pos[i], true_pos[i])
    print('AUC score for title %s is %f' % (novel_names[i],roc_auc[i]))



colors = cycle(['read', 'green', 'blue','yellow'])
for i, color in zip(range(class_num), colors):
    plt.plot(false_pos[i], true_pos[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for all classes')
plt.legend(loc="lower right")
plt.savefig('ROC.png')



