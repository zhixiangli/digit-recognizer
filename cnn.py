import pandas
import numpy
from keras.models import Sequential
from keras.layers.core import Dropout, Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils


numpy.random.seed(1337) # for reproducibility
images_size = 28


model = Sequential()

model.add(Convolution2D(32, 3, 3, input_shape=(1, images_size, images_size), dim_ordering='th'))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3, dim_ordering='th'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, dim_ordering='th'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, dim_ordering='th'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, init='normal'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10, init='normal'))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')


train = pandas.read_csv('./train.csv')
train_x = train.values[:, 1:].astype("float32")
train_x = train_x.reshape(train_x.shape[0], 1, 28, 28)
train_x = train_x / 255.0
train_y = np_utils.to_categorical(train.ix[:, 0].values.astype("int8"))

test = pandas.read_csv('./test.csv')
test_x = test.values.astype("float32")
test_x = test_x.reshape(test_x.shape[0], 1, 28, 28)
test_x = test_x / 255.0


model.fit(train_x, train_y, nb_epoch=32, batch_size=32, verbose=1, shuffle=True)


test_y = model.predict_classes(test_x)
pandas.DataFrame({"ImageId": range(1, len(test_y) + 1), "Label": test_y}).to_csv('./cnn.csv', index=False, header=True)
