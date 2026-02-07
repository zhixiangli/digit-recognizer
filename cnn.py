import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Dense, Dropout, Flatten, Input, MaxPooling2D,
    RandomRotation, RandomTranslation, RandomZoom,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

np.random.seed(1337)  # for reproducibility
images_size = 28


def load_train_data(train_csv):
    train = pandas.read_csv(train_csv)
    train_x = train.values[:, 1:].astype("float32")
    train_x = train_x.reshape(train_x.shape[0], images_size, images_size, 1)
    train_x = train_x / 255.0
    train_y = to_categorical(train.iloc[:, 0].values.astype("int8"))
    return train_x, train_y


def build_model():
    model = Sequential()

    model.add(Input(shape=(images_size, images_size, 1)))
    model.add(RandomRotation(20 / 360))
    model.add(RandomZoom(0.2))
    model.add(RandomTranslation(height_factor=0.2, width_factor=0.2))

    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=["accuracy"])
    return model


def train_model(model, train_x, train_y):
    hist = model.fit(train_x, train_y, batch_size=32, epochs=100, verbose=1)
    plt.plot(hist.history['loss'], color='b')
    plt.plot(hist.history['accuracy'], color='r')
    plt.show()


def predict(model, test_csv, result_csv):
    test = pandas.read_csv(test_csv)
    test_x = test.values.astype("float32")
    test_x = test_x.reshape(test_x.shape[0], images_size, images_size, 1)
    test_x = test_x / 255.0
    test_y = np.argmax(model.predict(test_x), axis=-1)
    pandas.DataFrame({"ImageId": range(1, len(test_y) + 1), "Label": test_y}).to_csv(result_csv, index=False,
                                                                                     header=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data', action="store", default="./train.csv")
    parser.add_argument('test_data', action="store", default="./test.csv")
    parser.add_argument('predict_data', action="store", default="./result.csv")
    args = parser.parse_args()

    train_x, train_y = load_train_data(args.train_data)
    model = build_model()
    train_model(model, train_x, train_y)
    predict(model, args.test_data, args.predict_data)
