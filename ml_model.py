import keras.models
from keras import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten


MODEL_PATH = 'models/model.npy'


def init_model() -> Model:
    model = Sequential(
        layers=[
            Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 1)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

    model.compile(optimizer="adam",
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def load_model() -> Model:
    return keras.models.load_model('model')
