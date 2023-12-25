from keras.utils import image_dataset_from_directory
import tensorflow as tf
import os, keras
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Activation, Add, Input, Conv2DTranspose
from keras.callbacks import EarlyStopping

current_directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_directory, 'image')
os.chdir(path)
channels = 3
cropped_width = 256
cropped_height = 256
upscale_factor = 2
input_width = cropped_width // upscale_factor
input_height = cropped_height // upscale_factor


def Model(channels):
    inputs = keras.Input(shape=(input_width, input_height, channels))

    X1 = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
    X2 = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(X1)
    X3 = MaxPool2D(pool_size=(2, 2), padding="same")(X2)
    X4 = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(X3)
    X5 = Conv2DTranspose(64, 4, padding='same', activation='relu', kernel_initializer='he_normal', strides=(2, 2))(X4)
    X6 = Add()([X5, X2])
    X7 = Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(X6)
    X8 = Conv2DTranspose(64, 4, padding='same', activation='relu', kernel_initializer='he_normal', strides=(2, 2))(X7)
    X9 = Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_normal')(X8)
    outputs = Conv2D(channels, 3, padding='same', activation='sigmoid', kernel_initializer='Orthogonal')(X9)

    model = keras.Model(inputs, outputs)
    return model


def normalize(image):
    image = image / 255.0
    return image


def process_features(input, new_width, new_height):
    return tf.image.resize(input, [new_width, new_height], method="area")


train_set = image_dataset_from_directory(
    path,
    image_size=(cropped_width, cropped_height),
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=123,
    label_mode=None
)

validation_set = image_dataset_from_directory(
    path,
    image_size=(cropped_width, cropped_height),
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=123,
    label_mode=None
)

train_set = train_set.map(normalize)
validation_set = validation_set.map(normalize)

train_set = train_set.map(lambda x: (process_features(x, input_width, input_height), x))
test_set = validation_set.map(lambda x: (process_features(x, input_width, input_height), x))

early_stopping = EarlyStopping(monitor='loss', patience=10, min_delta=0.0001)
model = Model(channels)

model.compile(optimizer='adam', loss='MSE')

model.summary()
model.fit(train_set, epochs=20, callbacks=[early_stopping], validation_data=test_set, verbose=1)

parent_directory = os.path.dirname(os.getcwd())
os.chdir(parent_directory)
os.chdir('models')
model.save('new_model.h5')
