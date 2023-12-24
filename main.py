from keras.utils import image_dataset_from_directory
import tensorflow as tf
import os, keras
from keras.layers import Conv2D
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


def Model(channels, upscale_factor):
    inputs = keras.Input(shape=(input_width, input_height, channels))
    X = Conv2D(128, 2, padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
    X = Conv2D(128, 2, padding='same', activation='relu', kernel_initializer='he_normal')(X)
    X = Conv2D(64, 2, padding='same', activation='relu', kernel_initializer='he_normal')(X)
    X = Conv2D(32, 2, padding='same', activation='relu', kernel_initializer='he_normal')(X)
    X = Conv2D(16, 2, padding='same', activation='relu', kernel_initializer='he_normal')(X)
    X = Conv2D(32, 2, padding='same', activation='relu', kernel_initializer='he_normal')(X)
    X = Conv2D(64, 2, padding='same', activation='relu', kernel_initializer='he_normal')(X)
    X = Conv2D(128, 2, padding='same', activation='relu', kernel_initializer='he_normal')(X)
    X = Conv2D(128, 1, padding='same', activation='relu', kernel_initializer='he_normal')(X)
    X = Conv2D(channels * (upscale_factor ** 2), 3, padding='same', activation='relu', kernel_initializer='Orthogonal')(
        X)

    outputs = tf.nn.depth_to_space(X, upscale_factor)
    return keras.Model(inputs, outputs)


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
model = Model(channels, upscale_factor)

model.compile(optimizer='adam', loss='MSE')

model.summary()
model.fit(train_set, epochs=20, callbacks=[early_stopping], validation_data=test_set, verbose=1)


parent_directory = os.path.dirname(os.getcwd())
os.chdir(parent_directory)
os.chdir('models')
model.save('new_model.h5')