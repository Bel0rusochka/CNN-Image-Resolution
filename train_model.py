from keras.utils import image_dataset_from_directory
import tensorflow as tf
import os, keras
from keras.layers import Conv2D, Add, Input, Conv2DTranspose, PReLU
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
    inputs = Input(shape=(input_width, input_height, channels))

    # Feature Extraction
    model = Conv2D(112, (5, 5), padding='same', kernel_initializer='he_normal')(inputs)
    model = PReLU()(model)

    model = Conv2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(model)
    model = PReLU()(model)

    model = Conv2D(24, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = PReLU()(model)
    recular = model
    model = Conv2D(24, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = PReLU()(model)
    model = Conv2D(24, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = PReLU()(model)
    model = Conv2D(24, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = PReLU()(model)
    model = Add()([model, recular])
    model = Conv2D(112, (1, 1), padding='same', kernel_initializer='he_normal')(model)
    model = PReLU()(model)

    outputs = Conv2DTranspose(3, (11, 11), strides=(2, 2), padding='same', activation='sigmoid')(model)

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
    batch_size=16,
    validation_split=0.2,
    subset="training",
    seed=123,
    label_mode=None
)

validation_set = image_dataset_from_directory(
    path,
    image_size=(cropped_width, cropped_height),
    batch_size=16,
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
model.fit(train_set, epochs=40, callbacks=[early_stopping], validation_data=test_set, verbose=1)

parent_directory = os.path.dirname(os.getcwd())
os.chdir(parent_directory)
os.chdir('models')
model.save('new_model.h5')
