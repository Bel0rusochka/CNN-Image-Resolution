from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import numpy as np
import time
import cv2
import random
import os, PIL, keras
from PIL import Image

current_directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_directory, 'image')
os.chdir(path)


def ConvertToRGB(image_path):
    try:

        with Image.open(image_path) as image:
            array = np.array(image)
            channels = array.shape[-1]

            if channels == 4:
                # Keep only the first three channels (RGB) and discard the alpha channel
                rgb_image = array[:, :, :3]
                Image.fromarray(rgb_image).save(image_path)

            elif channels < 3:
                # If the image has less than 3 channels, convert it to RGB
                rgb_image = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
                Image.fromarray(rgb_image).save(image_path)

    except Exception as e:
        print(1)
        os.remove(image_path)


def TransformDataSet():
    for image_file in os.listdir():
        try:
            with Image.open(image_file) as image:

                array = img_to_array(image)
                channels = array.shape[-1]
                width = image.size[0]
                height = image.size[1]

        except PIL.UnidentifiedImageError:
            print('Image is corrupted: ', image_file)
            os.remove(image_file)
            continue
        if channels != 3:
            ConvertToRGB(image_file)
        if width >= 800 or height >= 800:
            resized_image = image.resize((256, 256))
            resized_image.save(image_file)
        if width < 32 or height < 32:
            os.remove(image_file)
            print('Image is deleted: ', image_file)
        else:
            print('Image is ok: ', image_file)


def RenameAndAugmentImages():
    image_files = [image_file for image_file in os.listdir()]
    for index, image_file in enumerate(image_files):
        os.rename(dst=str(index) + '.jpg', src=image_file)
        image_file = str(index) + '.jpg'
        img = keras.preprocessing.image.load_img(image_file)
        img_array = img_to_array(img)
        for i in range(5):
            seed = int(time.time() * 1000) + i
            random.seed(seed)

            augmented_img_array = img_array.copy()

            # Apply random rotation
            augmented_img_array = keras.preprocessing.image.random_rotation(augmented_img_array, 370, row_axis=0,
                                                                            col_axis=1, channel_axis=2,
                                                                            fill_mode='nearest')

            # Apply random shift
            augmented_img_array = keras.preprocessing.image.random_shift(augmented_img_array, 0.3, 0.3, row_axis=0,
                                                                         col_axis=1, channel_axis=2,
                                                                         fill_mode='nearest')

            # Apply random zoom
            augmented_img_array = keras.preprocessing.image.random_zoom(augmented_img_array, (0.7, 1.3), row_axis=0,
                                                                        col_axis=1, channel_axis=2, fill_mode='nearest')

            # Apply random shear
            augmented_img_array = keras.preprocessing.image.random_shear(augmented_img_array, 0.3, row_axis=0,
                                                                         col_axis=1, channel_axis=2,
                                                                         fill_mode='nearest')

            # Apply random brightness
            augmented_img_array = keras.preprocessing.image.random_brightness(augmented_img_array, (0.7, 1.3))

            # Convert NumPy array back to image
            augmented_img = array_to_img(augmented_img_array)

            # Save augmented image
            output_file = f"{index}_aug_{i}.jpg"
            augmented_img.save(output_file)


def DeleteAugmentedFiles(folder_path):
    for file_name in os.listdir():
        if "_aug_" in file_name:
            file_path = os.path.join(folder_path, file_name)
            os.remove(file_path)


TransformDataSet()
RenameAndAugmentImages()
# DeleteAugmentedFiles(path)
