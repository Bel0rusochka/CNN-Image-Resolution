import time
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.models import load_model
import os, PIL
from PIL import Image
import numpy as np


model = load_model('./models/rgb1_16.h5')
channels = 3
model_width = 8
model_height = 8
upscale_factor = 2

img_path = "./test/12.jpg"
output_folder = "outputs"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image = PIL.Image.open(img_path)
width, height = image.size
if width < model_width or height < model_height:
    print("Image is too small")
    exit(1)

# resize image to model size(n*256) * (m*256) where n,m - integer(coefficient)
new_width = model_width * (width // model_width)
new_height = model_height * (height // model_height)
image = image.resize((new_width, new_height), PIL.Image.LANCZOS)

# dirivate image to small images 256x256
cropped_images = []
for i in range(0, image.size[0], model_width):
    row_images = []
    for j in range(0, image.size[1], model_height):
        row_images.append(image.crop((i, j, i + model_width, j + model_height)))
    cropped_images.append(row_images)

# predict small images
predicted_images = []
for i in range(len(cropped_images)):
    row_predicted_images = []
    for j in range(len(cropped_images[i])):
        img = cropped_images[i][j]
        img = img_to_array(img)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img)
        pred = np.squeeze(pred, axis=0)  # Remove the batch dimension
        pred = (pred * 255).astype('uint8')  # Convert back to 8-bit integer values
        pred = Image.fromarray(pred, 'RGB')  # Convert to Image
        row_predicted_images.append(pred)
    predicted_images.append(row_predicted_images)

# Concatenate images
concatenated_image = Image.new('RGB', (new_width * upscale_factor, new_height * upscale_factor))
for i, row in enumerate(predicted_images):
    for j, image in enumerate(row):
        concatenated_image.paste(image, (i * model_width * 2, j * model_height * 2))

concatenated_image.save(f"./outputs/image{time.time()}.jpg")