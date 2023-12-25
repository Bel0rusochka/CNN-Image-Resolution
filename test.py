from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.models import load_model
import os, PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# os.chdir("")
model = load_model('./models/new_model.h5')
channels = 3
cropped_width = 256
cropped_height = 256
upscale_factor = 2
input_width = cropped_width // upscale_factor
input_height = cropped_height // upscale_factor

test_images = sorted([os.path.join('test', img_name) for img_name in os.listdir('test')])

for img_path in test_images:
    HR_image = PIL.Image.open(img_path)
    HR_image = HR_image.resize((256, 256), PIL.Image.BICUBIC)

    LR_image = HR_image.resize((HR_image.size[0] // upscale_factor,
                                HR_image.size[1] // upscale_factor),
                               PIL.Image.BICUBIC)

    # Convert to RGB
    rgb = LR_image.convert("RGB")
    rgb = img_to_array(rgb)
    rgb = rgb.astype("float32") / 255.0

    input = rgb.reshape(1, rgb.shape[0], rgb.shape[1], rgb.shape[2])
    output = model.predict(input)

    output = output[0]
    output *= 255.0
    output = output.clip(0, 255)
    output = PIL.Image.fromarray(np.uint8(output))
    output = output.resize(HR_image.size, PIL.Image.Resampling.NEAREST)

    # Plot the images
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(np.uint8(LR_image))
    plt.title('Low Resolution (Stretched)')

    plt.subplot(1, 3, 2)
    plt.imshow(np.uint8(output))
    plt.title('Enhanced Resolution')

    plt.subplot(1, 3, 3)
    plt.imshow(np.uint8(HR_image))
    plt.title('High Resolution (Original Quality)')

    plt.show()
