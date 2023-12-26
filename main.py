import PIL
from PIL import Image, ImageTk
from tkinter import filedialog
import threading
import tkinter as tk
import os
import time
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


class ImageUpscalerApp:
    def __init__(self, master):
        self.channels = 3
        self.model_width = 8
        self.model_height = 8
        self.upscale_factor = 2

        self.model = load_model('./src/rgb1_16.h5')
        self.output_folder = "outputs"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)


        self.master = master
        self.master.title("Image Upscaler")

        # Variables
        self.loading_label = None

        # Interface
        self.image_label = tk.Label(self.master, text="Select an image")
        self.image_label.pack(pady=10)

        self.load_button = tk.Button(self.master, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=10)

        self.process_button = tk.Button(self.master, text="Process Image", command=self.process_image,
                                        state=tk.DISABLED)
        self.process_button.pack(pady=10)

        self.canvas = tk.Canvas(self.master, width=480, height=480)
        self.canvas.pack(pady=10)

    def load_image(self):
        self.img_path = filedialog.askopenfilename(title="Select an image", filetypes=[
            ('image files', '.png'),
            ('image files', '.jpg'),
            ('image files', '.jpeg'),
            ('all files', '.*')
        ])

        if self.img_path:
            self.image_label.config(text=f"Selected Image: {self.img_path}")
            self.process_button.config(state=tk.NORMAL)

    def resize_image(self, image):
        self.image_label.config(text="Processing, please wait...")

        width, height = image.size
        if width < self.model_width or height < self.model_height:
            self.image_label.config(text="Image is too small")

            exit(1)

        new_width = self.model_width * (width // self.model_width)
        new_height = self.model_height * (height // self.model_height)
        image = image.resize((new_width, new_height), PIL.Image.LANCZOS)
        return image, new_width, new_height

    def dirivate_image(self, image):
        cropped_images = []
        for i in range(0, image.size[0], self.model_width):
            row_images = []
            for j in range(0, image.size[1], self.model_height):
                row_images.append(image.crop((i, j, i + self.model_width, j + self.model_height)))
            cropped_images.append(row_images)
        return cropped_images

    def predict_image(self, cropped_images):

        predicted_images = []
        for i in range(len(cropped_images)):
            row_predicted_images = []
            for j in range(len(cropped_images[i])):
                img = cropped_images[i][j]
                img = img_to_array(img)
                img = img.astype('float32') / 255.0
                img = np.expand_dims(img, axis=0)
                pred = self.model.predict(img, verbose=0)
                pred = np.squeeze(pred, axis=0)
                pred = (pred * 255).astype('uint8')
                pred = PIL.Image.fromarray(pred, 'RGB')
                row_predicted_images.append(pred)
            predicted_images.append(row_predicted_images)

        return predicted_images

    def concatenate_images(self, predicted_images, new_width, new_height):
        concatenated_image = PIL.Image.new('RGB', (new_width * self.upscale_factor, new_height * self.upscale_factor))
        for i, row in enumerate(predicted_images):
            for j, image in enumerate(row):
                concatenated_image.paste(image, (i * self.model_width * self.upscale_factor, j * self.model_height * self.upscale_factor))
        concatenated_image.save(f"./{self.output_folder}/image{time.time()}.jpg")
        self.image_label.config(text="Image saved")

    def process_image(self):
        image = PIL.Image.open(self.img_path)
        image, new_width, new_height = self.resize_image(image)
        cropped_images = self.dirivate_image(image)
        self.master.after(10, lambda: self.process_image_thread(cropped_images, new_width, new_height))

    def process_image_thread(self, cropped_images, new_width, new_height):
        predicted_images = self.predict_image(cropped_images)
        self.concatenate_images(predicted_images, new_width, new_height)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageUpscalerApp(root)
    root.mainloop()
