# Introduction
The main idea of this project is to learn how to work with convolutional neural networks. 
By learning to work with CNN I mean working with a dataset(its creation, processing and use for model learning), 
studying different architectures of neural networks, learning the TensorFlow library, specifically the Keras interface, and finally creating a application for improving the resolution of photos based on the received model. 
All the sources that I have used for learning can be find in the References section. 
These sources were either theoretical or practical sources for understanding the basics of Keras.

# Architecture of the neural network(FSRCNN)
The architecture for my model based on FSRCNN. 
I came to the conclusion that this architecture shows very good results and is not demanding on hardware. 
I had to modify this architecture and add some changes, although even with such changes the training was fast enough. 

I would like to say that the final model architecture is the result of a large number of experiments.
## Explanation of the using architecture
![model](src/model.png)

# Dataset
At first, I created a script download_image.py to quickly create a large dataset, but I came to the conclusion that it would be better to use a ready-made variant, because my internet speed is not enough.

I downloaded a dataset of 5500 photos from the [ImagesNet](https://www.image-net.org/challenges/LSVRC/2017/2017-downloads.php). 
Then I created a script that transforms all photos to the same format and checks the photos to see if they can be opened. 
Based on the given photos, I then perform photo agmentation (from one photo I make 5 pieces including the original one), so I have used 23000 images to train the model.

# Results
I have trained the model for 40 epochs and the results are shown below.
## Origin image
![result](src/origin.png)

## Resolution 2x
![result](src/ResolutionX2.jpg)

## Resolution 4x
![result](src/ResolutionX4.jpg)

# Using libraries
* [TensorFlow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)
* [OpenCV](https://opencv.org/)
* [Numpy](https://numpy.org/)
* [Pillow](https://pillow.readthedocs.io/en/stable/)

# References
1. [Fast and Accurate Image Super Resolution by Deep CNN
with Skip Connection and Network in Network](https://arxiv.org/ftp/arxiv/papers/1707/1707.05425.pdf)
2. [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/pdf/1501.00092.pdf)
3. [Images-Resolution-Enhancement-Usiung-CNN](https://github.com/ahmadsallakh/Images-Resolution-Enhancement-Usiung-CNN)
4. [Image Super-Resolution Using Deep Convolutional Networks in Tensorflow](https://github.com/tegg89/SRCNN-Tensorflow)