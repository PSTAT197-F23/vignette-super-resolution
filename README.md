# PSTAT 197A FINAL PROJECT - ESPCNN (ISR)
Vignette on constructing an Efficient Sub-Pixel Convolutional Neural Network in Python for Image Super Resolution

**Contributors**: Jinran Jin, Yijiao Wang, Peng Zhao, Puyuan Zhang, Sichen Zhong

## Abstract
a brief description in a few sentences of your vignette topic, example data, and outcomes.
Our goal is to construct a model which that can reconstruct and de-blur images. The ESPCNN model, a type of ISR model, designed for image restoration, aiming to recover a high-resolution (HR) image from its corresponding low-resolution (LR) counterpart. Our dataset contains 4,739 images of wild animals, each with Each with dimensions of 3x512x512 (3 RGB color channel represents colored images instead of grey-scaled images, with 512x512 pixel length and width). We can find that the model performs well by comparing the input images and the final output images as shown below.
![image text](https://github.com/PSTAT197-F23/vignette-super-resolution/blob/main/image/reference.jpg)

Also, by the image of the visualization of the loss, we can find a decreasing trend. ### insert loss image

## Repository Contents
**Image Re-sizing** 

In order to launch our project effectively, we meticulously chose a substantial image dataset to thoroughly evaluate our model's performance by comparing the before and after images upon input. We pre-processed the images by resizing them into 3x128x128 images (found in main.py). 

**Loading Data**

Following preprocessing, we load the data into tensor objects and proceed to extract both training and testing datasets from these objects.

**Model**

Our model is a convolutional neural network (CNN) designed for image super-resolution. It consists of three convolutional layers (conv1, conv2, conv3) followed by a pixel shuffle operation (PixelShuffle). The model takes a low-resolution image (x) as input and outputs a high-resolution image. The convolutional layers learn hierarchical features, and the pixel shuffle operation is used for upscaling. The ReLU activation is applied after each convolutional layer, and the final output is clamped to ensure pixel values are within the valid range [0, 1]. The scale factor is a hyperparameter that controls the level of upscaling.

## References
The original ESPCN paper: https://arxiv.org/pdf/1609.05158.pdf

Another amazing ESPCN paper: https://arxiv.org/pdf/1904.07523.pdf

Pytorch implementations of the paper:
 - https://github.com/pytorch/examples/tree/main/super_resolution
 - https://github.com/Lornatang/ESPCN-PyTorch

## Re-evaluation
instructions on use and instructions on contributing to the repository
