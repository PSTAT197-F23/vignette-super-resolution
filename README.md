# PSTAT 197A FINAL PROJECT - ESPCNN (ISR)
Vignette on constructing an Efficient Sub-Pixel Convolutional Neural Network in Python for Image Super Resolution

**Contributors**: Jinran Jin, Yijiao Wang, Peng Zhao, Puyuan Zhang, Sichen Zhong

## Abstract
a brief description in a few sentences of your vignette topic, example data, and outcomes.

## Repository Contents
**Image Re-sizing** 

In order to launch our project effectively, we meticulously chose a substantial image dataset to thoroughly evaluate our model's performance by comparing the before and after images upon input. Our image dataset consists of 1,253 low-resolution images of wild animals, each with dimensions of 3x512x512 (3 RGB color channel represents colored images instead of grey-scaled images, with 512x512 pixel length and width). We then pre-processed the images by resizing them into 3x128x128 images (found in main.py). 

**Loading Data**

Following preprocessing, we load the data into tensor objects and proceed to extract both training and testing datasets from these objects.

**Model**

We create a CNN model...

## References
The original ESPCN paper: https://arxiv.org/pdf/1609.05158.pdf
Another amazing ESPCN paper: https://arxiv.org/pdf/1904.07523.pdf

Pytorch implementations of the paper:
 - https://github.com/pytorch/examples/tree/main/super_resolution
 - https://github.com/Lornatang/ESPCN-PyTorch

## Re-evaluation
instructions on use and instructions on contributing to the repository
