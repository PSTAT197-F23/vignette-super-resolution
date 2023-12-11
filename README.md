# PSTAT 197A FINAL PROJECT - ESPCNN (ISR)
Vignette on constructing an Efficient Sub-Pixel Convolutional Neural Network in Python for Image Super Resolution

**Contributors**: Jinran Jin, Yijiao Wang, Peng Zhao, Puyuan Zhang, Sichen Zhong

## Abstract

Our objective for the final project is to generate a high-resolution image from the original low-resolution image. To accomplish this, we selected the Efficient Sub-Pixel Convolutional Neural Network model, efficient as it reduces the network size by upsampling the image at the very last step. Our dataset from Kaggle contains aroud 4000 high resolution images of wild animal faces. Our Pytorch implementation mainly includes preprocessing the dataset, defining the ESPCNN model, and visualizing the training process.

## Repository Contents

```
├─data
│  ├─loss
│  │  └─[reference image]
│  ├─train
│  │  └─[training data]
│  └─val
│     └─[testing data]
├─image
│   ├─loss.jpg
│   ├─model_explanation.png
│   └─reference.jpg
├─model
│  └─model.pt
├─scripts
│  ├─dataset.py
│  ├─main.py
│  ├─model.py
│  └─drafts
└─vignette-super-resolution.ipynb
```

The detailed report is in the notebook `vignette-super-resolution.ipynb`.

To reproduce the result (change model architecture, change hyperparameters, use your image as reference, etc), update `model.py`, `main.py`, or `loss` folder as needed, and then simply run `main.py`. The generated high resolution reference image at the end of each epoch and the loss history will be saved in the `image` folder.


## References
The original ESPCN paper: https://arxiv.org/pdf/1609.05158.pdf

A survey of image super resolution models: https://arxiv.org/pdf/1904.07523.pdf

Pytorch implementations of the paper:
 - https://github.com/pytorch/examples/tree/main/super_resolution
 - https://github.com/Lornatang/ESPCN-PyTorch

Dataset: https://www.kaggle.com/datasets/dimensi0n/afhq-512?select=wild

