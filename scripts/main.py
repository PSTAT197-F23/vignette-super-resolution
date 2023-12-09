
from PIL import Image

import numpy as np

import pandas as pd 
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader


# import dataset.py and model.py
from dataset import *
from model import *


# main helper functions
def swap(img):
    img = img.swapaxes(0, 1)
    img = img.swapaxes(1, 2)
    return img

def train(epoch, model):
    epoch_loss = 0
    epoch_loss_history = []
    epoch_test_loss_history = []
    
    for iteration, batch in enumerate(train_dataloader, 1):
        img, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        loss = criterion(model(img), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        if iteration % 5 == 0:
            tbatch = next(iter(test_dataloader))
            timg, ttarget = tbatch[0].to(device), tbatch[1].to(device)
            tloss = criterion(model(timg), ttarget)
            
            epoch_loss_history.append(loss.item())
            epoch_test_loss_history.append(tloss.item())
            
            print("===> Epoch[{}]({}/{}): Loss: {:.6f}, Test Loss: {:.6f}".format(
                epoch+1, iteration, len(train_dataloader), loss.item(), tloss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.6f}".format(epoch+1, epoch_loss / len(train_dataloader)))
    
    return epoch_loss_history, epoch_test_loss_history


# set device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

    
# set hyperparameter
scale_factor    = 4
batch_size      = 32
epoch           = 5
learning_rate   = 0.0003
criterion       = nn.MSELoss()


# load data
train_data = DatasetFromFolder("../data/train/wild", scale_factor=scale_factor)
test_data = DatasetFromFolder("../data/train/wild", scale_factor=scale_factor)
ref_data = DatasetFromFolder("../data/loss", scale_factor=scale_factor)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
ref_dataloader = DataLoader(ref_data, batch_size=1, shuffle=False)


# create model
model = Model(scale_factor=scale_factor).to(device)


# train
figure, ax = plt.subplots(3, epoch)
figure.set_size_inches(20, 15)

loss_history = []
test_loss_history = []

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for i in range(epoch):
    epoch_loss_history, epoch_test_loss_history = train(i, model)
    loss_history = loss_history + epoch_loss_history
    test_loss_history = test_loss_history + epoch_test_loss_history
    del(epoch_loss_history)
    del(epoch_test_loss_history)
    
    ref = next(iter(ref_dataloader))[0]
    ref_tgt = next(iter(ref_dataloader))[1]
    ref_fit = model(ref.to(device)).cpu()
    
    ref = swap(ref.squeeze())
    ref_tgt = swap(ref_tgt.squeeze())
    ref_fit = swap(ref_fit.detach().numpy().squeeze())

    ax[0, i].imshow(ref)
    ax[1, i].imshow(ref_fit)
    ax[2, i].imshow(ref_tgt)

figure.savefig('../image/reference.jpg')


# plot loss
plt.yscale("log")
plt.plot(loss_history)
plt.plot(test_loss_history)
plt.savefig('../image/loss.jpg')


# save model
torch.save(model.state_dict(), '../model/model.pt')



