# FUNCTIONS

# Import libraries
from os import listdir
from os.path import join
from PIL import Image
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.io import read_image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    img = Image.open(filepath)
    return img
    
def swap(img):
    img = img.swapaxes(0, 1)
    img = img.swapaxes(1, 2)
    return img

class DatasetFromFolder(Dataset):
    def __init__(self, image_dir, scale_factor=4):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.tensor = transforms.ToTensor()

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
    
        input = self.tensor(input)
        target = self.tensor(target)
        
        height, width = transforms.functional.get_image_size(input)
        resize = transforms.Resize((int(height/scale_factor), int(width/scale_factor)), 
                                  transforms.InterpolationMode.BICUBIC, 
                                  antialias=True
                                 )
        input = resize(input)
        del(resize)
        
        return input, target
    
    def __len__(self):
        return len(self.image_filenames)

class Model(nn.Module):
    def __init__(self, scale_factor):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 3 * scale_factor ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.clamp(x, 0.0, 1.0)
        x = self.pixel_shuffle(x)
        return x


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
    