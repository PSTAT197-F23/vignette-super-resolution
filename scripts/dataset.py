
from os import listdir
from os.path import join

from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.io import read_image


# import dataset.py and model.py
from dataset import *
from model import *


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    img = Image.open(filepath)
    return img

class DatasetFromFolder(Dataset):
    def __init__(self, image_dir, scale_factor):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.tensor = transforms.ToTensor()
        self.scale_factor = scale_factor

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
    
        input = self.tensor(input)
        target = self.tensor(target)
        
        height, width = transforms.functional.get_image_size(input)
        resize = transforms.Resize((int(height/self.scale_factor), int(width/self.scale_factor)), 
                                  transforms.InterpolationMode.BICUBIC, 
                                  antialias=True
                                 )
        input = resize(input)
        del(resize)
        
        return input, target
    
    def __len__(self):
        return len(self.image_filenames)
