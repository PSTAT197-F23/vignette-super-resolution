import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os
from PIL import Image
from torch.utils.data import Dataset

class ESPCN(nn.Module):
    def __init__(self, scale_factor):
        super(ESPCN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 3 * scale_factor ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pixel_shuffle(x)
        return x

class ImageDataset(Dataset):
    def __init__(self, hr_folder, lr_folder, transform=None):
        self.hr_folder = hr_folder
        self.lr_folder = lr_folder
        self.transform = transform
        self.hr_images = sorted(os.listdir(hr_folder))
        self.lr_images = sorted(os.listdir(lr_folder))

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_image_path = os.path.join(self.hr_folder, self.hr_images[idx])
        lr_image_path = os.path.join(self.lr_folder, self.lr_images[idx])

        hr_image = Image.open(hr_image_path).convert('RGB')
        lr_image = Image.open(lr_image_path).convert('RGB')

        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        return hr_image, lr_image

# Example usage
transform = transforms.Compose([transforms.ToTensor()])
hr_folder = r'C:\Users\Lenovo\vignette-super-resolution\scripts\drafts\Jinran\wild_1'
lr_folder = r'C:\Users\Lenovo\vignette-super-resolution\scripts\drafts\Jinran\wild_128'

scale_factor = 4
lr = 0.001
batch_size = 64
epochs = 10

# Initialize model and optimizer
model = ESPCN(scale_factor=scale_factor)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

train_dataset = ImageDataset(hr_folder, lr_folder, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    for batch_idx, (hr_image, lr_image) in enumerate(train_loader):
        optimizer.zero_grad()

        # Forward pass
        sr_image = model(lr_image)

        # Compute loss
        loss = criterion(sr_image, hr_image)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Print training progress
        if batch_idx % 10 == 0:
            print(f"Epoch: {epoch+1}/{epochs}, Batch: {batch_idx+1}/{len(train_loader)}, Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), 'espcn_model.pth')
