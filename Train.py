# -*- coding: utf-8 -*-
"""
@author: Sharva Khandagale 
@author: Manas Dixit 

EE 5271 Robot Vision Course project
2D to 3D reconstruction using single image
"""
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import GLPNForDepthEstimation
from PIL import Image

class KITTIDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.images = [os.path.join(root, "images", f) for f in os.listdir(os.path.join(root, "images"))]
        self.targets = [os.path.join(root, "depths", f) for f in os.listdir(os.path.join(root, "images"))]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        target = Image.open(self.targets[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        return image, target

data_root = "E:/ee5271/data_depth_annotation/"
model_save_path = "E:/ee5271/models/glpn_model.pth"
batch_size = 16
learning_rate = 0.001
epochs = 300

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

data_loader = DataLoader(KITTIDataset(root=data_root, transform=transform), batch_size=batch_size, shuffle=True)

model = GLPNForDepthEstimation.from_pretrained('glpn-large').cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

def train():
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, depths in data_loader:
            images, depths = images.cuda(), depths.cuda()

            optimizer.zero_grad()
            outputs = model(images).predicted_depth
            loss = criterion(outputs, depths)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(data_loader):.4f}")
        torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":
    train()
