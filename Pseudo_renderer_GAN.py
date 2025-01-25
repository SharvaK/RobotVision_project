import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()

        def down_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        def up_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, dropout=False):
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)]
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        # Define encoder
        self.encoder = nn.ModuleList([
            down_block(1, 64, batch_norm=False),  # Input: 1 -> 64
            down_block(64, 128),                 # 64 -> 128
            down_block(128, 256),                # 128 -> 256
            down_block(256, 256),                # 256 -> 512
            down_block(256, 256),                # 512 -> 512
            down_block(256, 256),                # 512 -> 512
                             # 512 -> 512
        ])

        # Define decoder
        self.decoder = nn.ModuleList([
            up_block(256, 256, dropout=True),        
            up_block(256, 256, dropout=True),        
            up_block(256, 256, dropout=True),                      
            up_block(256, 128),
            up_block(128, 64),  
            up_block(64, 1)                     
        ])


        #self.final = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)  # Output: 64 -> 1

    def forward(self, x):
        skips = []
        for down in self.encoder:
            x = down(x)
            skips.append(x)
            print(f"Encoder Output Shape: {x.shape}")

        skips = skips[:-1][::-1]  # Reverse all skips except the last one

        for i,up in enumerate(self.decoder):
            print(f"Decoder Block {i}: Input Shape Before Concatenation = {x.shape}")
            if i < len(skips):
                #try :
                x = F.interpolate(x, size=skips[i].shape[2:], mode='nearest')  # Resize to match skip connection
                #except RuntimeError as e:
                #    print(f"Interpolation failed at Decoder Block {i}: {e}")
                #    pass

                print(f"Skip Connection {i}: Shape = {skips[i].shape}")
                #x = torch.cat([x, skips[i]], dim=1)
                #print(f"After Concatenation: Shape = {x.shape}")
                #x = reduction(x)  # Dynamically reduce channels
                #print(f"After Channel Reduction: Shape = {x.shape}")

                #if x.shape[2:] == skips[i].shape[2:]:  # Ensure shapes match before concatenation
                #    x = torch.cat([x, skips[i]], dim=1)
                #    print(f"After Concatenation: Shape = {x.shape}")
                #else:
                #    print(f"Shape mismatch: x shape {x.shape}, skip shape {skips[i].shape}. Skipping concatenation.")
                #    continue

            x = up(x)
            #x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

            print(f"Decoder Block {i}: Output Shape = {x.shape}")

        #return self.final(x)
        return x

class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # Depth map + RGB image as input
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )
        

    def forward(self, x):
        x = self.model(x)  # Pass through convolutional layers

        return x
    
adversarial_loss = nn.BCEWithLogitsLoss()  # For discriminator and generator adversarial loss
reconstruction_loss = nn.L1Loss()          # For generator depth reconstruction

class KITTIDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        """
        Args:
            root_dir (str): Path to the root of the dataset.
            mode (str): Either 'train' or 'val'.
            transform (callable, optional): Transform to be applied on the data.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.samples = self._collect_samples()

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {self.mode} dataset at {root_dir}")

    def _collect_samples(self):
        mode_dir = os.path.abspath(os.path.join(self.root_dir, self.mode))
        samples = []

        for date_dir in os.listdir(mode_dir):
            groundtruth_dir = os.path.join(
                mode_dir, date_dir, "proj_depth", "groundtruth", "image_02"
            )
            if not os.path.exists(groundtruth_dir):
                print(f"Groundtruth directory not found: {groundtruth_dir}")
                continue

            for file_name in os.listdir(groundtruth_dir):
                if not file_name.endswith(".png"):
                    continue
                groundtruth_path = os.path.join(groundtruth_dir, file_name)
                samples.append(groundtruth_path)

        print(f"Collected {len(samples)} samples for {self.mode} mode.")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        groundtruth_path = self.samples[idx]

        # Load the groundtruth depth map
        groundtruth_depth = Image.open(groundtruth_path)

        groundtruth_depth = np.array(groundtruth_depth, dtype=np.float32)  # Convert to float32
        groundtruth_depth /= 256.0  # Normalize 16-bit depth (KITTI depth is in millimeters)

        groundtruth_depth = Image.fromarray(groundtruth_depth)

        if self.transform:
            groundtruth_depth = self.transform(groundtruth_depth) # Add channel dimension
            groundtruth_depth = torch.tensor(np.array(groundtruth_depth), dtype=torch.float32).unsqueeze(0)

        return groundtruth_depth

# Paths to the dataset directories
TRAIN_DIR = "/home/sharva/ee5271/data_depth_annotated/"
VAL_DIR = "/home/sharva/ee5271/data_depth_annotated/"

# Define transforms
transform = Compose([
    Resize((256, 256)),  # Resize to consistent resolution
])

# Load datasets
train_dataset = KITTIDataset(TRAIN_DIR, mode="train", transform=transform)
val_dataset = KITTIDataset(VAL_DIR, mode="val", transform=transform)

# Debug number of samples
print(f"Number of samples in train dataset: {len(train_dataset)}")
print(f"Number of samples in val dataset: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

# Initialize models
generator = UNetGenerator().cuda()
discriminator = PatchDiscriminator().cuda()

# Optimizers and loss functions
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
adversarial_loss = nn.BCEWithLogitsLoss()
reconstruction_loss = nn.L1Loss()

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    generator.train()
    discriminator.train()

    for i, groundtruth_depth in enumerate(train_loader):
        groundtruth_depth = groundtruth_depth.cuda()

        # Simulate sparse depth maps (e.g., random dropout of p/home/sharva/.local/lib/python3.12/site-packages/torch/nn/modules/loss.py:128: UserWarning: Using a target size (torch.Size([8, 1, 256, 256])) that is different to the input size (torch.Size([8, 1, 512, 512])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.ixels)
        sparse_depth = groundtruth_depth * torch.bernoulli(torch.full_like(groundtruth_depth, 0.1))  # 10% sparse

        # Ground truth labels for adversarial loss
        valid = torch.ones((groundtruth_depth.size(0), 1)).cuda()
        fake = torch.zeros((groundtruth_depth.size(0), 1)).cuda()

        # Train Generator
        optimizer_G.zero_grad()
        gen_depth = generator(sparse_depth)
        gen_depth_resized = F.interpolate(gen_depth, size=groundtruth_depth.shape[2:], mode='bilinear', align_corners=False)
        g_adv_loss = adversarial_loss(discriminator(gen_depth), valid)
        g_rec_loss = reconstruction_loss(gen_depth_resized, groundtruth_depth)
        g_loss = g_adv_loss + 100 * g_rec_loss
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(groundtruth_depth), valid)
        fake_loss = adversarial_loss(discriminator(gen_depth.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# Save the model
#torch.save(generator.state_dict(), "generator.pth")

#print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# Save models
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")

# Visualize generated depth
import matplotlib.pyplot as plt

gen_dense_depth = generator(sparse_depth).detach().cpu().numpy()
plt.imshow(gen_dense_depth[0, 0], cmap='plasma')  # Visualize first sample
plt.colorbar()
plt.show()
