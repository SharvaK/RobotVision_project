# -*- coding: utf-8 -*-
"""
@author: Sharva Khandagale
@author: Manas Dixit

EE 5271 Robot Vision Course project
2D to 3D reconstruction using single image
"""

from Train import model_save_path
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import GLPNForDepthEstimation, GLPNImageProcessor
from PIL import Image
import open3d as o3d
import numpy as np

image_path = "E:/ee5271/bird1.jpg"
model = GLPNForDepthEstimation.from_pretrained('glpn-large').cuda()
model.load_state_dict(torch.load(model_save_path))
model.eval()


processor = GLPNImageProcessor()
image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt").to(torch.device("cuda"))

with torch.no_grad():
    depth_map = model(**inputs).predicted_depth.squeeze().cpu().numpy()

rgb_image = np.asarray(image)

h, w = depth_map.shape
fx, fy = 300.0, 300.0  # Focal lengths
cx, cy = w / 2.0, h / 2.0

points = []
colors = []
for v in range(h):
    for u in range(w):
        z = depth_map[v, u]
        if z > 0:
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append((x, y, z))
            colors.append(rgb_image[v, u] / 255.0)

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
point_cloud.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([point_cloud])


