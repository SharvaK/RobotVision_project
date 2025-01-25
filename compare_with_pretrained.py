# -*- coding: utf-8 -*-
"""
@author: Sharva Khandagale 
@author: Manas Dixit 

EE 5271 Robot Vision Course project
2D to 3D reconstruction using single image
"""
#%%
#import gradio
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import numpy as np
import open3d as o3d
#import io
#import base64
#import plotly.graph_objs as go
#import os
#%%

# Use a short temp directory for Gradio cache
#os.environ["GRADIO_TEMP_DIR"] = "E:/ee5271/gradio_cache"
feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

#%%


image = Image.open("E:/ee5271/bird1.jpg")
new_height = 480 if image.height > 480 else image.height 
new_height -= (new_height % 32)

new_width = int(new_height * image.width / image.height)
diff = new_width % 32

new_width = new_width - diff if diff < 16 else new_width + 32 - diff
new_size = (new_width, new_height)
image = image.resize(new_size)

inputs = feature_extractor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth
    
pad = 16
output = predicted_depth.squeeze().cpu().numpy() * 1000.0
output = output[pad: -pad, pad:-pad]
image = image.crop((pad, pad, image.width - pad, image.height - pad))

  #  return image, output
    
#%%    

#Visualize the prediction
fig, ax = plt.subplots(1,2)
ax[0].imshow(image)
ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax[1].imshow(output, cmap='plasma')
ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.tight_layout()
plt.draw()
plt.pause(5)

#%%


#%%

#def generate_point_cloud(image, output):
print(type(image))
width, height = image.size
depth_image = (output *255 / np.max(output)).astype('uint8')

image = np.array(image)

depth_o3d = o3d.geometry.Image(depth_image)
image_o3d = o3d.geometry.Image(image)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity = False)

camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsic.set_intrinsics(width, height, 300, 300 , width/2, height/2)

pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
#o3d.visualization.draw_geometries([pcd_raw])

cl, ind = pcd_raw.remove_statistical_outlier(nb_neighbors=20, std_ratio=10.0)
pcd = pcd_raw.select_by_index(ind)

pcd.estimate_normals()
pcd.orient_normals_to_align_with_direction()

o3d.visualization.draw_geometries([pcd])
# Extract point cloud data for Plotly
#    points = np.asarray(pcd.points)
#    colors = np.asarray(pcd.colors)

#    return points, colors

#%%

#def visualize_point_cloud(points, colors):
#    # Prepare data for Plotly scatter3d
#    x, y, z = points[:, 0], points[:, 1], points[:, 2]
#    r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
#    color = (r * 255, g * 255, b * 255)

#    trace = go.Scatter3d(
#        x=x, y=y, z=z,
#        mode='markers',
#        marker=dict(size=2, color=colors, opacity=0.8),
#    )
#    layout = go.Layout(
#        margin=dict(l=0, r=0, b=0, t=0),
#        scene=dict(aspectmode="data")
#    )
#    fig = go.Figure(data=[trace], layout=layout)
#    return fig

#%%

#def process_and_visualize(image):
#    color_image, depth_map = process_image(image)
    
    # Generate depth map visualization
#    plt.imshow(depth_map, cmap="plasma")
#   plt.axis("off")
#    buf = io.BytesIO()
#    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
#    buf.seek(0)
#    depth_map_img = base64.b64encode(buf.read()).decode("utf-8")
#    buf.close()
    
    # Generate point cloud and create Plotly figure
#    points, colors = generate_point_cloud(color_image, depth_map)
#    point_cloud_fig = visualize_point_cloud(points, colors)
    
#    return f"data:image/png;base64,{depth_map_img}", point_cloud_fig

#%%

# Gradio interface
#iface = gradio.Interface(
#    fn=process_and_visualize,
#    inputs=gradio.Image(type="pil"),
#    outputs=[gradio.Image(label="Depth Map"), gradio.Plot(label="Interactive Point Cloud")],
#    title="Depth Map and Interactive Point Cloud Generator"
#)

#iface.launch()

#%%

#mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10, n_threads = 1)[0]

#rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
#mesh.rotate(rotation, center=(0,0,0))

#o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

#mesh_uniform = mesh.paint_uniform_colour([0.9, 0.8, 0.9])
#mesh_uniform.compute_vertex_normals()
#o3d.visualization.draw_geometries([mesh_uniform], mesh_show_back_face=True)

                                             
#%%

#o3d.io.write_triangle_mesh("E:/ee5271/plane.ply",mesh)




