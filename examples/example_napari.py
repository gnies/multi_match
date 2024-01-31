### This is an example script to show how to use the multi_match package with napari and requires napari to be installed
### You can run this script by executing the command "python3 example_napari.py"

import multi_match
import napari
import numpy as np
from skimage import io
import os

viewer = napari.Viewer()

viewer.scale_bar.visible = True
viewer.scale_bar.unit = "nm"
pixel_per_nm = 0.04
nm_per_pixel = 1/0.04
scale = (nm_per_pixel, nm_per_pixel)

# First we read some example images

file_A = os.path.join('example_data', 'channel_A.tif')
file_B = os.path.join('example_data', 'channel_B.tif')
file_C = os.path.join('example_data', 'channel_C.tif')

image_A = io.imread(file_A)
image_B = io.imread(file_B)
image_C = io.imread(file_C)

new_layer = viewer.add_image(image_A, colormap='magenta', blending="additive", 
        scale=scale)
new_layer = viewer.add_image(image_B, colormap='green', blending="additive", 
        scale=scale)
new_layer = viewer.add_image(image_C, colormap='yellow', blending="additive", 
        scale=scale)

# We now can perform a point detection
x = multi_match.point_detection(image_A.T)
y = multi_match.point_detection(image_B.T)
z = multi_match.point_detection(image_C.T)
x = x*nm_per_pixel
y = y*nm_per_pixel
z = z*nm_per_pixel

# And find the matching within a certain distance
maxdist = 8.5*nm_per_pixel  # in pixel size
match = multi_match.Multi_Matching([x, y, z], maxdist)
l = match.get_all_links()
l = np.array(l)
pos = np.empty_like(l)
pos[:, 0, :] = l[:, 0, :]
pos[:, 1, :] = l[:, 1, :] - pos[:, 0, :]
# pos = np.transpose(pos, axes=[0, 2, 1])
vect = viewer.add_vectors(pos, blending="opaque", edge_color="white", vector_style="line", edge_width=1*nm_per_pixel)
points_layer = viewer.add_points(x, size=2*nm_per_pixel, face_color="violet")
points_layer = viewer.add_points(y, size=2*nm_per_pixel, face_color="green")
points_layer = viewer.add_points(z, size=2*nm_per_pixel, face_color="yellow")

ovals = []
r = maxdist/2
for points in match.point_list:
    for p in points:
        x = p[1]
        y = p[0]
        pos_1 = [y - r, x - r] 
        pos_2 = [y - r, x + r] 
        pos_3 = [y + r, x + r] 
        pos_4 = [y + r, x - r] 
        ovals.append([pos_1, pos_2, pos_3, pos_4])
napari_detections = np.asarray(ovals)
shapes_layer = viewer.add_shapes()
viewer.layers[-1].visible = False
shapes_layer.add(
    napari_detections,
    shape_type='ellipse',
    edge_width=0.1*nm_per_pixel,
    edge_color='white',
    face_color=[0]*4,
)
napari.run()
