
### This example requires the additonal package napari to run. 
### You can then execute it with the command "python3 -i napari_visualize_data.py".
import multi_match
import napari
import numpy as np
from skimage import io

viewer = napari.Viewer()

# x_int = [836, 1153]
# y_int = [341, 836]
# First we read some example images
image_A = io.imread('example_data/STAR RED_STED {7}.tif') # [x_int[0]: x_int[1], y_int[0]: y_int[1]]
image_B = io.imread('example_data/Alexa 488_STED {12}.tif') # [x_int[0]: x_int[1], y_int[0]: y_int[1]]
image_C = io.imread('example_data/Alexa 594_STED {7}.tif') # [x_int[0]: x_int[1], y_int[0]: y_int[1]]
new_layer = viewer.add_image(image_A, colormap='magenta', blending="additive")
new_layer = viewer.add_image(image_B, colormap='green', blending="additive")
new_layer = viewer.add_image(image_C, colormap='yellow', blending="additive")

# We now can perform a point detection
x = multi_match.point_detection(image_A.T)
y = multi_match.point_detection(image_B.T)
z = multi_match.point_detection(image_C.T)

# And find the matching within a certain distance
maxdist = 8.5 # in pixel size
match = multi_match.Multi_Matching([x, y, z], maxdist)
l = match._get_links()
l = np.array(l)
pos = np.empty_like(l)
pos[:, 0, :] = l[:, 0, :]
pos[:, 1, :] = l[:, 1, :] - pos[:, 0, :]
# pos = np.transpose(pos, axes=[0, 2, 1])
vect = viewer.add_vectors(pos, blending="opaque", edge_color="white")
points_layer = viewer.add_points(x, size=2, face_color="violet")
points_layer = viewer.add_points(y, size=2, face_color="green")
points_layer = viewer.add_points(z, size=2, face_color="yellow")



