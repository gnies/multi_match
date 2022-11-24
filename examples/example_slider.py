import numpy as np
from skimage import io
import multi_match
from multi_match.slider import max_dist_slider
from multi_match.colorify import multichannel_to_rgb

init_radius = 6
min_slider_value = 0
max_slider_value = 15

# First we read some example images
image_A = io.imread('example_data/STAR RED_STED {7}.tif')
image_B = io.imread('example_data/Alexa 488_STED {12}.tif')
image_C = io.imread('example_data/Alexa 594_STED {7}.tif')

# We now can perform a point detection
x = multi_match.point_detection(image_A)
y = multi_match.point_detection(image_B)
z = multi_match.point_detection(image_C)

background_image, _, __ = multichannel_to_rgb(images=np.stack([image_A, image_B, image_C]), cmaps=['pure_red','pure_green', 'pure_blue'])

multi_match.slider.max_dist_slider(x, y, z, init_radius, min_slider_value, max_slider_value, background_image)


