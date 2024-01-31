import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from matplotlib_scalebar.scalebar import ScaleBar
from multi_match.colorify import multichannel_to_rgb
import os 

import multi_match

min_slider_value = 0
max_slider_value = 10
max_dist_range = np.linspace(min_slider_value, max_slider_value, 30)

# First we read the example image

file_A = os.path.join('example_data', 'channel_A.tif')
file_B = os.path.join('example_data', 'channel_B.tif')
file_C = os.path.join('example_data', 'channel_C.tif')

image_A = io.imread(file_A)
image_B = io.imread(file_B)
image_C = io.imread(file_C)

# We now can perform a point detection
x = multi_match.point_detection(image_A)
y = multi_match.point_detection(image_B)
z = multi_match.point_detection(image_C)
mmr = multi_match.Multi_Matching_Over_Range([x, y, z], max_dist_range)

# number of chains detected for each max_dist value
range_match = multi_match.Multi_Matching_Over_Range([x, y, z], max_dist_range)
W = range_match.count_objects()
print(W)
range_match.plot_number_of_objects()

### We now plot the matching with a slider for the max distance
cmaps = ['pure_red','pure_green', 'pure_blue']
fig, ax = plt.subplots()
background_image, _, __ = multichannel_to_rgb(images=[image_A, image_B, image_C], cmaps=cmaps)
ax.imshow(background_image)
ax.axis("off")
scalebar = ScaleBar(0.04,
        "nm",
        length_fraction=0.10,
        box_color="black",
        color="white",
        location="lower right")
scalebar.dx = 25
ax.add_artist(scalebar)
background_image, _, __ = multichannel_to_rgb(images=np.stack([image_A, image_B, image_C]), cmaps=cmaps)

colors = ["tab:red", "palegreen", "cyan"]
mmr.plot_match(channel_colors=colors,
        circle_alpha=0.7, circle_color="white", scatter_size=4,
        scatter_edgecolors="white", segment_color="white")
