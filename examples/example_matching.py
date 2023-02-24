import matplotlib.pyplot as plt
from skimage import io
from matplotlib_scalebar.scalebar import ScaleBar
from multi_match.colorify import multichannel_to_rgb
import multi_match

# First we read some example images
image_A = io.imread('example_data/STAR RED_STED {7}.tif')
image_B = io.imread('example_data/Alexa 488_STED {12}.tif')
image_C = io.imread('example_data/Alexa 594_STED {7}.tif')

# We now can perform a point detection
x = multi_match.point_detection(image_A)
y = multi_match.point_detection(image_B)
z = multi_match.point_detection(image_C)

# And find the matching within a certain distance
maxdist = 3.5 # in pixel size
match = multi_match.Multi_Matching([x, y, z], maxdist)

# And count the number of different objects in the image:
num_obj = match.count_objects()
for key, value in num_obj.items():
    print(key, ' : ', value)

# the matches can be plotted
# match.plot_match(channel_colors=["tab:red", "palegreen", "cyan"], circle_alpha=0.5)
match.plot_circles(color="white")

# And the multicolor image can be plotted in the background
ax = plt.gca()

background_image, _, __ = multichannel_to_rgb(images=[image_A, image_B, image_C], cmaps=['pure_red','pure_green', 'pure_blue'])
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
# we can chose to focus only on a section of the image by setting the limits 
# plt.xlim([70, 170])
# plt.ylim([70, 170])

plt.tight_layout()
plt.show()

