import numpy as np
from skimage import io
from skimage.color import rgb2gray, rgba2rgb
import multi_match
import numpy as np

# First we read some example images
image_A = io.imread('example_data/STAR RED_STED {7}.tif')
image_B = io.imread('example_data/Alexa 488_STED {12}.tif')
image_C = io.imread('example_data/Alexa 594_STED {7}.tif')

# We now can perform a point detection
x = multi_match.point_detection(image_A)
y = multi_match.point_detection(image_B)
z = multi_match.point_detection(image_C)

# set a range of maximal matching distance values
maxdist_range = np.linspace(0, 10, num=31)

# match ponts for each value
range_match = multi_match.Multi_Matching_Over_Range([x, y, z], maxdist_range)
range_match.plot_number_of_objects()
W = range_match.count_objects()
print(W)


# some more functions to visulaize the match
range_match.plot_relative_abundances()
range_match.cumolative_plot_number_of_matches()


