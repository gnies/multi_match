import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import multi_match
import numpy as np
import os

s_A, s_B, s_C = 0.9, 0.9, 0.9 # setting some staining efficinecies

# First we read some example images
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

# find the matching within a certain distance  
maxdist = 8.5 # in pixel size

match = multi_match.Multi_Matching([x, y, z], maxdist)
# count the number of different objects in the image:
W = match.count_objects()
# compute abundances
N = match.estimate_abundances([s_A, s_B, s_C])
print(W)
print(N)
# alpha = 0.05
# this gives the ellipsoid matrix and the 1-alpha quantile of the chi-squared distribution with 6 degrees of freedom
# ellipsoid = match.compute_confidence_ellipsiod(s_A, s_B, s_C, alpha)
# # print(ellipsoid)
# 
# We can also compute the abbundances for multiple maxdist values
# set a range of maximal matching distance values

# match ponts for each value
maxdist_range = np.linspace(0, 10, num=15)
range_match = multi_match.Multi_Matching_Over_Range([x, y, z], maxdist_range)
W = range_match.count_objects()
print(W)
N = range_match.estimate_abundances([s_A, s_B, s_C])
print(N)
range_match.plot_number_of_objects()
range_match.plot_estimated_number_of_objects([s_A, s_B, s_C])

