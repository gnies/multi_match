import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import multi_match
import numpy as np

s_A, s_B, s_C = 0.9, 0.9, 0.9 # setting some staining efficinecies

# First we read some example images
image_A = io.imread('example_data/STAR RED_STED {7}.tif')
image_B = io.imread('example_data/Alexa 488_STED {12}.tif')
image_C = io.imread('example_data/Alexa 594_STED {7}.tif')

# We now can perform a point detection
x = multi_match.point_detection(image_A)
y = multi_match.point_detection(image_B)
z = multi_match.point_detection(image_C)

# find the matching within a certain distance  
maxdist = 7 # in pixel size
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
maxdist_range = np.linspace(0, 7, num=15)
range_match = multi_match.Multi_Matching_Over_Range([x, y, z], maxdist_range)
W = range_match.count_objects()
print(W)
N = range_match.estimate_abundances([s_A, s_B, s_C])
print(N)
# range_match.plot_relative_abundances_estimation(s_A, s_B, s_C)
# range_match.plot_relative_abundances()

