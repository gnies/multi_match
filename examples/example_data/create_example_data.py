#### This script generates a simple simulated three channel microscopy image with chain-like structures
###  We use this image to illustrate some basic functionallities of this package

import numpy as np
from multi_match.colorify import multichannel_to_rgb
import matplotlib.pyplot as plt

np.random.seed(0)
img_size = 220
shape = (img_size, img_size)  # shape of the simulated image

##### simulate coordinate positions of some chain-like structures
n = 70
centers = img_size*np.random.random(size=(n, 2)) # n random positions
angles = 2*np.pi*np.random.random(size=n)        # n random angles
length_of_rods = 7
angle_xy = np.array([[np.sin(angle), np.cos(angle)] for angle in angles])

# we simulate chains that do not fully allign by perturbating the angle
angle_noise = 0.1        
angle_yz = angle_xy + angle_noise*np.random.normal(size=(n, 2))
x_c = centers - 0.5*length_of_rods*angle_xy 
y_c = centers
z_c = centers + 0.5*length_of_rods*angle_yz 

# simulte marker positions by perturbating the positions as well
marker_std = 1
x = x_c + marker_std * np.random.normal(size=(n, 2))
y = y_c + marker_std * np.random.normal(size=(n, 2))
z = z_c + marker_std * np.random.normal(size=(n, 2))

# create a dataset of mixed type of chains by removing some centers 
x = x[0:29]
z = z[20:49]

# remove some of the points due to simulated staining efficiency
s = 0.9
x = x[np.random.choice([True, False], size=len(x), p=[s, 1-s])]
y = y[np.random.choice([True, False], size=len(y), p=[s, 1-s])]
z = z[np.random.choice([True, False], size=len(z), p=[s, 1-s])]

#### simulate images

# this works only with an extra package
from sdt.sim.fluo_image import simulate_gauss
from tifffile import imwrite

# points that fall outside of the image are ignored
amplitudes = 10
sigmas = 2

clannel_0 = simulate_gauss(shape, x, amplitudes, sigmas)
channel_0 = np.random.poisson(clannel_0)  # shot noise
channel_0 = np.asarray(channel_0, dtype=float)

clannel_1 = simulate_gauss(shape, y, amplitudes, sigmas)
channel_1 = np.random.poisson(clannel_1)  # shot noise
channel_1 = np.asarray(channel_1, dtype=float)

clannel_2 = simulate_gauss(shape, z, amplitudes, sigmas)
channel_2 = np.random.poisson(clannel_2)  # shot noise
channel_2 = np.asarray(channel_2, dtype=float)

imwrite("channel_A.tif", channel_0)
imwrite("channel_B.tif", channel_1)
imwrite("channel_C.tif", channel_2)

# Just to be sure, the multicolor image can be plotted 
ax = plt.gca()
overlay_image, _, __ = multichannel_to_rgb(images=[channel_0, channel_1, channel_2],
        cmaps=['pure_red','pure_green', 'pure_blue'])
ax.imshow(overlay_image)
ax.axis("off")
plt.show()

