# MultiMatch

This repository includes the Python package to perform object-based multi-color colocalization analysis on super-resolution microscopy data using tools from optimal transport.

#### Pip installation

One can install the package by navigating to the folder in which compressed `multi_match-0.0.1.tar.gz` is stored and then execute
```console
pip install multi_match-0.0.1.tar.gz
```

#### Short example

```python
import matplotlib.pyplot as plt
from skimage import io
from matplotlib_scalebar.scalebar import ScaleBar
from multi_match.colorify import multichannel_to_rgb
import multi_match

# First we read some example images
image_A = io.imread('examples/example_data/STAR RED_STED {7}.tif')
image_B = io.imread('examples/example_data/Alexa 488_STED {12}.tif')
image_C = io.imread('examples/example_data/Alexa 594_STED {7}.tif')

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
match.plot_match(channel_colors=["tab:red", "palegreen", "cyan"], circle_alpha=0.5)

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
plt.xlim([80, 140])
plt.ylim([80, 140])

plt.tight_layout()
plt.show()
```

w_ABC  :  251

w_AB  :  175

w_BC  :  172

w_A  :  211

w_B  :  52

w_C  :  260


![Example_MultiMatch](https://user-images.githubusercontent.com/72695751/202760361-afefdbbb-ea7b-4efe-b4bd-d014e76ac7ee.png)

#### References

To reproduce figures of the paper "MultiMatch: Optimal Matching Colocalization in Multi-Color Super-Resolution Microscopy" by Naas et al. run the scripts located in [Zenodo](https://doi.org/10.5281/zenodo.7221879).

