from skimage.feature import blob_log
from skimage.color import rgb2gray, rgba2rgb
import numpy as np

def point_detection(image, min_sigma=1.5, max_sigma=2, num_sigma=10, threshold=0.05):
    image = (image - image.min()) / (image.max()-image.min())
    blobs = blob_log(image, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)

    x2 = blobs[:, 0]
    x1 = blobs[:, 1]
    x = np.vstack([x1, x2]).T
    return x

