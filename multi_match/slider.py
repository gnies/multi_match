from multi_match import Multi_Matching
import matplotlib
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

# helper class, following example in
# https://stackoverflow.com/questions/53093347/dynamically-update-plot-of-patches-without-artistanimations-in-matplotlib
class UpdatablePatchCollection(matplotlib.collections.PatchCollection):
    def __init__(self, patches, *args, **kwargs):
        self.patches = patches
        matplotlib.collections.PatchCollection.__init__(self, patches, *args, **kwargs)

    def get_paths(self):
        self.set_paths(self.patches)
        return self._paths

def max_dist_slider(x, y, z, init_max_dist, min_slider_value, max_slider_value, background_image):
    
    fig, ax = plt.subplots()

    ax_slider =  plt.axes([0.25, 0.02, 0.4, 0.05])
    max_dist_slider = matplotlib.widgets.Slider(ax_slider, "max dist", min_slider_value, max_slider_value, valinit=init_max_dist, initcolor='none')
    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(bottom=0.15)
    if background_image is not None:
        ax.imshow(background_image)

    # the result of the point detection is scattered 
    s = 10
    ax.scatter(x[:, 0], x[:, 1], s=s, color="red", zorder=20, edgecolors="white")
    ax.scatter(y[:, 0], y[:, 1], s=s, color="palegreen", zorder=20, edgecolors="white")
    ax.scatter(z[:, 0], z[:, 1], s=s, color="blue", zorder=20, edgecolors="white")
    
    # Create the figure and the line that we will manipulate
    circles_x = [matplotlib.patches.Circle(point, radius=init_max_dist/2, fill=False, color="white") for point in x]
    circles_y = [matplotlib.patches.Circle(point, radius=init_max_dist/2, fill=False, color="white") for point in y]
    circles_z = [matplotlib.patches.Circle(point, radius=init_max_dist/2, fill=False, color="white") for point in z]
    collection_x = UpdatablePatchCollection(circles_x, match_original=True, alpha=0.5)
    collection_y = UpdatablePatchCollection(circles_y, match_original=True, alpha=0.5)
    collection_z = UpdatablePatchCollection(circles_z, match_original=True, alpha=0.5)
    ax.add_artist(collection_x)
    ax.add_artist(collection_y)
    ax.add_artist(collection_z)

    match = Multi_Matching([x, y, z], init_max_dist)
    num_obj = match.count_objects()

    ax.set_title('$W_{ABC}=%s, W_{AB}=%s, W_{BC}=%s, W_{A}=%s, W_{B}=%s, W_{C}=%s$' % (num_obj["w_ABC"], num_obj["w_AB"], num_obj["w_BC"], num_obj["w_A"], num_obj["w_B"], num_obj["w_C"]))

    segs = match._get_links()
    lines = LineCollection(segs, colors="white", alpha=1)
    ax.add_collection(lines)

    # The function to be called anytime a slider's value changes
    def update(max_matching_dist):
        radius = max_matching_dist/2
        for circle in circles_x:
            circle.set_radius(radius)
        for circle in circles_y:
            circle.set_radius(radius)
        for circle in circles_z:
            circle.set_radius(radius)
        match = Multi_Matching([x, y, z], max_matching_dist)
        num_obj = match.count_objects()
        ax.set_title('$W_{ABC}=%s, W_{AB}=%s, W_{BC}=%s, W_{A}=%s, W_{B}=%s, W_{C}=%s$' % (num_obj["w_ABC"], num_obj["w_AB"], num_obj["w_BC"], num_obj["w_A"], num_obj["w_B"], num_obj["w_C"]))
        segs = match._get_links()
        lines.set_segments(segs)
        fig.canvas.draw()
    
    # register the update function with each slider
    max_dist_slider.on_changed(update)
    ax.axis("off")

    plt.show()
