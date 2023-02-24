from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection, LineCollection
import numpy as np
import string

class chain:
    """ Any detected object is represented by the positions of the paricles, their color channels and a label"""
    def __init__(self, positions, color_channels):
        self.color_channels = color_channels
        self.positions = positions
        self.length = len(color_channels)

    def get_links(self):
        """ 
        Get a list of coordinates identifing the segments that join the matched points.
        This is particularly usefull for plotting the object.
        """
        links = []
        i = 0
        while len(self.positions[i:])>1:
            start = self.positions[i]
            end = self.positions[i+1]
            links.append((start, end))
            i += 1
        return links

class chain_collection:
    def __init__(self, list_of_chains, color_channel_num, interaction_radius=None):
        self.all_objects =  list_of_chains
        self.color_channel_num = color_channel_num
        self.point_list = self.get_marginals()
        self.interaction_radius = interaction_radius

        # we create a list of all possibe chain-like objects
        self.possible_objects = get_all_subchains(self.color_channel_num)

        # we create a string label for each possible object
        self.label_list =  create_label_list(self.possible_objects, prefix="w_") 

    def count_objects(self):
        """ Counts the total number of different objects that are found"""
        channel_indicies = [tuple(obj.color_channels) for obj in self.all_objects]
        counted = Counter(channel_indicies)
        res = {}
        for i in range(len(self.possible_objects)):
            obj_type = self.possible_objects[i]
            label = self.label_list[i]
            res[label] = counted[obj_type]
        return res

    def get_marginals(self):
        marginal_point_list = [[] for c in range(self.color_channel_num)]
        for obj in self.all_objects:
            channels = obj.color_channels
            positions = obj.positions
            length = obj.length
            for t in range(length):
                channel = channels[t]
                position = positions[t]
                marginal_point_list[channel].append(position)
        # every point list needs to be a numpy array
        marginal_point_list = [np.asarray(points) for points in marginal_point_list]
        return marginal_point_list
    
    def get_all_links(self):
        lst = []
        for obj in self.all_objects:
            obj_links = obj.get_links()
            lst.extend(obj_links)
        return lst

    def plot_marginals(self, ax=None, size=None, alpha=None, edgecolors=None,
            channel_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']):
        if ax is None:
            ax = plt.gca()
        for i in range(len(self.point_list)):
            points = self.point_list[i]
            zorder = 20 # hight zorder ensures that these will not be convered by the segments 
                        # or other lines
            if len(points)>0:
                ax.scatter(points[:, 0], points[:, 1], s=size, color=channel_colors[i],
                        zorder=zorder, edgecolors=edgecolors, alpha=alpha)

    def plot_segments(self, show=False, ax=None, segment_color="grey", linewidth=None, alpha=None):
        """Plot a collection (list) of chains"""
        if ax is None:
            ax = plt.gca()
        segs = self.get_all_links()
        lines = LineCollection(segs, colors=segment_color, linewidth=linewidth, alpha=alpha)
        ax.add_collection(lines)
        if show:
            plt.show()
        return lines

    def plot_circles(self, show=False, ax=None, color="grey", alpha=None):
        radius = self.interaction_radius
        if radius is None:
            raise Exception('You need to set a value for the interaction_radius')
        if ax is None:
            ax = plt.gca()
        collections = []
        for i in range(len(self.point_list)):
            points = self.point_list[i]
            if len(points)>0:
                circles = [Circle(point, radius=radius, fill=False, color=color, alpha=alpha)
                    for point in points]
            collection = UpdatablePatchCollection(circles, match_original=True)
            ax.add_artist(collection)
            collections.append(collection)
        if show:
            plt.show()
        return collections

    def plot_match(self, show=False, ax=None, 
        scatter_size=None, scatter_alpha=None, scatter_edgecolors=None, 
        channel_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'], 
        segment_color = "grey", segment_linewidth=None, segment_alpha=None,
        circle_color="grey", circle_alpha=None):

        if ax is None:
            ax = plt.gca()
        # plotting the marginals
        self.plot_marginals(size=scatter_size, ax=ax, alpha=scatter_alpha, edgecolors=scatter_edgecolors,
                channel_colors=channel_colors)
        # plotting the links connecting the points 
        self.plot_segments(ax=ax, segment_color=segment_color, linewidth=segment_linewidth, alpha=segment_alpha)
        if self.interaction_radius is not None:
            # plotting the circes 
            self.plot_circles(ax=ax, color=circle_color, alpha=circle_alpha)
        if show:
            plt.show()

def get_all_subchains(n):
    """
    This function returns a list of all possible subsequences of (1, \dots, n),
    such that the difference beween two consecutive elements is 1.
    If n=3, then the function returns [(0, 1, 2), (0, 1), (1, 2), (0), (1), (2)].
    """ 
    res = []
    for k in range(n):
        length = n - k
        for i in range(n - length + 1):
            chain = tuple(range(i, i+length))
            res.append(chain)
    return res 

def create_label_list(objects, prefix="w_"):
    """
    This function returns a list of the label for all the possible objects we can detect.
    If color_channel_num=3, then the object labels are 
    ["w_ABC", "w_AB", "w_BC", "w_A, "w_B", "w_C"].
    """ 
    label_list = []
    alphab = string.ascii_uppercase
    for obj in objects:
        channel_in_letters = [alphab[i] for i in obj]
        label = prefix + "".join(channel_in_letters)
        label_list.append(label)
    return label_list

class UpdatablePatchCollection(PatchCollection):
    def __init__(self, patches, *args, **kwargs):
        self.patches = patches
        PatchCollection.__init__(self, patches, *args, **kwargs)
    def get_paths(self):
        self.set_paths(self.patches)
        return self._paths
