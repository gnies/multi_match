import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pandas as pd
from scipy.spatial.distance import cdist
from .matching_methods import match_pairwise, match_all
from .abbundance_estimation import _estimate_abundance
from .confidence_ellipsoid import compute_confidence_ellipsiod_from_data, compute_confidence_box
from .chain_collection import get_all_subchains, create_label_list
from .chain_collection import chain, chain_collection

class Multi_Matching(chain_collection):
    """
    This class detects chains given a set of point clouds and an interaction radious.
    """
    def __init__(self, point_list, max_dist, method="triplets first", cost_list=None):
        """
        Initializes a new instance of the Multi_Matching class.
        
        Parameters:
            point_list (list): A list of numpy arrays, where each entry is an array of coordinates for a given point cloud.
            max_dist (float): A maximal interaction radius. Points will not be matched above this radius.
            method (str): The method used to detect chains. Can be "triplets first" or "pairwise".
            cost_list (list): A list of cost matrices, where each entry is the cost matrix between consecutive point clouds in point_list.
                              if cost_list is None, the cost matrices are computed using the euclidean distance.
        """
        # Convert each array in point_list to a numpy array
        self.point_list = [np.asarray(points) for points in point_list]
        self.max_dist = max_dist
        self.interaction_radius = max_dist/2
        self.method = method

        self.color_channel_num = len(point_list)

        # We create a list of all possible channel inidicies for a chain-like objects.
        self.possible_objects = get_all_subchains(self.color_channel_num)

        # we create a string label for each possible object
        self.label_list =  create_label_list(self.possible_objects, prefix="w_") 

        # we compute the assignment matrix
        self.index_assignment = compute_assignment(self.point_list, max_dist, method, cost_list=cost_list)

        # we compute a list with all objects we detect
        self.all_objects = get_all_objects(self.point_list, self.index_assignment)

    def estimate_abundances(self, s_list, enforce_positive_integers=True):
        """ Estimates the abbundances given a list ov values that determine the staining efficinency for each channel"""
        data = self.count_objects()
        W = np.asarray([data[label] for label in self.label_list], dtype=int)
        N = _estimate_abundance(W, self.possible_objects, self.color_channel_num, s_list,
                enforce_positive_integers=enforce_positive_integers)
        # the result is returned as a dictionary, with the correspondend object label as a key 
        res = {}
        for i in range(len(self.label_list)):
            label = self.label_list[i]
            new_label = "n" + label[1:]
            res[new_label] = N[i]
        return res

    def confidence_ellipsiod(self, s_list, test_alpha,
            enforce_positive_integers=True):
        N_data = self.estimate_abundances(s_list, enforce_positive_integers=True)
        label_list =  create_label_list(self.possible_objects, prefix="n_") 
        N = np.asarray([N_data[label] for label in label_list])
        return compute_confidence_ellipsiod_from_data(N, s_list, self.color_channel_num, self.possible_objects, test_alpha)

    def confidence_box(self, s_list, test_alpha,
            enforce_positive_integers=True):

        N_data = self.estimate_abundances(s_list, enforce_positive_integers=True)
        label_list =  create_label_list(self.possible_objects, prefix="n_") 
        N = np.asarray([N_data[label] for label in label_list])
        return compute_confidence_box(N, s_list, self.color_channel_num, self.possible_objects, test_alpha)

class Multi_Matching_Over_Range:
    """ This class serves to plot and count the number of chains given a set of point clouds and an range of maximal interaction distance values."""
    def __init__(self, point_list, max_dist_range, method="triplets first", cost_list=None):
        self.point_list = [np.asarray(points) for points in point_list]
        self.max_dist_range = max_dist_range
        self.method = method
        self.color_channel_num = len(point_list)
        self.cost_list = cost_list

        # we create a list of all possibe chain-like objects
        self.possible_objects = get_all_subchains(self.color_channel_num)

        # we create a string label for each possible object
        self.label_list =  create_label_list(self.possible_objects, prefix="w_") 

        self.number_of_objects = None

    def plot_match(self, fig=None, ax=None, show=False, slider_window=[0.25, 0.02, 0.4, 0.05], fontsize=None,
        scatter_size=None, scatter_alpha=None, scatter_edgecolors=None, 
        channel_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'], 
        segment_color = "grey", segment_linewidth=None, segment_alpha=None,
        circle_color="grey", circle_alpha=None):
        """ 
        Function to plot the matching with a slider.

        Slider_window gives the location of the slider.
        left, bottom, width, height = slider_window."""
        if fig is None:
            fig = plt.gcf()
        if ax is None:
            ax = plt.gca()
        ax_slider =  plt.axes(slider_window)
        min_slider_val = self.max_dist_range[0]
        max_slider_val = self.max_dist_range[-1]
        init_max_dist = (min_slider_val + max_slider_val)/2
        init_match = Multi_Matching(self.point_list, init_max_dist, method=self.method, cost_list=self.cost_list)
        slid = Slider(ax_slider, "max dist",
                min_slider_val, max_slider_val, valinit=init_max_dist,
                initcolor='none')
        # adjust the main plot to make room for the sliders
        plt.subplots_adjust(bottom=0.15)
        init_match.plot_marginals(ax=ax, size=scatter_size, alpha=scatter_alpha, edgecolors=scatter_edgecolors, channel_colors=channel_colors)
        circle_collection_lst = init_match.plot_circles(ax=ax, color=circle_color, alpha=circle_alpha)
        line_collection = init_match.plot_segments(ax=ax, segment_color=segment_color, linewidth=segment_linewidth, alpha=segment_alpha)
        ax.set_title(str(init_match.count_objects()), fontsize=fontsize) # change this as well

        # The function to be called anytime a slider's value changes
        def update(max_matching_dist):
            radius = max_matching_dist/2
            for circle_collection in circle_collection_lst:
                for circle in circle_collection.patches:
                    circle.set_radius(radius)
            current_match = Multi_Matching(self.point_list, max_matching_dist,
                    method=self.method, cost_list=self.cost_list)
            num_obj = current_match.count_objects()
            ax.set_title(str(num_obj), fontsize=fontsize)
            segs = current_match.get_all_links()
            line_collection.set_segments(segs)
            fig.canvas.draw()
        
        # register the update function with each slider
        slid.on_changed(update)
        plt.show()


    def count_objects(self):
        """ This function counts the number of chains detected for each maximal interaction distance value"""
        columns = self.label_list.copy()
        columns.append("maxdist")
        data = pd.DataFrame(columns=columns)
        for i in range(len(self.max_dist_range)):
            maxdist = self.max_dist_range[i]
            match = Multi_Matching(self.point_list, maxdist, method=self.method, cost_list=self.cost_list)
            vals = match.count_objects()
            data.loc[i] = np.asarray([vals[col] if col !="maxdist" else maxdist for col in columns])
        self.number_of_objects = data
        return data

    def plot_number_of_objects(self, ax=None, show=True):
        """ This function plots the number of chains detected for each maximal interaction distance value with a cumulative plot"""
        if self.number_of_objects is None:
            self.count_objects()
        data = self.number_of_objects
        if ax is None:
            ax = plt.gca()
        maxdist_range = data["maxdist"].values

        alpha = 0.7
        lower = 0
        for label in data.columns:
            if label != "maxdist":
                upper = lower + data[label]
                ax.fill_between(maxdist_range, lower, upper, label=label, alpha=alpha)
                # the plot is cumulative, therefore the lower curve is updated
                lower = upper

        ax.set_title("Comulative plot of detected objets against maximal matching distance")
        ax.set_ylabel("number of objects")
        ax.set_xlabel("maximal  matching distance")
        ax.legend()
        if show == True:
            plt.show()

    def estimate_abundances(self, s_list, enforce_positive_integers=True):
        """ Estimates the abbundances given a list of the staining efficiency coefficient per channel """
    
        if self.number_of_objects is None:
            self.count_objects()
        data = self.number_of_objects
        W = data.loc[:, self.label_list].values
        N = _estimate_abundance(W, self.possible_objects, self.color_channel_num, s_list,
                enforce_positive_integers=enforce_positive_integers)
        # the result is returned as a dictionary, with the correspondend object label as a key 
        res = {}
        for i in range(len(self.label_list)):
            label = self.label_list[i]
            new_label = "n" + label[1:]
            res[new_label] = N[:, i]
        res["maxdist"] = data["maxdist"]
        return pd.DataFrame(res)

    def plot_estimated_number_of_objects(self,s_list, ax=None, show=True):
        """ This function plots the number of chains detected for each maximal interaction distance value with a cumulative plot"""
        data = self.estimate_abundances(s_list)
        if ax is None:
            ax = plt.gca()
        maxdist_range = data["maxdist"].values

        alpha = 0.7
        lower = 0
        for label in data.columns:
            if label != "maxdist":
                upper = lower + data[label]
                ax.fill_between(maxdist_range, lower, upper, label=label, alpha=alpha)
                # the plot is cumulative, therefore the lower curve is updated
                lower = upper

        ax.set_title("Comulative plot of estimated abundances against maximal matching distance")
        ax.set_ylabel("number of estimated chains")
        ax.set_xlabel("maximal  matching distance")
        ax.legend()
        if show == True:
            plt.show()

def compute_assignment(point_list, max_dist, method, cost_list=None):
    if cost_list is None:
        cost_list = [cdist(point_list[i], point_list[i+1]) for i in range(len(point_list)-1)]
    if method == "triplets first":
        if len(point_list) != 3:
            raise Exception("This method currently works for matching three point clouds only. Try setting method=\"pairwise\"")
        else:
            c_xy, c_yz = cost_list
            assignment = match_all(c_xy, c_yz, max_dist)
    elif method == "pairwise":
        assignment = match_pairwise(cost_list, max_dist)
    else:
        raise Exception("Method not implemented, try setting method=\"pairwise\" or method=\"triplets first\"")
    return assignment
    
def get_all_objects(point_list, index_assignment):
    """ Get a list of objects, given the marginal point clouds, an index assignment"""
    objects = []
    n = len(index_assignment)
    for i in range(n):
        row = index_assignment[i]
        channels = np.where(row >= 0)[0].tolist()
        channels = tuple(channels)
        positions = []
        for j in channels:
            positions.append(point_list[j][index_assignment[i, j]].tolist())
        obj = chain(positions, channels)
        objects.append(obj)
    return objects
