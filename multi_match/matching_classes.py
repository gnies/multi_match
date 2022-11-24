import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .matching_methods import match_pairwise, match_all
from .confidence_ellipsoid import compute_confidence_ellipsiod_from_data
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection, LineCollection

# works for triplets, but it would be possible to build a more general class that covers n-lets
class Multi_Matching:
    def __init__(self, point_lst, max_dist, method="triplets first"):

        self.point_lst = [np.array(points) for points in point_lst]
        self.max_dist = max_dist
        self.method = method
        self.number_of_color_channels = len(point_lst)

        if method == "triplets first":
            if len(point_lst) != 3:
                raise Exception("This method currently works for matching three point clouds only. Try setting method=\"pairwise\"")
            else:
                x, y, z = point_lst
                match = match_all(x, y, z, max_dist)
        elif method == "pairwise":
            match = match_pairwise(point_lst, max_dist)
        else:
            raise Exception("Method not implemented (yet!)")
        self.index_assignment = match

    def get_object_channel_indicies(self):
        channels = []
        for obj in self.index_assignment:
            channels.append(np.where(obj >= 0)[0].tolist())
        return channels

    def get_object_locations(self):
        channels = self.get_object_channel_indicies()
        ind = self.index_assignment
        locs = []
        for i in range(len(channels)):
            loc_list = []
            for j in channels[i]:
                loc_list.append(self.point_lst[j][ind[i, j]].tolist())
            locs.append(loc_list)
        return locs

    def _get_links(self):
        lst = []
        locs = self.get_object_locations()
        for positions in locs:
            i = 0
            while len(positions[i:])>1:
                start = positions[i]
                end = positions[i+1]
                lst.append((start, end))
                i += 1
        return lst
    
    def plot_match(self, ax=None, scatter_dot_size=None, scatter_alpha=0.7, edgecolors=None, linewidth=None,
            color='white', circle_alpha=0.5,
            channel_colors=plt.rcParams['axes.prop_cycle'].by_key()['color']):
        edgecolors = None
        if ax is None:
            ax = plt.gca()
        segs = self._get_links()
        lines = LineCollection(segs, colors=color, alpha=1, linewidth=2)
        ax.add_collection(lines)
        # the result of the point detection is scattered

        for i in range(len(self.point_lst)):
            points = self.point_lst[i]
            if len(points)>0:
                ax.scatter(points[:, 0], points[:, 1], s=scatter_dot_size, color=channel_colors[i],
                        zorder=20, edgecolors=edgecolors)
            radius = self.max_dist/2
            circles = [Circle(point, radius=radius, fill=False, color=color, alpha=circle_alpha)
                    for point in points]
            collection = PatchCollection(circles, match_original=True)
            ax.add_artist(collection)

    def count_objects(self):
        channel_indicies = self.get_object_channel_indicies()
        n_abc = 0
        n_ab = 0
        n_bc = 0
        n_a = 0
        n_b = 0
        n_c = 0
        for ob in channel_indicies:
            if ob == [0, 1, 2]:
                n_abc += 1
            elif ob == [0, 1]:
                n_ab += 1
            elif ob == [1, 2]:
                n_bc += 1
            elif ob == [0]:
                n_a += 1
            elif ob == [1]:
                n_b += 1
            elif ob == [2]:
                n_c += 1
        return {"w_ABC":int(n_abc), "w_AB":int(n_ab), "w_BC":int(n_bc), "w_A":int(n_a), "w_B":int(n_b), "w_C":int(n_c)}

    def estimate_abundances(self, s_A, s_B, s_C, enforce_positive_integers=True):
        """ Estimates the abbundances given staining efficinencies s_A, s_B, s_C """
        data = self.count_objects()
        W = [data['w_ABC'], data['w_AB'], data['w_BC'], data['w_A'], data['w_B'], data['w_C']]
        theta_mu_inverse = np.array([
                          [1/(s_A*s_B*s_C)               , 0                 , 0                 , 0    , 0     , 0     ],
                          [(s_C-1)/(s_A*s_B*s_C)         , 1/(s_A*s_B)       , 0                 , 0    , 0     , 0     ],
                          [(s_A-1)/(s_A*s_B*s_C)         , 0                 , 1/(s_B*s_C)       , 0    , 0     , 0     ],
                          [(s_B-1)/(s_A*s_B)             , (s_B-1)/(s_A*s_B) , 0                 , 1/s_A, 0     , 0     ],
                          [(s_A-1)*(s_C-1)/(s_A*s_B*s_C) , (s_A-1)/(s_A*s_B) , (s_C-1)/(s_B*s_C) , 0    , 1/s_B , 0     ],
                          [(s_B-1)/(s_B*s_C)             , 0                 , (s_B-1)/(s_B*s_C) , 0    , 0     , 1/s_C ]
                          ])
        N = theta_mu_inverse @ W
        if enforce_positive_integers:
            N = np.rint(N)   # round to integer
            N = np.maximum(N, 0) # no negative values
            N = N.astype(int)
        return {'n_ABC':N[0], 'n_AB':N[1], 'n_BC':N[2], 'n_A':N[3], 'n_B':N[4], 'n_C':N[5]}

    def compute_confidence_ellipsiod(self, s_A, s_B, s_C, alpha):
        N_data = self.estimate_abundances(s_A, s_B, s_C, enforce_positive_integers=True)
        N = np.array([N_data['n_ABC'], N_data['n_AB'], N_data['n_BC'], N_data['n_A'], N_data['n_B'], N_data['n_C']])

    # putting them into an array
        return compute_confidence_ellipsiod_from_data(N, s_A, s_B, s_C, alpha)

class Multi_Matching_Over_Range:
    def __init__(self, point_lst, max_dist_range, method="triplets first"):

        self.point_lst = [np.array(points) for points in point_lst]
        self.max_dist_range = max_dist_range
        self.method = method
        self.number_of_color_channels = len(point_lst)
        self.number_of_objects = None

    def count_objects(self):
        columns=["w_ABC", "w_AB", "w_BC", "w_A", "w_B", "w_C", "maxdist"]
        data = pd.DataFrame(columns=columns)
        for i in range(len(self.max_dist_range)):
            maxdist = self.max_dist_range[i]
            match = Multi_Matching(self.point_lst, maxdist, method=self.method)
            vals = match.count_objects()
            data.loc[i] = np.array([vals[col] if col !="maxdist" else maxdist for col in columns])
        self.number_of_objects = data
        return data

    def plot_number_of_objects(self, ax=None, show=True):
        if self.number_of_objects is None:
            self.count_objects()
        data = self.number_of_objects
        if ax is None:
            ax = plt.gca()
        maxdist_range = data["maxdist"].values
        abc = data["w_ABC"]
        ab = data["w_AB"]
        bc = data["w_BC"]
        a = data["w_A"]
        b = data["w_B"]
        c = data["w_C"]
    
        alpha = 0.7
        ax.set_title("Comulative plot of detected objets against maximal matching distance")
        ax.fill_between(maxdist_range, 0, a, label="$W_A$", alpha=alpha)
        ax.fill_between(maxdist_range, a, a+b, label="$W_B$", alpha=alpha)
        ax.fill_between(maxdist_range, a+b, a+b+c, label="$W_C$", alpha=alpha)
        ax.fill_between(maxdist_range, a+b+c, a+b+c+ab, label="$W_{AB}$", alpha=alpha)
        ax.fill_between(maxdist_range, a+b+c+ab, a+b+c+ab+bc, label="$W_{BC}$", alpha=alpha)
        ax.fill_between(maxdist_range, a+b+c+ab+bc, a+b+c+ab+bc+abc, label="$W_{ABC}$", alpha=alpha)
        ax.set_ylabel("number of objects")
        ax.set_xlabel("maximal  matching distance")
        ax.legend()
        if show == True:
            plt.show()

    def estimate_abundances(self, s_A, s_B, s_C):
        """ Estimates the abbundances given staining efficinencies s_A, s_B, s_C """
    
        if self.number_of_objects is None:
            self.count_objects()
        data = self.number_of_objects
        W = data.loc[:, ['w_ABC', 'w_AB', 'w_BC', 'w_A', 'w_B', 'w_C']].values
        theta_mu_inverse = np.array([
                          [1/(s_A*s_B*s_C)               , 0                 , 0                 , 0    , 0     , 0     ],
                          [(s_C-1)/(s_A*s_B*s_C)         , 1/(s_A*s_B)       , 0                 , 0    , 0     , 0     ],
                          [(s_A-1)/(s_A*s_B*s_C)         , 0                 , 1/(s_B*s_C)       , 0    , 0     , 0     ],
                          [(s_B-1)/(s_A*s_B)             , (s_B-1)/(s_A*s_B) , 0                 , 1/s_A, 0     , 0     ],
                          [(s_A-1)*(s_C-1)/(s_A*s_B*s_C) , (s_A-1)/(s_A*s_B) , (s_C-1)/(s_B*s_C) , 0    , 1/s_B , 0     ],
                          [(s_B-1)/(s_B*s_C)             , 0                 , (s_B-1)/(s_B*s_C) , 0    , 0     , 1/s_C ]
                          ])
        N = np.matmul(W, theta_mu_inverse.T)
        # enforce positive integer solutions
        N = np.rint(N)   # round to integer
        N = np.maximum(N, 0) # no negative values
        N = N.astype(int)
        return pd.DataFrame({'n_ABC':N[:, 0], 'n_AB':N[:, 1], 'n_BC':N[:, 2], 'n_A':N[:, 3], 'n_B':N[:, 4], 'n_C':N[:, 5], 'maxdist':data['maxdist'].values})

    def plot_number_of_matches(self):
        """ Plots number of matches as a function of the maximal matching distance given a data frame."""
        if self.number_of_objects is None:
            self.count_objects()
        data = self.number_of_objects
    
        maxdist_range = data["maxdist"].values
        n_abc = data["w_ABC"].values
        n_ab = data["w_AB"].values
        n_bc = data["w_BC"].values
        n_x = data["w_ABC"].values[0] + data["w_AB"].values[0] + data["w_A"].values[0]
        n_y = data["w_ABC"].values[0] + data["w_AB"].values[0] + data["w_BC"].values[0] + data["w_B"].values[0]
        n_z = data["w_ABC"].values[0] + data["w_BC"].values[0] + data["w_C"].values[0]

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    
        ax[0].plot(maxdist_range, n_abc/n_x, label="ABC", color="tab:orange")
        ax[0].plot(maxdist_range, n_ab/n_x, label="AB", color="tab:blue")
        title_x = "A"
        ax[0].set_title(title_x)
        ax[0].legend(loc="upper left")
        ax[0].set_xlabel("Maximum matching distance in pixel")
        ax[0].set_ylabel("Ratio of matched points")
    
        title_y = "B"
        ax[1].set_title(title_y)
        ax[1].plot(maxdist_range, n_abc/n_y, label="ABC", color="tab:orange")
        ax[1].plot(maxdist_range, n_ab/n_y, label="AB", color="tab:red")
        ax[1].plot(maxdist_range, n_bc/n_y, label="BC", color="tab:green")
        ax[1].legend(loc="upper left")
        ax[1].set_xlabel("Maximum matching distance in pixel")
    
        title_z = "C"
        ax[2].set_title(title_z)
        ax[2].plot(maxdist_range, n_abc/n_z, label="ABC", color="tab:orange")
        ax[2].plot(maxdist_range, n_bc/n_z, label="BC", color="tab:blue")
        ax[2].legend(loc="upper left")
        ax[2].set_xlabel("Maximum matching distance in pixel")
        plt.show()
    
    def plot_relative_abundances_estimation(self, s_a, s_b, s_c, alpha=0.05):
        """ Plots number of matches as a function of the maximal matching distance
            given a data frame."""

        # abundance estimation
        N = self.estimate_abundances(s_a, s_b, s_c)

        maxdist_range = N["maxdist"].values
        n_abc = N["n_ABC"].values
        n_ab = N["n_AB"].values
        n_bc = N["n_BC"].values
        n_x = N["n_ABC"].values + N["n_AB"] + N["n_A"]
        n_y = N["n_ABC"].values + N["n_AB"] + N["n_BC"] + N["n_B"]
        n_z = N["n_ABC"].values + N["n_BC"] + N["n_C"]

        # compute confidence box for each maxdist value
        N = np.asarray(N)
        n_abc_conf_radius = []
        n_ab_conf_radius = []
        n_bc_conf_radius = []
        for n_row in N[:, :-1]:
            d = compute_confidence_ellipsiod_from_data(n_row, s_a, s_b, s_c, alpha)
            matrix = d["ellipsoid_matrix"]
            t = d["threshold"]
            if matrix[0, 0] > 0:
                n_abc_conf_radius.append((t/matrix[0, 0])**0.5)
            else:
                n_abc_conf_radius.append(0)
            if matrix[1, 1]>0:
                n_ab_conf_radius.append((t/matrix[1, 1])**0.5)
            else:
                n_ab_conf_radius.append(0)
            if matrix[2, 2]>0:
                n_bc_conf_radius.append((t/matrix[2, 2])**0.5)
            else:
                n_bc_conf_radius.append(0)
        n_abc_conf_radius = np.asarray(n_abc_conf_radius)
        n_ab_conf_radius = np.asarray(n_ab_conf_radius)
        n_bc_conf_radius = np.asarray(n_bc_conf_radius)

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    
        ax[0].plot(maxdist_range, n_abc/n_x, label="ABC", color="tab:orange")
        ax[0].plot(maxdist_range, n_ab/n_x, label="AB", color="tab:red")
        ax[0].set_xlim((maxdist_range[0], maxdist_range[-1]))
        ax[0].set_ylim((-0.00, 1.00))
        title_x = "Channel A"
        ax[0].set_title(title_x)
        ax[0].legend(loc="upper left")
        ax[0].set_xlabel("colocalization distance $t$ in pixels")
        ax[0].set_ylabel("relative abundance")
        ax[0].fill_between(maxdist_range, (n_abc - n_abc_conf_radius)/n_x, (n_abc + n_abc_conf_radius)/n_x, color="tab:orange", alpha=0.3)
        ax[0].fill_between(maxdist_range, (n_ab - n_ab_conf_radius)/n_x, (n_ab + n_ab_conf_radius)/n_x, color="tab:red", alpha=0.3)
        ax[0].grid(axis="both")
        box = ax[0].get_position()
        ax[0].set_position([box.x0, box.y0, box.width * 0.85, box.height])
    
        title_y = "Channel B"
        ax[1].set_title(title_y)
        ax[1].plot(maxdist_range, n_abc/n_y, label="ABC", color="tab:orange")
        ax[1].plot(maxdist_range, n_ab/n_y, label="AB", color="tab:red")
        ax[1].plot(maxdist_range, n_bc/n_y, label="BC", color="tab:blue")
        ax[1].fill_between(maxdist_range, (n_abc - n_abc_conf_radius)/n_y, (n_abc + n_abc_conf_radius)/n_y, color="tab:orange", alpha=0.3)
        ax[1].fill_between(maxdist_range, (n_ab - n_ab_conf_radius)/n_y, (n_ab + n_ab_conf_radius)/n_y, color="tab:red", alpha=0.3)
        ax[1].fill_between(maxdist_range, (n_bc - n_bc_conf_radius)/n_y, (n_bc + n_bc_conf_radius)/n_y, color="tab:blue", alpha=0.3)
        ax[1].legend(loc="upper left")
        ax[1].set_xlabel("Maximum matching distance in pixel")
        ax[1].set_xlabel("colocalization distance $t$ in pixels")
        ax[1].set_ylabel("relative abundance")
        box = ax[1].get_position()
        ax[1].set_position([box.x0, box.y0, box.width * 0.85, box.height])
        ax[1].grid(axis="both")
        ax[1].set_xlim((maxdist_range[0], maxdist_range[-1]))
        ax[1].set_ylim((-0.00, 1.00))

        title_z = "Channel C"
        ax[2].set_title(title_z)
        ax[2].plot(maxdist_range, n_abc/n_z, label="ABC", color="tab:orange")
        ax[2].plot(maxdist_range, n_bc/n_z, label="BC", color="tab:blue")
        ax[2].fill_between(maxdist_range, (n_abc - n_abc_conf_radius)/n_z, (n_abc + n_abc_conf_radius)/n_z, color="tab:orange", alpha=0.3)
        ax[2].fill_between(maxdist_range, (n_bc - n_bc_conf_radius)/n_z, (n_bc + n_bc_conf_radius)/n_z, color="tab:blue", alpha=0.3)
        ax[2].legend(loc="upper left")
        ax[2].set_xlabel("Maximum matching distance in pixel")
        ax[2].set_xlabel("colocalization distance $t$ in pixels")
        ax[2].set_ylabel("relative abundance")
        box = ax[2].get_position()
        ax[2].set_position([box.x0, box.y0, box.width * 0.85, box.height])
        ax[2].grid(axis="both")
        ax[2].set_xlim((maxdist_range[0], maxdist_range[-1]))
        ax[2].set_ylim((-0.00, 1.00))
        plt.show()

    def cumolative_plot_number_of_matches(self):
        """ Plots number of matches as a function of the maximal matching distance given a data frame."""
        if self.number_of_objects is None:
            self.count_objects()
        data = self.number_of_objects
    
        maxdist_range = data["maxdist"].values
        n_abc = data["w_ABC"].values
        n_ab = data["w_AB"].values
        n_bc = data["w_BC"].values
        n_x = data["w_ABC"].values[0] + data["w_AB"].values[0] + data["w_A"].values[0]
        n_y = data["w_ABC"].values[0] + data["w_AB"].values[0] + data["w_BC"].values[0] + data["w_B"].values[0]
        n_z = data["w_ABC"].values[0] + data["w_BC"].values[0] + data["w_C"].values[0]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    
        # A
        ax[0].set_box_aspect(1)
        ax[0].set_title("A")
        ax[0].fill_between(maxdist_range, np.zeros(len(maxdist_range)),n_abc/n_x, color="tab:orange", alpha=0.3, label='$\\frac{W_{ABC}}{W_{ABC}+ W_{AB} + W_A}$')
        ax[0].plot(maxdist_range, n_abc/n_x, color="tab:orange")
        ax[0].fill_between(maxdist_range, n_abc/n_x, (n_abc + n_ab)/n_x, color="tab:red", alpha=0.3, label='$\\frac{W_{AB}}{W_{ABC}+ W_{AB} + W_A}$')
        ax[0].plot(maxdist_range, (n_abc + n_ab)/n_x, color="tab:red")
        ax[0].set_ylabel("Percentage of matched points")
        ax[0].set_xlabel("Value of $\gamma$ in pixel")
        ax[0].vlines(x=2.8, ymin=0, ymax=1, color="grey", alpha=0.5)
        ax[0].hlines(y=1, xmin=0, xmax=maxdist_range.max(), color="grey", alpha=0.5)
        ax[0].legend()
        def prob_to_abs(p):
            return p * n_x
    
        def abs_to_prob(a):
            return a / n_x
        secax = ax[0].secondary_yaxis('right', functions=(prob_to_abs, abs_to_prob))
        ax[0].grid(axis="y")
        # secax.set_ylabel("Number of points")
    
        # B
        ax[1].set_title("B")
        ax[1].set_box_aspect(1)
        ax[1].fill_between(maxdist_range, np.zeros(len(maxdist_range)),n_abc/n_y, color="tab:orange", alpha=0.3, label='$\\frac{W_{ABC}}{W_{ABC}+ W_{AB} + W_{BC} + W_B}$')
        ax[1].plot(maxdist_range, n_abc/n_y, color="tab:orange")
        ax[1].fill_between(maxdist_range, n_abc/n_y, (n_abc + n_ab)/n_y, color="tab:red", alpha=0.3, label='$\\frac{W_{AB}}{W_{ABC}+ W_{AB} + W_{BC} + W_B}$')
        ax[1].plot(maxdist_range, (n_abc + n_ab)/n_y, color="tab:red")
        ax[1].fill_between(maxdist_range, (n_abc + n_ab)/n_y, (n_abc + n_ab+ n_bc)/n_y, color="tab:blue", alpha=0.3, label='$\\frac{W_{BC}}{W_{ABC}+ W_{AB} + W_{BC} + W_B}$')
        ax[1].plot(maxdist_range, (n_abc + n_ab+ n_bc)/n_y, color="tab:blue")
        ax[1].legend()
        # ax[1].set_ylabel("Percentage of matched points")
        ax[1].vlines(x=2.8, ymin=0, ymax=1, color="grey", alpha=0.5)
        ax[1].hlines(y=1, xmin=0, xmax=maxdist_range.max(), color="grey", alpha=0.5)
        ax[1].set_xlabel("Value of $\gamma$ in pixel")
    
        def prob_to_abs(p):
            return p * n_y
    
        def abs_to_prob(a):
            return a / n_y
        ax[1].grid(axis="y")
        
        secax = ax[1].secondary_yaxis('right', functions=(prob_to_abs, abs_to_prob))
        # secax.set_ylabel("Number of points")
    
        # C
        ax[2].set_title("C")
        ax[2].set_box_aspect(1)
        ax[2].fill_between(maxdist_range, np.zeros(len(maxdist_range)), n_abc/n_z, color="tab:orange", alpha=0.3, label='$\\frac{W_{ABC}}{W_{ABC}+ W_{BC} + W_{B}}$')
        ax[2].plot(maxdist_range, n_abc/n_z, color="tab:orange")
        ax[2].fill_between(maxdist_range, (n_abc )/n_z, (n_abc + n_bc)/n_z, color="tab:blue", alpha=0.3, label='$\\frac{W_{BC}}{W_{ABC}+ W_{BC} + W_{B}}$')
        ax[2].plot(maxdist_range, (n_abc + n_bc)/n_z, color="tab:blue")
        # ax[2].set_ylabel("Percentage of matched points")
        ax[2].vlines(x=2.8, ymin=0, ymax=1, color="grey", alpha=0.5)
        ax[2].hlines(y=1, xmin=0, xmax=maxdist_range.max(), color="grey", alpha=0.5)
        ax[2].legend()
        ax[2].grid(axis="y")
        ax[2].set_xlabel("Value of $\gamma$ in pixel")
        def prob_to_abs(p):
            return p * n_z
        def abs_to_prob(a):
            return a / n_z
        secax = ax[2].secondary_yaxis('right', functions=(prob_to_abs, abs_to_prob))
        secax.set_ylabel("Number of points")
        plt.show()

    def plot_relative_abundances(self):
        """ Plots number of matches as a function of the maximal matching distance given a data frame."""
        # get number of objects
        if self.number_of_objects is None:
            self.count_objects()
        N = self.number_of_objects

        maxdist_range = N["maxdist"].values
        n_abc = N["w_ABC"].values
        n_ab = N["w_AB"].values
        n_bc = N["w_BC"].values
        n_x, n_y, n_z = (len(self.point_lst[0]), len(self.point_lst[1]), len(self.point_lst[2]))

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    
        ax[0].plot(maxdist_range, n_abc/n_x, label="ABC", color="tab:orange")
        ax[0].plot(maxdist_range, n_ab/n_x, label="AB", color="tab:red")
        ax[0].set_xlim((maxdist_range[0], maxdist_range[-1]))
        ax[0].set_ylim((-0.00, 1.00))
        title_x = "Channel A"
        ax[0].set_title(title_x)
        ax[0].legend(loc="upper left")
        ax[0].set_xlabel("colocalization distance $t$ in pixels")
        ax[0].set_ylabel("relative abundance")
        ax[0].grid(axis="both")
        box = ax[0].get_position()
        ax[0].set_position([box.x0, box.y0, box.width * 0.85, box.height])
        def prob_to_abs(p):
            return p * n_x 
        def abs_to_prob(a):
            return a /  n_x
        secax = ax[0].secondary_yaxis('right', functions=(prob_to_abs, abs_to_prob))
        secax.set_ylabel("Number of points")

    
        title_y = "Channel B"
        ax[1].set_title(title_y)
        ax[1].plot(maxdist_range, n_abc/n_y, label="ABC", color="tab:orange")
        ax[1].plot(maxdist_range, n_ab/n_y, label="AB", color="tab:red")
        ax[1].plot(maxdist_range, n_bc/n_y, label="BC", color="tab:blue")
        ax[1].legend(loc="upper left")
        ax[1].set_xlabel("Maximum matching distance in pixel")
        ax[1].set_xlabel("colocalization distance $t$ in pixels")
        ax[1].set_ylabel("relative abundance")
        box = ax[1].get_position()
        ax[1].set_position([box.x0, box.y0, box.width * 0.85, box.height])
        ax[1].grid(axis="both")
        ax[1].set_xlim((maxdist_range[0], maxdist_range[-1]))
        ax[1].set_ylim((-0.00, 1.00))

        def prob_to_abs(p):
            return p * n_y 
        def abs_to_prob(a):
            return a /  n_y
        secax = ax[1].secondary_yaxis('right', functions=(prob_to_abs, abs_to_prob))
        secax.set_ylabel("Number of points")
    
        title_z = "Channel C"
        ax[2].set_title(title_z)
        ax[2].plot(maxdist_range, n_abc/n_z, label="ABC", color="tab:orange")
        ax[2].plot(maxdist_range, n_bc/n_z, label="BC", color="tab:blue")
        ax[2].legend(loc="upper left")
        ax[2].set_xlabel("Maximum matching distance in pixel")
        ax[2].set_xlabel("colocalization distance $t$ in pixels")
        ax[2].set_ylabel("relative abundance")
        box = ax[2].get_position()
        ax[2].set_position([box.x0, box.y0, box.width * 0.85, box.height])
        ax[2].grid(axis="both")
        ax[2].set_xlim((maxdist_range[0], maxdist_range[-1]))
        ax[2].set_ylim((-0.00, 1.00))
        def prob_to_abs(p):
            return p * n_z 
        def abs_to_prob(a):
            return a /  n_z
        secax = ax[2].secondary_yaxis('right', functions=(prob_to_abs, abs_to_prob))
        secax.set_ylabel("Number of points")
        plt.show()

