import matplotlib.pyplot as plt
import numpy as np 
from .confidence_ellipsoid import compute_confidence_ellipsiod_from_data

def plot_relative_abundances_estimation(mm, s_a, s_b, s_c, alpha=0.05):
    """ Plots number of matches as a function of the maximal matching distance
        given a data frame."""

    # abundance estimation
    N = mm.estimate_abundances([s_a, s_b, s_c])

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
        # d = compute_confidence_ellipsiod(n_row, , alpha)
        s_list = [s_a, s_b, s_c]
        d = compute_confidence_ellipsiod_from_data(n_row, s_list,
                mm.color_channel_num, mm.possible_objects, alpha)
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

def cumulative_plot_number_of_matches(mm, show=True):
    """ Plots number of matches as a function of the maximal matching distance given a data frame."""
    if mm.number_of_objects is None:
        mm.count_objects()
    data = mm.number_of_objects

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
    if show:
        plt.show()

def plot_relative_abundances(mm):
    """ Plots number of matches as a function of the maximal matching distance given a data frame."""
    # get number of objects
    if mm.number_of_objects is None:
        mm.count_objects()
    N = mm.number_of_objects

    maxdist_range = N["maxdist"].values
    n_abc = N["w_ABC"].values
    n_ab = N["w_AB"].values
    n_bc = N["w_BC"].values
    n_x, n_y, n_z = (len(mm.point_list[0]), len(mm.point_list[1]), len(mm.point_list[2]))

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

def plot_number_of_matches(mm, show=True):
    """ Plots number of matches as a function of the maximal matching distance given a data frame."""
    if mm.number_of_objects is None:
        mm.count_objects()
    data = mm.number_of_objects

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
    if show:
        plt.show()
