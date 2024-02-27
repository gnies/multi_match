from .min_cost_flow import min_cost_flow_ortools as mcf
import numpy as np
from scipy.spatial.distance import cdist

def maximal_pair_match(cost, maxdist):
    """Match maximal number of pairs x-y under a certain distance"""
    n_x, n_y = cost.shape
    if n_x==0 or n_y==0:
        res = [], []
    else:
        I, J = np.where(cost <= maxdist)
        if len(I)==0:
            res = [], []
        else:
            # nodes = np.arange(n_x + n_y)
            supplies = np.hstack([np.ones(n_x), -np.ones(n_y)])
            start_nodes = I 
            end_nodes = J + n_x

            supplies = np.hstack([np.ones(n_x), -np.ones(n_y)])
            costs = cost[I, J]
            capacities = np.ones_like(costs)

            flow = mcf(start_nodes, end_nodes, costs, capacities, supplies, max_flow=True)
            res = I[np.where(flow>0)], J[np.where(flow>0)]
    return res

def penalized_pair_match(cost, maxdist, lam):
    n_x, n_y = cost.shape
    if n_x==0 or n_y==0:
        res = [], []
    else:
        I, J = np.where(cost <= maxdist)
        if len(I)==0:
            res = [], []
        else:
            # nodes = np.arange(n_x + n_y)
            supplies = np.hstack([np.ones(n_x), -np.ones(n_y)])
            start_nodes = I 
            end_nodes = J + n_x

            supplies = np.hstack([np.zeros(n_x), np.zeros(n_y)])
            costs = cost[I, J]
            capacities = np.ones_like(costs)
            
            # add start node indexed by (n_x + n_y) and connect it to all x nodes
            supplies = np.append(n_x+n_y, supplies)
            start_nodes = np.hstack([start_nodes, (n_x+n_y)*np.ones(n_x+n_y)])
            end_nodes = np.hstack([end_nodes, np.arange(n_x)])
            costs = np.hstack([costs, np.zeros(n_x)])
            capacities = np.hstack([capacities, np.ones(n_x)])

            # add end node indexed by (n_x + n_y + 1) and connect it to all y nodes
            supplies = np.append(supplies, -n_x-n_y)
            start_nodes = np.hstack([start_nodes, np.arange(start=n_x, stop=n_x+n_y)])
            end_nodes = np.hstack([end_nodes, (n_x+n_y+1)*np.ones(n_y)])
            costs = np.hstack([costs, np.zeros(n_y)])
            capacities = np.hstack([capacities, np.ones(n_y)])

            # add edge connecting start node to end node
            start_nodes = np.append(start_nodes, n_x+n_y)
            end_nodes = np.append(end_nodes, n_x+n_y+1)
            costs = np.append(costs, lam)
            capacities = np.append(capacities, n_x+n_y)

            flow = mcf(start_nodes, end_nodes, costs, capacities, supplies, max_flow=False)
            # remove the start and end node from the result by keeping only the edges between x and y
            flow = flow[:len(I)]

            res = I[np.where(flow>0)], J[np.where(flow>0)]
    return res

def create_partial_match_matrix(cost, I, J):
    """contains a list of points matched, a point i matched to the trash can will be saved as [i, -1], where -1 is the index of the bin"""
    nx, ny = cost.shape
    mask_I = np.zeros(nx, dtype=bool)
    mask_J = np.zeros(ny, dtype=bool)
    mask_I[I] = True
    mask_J[J] = True
    I_s = np.arange(nx)[~mask_I]
    J_s = np.arange(ny)[~mask_J]

    IJ_pairs = np.vstack([I, J]).T

    I_bin = - np.ones_like(I_s, int)
    I_singlets = np.stack([I_s, I_bin]).T

    J_bin = - np.ones_like(J_s, int)
    J_singlets = np.stack([J_bin, J_s]).T

    IJ_match = np.vstack([IJ_pairs, I_singlets, J_singlets])
    return IJ_match
