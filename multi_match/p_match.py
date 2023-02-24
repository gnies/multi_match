from .min_cost_flow import min_cost_flow_ortools as mcf
import numpy as np
from scipy.spatial.distance import cdist

def maximal_pair_match(x, y, maxdist, cost_function="euclidean"):
    """Match maximal number of pairs x-y under a certain distance"""
    n_x = len(x)
    n_y = len(y)
    if n_x==0 or n_y==0:
        res = [], []
    else:
        cost = cdist(x, y, cost_function)
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

def create_partial_match_matrix(x, y, I, J):
    """contains a list of points matched, a point i matched to the trash can will be saved as [i, -1], where -1 is the index of the bin"""
    mask_I = np.zeros(len(x), dtype=bool)
    mask_J = np.zeros(len(y), dtype=bool)
    mask_I[I] = True
    mask_J[J] = True
    I_s = np.arange(len(x))[~mask_I] 
    J_s = np.arange(len(y))[~mask_J]
    IJ_pairs = np.vstack([I, J]).T

    I_bin = - np.ones_like(I_s, int)
    I_singlets = np.stack([I_s, I_bin]).T

    J_bin = - np.ones_like(J_s, int)
    J_singlets = np.stack([J_bin, J_s]).T

    IJ_match = np.vstack([IJ_pairs, I_singlets, J_singlets])
    return IJ_match

if __name__=="__main__":
    np.random.seed(0)

    distance = "euclidean" # ground distance considered
    penal = 0.1 # penalisation parameter lambda
    
    # unbalanced number of points
    n = 20
    m = 18
    
    # select random points
    x = np.random.random(size = (n, 2)) 
    y = np.random.random(size = (m, 2)) 

    maxdist = 0.2 
    res = p_match(x, y, maxdist)
    print(res)
