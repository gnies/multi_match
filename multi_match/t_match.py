from .min_cost_flow import min_cost_flow_ortools as mcf
import numpy as np
from scipy.spatial.distance import cdist

def maximal_triplet_match(cost_xy, cost_yz, maxdist):
    """Match maximal number of triples x-y-z under a certain distance"""
    n_x, n_y = cost_xy.shape
    n_y, n_z = cost_yz.shape

    if n_x==0 or n_y==0 or n_z==0:
        res = [], [], []
    else:
        I, J1 = np.where(cost_xy <= maxdist)
        J2, K = np.where(cost_yz <= maxdist)
        if len(I)==0 or len(J2)==0:
            res = [], [], []
        else:
            # the y-layer needs to be double in order to enforce marginal constrain
            supplies = np.hstack([np.ones(n_x), np.zeros(2*n_y), -np.ones(n_z)])
    
            start_nodes = np.hstack([I, np.arange(n_y)+n_x, J2+n_x+n_y])
            end_nodes = np.hstack([J1+n_x, np.arange(n_y)+n_x+n_y, K+n_x+2*n_y])
            costs = np.hstack([cost_xy[I, J1], np.zeros(n_y), cost_yz[J2, K]])
            capacities = np.ones_like(costs)
    
            flow = mcf(start_nodes, end_nodes, costs, capacities, supplies, max_flow=True)
            A = I[np.where(flow[:len(I)]>0)]
            B1 = J1[np.where(flow[:len(I)]>0)]
    
            B2 = J2[np.where(flow[len(I)+n_y: len(I)+n_y+len(J2)]>0)]
            C = K[np.where(flow[len(I)+n_y: len(I)+n_y+len(J2)]>0)]
            # the index pairs determined by A, B1 and B2, C are converted to triplets of indicies by reordering
            A, B, C = glue_matches(A, B1, B2, C)
            res = A, B, C
    return res

def penalized_triplet_match(cost_xy, cost_yz, maxdist, lam):
    n_x, n_y = cost_xy.shape
    n_y, n_z = cost_yz.shape

    if n_x==0 or n_y==0 or n_z==0:
        res = [], [], []
    else:
        I, J1 = np.where(cost_xy <= maxdist)
        J2, K = np.where(cost_yz <= maxdist)
        if len(I)==0 or len(J2)==0:
            res = [], [], []
        else:
            # the y-layer needs to be double in order to enforce marginal constrain
            supplies = np.hstack([np.zeros(n_x), np.zeros(2*n_y), -np.zeros(n_z)])
            start_nodes = np.hstack([I, np.arange(n_y)+n_x, J2+n_x+n_y])
            end_nodes = np.hstack([J1+n_x, np.arange(n_y)+n_x+n_y, K+n_x+2*n_y])
            costs = np.hstack([cost_xy[I, J1], np.zeros(n_y), cost_yz[J2, K]])
            capacities = np.ones_like(costs)
            
            # add start node indexed by (n_x + n_y + n_z) and connect it to all x nodes
            supplies = np.append(supplies, np.min([n_x, n_y, n_z]))
            start_nodes = np.hstack([start_nodes, (n_x+n_y+n_z)*np.ones(n_x)])
            end_nodes = np.hstack([end_nodes, np.arange(n_x)])
            costs = np.hstack([costs, np.zeros(n_x)])
            capacities = np.hstack([capacities, np.ones(n_x)])

            # add end node indexed by (n_x + n_y + n_z + 1) and connect it to all z nodes
            supplies = np.append(supplies, -np.min([n_x, n_y, n_z]))
            start_nodes = np.hstack([start_nodes, np.arange(start=n_x+n_y, stop=n_x+n_y+n_z)])
            end_nodes = np.hstack([end_nodes, (n_x+n_y+n_z+1)*np.ones(n_z)])
            costs = np.hstack([costs, np.zeros(n_z)])
            capacities = np.hstack([capacities, np.ones(n_z)])

            # add edge connecting start node to end node
            start_nodes = np.append(start_nodes, n_x+n_y+n_z)
            end_nodes = np.append(end_nodes, n_x+n_y+n_z+1)
            costs = np.append(costs, lam)
            capacities = np.append(capacities, np.min([n_x, n_y, n_z]))

            flow = mcf(start_nodes, end_nodes, costs, capacities, supplies, max_flow=False)
            A = I[np.where(flow[:len(I)]>0)]
            B1 = J1[np.where(flow[:len(I)]>0)]
    
            B2 = J2[np.where(flow[len(I)+n_y: len(I)+n_y+len(J2)]>0)]
            C = K[np.where(flow[len(I)+n_y: len(I)+n_y+len(J2)]>0)]
            # the index pairs determined by A, B1 and B2, C are converted to triplets of indicies by reordering
            A, B, C = glue_matches(A, B1, B2, C)
            res = A, B, C
    return res


def glue_matches(A, B1, B2, C):
    if len(A)==0:
        res = [], [], []
    else:
        tf = B1[np.newaxis, :]==B2[:, np.newaxis]
        _, V = np.where(tf>0)
        perm = np.argsort(V)
        C_perm = C[perm]
        res = A, B1, C_perm
    return res

