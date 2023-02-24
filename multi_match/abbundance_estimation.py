import numpy as np
from itertools import chain as iter_chain
from itertools import combinations

def construct_Theta(possible_objects, color_channel_num):
    power_s = powerset([i for i in range(color_channel_num)])
    n = len(possible_objects)
    m = 2**color_channel_num
    Theta = np.zeros((n, m))
    for j in range(m):
        obj = next(power_s)

        # split object into chains
        obj_chains = [[]]
        k = 0
        while k < len(obj):
            channel = obj[k]
            prev_channel = channel-1
            current_chain = obj_chains[-1]
            if len(current_chain) == 0:
                obj_chains[-1].append(channel)
            elif current_chain[-1] == prev_channel:
                obj_chains[-1].append(channel)
            else:
                obj_chains.append([channel])
            k = k+1
        obj_chains = tuple([tuple(chain) for chain in obj_chains])
        for i in range(n):
            detectable = possible_objects[i]
            if detectable in obj_chains:
                Theta[i, j] = 1
    return Theta

def construct_mu(possible_objects, color_channel_num, s_list):
    power_s = powerset([i for i in range(color_channel_num)])
    n = len(possible_objects)
    m = 2**color_channel_num
    mut = np.zeros((n, m))
    for j in range(m):
        observable = next(power_s)
        for i in range(n):
            obj = possible_objects[i]
            if set(observable).issubset(obj):
                detected = set(observable).intersection(obj)
                undetected = set(obj).difference(detected)
                val = 1
                for k in detected:
                    val = val*s_list[k]
                for k in undetected:
                    val = val*((1-s_list[k]))
                mut[i, j] = val
    return mut.T

def _estimate_abundance(W, possible_objects, color_channel_num, s_list, enforce_positive_integers=True):
    mu = construct_mu(possible_objects, color_channel_num, s_list)
    Theta = construct_Theta(possible_objects, color_channel_num)
    theta_mu = Theta @ mu
    theta_mu_inverse = np.linalg.inv(theta_mu)
    N = W @ theta_mu_inverse.T 

    if enforce_positive_integers:
        N = np.rint(N)   # round to integer
        N = np.maximum(N, 0) # no negative values
        N = N.astype(int)
    return N

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return iter_chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

