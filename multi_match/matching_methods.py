import numpy as np
from multi_match.p_match import maximal_pair_match, create_partial_match_matrix, penalized_pair_match
from multi_match.t_match import maximal_triplet_match, penalized_triplet_match

def match_all(c_xy, c_yz, maxdist, lam):
    """Find first triplets ABC, then pairs AB and BC. """

    n_x, n_y = c_xy.shape
    n_y, n_z = c_yz.shape

    # Find maximal number of triplets and add them
    if lam == "max":
        I, J, K = maximal_triplet_match(c_xy, c_yz, maxdist)
    else:
        I, J, K = penalized_triplet_match(c_xy, c_yz, maxdist, lam)
    if len(I) == 0:
        res = match_pairwise([c_xy, c_yz], maxdist, lam)
    else:
        mask_I = np.zeros(n_x, dtype=bool)
        mask_J = np.zeros(n_y, dtype=bool)
        mask_K = np.zeros(n_z, dtype=bool)
        mask_I[I] = True
        mask_J[J] = True
        mask_K[K] = True
        c_xy = c_xy[~mask_I, :]
        c_xy = c_xy[:, ~mask_J]
        c_yz = c_yz[~mask_J, :]
        c_yz = c_yz[:, ~mask_K]

        sub_match = match_pairwise([c_xy, c_yz], maxdist, lam)

        # convert indicies of match between x_new, y_new and z_new to indicies of x, y and z:
        IJK = np.vstack([I, J, K]).T
        # remove matched points

        converted_match = - np.ones_like(sub_match, dtype=int)
        w0 = np.where(sub_match[:,0] != -1)
        a0 = np.where(~mask_I)[0]
        converted_match[w0, 0] = a0[sub_match[w0, 0]]
        w1 = np.where(sub_match[:,1] != -1)
        a1 = np.where(~mask_J)[0]
        converted_match[w1, 1] = a1[sub_match[w1, 1]]
        w2 = np.where(sub_match[:,2] != -1)
        a2 = np.where(~mask_K)[0]
        converted_match[w2, 2] = a2[sub_match[w2, 2]]

        # adding the triplets
        res = np.vstack([IJK, converted_match])
        res = res.astype(int)
    return res

def match_pairwise(cost_list, maxdist, lam):
    # create all pairwise matchings along the chain
    lst_of_partial_matchings = []
    i = 0
    for c in cost_list:
        if lam == "max":
            I, J = maximal_pair_match(c, maxdist)
        else:
            I, J = penalized_pair_match(c, maxdist, lam)
        IJ = create_partial_match_matrix(c, I, J)
        lst_of_partial_matchings.append(IJ)
        i = i+1

    # glue them all together
    i = 0
    match = lst_of_partial_matchings[0]
    while len(lst_of_partial_matchings[i:])>1:
        match = glue_unbalanced(match, lst_of_partial_matchings[i+1])
        i = i+1
    match = match.astype(int)
    return match

def glue_unbalanced(IJ, JK):
    """Gluing of two unbalanced matches"""
    # we first look for the permutation that correctly "glues" the J indexes of IJ with the ones of JK
    J_loc_IJ = IJ[:, -1] != -1
    J_loc_JK = JK[:, 0] != -1
    permutation_IJ = np.argsort(IJ[J_loc_IJ, -1])
    permutation_JK = np.argsort(JK[J_loc_JK, 0])

    paired = np.hstack([IJ[J_loc_IJ, :][permutation_IJ], JK[J_loc_JK, 1][permutation_JK][:, np.newaxis]])

    # now we just have to append the remaining
    unmached_I = IJ[~ J_loc_IJ, :]
    I_bin = - np.ones((unmached_I.shape[0], 1), int)
    unmached_I = np.hstack([unmached_I, I_bin])

    # now we just have to append the remaining
    unmached_K = JK[~ J_loc_JK, :]
    K_bin = - np.ones((unmached_K.shape[0], IJ.shape[1]-1), dtype=int)
    K_bin.fill(-1)
    unmached_K = np.hstack([K_bin, unmached_K])
    glued = np.vstack([paired, unmached_I, unmached_K])
    return glued
