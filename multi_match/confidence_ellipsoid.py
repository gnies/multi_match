import numpy as np
from scipy.stats import chi2
from .abbundance_estimation import construct_Theta, construct_mu 

def construct_Sigma_n(mu, N):
    # for each collumn
    n, m = mu.shape
    Sigma_n = np.zeros((n, n)) 
    for j in range(m):
        pi = mu[:, j]
        diagpi = np.diag(pi)
        pipit = np.einsum("i,j->ij", pi, pi)
        nj = N[j]
        Sigma_n = Sigma_n + nj*(diagpi - pipit)
    return Sigma_n

def compute_confidence_ellipsiod_from_data(N, s_list, color_channel_num, possible_objects, test_alpha):
    mu = construct_mu(possible_objects, color_channel_num, s_list)
    Theta = construct_Theta(possible_objects, color_channel_num)
    theta_mu = Theta @ mu
    theta_mu_inverse = np.linalg.inv(theta_mu)
    S = construct_Sigma_n(mu, N)

    covariance_matrix = theta_mu_inverse @ Theta @ S @ Theta.T @ theta_mu_inverse.T
    rank = np.linalg.matrix_rank(covariance_matrix)
    ellipsoid_matrix = np.linalg.pinv(covariance_matrix, hermitian=True)
    ellipsoid_treshhold = chi2.ppf(1-test_alpha, df=rank)
    return {"ellipsoid_matrix":ellipsoid_matrix, "threshold":ellipsoid_treshhold}

def compute_confidence_box(N, s_list, color_channel_num, possible_objects, test_alpha): 
    mu = construct_mu(possible_objects, color_channel_num, s_list)
    Theta = construct_Theta(possible_objects, color_channel_num)
    theta_mu = Theta @ mu
    theta_mu_inverse = np.linalg.inv(theta_mu)
    S = construct_Sigma_n(mu, N)

    covariance_matrix = theta_mu_inverse @ Theta @ S @ Theta.T @ theta_mu_inverse.T
    rank = np.linalg.matrix_rank(covariance_matrix)
    ellipsoid_treshhold = chi2.ppf(1-test_alpha, df=rank)
    dvals = np.diag(covariance_matrix)
    box = []
    for i in range(len(dvals)):
        l = N[i] - ellipsoid_treshhold/np.sqrt(dvals[i])
        r = N[i] + ellipsoid_treshhold/np.sqrt(dvals[i])
        box.append((l, r))
    return box

