import numpy as np
from scipy.stats import chi2

def compute_confidence_ellipsiod_from_data(N_data, s_A, s_B, s_C, alpha):
    # getting all values
    theta_mu_inverse = np.array([
                      [1/(s_A*s_B*s_C)               , 0                 , 0                 , 0    , 0     , 0     ],
                      [(s_C-1)/(s_A*s_B*s_C)         , 1/(s_A*s_B)       , 0                 , 0    , 0     , 0     ],
                      [(s_A-1)/(s_A*s_B*s_C)         , 0                 , 1/(s_B*s_C)       , 0    , 0     , 0     ],
                      [(s_B-1)/(s_A*s_B)             , (s_B-1)/(s_A*s_B) , 0                 , 1/s_A, 0     , 0     ],
                      [(s_A-1)*(s_C-1)/(s_A*s_B*s_C) , (s_A-1)/(s_A*s_B) , (s_C-1)/(s_B*s_C) , 0    , 1/s_B , 0     ],
                      [(s_B-1)/(s_B*s_C)             , 0                 , (s_B-1)/(s_B*s_C) , 0    , 0     , 1/s_C ]
                      ])
    
    # probablity vectors of multinomial distributions
    p_ABC = np.array((s_A*s_B*s_C,
        s_A*s_B*(1-s_C),
        (1-s_A)*s_B*s_C,
        s_A*(1-s_B)*s_C,
        s_A*(1-s_B)*(1-s_C),
        (1-s_A)*s_B*(1-s_C),
        (1-s_A)*(1-s_B)*s_C,
        (1-s_A)*(1-s_B)*(1-s_C)))
    
    p_AB = np.array((0,
        s_A*s_B,
        0,
        0,
        s_A*(1-s_B),
        (1-s_A)*s_B,
        0,
        (1-s_A)*(1-s_B)))
    
    p_BC = np.array((0,
        0,
        s_B*s_C,
        0,
        0,
        s_B*(1-s_C),
        (1-s_B)*s_C,
        (1-s_B)*(1-s_C)))
    
    p_A = np.array((0,
        0,
        0,
        0,
        s_A,
        0,
        0,
        (1-s_A)))
    
    p_B = np.array((0,
        0,
        0,
        0,
        0,
        s_B,
        0,
        (1-s_B)))
    
    p_C = np.array((0,
        0,
        0,
        0,
        0,
        0,
        s_C,
        (1-s_C)))
    
    mu = np.vstack([p_ABC, p_AB, p_BC, p_A, p_B, p_C]).T
    theta = np.array(((1,0,0,0,0,0,0,0),
                      (0,1,0,0,0,0,0,0),
                      (0,0,1,0,0,0,0,0),
                      (0,0,0,1,1,0,0,0),
                      (0,0,0,0,0,1,0,0),
                      (0,0,0,1,0,0,1,0)))
          
    theta_mu = theta @ mu
    
    # small check
    # print("Error of inversion = ", np.sum(np.abs(np.matmul(theta_mu_inverse, theta_mu)-np.eye(len(theta_mu)))))
    
    (n_ABC, n_AB, n_BC, n_A, n_B, n_C) = N_data
    s = np.diag(n_ABC*p_ABC +
              n_AB*p_AB +
              n_BC*p_BC +
              n_A*p_A +
              n_B*p_B +
              n_C*p_C
              ) - (n_ABC*np.outer(p_ABC, p_ABC.T) +
              n_AB * np.outer(p_AB , p_AB.T) +
              n_BC * np.outer(p_BC , p_BC.T) +
              n_A  * np.outer(p_A  , p_A.T) +
              n_B  * np.outer(p_B  , p_B.T) +
              n_C  * np.outer(p_C  , p_C.T)
              )
    # print(np.linalg.eigvals(s))
    covariance_matrix = theta_mu_inverse @ theta @ s @ theta.T @ theta_mu_inverse.T
    rank = np.linalg.matrix_rank(covariance_matrix)
    ellipsoid_matrix = np.linalg.pinv(covariance_matrix, hermitian=True)
    ellipsoid_treshhold = chi2.ppf(1-alpha, df=rank)
    return {"ellipsoid_matrix":ellipsoid_matrix, "threshold":ellipsoid_treshhold}
