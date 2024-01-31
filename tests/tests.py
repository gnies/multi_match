import unittest
import numpy as np
import multi_match
import sys


# Note, many of these tests rely on the numpy set seed functionality

class TestSum(unittest.TestCase):

        def test_min_cost_flow(self):
                # small test example
                supplies = [20, 0, 0, -5, -15]

                start_nodes = [0, 0, 1, 1, 1, 2, 2, 3, 4]
                end_nodes = [1, 2, 2, 3, 4, 3, 4, 4, 2]

                capacities = [15, 8, 20, 4, 10, 15, 4, 20, 5]
                costs = [4, 4, 2, 2, 6, 1, 3, 2, 3]

                # Define an array of supplies at each node.

                flows = multi_match.min_cost_flow.min_cost_flow_ortools(start_nodes,
                        end_nodes, costs, capacities, supplies,
                        max_flow=False, return_flow=True)
                np.testing.assert_almost_equal(np.sum(flows*costs), 150,
                err_msg="min cost flow problem is not being solved correctly", decimal=5)

        def test_object_counting(self):
                np.random.seed(2)
                ns = [11, 23, 13, 9]
                max_dist = 0.1
                point_lst = [np.random.random(size= (n_j, 2)) for n_j in ns]
                match = multi_match.Multi_Matching(point_lst, max_dist=max_dist,
                        method="pairwise")
                # And count the number of different objects in the image:
                num_obj = match.count_objects()
                true_num_obj = {'w_ABCD': 1, 'w_ABC': 3, 'w_BCD': 2, 'w_AB': 2,
                        'w_BC': 4, 'w_CD': 1, 'w_A': 5, 'w_B': 11,
                        'w_C': 2, 'w_D': 5}
                assert true_num_obj == num_obj
                
        # def test_costum_cost_pairwise(self):
        #         from scipy.spatial.distance import cdist
        #         np.random.seed(2)
        #         ns = [11, 23, 13, 9]
        #         max_dist = 0.1
        #         point_lst = [np.random.random(size= (n_j, 2)) for n_j in ns]
        #         cost_lst = [cdist(point_lst[i], point_lst[i+1]) for i in range(len(point_lst)-1)]
        #         match = multi_match.Multi_Matching(point_lst, cost_matrix_lst=cost_lst, max_dist=max_dist,
        #         method="pairwise")
        #         # And count the number of different objects in the image:
        #         num_obj = match.count_objects()
        #         true_num_obj = {'w_ABCD': 1, 'w_ABC': 3, 'w_BCD': 2, 'w_AB': 2,
        #         'w_BC': 4, 'w_CD': 1, 'w_A': 5, 'w_B': 11,
        #         'w_C': 2, 'w_D': 5}
        #         assert true_num_obj == num_obj

        # def test_costum_cost_triplets(self):
        #         from scipy.spatial.distance import cdist
        #         np.random.seed(2)
        #         ns = [11, 23, 13]
        #         max_dist = 0.1
        #         point_lst = [np.random.random(size= (n_j, 2)) for n_j in ns]
        #         cost_lst = [cdist(point_lst[i], point_lst[i+1]) for i in range(len(point_lst)-1)]
        #         match1 = multi_match.Multi_Matching(point_lst, max_dist=max_dist, cost_matrix_lst=cost_lst,
        #         method="triplets first")
        #         match2 = multi_match.Multi_Matching(point_lst, max_dist=max_dist, cost_matrix_lst=None, 
        #         method="triplets first")
        #         # And count the number of different objects in the image:
        #         num_obj = match1.count_objects()
        #         true_num_obj = match2.count_objects()
        #         assert true_num_obj == num_obj

if __name__ == '__main__':
        unittest.main()


