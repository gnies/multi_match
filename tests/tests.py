import unittest
import numpy as np
import multi_match

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


if __name__ == '__main__':
    unittest.main()


