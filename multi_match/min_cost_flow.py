from ortools.graph.python import min_cost_flow
import numpy as np


def min_cost_flow_ortools(start_nodes, end_nodes, costs, capacities,
        supplies, max_flow=False, return_flow=True):
    """ Solve the min cost flow problem using the ortools-solver after converting the costs to integers."""
    int_costs, scaling_factor = scale_to_int(costs)

    # Instantiate a SimpleMinCostFlow solver.
    smcf = min_cost_flow.SimpleMinCostFlow()

    # setting edges
    for k in range(len(start_nodes)):
        start_node = int(start_nodes[k])
        end_node = int(end_nodes[k])
        capacity = int(capacities[k])
        cost = int(int_costs[k])
        # print(start_node, end_node, capacity, cost)
        smcf.add_arcs_with_capacity_and_unit_cost(start_node, end_node, capacity, cost)

    # setting supply
    for k in range(len(supplies)):
        node = k 
        supply = int(supplies[k])
        # print(node, supply)
        smcf.set_node_supply(node, supply)

    # solve network flow problem
    if max_flow == True:
        check = smcf.solve_max_flow_with_min_cost()
    else:
        check =  smcf.solve()

    if check != smcf.OPTIMAL:
        raise Exception('There was an issue with the min cost flow input.')

    if return_flow:
        # get flow solution
        flows = []
        for k in range(len(start_nodes)):
            flow = smcf.flow(k)
            flows.append(flow)
        res = np.array(flows)
    else:
        res = smcf.optimal_cost() * scaling_factor_for_cost * scaling_factor_for_supplies
    return res

def isinteger(x):
    return np.all(np.equal(np.mod(x, 1), 0))

def scale_to_int(cost, max_int=2**16-2):
    """ Convert float cost to numpy int32 by rescaling to a certain large maximal integer value and rounding (but only if necessary)."""
    cost = np.array(cost) # convert to numpy
    if isinteger(cost):
        scaling_factor = 1
        int_cost = cost
    else:
        max_abs_cost = np.max(np.abs(cost))
        scaling_factor = (max_int/max_abs_cost)
        int_cost = cost*scaling_factor

        int_cost = np.rint(int_cost) # covnert to integers

    int_cost = int_cost.astype(np.int32).tolist()
    return int_cost, 1/scaling_factor


