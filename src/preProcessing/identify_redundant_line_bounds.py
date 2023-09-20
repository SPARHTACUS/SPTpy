"""
@author: Colonetti
"""

from sys import executable
from time import time
from mpi4py import MPI
import numpy as np

from network import get_buses_bounds_on_injections
from preProcessing.identify_redund_flows_DC import _remove_redundant_flow_limits_angles

def remove_redundant_flow_limits_without_opt(params, hydros, thermals, network):
    """
        try to identify redundant flow bounds by only looking at the line's sensibilities to
        power injections and the minimum and maximum power injections of each bus
        in this approach, the optimization models can be solved analytically just by looking
        at the sign of the PTDF coefficient of the line and then accordingly setting
        the bus' power injection to either its minimum or maximum
    """

    time_0 = time()

    new_unreachable_bounds = 0

    assert network.PTDF.shape == (len(network.LINE_ID), len(network.BUS_ID)),\
            ("The shape of the numpy array network.PTDF must be " +
            f"({len(network.LINE_ID)}, {len(network.BUS_ID)}): the number of rows is the number "+
                "and the number of columns equals the number of buses")

    line_sensitivities_arr = network.PTDF[:]
    line_sensitivities_arr[np.where(abs(network.PTDF) < params.PTDF_COEFF_TOL)] = 0

    neg_coeff, pos_coeff = {}, {}
    for l_idx in [l_idx for l_idx in range(len(network.LINE_ID))
                                                if network.ACTIVE_BOUNDS[network.LINE_ID[l_idx]]]:
        l = network.LINE_ID[l_idx]

        neg_coeff[l] = np.where(line_sensitivities_arr[l_idx, :] < 0)[0]
        pos_coeff[l] = np.where(line_sensitivities_arr[l_idx, :] > 0)[0]

    # get the bounds on the injections at each bus
    (_0, _1, min_inj_per_period, max_inj_per_period) = get_buses_bounds_on_injections(
                                                                params, network, thermals, hydros)

    for l in network.LINK_ID:
        for t in range(params.T):
            min_inj_per_period[network.LINK_F_T[l][0]][t] -= network.LINK_MAX_P[l]
            max_inj_per_period[network.LINK_F_T[l][0]][t] += network.LINK_MAX_P[l]
            min_inj_per_period[network.LINK_F_T[l][1]][t] -= network.LINK_MAX_P[l]
            max_inj_per_period[network.LINK_F_T[l][1]][t] += network.LINK_MAX_P[l]

    innactive_lines_per_period = {t: 0 for t in range(params.T)}

    for t in range(params.T):
        p_inj_ub = np.array([max_inj_per_period[bus][t] for bus in network.BUS_ID], dtype = 'd')
        p_inj_lb = np.array([min_inj_per_period[bus][t] for bus in network.BUS_ID], dtype = 'd')

        for l_idx in [l_idx for l_idx in range(len(network.LINE_ID))
                                                if network.ACTIVE_BOUNDS[network.LINE_ID[l_idx]]]:
            l = network.LINE_ID[l_idx]

            # minimize the flow in the line (i.e., try to make the flow as negative as possible)
            objVal = np.inner(line_sensitivities_arr[l_idx, neg_coeff[l]], p_inj_ub[neg_coeff[l]])+\
                    np.inner(line_sensitivities_arr[l_idx, pos_coeff[l]], p_inj_lb[pos_coeff[l]])

            if objVal > (network.LINE_FLOW_LB[l][t] + 1e-18):
                # the lower bound cannot possibly be reached in this period
                network.ACTIVE_LB_PER_PERIOD[l][t] = False

            # now maximize the flow
            objVal = (
                    np.inner(line_sensitivities_arr[l_idx,neg_coeff[l]],p_inj_lb[neg_coeff[l]])+
                    np.inner(line_sensitivities_arr[l_idx,pos_coeff[l]],p_inj_ub[pos_coeff[l]]))

            if objVal < (network.LINE_FLOW_UB[l][t] - 1e-18):
                innactive_lines_per_period[t] += 1 # not binding for this period
                network.ACTIVE_UB_PER_PERIOD[l][t] = False

    old_active_bounds = {l: network.ACTIVE_BOUNDS[l] for l in network.LINE_ID}

    for l in network.LINE_ID:
        network.ACTIVE_UB[l] = any(network.ACTIVE_UB_PER_PERIOD[l].values())

    for l in network.LINE_ID:
        network.ACTIVE_LB[l] = any(network.ACTIVE_LB_PER_PERIOD[l].values())

    for l in network.LINE_ID:
        network.ACTIVE_BOUNDS[l] = max(network.ACTIVE_UB[l], network.ACTIVE_LB[l])

    time_end = time()

    new_unreachable_bounds = len([l for l in network.LINE_ID
                                            if old_active_bounds[l] != network.ACTIVE_BOUNDS[l]])

    print(f"\nThe total time in remove_redundant_flow_limits_without_opt is {time_end-time_0:,.4f}"
            f" seconds.\nThis function has identified {new_unreachable_bounds} more bounds that "
            + "can be removed.", flush = True)


def _create_list_of_jobs(params, network) -> list:
    """
        organize the list of lines in such a way that lines connected to buses who have few
        connections are prioritized
    """
    possibly_binding_lines = [l for l in network.LINE_ID if network.ACTIVE_BOUNDS[l]]
    end_points_n_connecs = {l: (len(network.LINES_FROM_BUS[network.LINE_F_T[l][0]]
                                            + network.LINES_TO_BUS[network.LINE_F_T[l][0]]),
                                len(network.LINES_FROM_BUS[network.LINE_F_T[l][1]]
                                            + network.LINES_TO_BUS[network.LINE_F_T[l][1]]))
                                                for l in possibly_binding_lines}
    complete_list_jobs = len(possibly_binding_lines)*[-1e12]

    lines_per_process = int(len(complete_list_jobs)/params.MAX_PROCES)

    i, p, count_all = 0, 0, 0
    for nc in range(1,max(n for l in possibly_binding_lines for n in end_points_n_connecs[l])+1,1):
        for l in [l for l in possibly_binding_lines if nc in end_points_n_connecs[l]]:
            if l not in complete_list_jobs:
                count_all += 1
                if count_all > lines_per_process*params.MAX_PROCES:
                    complete_list_jobs[count_all - 1] = l
                else:
                    complete_list_jobs[i + p*lines_per_process] = l
                    p += 1
                    if p == params.MAX_PROCES:
                        p = 0
                        i += 1
    return complete_list_jobs

def _get_back_flags(params, network, CHILD_COMM):
    """
        get back from the spawn workers the flags indicating whether the line bounds can be binding
        and then update the flags
    """
    aux_active_bounds = np.array([int(v) for v in network.ACTIVE_BOUNDS.values()], dtype='int')
    CHILD_COMM.Reduce(None, [aux_active_bounds, MPI.INT], op = MPI.MIN, root = MPI.ROOT)

    aux_ubs = np.array([int(v) for v in network.ACTIVE_UB.values()], dtype='int')
    CHILD_COMM.Reduce(None, [aux_ubs, MPI.INT], op = MPI.MIN, root = MPI.ROOT)

    aux_lbs = np.array([int(v) for v in network.ACTIVE_LB.values()], dtype='int')
    CHILD_COMM.Reduce(None, [aux_lbs, MPI.INT], op = MPI.MIN, root = MPI.ROOT)

    aux_actibe_ubs_per_period = np.zeros((len(network.LINE_ID), params.T), dtype = 'int')
    CHILD_COMM.Reduce(None, [aux_actibe_ubs_per_period, MPI.INT], op = MPI.MIN, root = MPI.ROOT)

    aux_actibe_lbs_per_period = np.zeros((len(network.LINE_ID), params.T), dtype = 'int')
    CHILD_COMM.Reduce(None, [aux_actibe_lbs_per_period, MPI.INT], op = MPI.MIN, root = MPI.ROOT)

    # convert the results back to bool. note that bool(1) = True, and bool(0) = False
    # similarly, int(True) = 1
    i = 0
    for l in network.ACTIVE_BOUNDS.keys():
        network.ACTIVE_BOUNDS[l] = bool(aux_active_bounds[i])
        i += 1

    i = 0
    for l in network.ACTIVE_UB.keys():
        network.ACTIVE_UB[l] = bool(aux_ubs[i])
        i += 1

    i = 0
    for l in network.ACTIVE_LB.keys():
        network.ACTIVE_LB[l] = bool(aux_lbs[i])
        i += 1

    i = 0
    for l in network.ACTIVE_UB_PER_PERIOD.keys():
        for t in range(params.T):
            network.ACTIVE_UB_PER_PERIOD[l][t] = bool(aux_actibe_ubs_per_period[i][t])
        i += 1

    i = 0
    for l in network.ACTIVE_LB_PER_PERIOD.keys():
        for t in range(params.T):
            network.ACTIVE_LB_PER_PERIOD[l][t] = bool(aux_actibe_lbs_per_period[i][t])
        i += 1

def redundant_line_bounds(params, hydros, thermals, network,
                                time_limit:float = 360,
                                    run_single_period_models:bool = True):
    """
        Through a series of steps, try to identify line flow limits that can never be reached, and
        thus are redundant and can be removed from the model
    """

    time_limit = float(time_limit)

    complete_list_jobs = _create_list_of_jobs(params, network)

    # initial number of redundant transmission line bounds
    i_redund_b = len([l for l in network.LINE_ID if not(network.ACTIVE_BOUNDS[l])])

    t_0 = time()

    if params.MAX_PROCES > 1:
        # spawn at most params.MAX_PROCES child processes
        CHILD_COMM = MPI.COMM_SELF.Spawn(executable,
                                            args = ["preProcessing/identify_redund_flows_DC.py"],
                                                maxprocs = params.MAX_PROCES)
        # share with them the necessary info
        CHILD_COMM.bcast(params, root = MPI.ROOT)
        CHILD_COMM.bcast(hydros, root = MPI.ROOT)
        CHILD_COMM.bcast(network, root = MPI.ROOT)
        CHILD_COMM.bcast(thermals, root = MPI.ROOT)
        CHILD_COMM.bcast(run_single_period_models, root = MPI.ROOT)
        CHILD_COMM.bcast(complete_list_jobs, root = MPI.ROOT)
        CHILD_COMM.bcast(time_limit, root = MPI.ROOT)

        # get back the results
        _get_back_flags(params, network, CHILD_COMM)

        CHILD_COMM.Disconnect()

    else:
        _remove_redundant_flow_limits_angles(params, network, thermals, hydros,
                                                time_limit = time_limit,
                                                list_of_jobs = complete_list_jobs,
                                                print_to_console = True,
                                                run_single_period_models = run_single_period_models)

    # final number of redundant transmission line bounds
    f_redund_b = len([l for l in network.LINE_ID if not(network.ACTIVE_BOUNDS[l])])

    print(f"\n\nIt took {time() - t_0:,.4f} sec to execute identify_redund_flows_DC.py")
    print(f"{f_redund_b-i_redund_b} more redundant line bounds have been identifed\n\n",flush=True)
