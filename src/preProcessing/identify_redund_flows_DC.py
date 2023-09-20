"""
@author: Colonetti
"""

import sys
from os import path
from time import time
from mpi4py import MPI
import numpy as np

if __name__ == '__main__':
    ROOT_FOLDER = path.abspath(path.join(__file__ , "../..")).replace("\\","/")

    sys.path.append(ROOT_FOLDER + "/")

from network import get_buses_bounds_on_injections
from solver_interface.opt_model import Model


def _remove_redundant_flow_limits_angles(params, network,
                                            thermals, hydros,
                                                time_limit:float = 360,
                                                    list_of_jobs:list = None,
                                                        print_to_console:bool = True,
                                                            run_single_period_models:bool = True):
    """
        Use the complete DC model to identify more redundant limits
    """

    time_0 = time()
    last_time = time_0 + time_limit

    if list_of_jobs is None:
        list_of_jobs = [l for l in network.LINE_ID if network.ACTIVE_BOUNDS[l]]

    m = Model(model_name = 'm', solver_name = params.SOLVER, package = params.PACKAGE)

    flow = {k:
                m.add_var(
                            lb = np.min(network.LINE_FLOW_LB[k[-1]]),
                            ub = np.max(network.LINE_FLOW_UB[k[-1]]),
                            var_type = 'C', name = f"flow_{k}")
            for k in [(network.LINE_F_T[l][0], network.LINE_F_T[l][1], l) for l in network.LINE_ID
                                                                    if network.ACTIVE_BOUNDS[l]]}

    flow_DC = {k:
                m.add_var(
                            lb = -network.LINK_MAX_P[k[-1]],
                            ub = network.LINK_MAX_P[k[-1]],
                            var_type = 'C', name = f"flow_DC_{k}")
            for k in [(network.LINK_F_T[l][0], network.LINK_F_T[l][1], l)
                            for l in network.LINK_F_T.keys()]}

    theta = {k: m.add_var(lb = -10000, var_type = 'C', name = f"theta_bus_{k}")
                for k in network.BUS_ID}

    for bus in network.REF_BUS_ID:
        m.set_lb(theta[bus], 0)
        m.set_ub(theta[bus], 0)

    for l in [l for l in network.LINE_ID if network.ACTIVE_BOUNDS[l]]:
        ADMT = 1/network.LINE_X[l]
        m.add_constr(flow[network.LINE_F_T[l][0], network.LINE_F_T[l][1], l]
                                    == ADMT*(theta[network.LINE_F_T[l][0]] -
                                        theta[network.LINE_F_T[l][1]]),
                            name=f"AC_flow_{network.LINE_F_T[l][0]}_{network.LINE_F_T[l][1]}_{l}")

    for l in [l for l in network.LINE_ID if not(network.ACTIVE_BOUNDS[l])]:
        ADMT = 1/network.LINE_X[l]
        flow[network.LINE_F_T[l][0],network.LINE_F_T[l][1],l] = (ADMT*(theta[network.LINE_F_T[l][0]]
                                                                - theta[network.LINE_F_T[l][1]]))

    power_inj = {k: m.add_var(
                                lb = 0, ub = 0, obj = 0,
                                    var_type = 'C',
                                        name = f"power_inj_{k}"
                            )
                                for k in network.BUS_ID
                }

    # get the bounds on the injections at each bus
    (min_inj, max_inj,
            min_inj_per_period, max_inj_per_period) = get_buses_bounds_on_injections(
                                                                params, network, thermals, hydros)

    for bus in network.BUS_ID:
        m.set_lb(power_inj[bus], min_inj[bus])
        m.set_ub(power_inj[bus], max_inj[bus])

    power_balance_constrs = []

    for bus in network.BUS_ID:
        power_balance_constrs.append(m.add_constr(power_inj[bus]
                            - m.xsum(flow[network.LINE_F_T[l][0], network.LINE_F_T[l][1], l]
                                                            for l in network.LINES_FROM_BUS[bus])
                            + m.xsum(flow[network.LINE_F_T[l][0], network.LINE_F_T[l][1], l]
                                                            for l in network.LINES_TO_BUS[bus])
                            - m.xsum(flow_DC[network.LINK_F_T[l][0],
                                            network.LINK_F_T[l][1], l]
                                                for l in network.LINKS_FROM_BUS[bus])
                            + m.xsum(flow_DC[network.LINK_F_T[l][0],
                                            network.LINK_F_T[l][1], l]
                                                for l in network.LINKS_TO_BUS[bus])
                                                == 0,
                                                name = f"power_balance_{bus}"))

    m.verbose = False

    test_PTDF = False
    if test_PTDF:
        m.optimize()
        if m.status.value != 0:
            m.write("infeas_angles.lp")
            m.write("infeas_angles.mps")
            raise ValueError("reduce network angle model is infeasible")

        max_diff = 0
        for l_index in range(len(network.LINE_ID)):
            l = network.LINE_ID[l_index]
            non_zeros = np.where(abs(network.PTDF[l_index, :]) > 0)[0]
            flow_ptdf=\
                    sum(network.PTDF[l_index,b_index]*power_inj[network.BUS_ID[b_index]].x
                                                            for b_index in non_zeros)\
                    + sum(flow_DC[network.LINK_F_T[link][0],
                                    network.LINK_F_T[link][1], link].x*(
                network.PTDF[l_index, network.BUS_ID.index(network.LINK_F_T[link][1])]
                -network.PTDF[l_index,network.BUS_ID.index(network.LINK_F_T[link][0])])
                                            for link in network.LINK_ID)
            diff = flow[network.LINE_F_T[l][0], network.LINE_F_T[l][1], l].x - flow_ptdf
            print(diff)
            if abs(diff) > 1:
                print(f"{flow[network.LINE_F_T[l][0], network.LINE_F_T[l][1], l].x} {flow_ptdf}")
                print(f"({network.LINE_F_T[l][0]}, {network.LINE_F_T[l][1]})")
            max_diff = max(max_diff, abs(diff))

        print(f"The maximum difference is {max_diff}")

        raise ValueError("End")

    total_n_jobs = params.T*len(list_of_jobs)
    next_print = 0.001
    new_unreachable_bounds = 0
    counter = 0
    if print_to_console:
        print(f"Total of {total_n_jobs} jobs to perform. " +
                f"(Out of {params.T*len(network.LINE_ID)} jobs.)", flush = True)

    for l in list_of_jobs:

        #### firstly, try to reach the bounds by considering the whole range of the power injections
        #### at the buses. if the bounds cannot be reached under this assumption, then they can
        #### certainly not be reached when the bus injections are limited to the specific
        #### injections of each period. also, in these model, use both the most restrictive
        #### lower bound and the most restrictive upper bound. if these bounds are not reached, then
        #### the ones less restrictive ones also will not be reached

        if network.ACTIVE_LB[l]:
            m.objective = flow[network.LINE_F_T[l][0], network.LINE_F_T[l][1], l]
            m.optimize(max_seconds = max(last_time - time(), 0))

        if not(network.ACTIVE_LB[l]) or (network.ACTIVE_LB[l] and (m.status.value == 0)):
            if (not(network.ACTIVE_LB[l])
                    or (m.objective_value > (np.max(network.LINE_FLOW_LB[l]) + 1e-6))):
                network.ACTIVE_LB[l] = False
                network.ACTIVE_LB_PER_PERIOD[l] = {t: False for t in range(params.T)}
        elif m.status.value == 10000 and (last_time - time() <= 0):
            pass
        else:
            m.write("infeas_angles.lp")
            m.write("infeas_angles.mps")
            raise ValueError("reduce network angle model is infeasible")

        if network.ACTIVE_UB[l]:
            m.objective = -1*flow[network.LINE_F_T[l][0], network.LINE_F_T[l][1], l]
            m.optimize(max_seconds = max(last_time - time(), 0))

        if not(network.ACTIVE_UB[l]) or (network.ACTIVE_UB[l] and (m.status.value == 0)):
            if (not(network.ACTIVE_UB[l])
                    or (-1*m.objective_value < (np.min(network.LINE_FLOW_UB[l]) - 1e-6))):
                network.ACTIVE_UB[l] = False
                network.ACTIVE_UB_PER_PERIOD[l] = {t: False for t in range(params.T)}
        elif m.status.value == 10000 and (last_time - time() <= 0):
            pass
        else:
            m.write("infeas_angles.lp")
            m.write("infeas_angles.mps")
            raise ValueError("reduce network angle model is infeasible")

        if not(network.ACTIVE_LB[l]) and not(network.ACTIVE_UB[l]):
            # if neither the UB nor the LB can be reached
            network.ACTIVE_BOUNDS[l] = False
            new_unreachable_bounds += 1
            counter += params.T

    if not(run_single_period_models):
        return network.ACTIVE_BOUNDS

    # compute the injection bounds removing the load
    min_power_inj_no_load = {bus: {t:
                        (min_inj_per_period[bus][t] + network.NET_LOAD[network.BUS_HEADER[bus], t])
                                            for t in range(params.T)} for bus in network.BUS_ID}
    max_power_inj_no_load = {bus: {t:
                        (max_inj_per_period[bus][t] + network.NET_LOAD[network.BUS_HEADER[bus], t])
                                            for t in range(params.T)} for bus in network.BUS_ID}

    for l in [l for l in list_of_jobs if network.ACTIVE_BOUNDS[l]]:
        for t in [t for t in range(params.T - 1, -1, -1) if network.ACTIVE_LB_PER_PERIOD[l][t] or
                                                                network.ACTIVE_UB_PER_PERIOD[l][t]]:

            for b, bus in enumerate(network.BUS_ID):
                m.set_lb(power_inj[bus], min_power_inj_no_load[bus][t])
                m.set_ub(power_inj[bus], max_power_inj_no_load[bus][t])
                power_balance_constrs[b].rhs = network.NET_LOAD[network.BUS_HEADER[bus], t]

            #### try minimizing the flow, i.e., try reaching the LB
            if network.ACTIVE_LB[l] and network.ACTIVE_LB_PER_PERIOD[l][t]:
                m.objective = flow[network.LINE_F_T[l][0], network.LINE_F_T[l][1], l]
                m.optimize(max_seconds = max(last_time - time(), 0))
            if ((not(network.ACTIVE_LB[l])
                    or not(network.ACTIVE_LB_PER_PERIOD[l][t]))
                        or m.status.value == 0):
                if ((not(network.ACTIVE_LB[l])
                        or not(network.ACTIVE_LB_PER_PERIOD[l][t]))
                            or m.objective_value > (network.LINE_FLOW_LB[l][t] + 1e-6)):
                    # then the LB of line l in period t cannot be reached
                    network.ACTIVE_LB_PER_PERIOD[l][t] = False
            elif m.status.value == 10000 and (last_time - time() <= 0):
                break
            else:
                m.write("infeas_angles.lp")
                m.write("infeas_angles.mps")
                raise ValueError("reduce network angle model is infeasible")

            #### now try maximizing the flow, i.e., try reaching the UB
            if network.ACTIVE_UB[l] and network.ACTIVE_UB_PER_PERIOD[l][t]:
                m.objective = -1*flow[network.LINE_F_T[l][0], network.LINE_F_T[l][1], l]
                m.optimize(max_seconds = max(last_time - time(), 0))

            if ((not(network.ACTIVE_UB[l])
                    or not(network.ACTIVE_UB_PER_PERIOD[l][t]))
                        or m.status.value == 0):
                if ((not(network.ACTIVE_UB[l])
                        or not(network.ACTIVE_UB_PER_PERIOD[l][t]))
                            or -1*m.objective_value < (network.LINE_FLOW_UB[l][t] - 1e-6)):
                    # then the UB of line l in period t cannot be reached
                    network.ACTIVE_UB_PER_PERIOD[l][t] = False

            elif m.status.value == 10000 and (last_time - time() <= 0):
                break
            else:
                m.write("infeas_angles.lp")
                m.write("infeas_angles.mps")
                raise ValueError("reduce network angle model is infeasible")

            counter += 1

        network.ACTIVE_LB[l] = any(network.ACTIVE_LB_PER_PERIOD[l].values())
        network.ACTIVE_UB[l] = any(network.ACTIVE_UB_PER_PERIOD[l].values())
        network.ACTIVE_BOUNDS[l] = max(network.ACTIVE_LB[l], network.ACTIVE_UB[l])
        new_unreachable_bounds += 1
        counter += 1

        if (counter/total_n_jobs) >= next_print and print_to_console:
            print(f"{100*((counter/total_n_jobs)):,.4f}% of all lines done. "+
                                f"{new_unreachable_bounds} more unreachable bounds " +
                                f"found in {time()-time_0:,.4f} sec"+
                                f". Average of {(time()-time_0)/(counter/params.T):,.4f} seconds " +
                                f"per line for {counter} jobs "+
                                "performed", flush = True)
            next_print += 0.001

    del m

    time_end = time()

    if print_to_console:
        print(f"\nTotal time in _remove_redundant_flow_limits_angles is {time_end-time_0:,.4f} sec"+
                            f" {new_unreachable_bounds} more bounds can be removed.", flush = True)

def _initialize_child_processes(run_single_period_models:bool = None):
    """
        child processes have been spawned and here they get their intercommunicator with their
        parent process, their size and ranks, and also receive through broadcasting the necessary
        data to run their tasks


    """
    CHILD_COMM_ = MPI.Comm.Get_parent() # the intercommunicator to communicate with the parent p
    SIZE_ = CHILD_COMM_.Get_size()       # size in the intercommunicator
    CHILD_RANK_ = CHILD_COMM_.Get_rank() # rank of the process in the intercommunicator
                                        # the rank goes from 0 to (SIZE - 1)

    params_, thermals_, network_, hydros_ = 4*[None]

    params_ = CHILD_COMM_.bcast(params_, root = 0)
    hydros_ = CHILD_COMM_.bcast(hydros_, root = 0)
    network_ = CHILD_COMM_.bcast(network_, root = 0)
    thermals_ = CHILD_COMM_.bcast(thermals_, root = 0)

    if run_single_period_models is not None:
        run_single_period_models = CHILD_COMM_.bcast(run_single_period_models, root = 0)

    complete_list_jobs = None

    complete_list_jobs = CHILD_COMM_.bcast(complete_list_jobs, root = 0)

    jobs_per_proc = int(len(complete_list_jobs)/SIZE_)

    if CHILD_RANK_ == (SIZE_ - 1):
        # the process with the `last` rank gets jobs_per_proc jobs plus whatever remains
        jobs_ = complete_list_jobs[int(CHILD_RANK_*jobs_per_proc):]
    else:
        jobs_ =complete_list_jobs[int(CHILD_RANK_*jobs_per_proc):int((CHILD_RANK_+1)*jobs_per_proc)]

    time_limit_ = 360.000

    time_limit_ = CHILD_COMM_.bcast(time_limit_, root = 0)

    return (CHILD_COMM_, SIZE_, CHILD_RANK_,
                params_, thermals_, network_, hydros_,
                    jobs_,
                        time_limit_, run_single_period_models)

def _share_results_with_parent(CHILD_COMM_, network_):
    """
        the child processes have finished their jobs and now they will send the results to the
        parent through reduce
    """
    aux_active_bounds = np.array([int(v) for v in network_.ACTIVE_BOUNDS.values()], dtype = 'int')
    CHILD_COMM_.Reduce([aux_active_bounds, MPI.INT], None, op = MPI.MIN, root = 0)

    aux_ubs = np.array([int(v) for v in network_.ACTIVE_UB.values()], dtype='int')
    CHILD_COMM_.Reduce([aux_ubs, MPI.INT], None, op = MPI.MIN, root = 0)

    aux_lbs = np.array([int(v) for v in network_.ACTIVE_LB.values()], dtype='int')
    CHILD_COMM_.Reduce([aux_lbs, MPI.INT], None, op = MPI.MIN, root = 0)

    aux_actibe_ubs_per_period = np.array([[network_.ACTIVE_UB_PER_PERIOD[l][t]
                                            for t in network_.ACTIVE_UB_PER_PERIOD[l].keys()]
                                                for l in network_.LINE_ID],
                                                    dtype = 'int')
    CHILD_COMM_.Reduce([aux_actibe_ubs_per_period, MPI.INT], None, op = MPI.MIN, root = 0)

    aux_actibe_lbs_per_period = np.array([[network_.ACTIVE_LB_PER_PERIOD[l][t]
                                            for t in network_.ACTIVE_LB_PER_PERIOD[l].keys()]
                                                for l in network_.LINE_ID],
                                                    dtype = 'int')
    CHILD_COMM_.Reduce([aux_actibe_lbs_per_period, MPI.INT], None, op = MPI.MIN, root = 0)

if __name__ == '__main__':
    # __name__ will be `__main__` if this script is the first executed by the python process

    (CHILD_COMM, SIZE, CHILD_RANK,
                    params, thermals, network, hydros,
                        jobs,
                            time_limit, run_single_period_models
                                ) =_initialize_child_processes(run_single_period_models=False)

    _remove_redundant_flow_limits_angles(params, network, thermals, hydros,
                                        time_limit = time_limit,
                                        list_of_jobs = jobs,
                                        print_to_console = False,
                                        run_single_period_models = run_single_period_models)

    _share_results_with_parent(CHILD_COMM, network)

    CHILD_COMM.Disconnect()
