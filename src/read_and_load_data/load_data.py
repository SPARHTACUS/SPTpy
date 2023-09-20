# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""

import os
from time import sleep
import numpy as np
from mpi4py import MPI

from optoptions import OptOptions
from read_and_load_data.read_csv import (
                                    read_generators, read_hydro_generating_units,
                                    read_network, gross_load_and_renewableGen,
                                    read_ini_state_thermal,
                                    read_aggreg_HPF, read_cost_to_go_function,
                                    read_trajectories,
                                    read_bounds_on_gen_of_thermals,
                                    read_bounds_on_gen_of_hydros,
                                    read_inflows, reset_gen_costs_of_thermals,
                                    reset_volume_bounds,
                                    read_previous_state_of_hydro_plants
                                    )
from read_and_load_data.conver_json import convert_json
from network import Hydros, Thermals, Network
from preProcessing.remove_end_of_line_buses_with_inj import remove_end_of_line_buses_with_inj
from preProcessing.build_ptdf import build_ptdf
from preProcessing.reduce_system import reduce_system
from preProcessing.reduce_network import reduce_network
from optoptions import NetworkModel
from preProcessing.identify_redundant_line_bounds import (remove_redundant_flow_limits_without_opt,
                                                            redundant_line_bounds)

def create_comm(params, worldComm, w_rank, w_size):
    """
        Split the world communicator into primal and dual communicators
        Also create the windows for one-sided communications
    """

    ### Windows
    # Best solution
    if w_size > 1:
        params.winBestSol = MPI.Win.Create(np.zeros(params.N_COMPL_VARS, dtype = 'd'),
                                                        1, info = MPI.INFO_NULL, comm = worldComm)
    # The red flag, upper bound, and lower bound windows are created in mainSolver.py

    #### Forward communicator
    if ((w_rank in params.FORWARD_WORKERS) or (w_rank == 0)):
        color = 0
    else:
        color = MPI.UNDEFINED

    f_comm = worldComm.Split(color = color, key = w_rank)

    if ((w_rank in params.FORWARD_WORKERS) or (w_rank == 0)):
        f_rank = f_comm.Get_rank()
        f_size = f_comm.Get_size()
    else:
        f_rank = -1e3
        f_size = -1e3
    ############################################################################

    #### Backward communicator
    if ((w_rank in params.BACKWARD_WORKERS) or (w_rank == 0)):
        color = 0
    else:
        color = MPI.UNDEFINED

    bComm = worldComm.Split(color = color, key = w_rank)

    if ((w_rank in params.BACKWARD_WORKERS) or (w_rank == 0)):
        bRank = bComm.Get_rank()
        bSize = bComm.Get_size()
    else:
        bRank = -1e3
        bSize = -1e3
    ############################################################################

    return(f_comm, f_rank, f_size, bComm, bRank, bSize)

def set_params(root_folder, w_rank, w_size, experiment, exp_name):
    """
        Create an instance of OptOptions and set the initial values for its attributes
    """

    params = OptOptions(
                        root_folder,
                        w_rank, w_size,
                        experiment['case'],
                        n_subhorizons = experiment['nSubhorizonsPerProcess'],
                        forward_workers = experiment['forwardWs'],
                        backward_workers = experiment['backwardWs'],
                        exp_name = exp_name,
                        solve_only_first_subhorizon = experiment['solveOnlyFirstSubhorizonInput']
                    )

    for k in [k for k in experiment.keys() if hasattr(params, k)]:
        oldValue = getattr(params, k)
        if isinstance(oldValue, list):
            setattr(params,k, [experiment[k] for i in range(len(oldValue))])
        else:
            setattr(params, k, experiment[k])

    params.OUT_DIR = (
                        root_folder + '/output/' + params.PS + '/case '
                            + params.CASE + '/' + exp_name + '/'
                        )

    if w_rank == 0:
        if not(os.path.isdir(params.OUT_DIR)):
            os.makedirs(params.OUT_DIR)

        if (len(experiment) > 0):
            f = open(params.OUT_DIR + '/experiment - ' + params.PS +\
                                    ' - case ' + str(params.CASE) + '.csv', 'w', encoding = 'utf-8')
            f.write('key;value\n')
            for k in experiment.keys():
                f.write(str(k) + ';' + str(experiment[k]) + '\n')
            f.close()
            del f

    return params

def share_backward_aggrs(params, w_comm, w_rank, w_size):
    """
        Make sure that all processes know how many subhorizons each of the backward processes has,
        and the periods in each of them. This will facilitate communication later on
    """

    # Create a numpy-array version of self.PERIODS_PER_SUBH
    periods_per_subh = [list(params.PERIODS_PER_SUBH[b]) for b in range(params.N_SUBHORIZONS)]

    # Note that, for an array, all rows must have the same number of columns
    # Thus, for rows with less than the maximum number of columns, i.e.,
    # the maximum number of periods in a single subhorizon, add periods 1e6
    max_n_periods = max((len(periods_per_subh[b]) for b in range(params.N_SUBHORIZONS)))

    for b in range(params.N_SUBHORIZONS):
        periods_per_subh[b] = (periods_per_subh[b] +
                                            int(max_n_periods - len(periods_per_subh[b]))*[1e6])

    for b in range(params.N_SUBHORIZONS):
        periods_per_subh[b].sort()

    periods_per_subh = np.array(periods_per_subh, dtype = 'int')


    if params.I_AM_A_BACKWARD_WORKER and (w_size > 1):
        bw_package_send = np.array([periods_per_subh.shape[0],
                                    periods_per_subh.shape[1]], dtype = 'int')
        for r in [0] + [r for r in params.BACKWARD_WORKERS if r != w_rank] + params.FORWARD_WORKERS:
            # send to everyone except the process itself
            w_comm.Isend([bw_package_send, MPI.INT], dest = r, tag = 35)
            w_comm.Isend([periods_per_subh, MPI.INT], dest = r, tag = 36)

    if (w_size > 1):
        # correspondence between w_rank, r, and b_rank
        corr = {r: 1 + c for c, r in enumerate(params.BACKWARD_WORKERS)}

        if w_rank != 0 and not(params.I_AM_A_FORWARD_WORKER):
            params.PERIODS_OF_BACKWARD_WORKERS[corr[w_rank]] = periods_per_subh

        buffer_size = {r: np.array([0, 0], dtype = 'int') for r in params.BACKWARD_WORKERS}

        # receive from all backward processes. however, if the process is itself a backward
        # process, then it does not need to receive the info
        n_msgs_to_recv = len(params.BACKWARD_WORKERS) if not(params.I_AM_A_BACKWARD_WORKER) else\
                                                                len(params.BACKWARD_WORKERS) - 1
        n_msgs_recvd = 0

        status = MPI.Status()

        while n_msgs_recvd < n_msgs_to_recv:
            if (w_comm.Iprobe(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)):
                src = status.Get_source()
                tg = status.Get_tag()

                if tg == 35:
                    w_comm.Recv([buffer_size[src], MPI.INT], source = src, tag = 35)
                elif tg == 36:
                    c = corr[src]
                    params.PERIODS_OF_BACKWARD_WORKERS[c] = np.zeros((buffer_size[src][0],
                                                            buffer_size[src][1]), dtype = 'int')
                    w_comm.Recv(
                                [params.PERIODS_OF_BACKWARD_WORKERS[c], MPI.INT],
                                    source = src,
                                        tag = 36
                                )
                    n_msgs_recvd += 1
                else:
                    raise ValueError(f"Im W_RANK {w_rank} . Ive got a message from" +
                                        f"{src} with tag {tg}, and I dont know what to do with it.")
            else:
                sleep(1)

def define_complicating_vars(params, hydros, thermals):
    """
        Creates arrays to store the values of the coupling variables
    Note that the primal decisions and dual variables are shared among processes through
    large one-dimensional arrays. To correctly recover demanded values, create dictionaries
    that store the index for each decision variable. For instance, if one is interested in the
    reservoir volume of hydro plant 1 in time period 6, then a map that takes volume, the index
    of the plant, and the index of the time period will return the index of the reservoir volume
    in the array.
    """

    # Populate the map params.mp with the time-coupling variables

    #### Thermal generating units
    # Start-up decision
    params.MAP['stUpTG'] = {(g, t): -1e6 for g in thermals.ID for t in range(params.T)}
    # Shut-down decision
    params.MAP['stDwTG'] = {(g, t): -1e6 for g in thermals.ID for t in range(params.T)}
    # Dispatch phase
    params.MAP['DpTG'] = {(g, t): -1e6 for g in thermals.ID for t in range(params.T)}
    # Generation in the dispatch phase
    params.MAP['DpGenTG'] = {(g, t): -1e6 for g in thermals.ID for t in range(params.T)}

    #### Hydro plants
    # Reservoir volume
    params.MAP['v'] = {(h, t): -1e6 for h in hydros.ID for t in range(params.T)}
    # Turbine discharge
    params.MAP['q'] = {(h, t): -1e6 for h in hydros.ID for t in range(params.T)}
    # Spillage
    params.MAP['s'] = {(h, t): -1e6 for h in hydros.ID for t in range(params.T)}
    # Water transfer
    params.MAP['QbyPass'] = {(h, t): -1e6 for h in [h for h in hydros.ID
                            if hydros.DOWN_RIVER_BY_PASS[h] is not None] for t in range(params.T)}
    # Pumps
    params.MAP['pump'] = {(h, t): -1e6 for h in [h for h in hydros.ID
                                    if hydros.TURB_OR_PUMP[h] == 'Pump'] for t in range(params.T)}

    i = 0   # index of the time-coupling variable in the one-dimensional array that will be
            # created later

    for t in range(0, params.T, 1):
        for g in thermals.ID:
            params.MAP['stUpTG'][g, t] = i
            params.VARS_PER_PERIOD[t].append(i)
            params.BIN_VARS_PER_PERIOD[t].append(i)
            params.LB_ON_COUPL_VARS.append(0)
            params.UB_ON_COUPL_VARS.append(1)
            i += 1

        for g in thermals.ID:
            params.MAP['stDwTG'][g, t] = i
            params.VARS_PER_PERIOD[t].append(i)
            params.BIN_VARS_PER_PERIOD[t].append(i)
            params.LB_ON_COUPL_VARS.append(0)
            params.UB_ON_COUPL_VARS.append(1)
            i += 1
        for g in thermals.ID:
            params.MAP['DpTG'][g, t] = i
            params.VARS_PER_PERIOD[t].append(i)
            params.BIN_VARS_PER_PERIOD[t].append(i)
            params.BIN_DISP_VARS_PER_PERIOD[t].append(i)
            params.LB_ON_COUPL_VARS.append(0)
            params.UB_ON_COUPL_VARS.append(1)
            i += 1

    for t in range(0, params.T, 1):
        for g in thermals.ID:
            params.MAP['DpGenTG'][g, t] = i
            params.VARS_PER_PERIOD[t].append(i)
            params.CON_VARS_PER_PERIOD[t].append(i)
            params.DISP_GEN_VARS_PER_PERIOD[t].append(i)
            params.LB_ON_COUPL_VARS.append(0)
            params.UB_ON_COUPL_VARS.append(thermals.MAX_P[g] - thermals.MIN_P[g])
            i += 1

        for h in hydros.ID:
            params.MAP['v'][h, t] = i
            params.VARS_PER_PERIOD[t].append(i)
            params.CON_VARS_PER_PERIOD[t].append(i)
            params.LB_ON_COUPL_VARS.append(hydros.MIN_VOL[h])
            params.UB_ON_COUPL_VARS.append(hydros.MAX_VOL[h])
            i += 1

        for h in hydros.ID:
            params.MAP['q'][h, t] = i
            params.VARS_PER_PERIOD[t].append(i)
            params.CON_VARS_PER_PERIOD[t].append(i)
            params.LB_ON_COUPL_VARS.append(0)

            if sum(hydros.UNIT_MAX_P[h].values()) > 0:
                params.UB_ON_COUPL_VARS.append(sum(hydros.UNIT_MAX_TURB_DISCH[h].values()))
            else:
                params.UB_ON_COUPL_VARS.append(0)
            i += 1

        for h in hydros.ID:
            params.MAP['s'][h, t] = i
            params.VARS_PER_PERIOD[t].append(i)
            params.CON_VARS_PER_PERIOD[t].append(i)
            params.LB_ON_COUPL_VARS.append(0)
            params.UB_ON_COUPL_VARS.append(hydros.MAX_SPIL[h])
            i += 1

        for h in [h for h in hydros.ID if hydros.DOWN_RIVER_BY_PASS[h] is not None]:
            params.MAP['QbyPass'][h, t] = i
            params.VARS_PER_PERIOD[t].append(i)
            params.CON_VARS_PER_PERIOD[t].append(i)
            params.LB_ON_COUPL_VARS.append(0)
            params.UB_ON_COUPL_VARS.append(hydros.MAX_BY_PASS[h])
            i += 1

        for h in [h for h in hydros.ID if hydros.TURB_OR_PUMP[h] == 'Pump']:
            params.MAP['pump'][h, t] = i
            params.VARS_PER_PERIOD[t].append(i)
            params.CON_VARS_PER_PERIOD[t].append(i)
            params.LB_ON_COUPL_VARS.append(0)
            params.UB_ON_COUPL_VARS.append(sum(hydros.UNIT_MAX_TURB_DISCH[h].values()))

            i += 1

    params.N_COMPL_VARS = i   # total number of time-coupling variables

    # The following bounds will be used for rounding the time-coupling variables in
    # order to prevent numerical errors.
    params.UB_ON_COUPL_VARS = np.array(params.UB_ON_COUPL_VARS, dtype = 'd')
    params.LB_ON_COUPL_VARS = np.array(params.LB_ON_COUPL_VARS, dtype = 'd')

    for b in range(params.N_SUBHORIZONS):
        #### Binary variables
        for t in params.PERIODS_PER_SUBH[b]:
            params.VARS_PER_SUBH[b].extend(params.BIN_VARS_PER_PERIOD[t])
            params.BIN_VARS_PER_SUBH[b].extend(params.BIN_VARS_PER_PERIOD[t])
            params.BIN_DISP_VARS_PER_SUBH[b].extend(params.BIN_DISP_VARS_PER_PERIOD[t])

        #### Continuous variables
        for t in params.PERIODS_PER_SUBH[b]:
            params.VARS_PER_SUBH[b].extend(params.CON_VARS_PER_PERIOD[t])
            params.CON_VARS_PER_SUBH[b].extend(params.CON_VARS_PER_PERIOD[t])

        #### Generation of thermal units in dispatch phase
        for t in params.PERIODS_PER_SUBH[b]:
            params.DISP_GEN_VARS_PER_SUBH[b].extend(params.DISP_GEN_VARS_PER_PERIOD[t])

        # Get all variables associated with time periods in the current subhorizon
        # and subhorizons that come before b
        for b2 in range(0, b + 1, 1):
            params.VARS_IN_PREVIOUS_AND_CURRENT_SUBHS[b].extend(params.VARS_PER_SUBH[b2])
            params.CON_VARS_IN_PREVIOUS_AND_CURRENT_SUBH[b].extend(params.CON_VARS_PER_SUBH[b2])
            params.BIN_VARS_IN_PREVIOUS_AND_CURRENT_SUBH[b].extend(params.BIN_VARS_PER_SUBH[b2])

        # Get all variables associated with previous time periods
        for b2 in range(0, b, 1):
            params.VARS_IN_PREVIOUS_SUBHS[b].extend(params.VARS_PER_SUBH[b2])

    for b in range(params.N_SUBHORIZONS):
        params.CON_VARS_IN_PREVIOUS_AND_CURRENT_SUBH[b] = np.array(
                                    params.CON_VARS_IN_PREVIOUS_AND_CURRENT_SUBH[b], dtype = 'int')

        params.BIN_VARS_IN_PREVIOUS_AND_CURRENT_SUBH[b] = np.array(
                                    params.BIN_VARS_IN_PREVIOUS_AND_CURRENT_SUBH[b], dtype = 'int')

def load_data(root_folder, w_comm, w_size, w_rank, experiment, exp_name):
    """
        Read csv files with system's data and operating conditions
    """

    # create an instance of OptOptions (optoptions.py) with all parameters for the problem
    # and the solution process
    params = set_params(root_folder, w_rank, w_size, experiment, exp_name)

    share_backward_aggrs(params, w_comm, w_rank, w_size)

    if w_rank != 0:
        # create objects for the configurations of hydro plants, thermal plants,
        # and the network model
        hydros, thermals, network = None, None, None

    if w_rank == 0:
        # create objects for the configurations of hydro plants, thermal plants,
        # and the network model
        hydros, thermals, network = Hydros(), Thermals(), Network()

        if os.path.isfile(params.IN_DIR + params.PS + '.json'):
            print("Reading and converting json file", flush = True)
            convert_json(params, params.PS, params.IN_DIR + params.PS + '.json',
                            min_gen_cut_MW = params.MIN_GEN_CUT_MW)

        # read the parameters of the transmission network
        read_network(params.IN_DIR + 'network - ' + params.PS +'.csv', params, network)

        # read data for the thermal and hydro generators
        read_generators(params.IN_DIR + 'powerPlants - ' + params.PS +'.csv',
                                                                params, network, hydros, thermals)

        # read the data of hydro generating units
        if len(hydros.ID) > 0:
            read_hydro_generating_units(params.IN_DIR + 'data_of_hydro_generating_units - ' +
                                                                params.PS +'.csv', params, hydros)

        for b, bus in enumerate(network.BUS_ID):
            network.BUS_HEADER[bus] = b

        # read the gross load and renewable generation
        gross_load_and_renewableGen(
                            params.IN_DIR + 'case ' + str(params.CASE) +'/' +'gross load - '+
                            params.PS + ' - case ' + str(params.CASE) + '.csv',
                            params.IN_DIR + 'case ' + str(params.CASE) +'/'
                            + 'renewable generation - ' +
                            params.PS + ' - case ' + str(params.CASE) + '.csv' , params, network)

        # read the start-up and shut-down trajectories of thermal units
        if os.path.isfile(params.IN_DIR + 'trajectories - ' + params.PS +'.csv'):
            read_trajectories(params.IN_DIR + 'trajectories - ' + params.PS +'.csv',
                                                                                params, thermals)
        else:
            print("No file of start-up and shut-down trajectories found", flush = True)

        # read the incremental inflows to the reservoirs
        if len(hydros.ID) > 0:
            read_inflows(params.IN_DIR + 'case ' + str(params.CASE) + '/' + 'inflows - ' +
                        params.PS + ' - case ' + str(params.CASE) + '.csv', params, hydros)

        # bounds on the generation of groups of thermal units
        if os.path.isfile(params.IN_DIR + 'case ' + str(params.CASE) +'/'+
                                'bounds on generation of groups of thermal units - ' + params.PS +
                                    ' - case ' + str(params.CASE) + '.csv'):
            read_bounds_on_gen_of_thermals(params.IN_DIR + 'case ' + str(params.CASE) +'/'+
                                'bounds on generation of groups of thermal units - ' + params.PS +
                                    ' - case ' + str(params.CASE) + '.csv', params, thermals)
        else:
            print("No file of additional bounds on thermal generation found", flush = True)

        # bounds on the generation of groups of hydro plants
        if len(hydros.ID) > 0:
            if os.path.isfile(params.IN_DIR + 'case ' + str(params.CASE) + '/' +
                                'bounds on generation of groups of hydro plants - ' + params.PS +
                                    ' - case ' + str(params.CASE) + '.csv'):
                read_bounds_on_gen_of_hydros(params.IN_DIR + 'case ' + str(params.CASE) + '/' +
                                'bounds on generation of groups of hydro plants - ' + params.PS +
                                    ' - case ' + str(params.CASE) + '.csv', params, hydros)
            else:
                print("No file of additional bounds on hydro generation found", flush = True)

        # read the initial state of the thermal units
        read_ini_state_thermal(params.IN_DIR + 'case ' + str(params.CASE) + '/' +
                                        'initial states of thermal units - ' + params.PS +
                                        ' - case ' + str(params.CASE) + '.csv', params, thermals)

        # reset generation costs
        if os.path.isfile(params.IN_DIR + 'case ' + str(params.CASE) + '/' +
                                        'reset generation costs of thermal units - ' + params.PS +
                                        ' - case ' + str(params.CASE) + '.csv'):
            reset_gen_costs_of_thermals(params.IN_DIR + 'case ' + str(params.CASE) + '/' +
                                        'reset generation costs of thermal units - ' + params.PS +
                                        ' - case ' + str(params.CASE) + '.csv', params, thermals)
        else:
            print("No file of new unitary generation costs found. Using default costs", flush=True)

        if len(hydros.ID) > 0:
            # reset bounds on reservoir volumes
            reset_volume_bounds(params.IN_DIR + 'case ' + str(params.CASE) + '/' +
                                            'reset bounds on reservoir volumes - ' + params.PS +
                                            ' - case ' + str(params.CASE) + '.csv', params, hydros)

            # read the previous states of hydro plants
            read_previous_state_of_hydro_plants(params.IN_DIR + 'case '
                                        + str(params.CASE) + '/' +
                                        'initial reservoir volumes - ' + params.PS +
                                        ' - case ' + str(params.CASE) + '.csv',
                                        params.IN_DIR + 'case ' + str(params.CASE) + '/' +
                                        'previous water discharges of hydro plants - ' + params.PS+
                                        ' - case ' + str(params.CASE) + '.csv', params, hydros)

            read_aggreg_HPF(params.IN_DIR + 'case ' + str(params.CASE)
                                +'/aggregated_3Dim - ' +
                                    params.PS +' - '+
                                    'case ' + str(params.CASE) + ' - HPF without binaries.csv',
                                        params, hydros)

            read_cost_to_go_function(params.IN_DIR + 'case ' + str(params.CASE) +'/' +
                                        'cost-to-go function - ' + params.PS +
                                        ' - case ' + str(params.CASE) + '.csv', params, hydros)

        if params.REDUCE_SYSTEM:

            reduce_network(params, hydros, thermals, network)

            assert len(network.LINE_F_T.keys()) > 0, ("After reducing the network, there are no "+
                                                        "transmission lines left in the system. "+
                                                        "Either use the single bus model " +
                                                        "or disable network reduction")

            build_ptdf(network)

            remove_redundant_flow_limits_without_opt(params, hydros, thermals, network)

            reduce_network(params, hydros, thermals, network)

            assert len(network.LINE_F_T.keys()) > 0, ("After reducing the network, there are no "+
                                                        "transmission lines left in the system. "+
                                                        "Either use the single bus model " +
                                                        "or disable network reduction")

            build_ptdf(network)

            redundant_line_bounds(params, hydros, thermals, network,
                                        time_limit = 360,
                                            run_single_period_models = False)

            reduce_network(params, hydros, thermals, network)

            assert len(network.LINE_F_T.keys()) > 0, ("After reducing the network, there are no "+
                                                        "transmission lines left in the system. "+
                                                        "Either use the single bus model " +
                                                        "or disable network reduction")

            if params.NETWORK_MODEL == NetworkModel.PTDF:
                build_ptdf(network)

        else:
            if params.NETWORK_MODEL == NetworkModel.PTDF:
                build_ptdf(network)

    hydros = w_comm.bcast(hydros, root = 0)
    network = w_comm.bcast(network, root = 0)
    thermals = w_comm.bcast(thermals, root = 0)

    define_complicating_vars(params, hydros, thermals)

    if w_rank == 0:
        f = open(params.OUT_DIR + 'Indices of complicating variables - ' + params.PS +
                            ' - case ' + str(params.CASE) + '.csv', 'w', encoding = 'utf-8')
        f.write('varType;Plant;Time;Index;Lower bound on var;Upper bound on var\n')
        for key, _ in [(k, v) for k,v in params.MAP.items() if v is not {}]:
            for key2, value2 in params.MAP[key].items():
                f.write(key + ';' + str(key2[0]) + ';' + str(key2[1]))
                f.write(';' + str(value2))
                f.write(';' + str(params.LB_ON_COUPL_VARS[value2]))
                f.write(';' + str(params.UB_ON_COUPL_VARS[value2]))
                f.write('\n')
        f.close()
        del f

    # create communicators for the forward processes and backward processes. these will be used
    # for the workers to communicate with the general coordinator
    (f_comm, f_rank, f_size, b_comm, b_rank, b_size) = create_comm(params, w_comm, w_rank, w_size)

    params.DEFICIT_COST = network.DEFICIT_COST

    return(params, hydros, thermals, network, f_comm, f_rank, f_size, b_comm, b_rank, b_size)
