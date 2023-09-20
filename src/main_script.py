# -*- coding: utf-8 -*-
"""
    @author: Colonetti
"""
import mpi4py
mpi4py.rc.thread_level = 'single'
import itertools
import os
from timeit import default_timer
from mpi4py import MPI
import numpy as np

from read_and_load_data.load_data import load_data
from solver.mainSolver import run_solver
from postOptimization import postOptimization
from solver.write import writeDDiPdata

W_COMM = MPI.COMM_WORLD
W_RANK = W_COMM.Get_rank()
W_SIZE = W_COMM.Get_size()
host = MPI.Get_processor_name()

def main(exp_list, exp_names):
    """
        main function
    """

    rootFolder = os.path.abspath(os.path.join(__file__ , "../..")).replace("\\","/")

    (params, hydros, thermals, network,
        f_comm, f_rank, f_size, b_comm, b_rank, b_size) = load_data(rootFolder,
                                                                        W_COMM, W_SIZE, W_RANK,
                                                                            exp_list, exp_names)

    W_COMM.Barrier()
    params.START = default_timer()
    params.LAST_TIME = params.START + params.TIME_LIMIT
    W_COMM.Barrier()

    if W_RANK == 0:
        n_p_act = len([l for l in network.LINE_ID if network.ACTIVE_BOUNDS[l]])
        print(f"The total number of possibly active lines is {n_p_act}", flush = True)

        ###########################################################################
        print(f"{'':#<70}")
        print(f"{' Overview of the system ':#^70}")
        print("System: " + params.PS + "\tcase: " + params.CASE)
        print(f"Planning horizon in hours: {params.T*params.DISCRETIZATION}")
        print(f"Time steps: {params.T}")
        print(f"{len(thermals.ID)} thermal plants with installed capacity "
                                        f"{sum(thermals.MAX_P.values())*params.POWER_BASE:.0f} MW")
        hydro_cap = sum(sum(units_max_p.values()) for units_max_p in hydros.UNIT_MAX_P.values())
        print(f"{len(hydros.ID)} hydropower plants with installed capacity " +
                                                            f"{hydro_cap*params.POWER_BASE:.0f} MW")
        print("Total installed capacity (MW): " +
                            f"{(sum(thermals.MAX_P.values()) + hydro_cap)*params.POWER_BASE:.0f}")
        print(f"Buses: {len(network.BUS_ID)}")
        print(f"AC transmission lines: {len(network.LINE_F_T)}")
        print(f"DC links: {len(network.LINK_F_T)}")
        print("Peak net load (MW): " +
                                 f"{np.max(np.sum(network.NET_LOAD,axis=0))*params.POWER_BASE:.0f}")
        if W_SIZE == 1:
            print(f"Subhorizons: {params.N_SUBHORIZONS}" + "\n" +
                                    f"Periods in each subhorizon: {params.PERIODS_PER_SUBH}")
        print(f'The total number of processes is {W_SIZE}')
        if W_SIZE > 1:
            print(f'Forward processes: {params.FORWARD_WORKERS}')
            print(f'Backward processes: {params.BACKWARD_WORKERS}')
        else:
            print('Forward processes: [0]')
            print('Backward processes: [0]')
        if params.ASYNCHRONOUS:
            print('Asynchronous optimization')
        else:
            print('Synchronous optimization')
        print(f'Solver:\t{params.SOLVER}')
        print(flush = True)

    f = open(params.OUT_DIR + '/params - ' + params.PS + ' - case ' +
                        str(params.CASE) + ' - rank ' + str(W_RANK) + '.csv', 'w', encoding="utf-8")
    f.write('attribute;value\n')

    DO_NOT_PRINT = ['lbOnCouplVars', 'ubOnCouplVars', 'map', 'varsPerPeriod', 'conVarsPerPeriod',
                    'binVarsPerPeriod', 'dispGenVarsPerPeriod', 'binDispVarsPerPeriod',
                        'varsInPreviousSubhs', 'varsInPreviousAndCurrentSubh', 'binVarsPerSubh',
                            'varsPerSubh', 'binDispVarsPerSubh',
                                'conVarsPerSubh', 'dispGenVarsPerSubh', 'genConstrsPerPeriod',
                                    'contVarsInPreviousAndCurrentSubh',
                                        'binVarsInPreviousAndCurrentSubh', 'periodsOfBackwardWs',
                                            'delayedCuts']
    for attr in dir(params):
        if attr[-2:] != '__' and attr not in DO_NOT_PRINT:
            if isinstance(getattr(params, attr), np.ndarray):
                f.write(attr + ';' + str(list(getattr(params, attr)))+ '\n')
            else:
                f.write(attr + ';' + str(getattr(params, attr)) + '\n')
    f.close()
    del f

    bestSol, ub, pLog, subhorizonInfo, backwardInfo = run_solver(
                                                            params, hydros, thermals, network,
                                                                W_COMM, W_RANK, W_SIZE,
                                                                    f_comm, f_rank, f_size,
                                                                        b_comm, b_rank, b_size)

    writeDDiPdata(params, pLog, subhorizonInfo, backwardInfo, W_RANK)

    if W_RANK == 0 and ub < 1e12:
        print('\n\nPrint the results\n\n', flush = True)
        postOptimization(params, thermals, hydros, network, bestSol, ub)

    W_COMM.Barrier()

if __name__ == '__main__':

    experiments = {'case': [str(i) for i in range(1, 4, 1)],
                    'nSubhorizonsPerProcess': [[4, 3, 6, 3, 6]],
                    'forwardWs': [[1, 2]],
                    'backwardWs': [[3, 4]],
                    'solveOnlyFirstSubhorizonInput': [4*[False] + [False] + 20*[False]],
                    'trials': [0]}

    experiment = {k: None for k in experiments}

    exp_id = 0

    keys, values = zip(*experiments.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for perm in permutations_dicts:
        for key in experiments:
            experiment[key] = perm[key]

        # Run the experiment
        exp_name = 'exp' + str(exp_id) + '_trial' + str(experiment['trials'])

        if ((W_SIZE > 1) and
                    (W_SIZE != (1 + len(experiment['forwardWs']) + len(experiment['backwardWs'])))):
            tw = len(experiment['forwardWs']) + len(experiment['backwardWs'])
            raise ValueError('Number of processes does not match number of ' +
                            f'forward and backward workers. For {tw} workers, ' +
                            f'there should be {tw + 1} processes.')

        if ((W_SIZE > 1) and (W_SIZE != (len(experiment['nSubhorizonsPerProcess'])))):
            fw, bw = len(experiment['forwardWs']), len(experiment['backwardWs'])
            raise ValueError('Number of processes does not match number of ' +
                            f'partitions provided. There are {W_SIZE} processes ' +
                            f"({fw} forward + {bw} backward workers + general coordinator), " +
                            f"but {len(experiment['nSubhorizonsPerProcess'])} partitions were " +
                            "provided.")

        main(experiment, exp_name)

        exp_id += 1

        W_COMM.Barrier()
