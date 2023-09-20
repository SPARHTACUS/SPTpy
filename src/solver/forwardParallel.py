# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""

from timeit import default_timer as dt
from time import sleep
from copy import deepcopy
import numpy as np
from mpi4py import MPI
from solver.forward import forwardStep, recv_and_add_cuts
from solver.write import write_event_tracker

def send_info_to_gen_coord(params, redFlag, ub, lb, it, bufferForward,
                        presentCosts, futureCosts, fixedVars, fRank, fComm, event_tracker):
    """Send information to the general coordinator"""

    bufferForward[('package', it, params.N_SUBHORIZONS - 1)] =\
                                        [np.array([redFlag, ub, lb, it, params.N_SUBHORIZONS-1,
                                        sum(presentCosts), futureCosts[-1]], dtype = 'd'), None]
    bufferForward[('sol', it, params.N_SUBHORIZONS - 1)] = [deepcopy(fixedVars), None]
    bufferForward[('package', it, params.N_SUBHORIZONS - 1)][1] = fComm.Isend(
                                    [bufferForward[('package', it, params.N_SUBHORIZONS - 1)][0],
                                                                MPI.DOUBLE], dest = 0, tag = 21)
    bufferForward[('sol', it, params.N_SUBHORIZONS - 1)][1] = fComm.Isend(
                                    [bufferForward[('sol', it, params.N_SUBHORIZONS - 1)][0],
                                                                MPI.DOUBLE], dest = 0, tag = 22)

    event_tracker.append(('primalSolSent', '29', sum(presentCosts) + futureCosts[-1], lb,
                                ' ', ' ', fRank, ' ',
                                    it, params.N_SUBHORIZONS - 1, ' ', ' ', dt() - params.START))
    # sum(presentCosts) + futureCosts[-1] is the upper bound associated with the primal solution
    # being sent
    return redFlag

def recv_info_from_gen_coord(params, it, redFlag, status, optModels, couplVars, beta,
                        objValuesRelaxs, subgrads, evaluatedSol, fwPackageRecv,
                        fRank, fComm, event_tracker):
    """Receive information from the general coordinator"""

    recvAllBackwards, counter, time_communicating, timeAddingCuts = False, 0, 0, 0

    while (redFlag != 1) and not(recvAllBackwards):
        while redFlag!=1 and not(fComm.Iprobe(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)):
            sleep(0.05)

        if (redFlag != 1):

            (recvAllBackwards, backwardSrc, counter, time_communicating_, timeAddingCuts_) =\
                            recv_and_add_cuts(params, it, optModels, couplVars, beta, objValuesRelaxs,
                                            subgrads, evaluatedSol, fwPackageRecv, fRank, fComm,
                                                event_tracker, counter, recvAllBackwards)

            time_communicating += time_communicating_
            timeAddingCuts += timeAddingCuts_

            if ((params.ASYNCHRONOUS and
                (params.PERIODS_OF_BACKWARD_WORKERS[backwardSrc].shape[0] == params.N_SUBHORIZONS))
                or (not(params.ASYNCHRONOUS) and (counter == params.N_CUTS_TO_BE_RECVD))):
                # If the current forward worker has received subgradients from a backward worker
                # with the same aggregation, then, there is enough information to continue without
                # more cuts. However, if synchronous optimization is used, then complete dual
                # information from all backward workers must be received before proceeding
                recvAllBackwards = True

    return(redFlag, time_communicating, timeAddingCuts)

def forwardStepPar(params, thermals, it, subhorizonInfo, couplVars, fixedVars, previousSol,\
                couplConstrs, optModels, bestSol, presentCosts, futureCosts,\
                alpha, beta, redFlag, ub, lb, gap, W_RANK, fComm, fRank, fSize):
    """Parallel forward step of the DDiP"""

    event_tracker = []

    status = MPI.Status()

    ini_check = dt()

    # The following dictionaries are used to temporarily store messages
    # sent to the GenCoord. The messages are stored with their respective
    # Isend's req, and they are identified by unique dictionary's keys.
    # for instance, buffPackagesForw[key] = [message, req]
    bufferForward = {}

    objValuesRelaxs = [[]] + [np.zeros(params.PERIODS_OF_BACKWARD_WORKERS[r].shape[0],
                                dtype = 'd') for r in range(1, len(params.BACKWARD_WORKERS) + 1, 1)]
    subgrads = [[]] + [np.zeros((params.PERIODS_OF_BACKWARD_WORKERS[r].shape[0],
                                    params.N_COMPL_VARS),dtype='d')
                                    for r in range(1, len(params.BACKWARD_WORKERS) + 1, 1)]

    evaluatedSol = np.zeros(params.N_COMPL_VARS, dtype = 'd')

    fwPackageRecv = np.array([0, 0, 0, 0, 0, 0], dtype = 'd')

    while redFlag != 1:
        (ub, lb, gap, redFlag, _1, previousSol, dualInfoFromMatchRecv) = forwardStep(params,
                                            thermals, it,
                                                subhorizonInfo, couplVars, fixedVars, previousSol,
                                                    couplConstrs, optModels, bestSol,
                                                        presentCosts, futureCosts,
                                                            alpha, beta, redFlag, ub, lb, gap,
                                                                bufferForward,
                                                                    status, objValuesRelaxs,
                                                                        subgrads, evaluatedSol,
                                                                            fwPackageRecv,
                                                                            fComm, fRank, fSize,
                                                                                event_tracker)

        if (redFlag != 1):
            redFlag = send_info_to_gen_coord(params, redFlag, ub, lb, it, bufferForward,
                                presentCosts, futureCosts, fixedVars, fRank, fComm, event_tracker)

        if (redFlag != 1) and not(dualInfoFromMatchRecv):
            # If matching dual information was not received during the last forward step, then
            # wait for the general coordinator to send it in the following function
            redFlag, time_communicating, timeAddingCuts = recv_info_from_gen_coord(params, it,
                                                    redFlag, status, optModels, couplVars,
                                                    beta, objValuesRelaxs, subgrads, evaluatedSol,
                                                    fwPackageRecv, fRank, fComm, event_tracker)

            subhorizonInfo[0]['communication'][-1] += time_communicating
            subhorizonInfo[0]['timeToAddCuts'][-1] += timeAddingCuts

        if (dt() - ini_check) >= 30:
            # delete buffers that were already sent
            keysToDeleteForward = []

            for k, item in bufferForward.items():
                if item[1].Test():
                    keysToDeleteForward.append(k)

            for k in keysToDeleteForward:
                del bufferForward[k]

            ini_check = dt()

        if not(params.ASYNCHRONOUS):
            event_tracker.append(('EndOfIteration', '179', ' ', ' ',
                                    ' ', ' ', ' ', ' ',
                                        it, ' ', ' ', ' ', dt() - params.START))

        it += 1

    write_event_tracker(params, event_tracker, W_RANK)

    return (ub, lb, gap, redFlag)
