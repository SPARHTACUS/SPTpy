# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""

from timeit import default_timer as dt
from time import sleep
from copy import deepcopy
import numpy as np
from mpi4py import MPI
from solver.forward import forwardStep, recvAndAddCuts
from solver.write import writeEventTracker

def sendInfoToGenCoord(params, redFlag, ub, lb, it, bufferForward,\
                        presentCosts, futureCosts, fixedVars, fRank, fComm, eventTracker):
    '''Send information to the general coordinator'''

    bufferForward[('package', it, params.nSubhorizons - 1)] =\
                                        [np.array([redFlag, ub, lb, it, params.nSubhorizons-1,\
                                        sum(presentCosts), futureCosts[-1]], dtype = 'd'), None]
    bufferForward[('sol', it, params.nSubhorizons - 1)] = [deepcopy(fixedVars), None]
    bufferForward[('package', it, params.nSubhorizons - 1)][1] = fComm.Isend(\
                                    [bufferForward[('package', it, params.nSubhorizons - 1)][0],\
                                                                MPI.DOUBLE], dest = 0, tag = 21)
    bufferForward[('sol', it, params.nSubhorizons - 1)][1] = fComm.Isend(\
                                    [bufferForward[('sol', it, params.nSubhorizons - 1)][0],\
                                                                MPI.DOUBLE], dest = 0, tag = 22)

    eventTracker.append(('primalSolSent', '29', sum(presentCosts) + futureCosts[-1], lb,\
                                ' ', ' ', fRank, ' ',\
                                    it, params.nSubhorizons - 1, ' ', ' ', dt() - params.start))
    # sum(presentCosts) + futureCosts[-1] is the upper bound associated with the primal solution
    # being sent
    return(redFlag)

def recvInfoFromGenCoord(params, it, redFlag, status, optModels, couplVars, beta,\
                        objValuesRelaxs, subgrads, evaluatedSol, fwPackageRecv,\
                        fRank, fComm, eventTracker):
    '''Receive information from the general coordinator'''

    recvAllBackwards, counter, timeCommunicating, timeAddingCuts = False, 0, 0, 0

    while (redFlag != 1) and not(recvAllBackwards):
        while redFlag!=1 and not(fComm.Iprobe(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)):
            sleep(0.05)

        if (redFlag != 1):

            recvAllBackwards, backwardSrc, counter, timeCommunicating_, timeAddingCuts_ =\
                            recvAndAddCuts(params, it, optModels, couplVars, beta, objValuesRelaxs,\
                                            subgrads, evaluatedSol, fwPackageRecv, fRank, fComm,\
                                                eventTracker, counter, recvAllBackwards)

            timeCommunicating += timeCommunicating_
            timeAddingCuts += timeAddingCuts_

            if (params.asynchronous and\
                    (params.periodsOfBackwardWs[backwardSrc].shape[0] == params.nSubhorizons)) or\
                (not(params.asynchronous) and (counter == params.nCutsTobeReceived)):
                # If the current forward worker has received subgradients from a backward worker
                # with the same aggregation, then, there is enough information to continue without
                # more cuts. However, if synchronous optimization is used, then complete dual
                # information from all backward workers must be received before proceeding
                recvAllBackwards = True

    return(redFlag, timeCommunicating, timeAddingCuts)

def forwardStepPar(params, thermals, it, subhorizonInfo, couplVars, fixedVars, previousSol,\
                couplConstrs, optModels, bestSol, presentCosts, futureCosts,\
                alpha, beta, redFlag, ub, lb, gap, wRank, fComm, fRank, fSize):
    '''Parallel forward step of the DDiP'''

    eventTracker = []

    status = MPI.Status()

    iniCheck = dt()

    # The following dictionaries are used to temporarily store messages
    # sent to the GenCoord. The messages are stored with their respective
    # Isend's req, and they are identified by unique dictionary's keys.
    # for instance, buffPackagesForw[key] = [message, req]
    bufferForward = {}

    objValuesRelaxs = [[]] + [np.zeros(params.periodsOfBackwardWs[r].shape[0],\
                                dtype = 'd') for r in range(1, len(params.backwardWorkers) + 1, 1)]
    subgrads = [[]]+[np.zeros((params.periodsOfBackwardWs[r].shape[0],params.nComplVars),dtype='d')\
                                    for r in range(1, len(params.backwardWorkers) + 1, 1)]

    evaluatedSol = np.zeros(params.nComplVars, dtype = 'd')

    fwPackageRecv = np.array([0, 0, 0, 0, 0, 0], dtype = 'd')

    while redFlag != 1:
        ub, lb, gap, redFlag, _1, previousSol, dualInfoFromMatchRecv = forwardStep(params,\
                                            thermals, it,\
                                                subhorizonInfo, couplVars, fixedVars, previousSol,\
                                                    couplConstrs, optModels, bestSol,\
                                                        presentCosts, futureCosts,\
                                                            alpha, beta, redFlag, ub, lb, gap,\
                                                                bufferForward,\
                                                                    status, objValuesRelaxs,\
                                                                        subgrads, evaluatedSol,\
                                                                            fwPackageRecv,\
                                                                            fComm, fRank, fSize,\
                                                                                eventTracker)

        if (redFlag != 1):
            redFlag = sendInfoToGenCoord(params, redFlag, ub, lb, it, bufferForward,\
                                presentCosts, futureCosts, fixedVars, fRank, fComm, eventTracker)

        if (redFlag != 1) and not(dualInfoFromMatchRecv):
            # If matching dual information was not received during the last forward step, then
            # wait for the general coordinator to send it in the following function
            redFlag, timeCommunicating, timeAddingCuts = recvInfoFromGenCoord(params, it,\
                                                    redFlag, status, optModels, couplVars,\
                                                    beta, objValuesRelaxs, subgrads, evaluatedSol,\
                                                    fwPackageRecv, fRank, fComm, eventTracker)

            subhorizonInfo[0]['communication'][-1] += timeCommunicating
            subhorizonInfo[0]['timeToAddCuts'][-1] += timeAddingCuts

        if (dt() - iniCheck) >= 30:
            # delete buffers that were already sent
            keysToDeleteForward = []

            for k, item in bufferForward.items():
                if item[1].Test():
                    keysToDeleteForward.append(k)

            for k in keysToDeleteForward:
                del bufferForward[k]

            iniCheck = dt()

        if not(params.asynchronous):
            eventTracker.append(('EndOfIteration', '179', ' ', ' ',\
                                    ' ', ' ', ' ', ' ',\
                                        it, ' ', ' ', ' ', dt() - params.start))

        it += 1

    writeEventTracker(params, eventTracker, wRank)

    return (ub, lb, gap, redFlag)
