# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""

from timeit import default_timer as dt
from copy import deepcopy
from time import sleep
import numpy as np
from mpi4py import MPI
from solver.backward import backwardStep, recvSubgradFromAnotherBackwardWorker
from solver.write import write_event_tracker

def recvMessageFromGenCoord(params, it, redFlag, status, fixedVars,
                            optModelsRelax, couplVarsRelax, betaRelax,
                            bComm, bRank, event_tracker):
    """Receive messages from the general coordinator"""

    fixedVarsRecv = False
    bwPackageRecv = np.array([0, 0, 0, 0, 0, 0], dtype = 'd')

    totaltime_communicating, totalAddingCuts = 0, 0

    while (redFlag != 1) and not(fixedVarsRecv):
        while ((redFlag != 1) and
                    not(bComm.Iprobe(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status))):
            sleep(0.05)

        if (redFlag != 1):
            src = status.Get_source()
            tg = status.Get_tag()

            if src == 0:
                if tg == 3:
                    bComm.Recv([bwPackageRecv, MPI.DOUBLE], source = 0, tag = tg)

                elif tg == 30:
                    bComm.Recv([fixedVars, MPI.DOUBLE], source = 0, tag = tg)
                    fixedVarsRecv = True
                    event_tracker.append(('primalSolRecv', '40', ' ', ' ',
                                        ' ', bRank, bwPackageRecv[1], ' ',
                                            bwPackageRecv[2], bwPackageRecv[3],
                                                it, ' ', dt() - params.START))

                elif tg == 31:
                    time_communicating, timeAddingCuts = recvSubgradFromAnotherBackwardWorker(
                                                        params, it, optModelsRelax, couplVarsRelax,
                                                        betaRelax, bwPackageRecv, bComm, bRank,
                                                        event_tracker)
                    totaltime_communicating += time_communicating
                    totalAddingCuts += timeAddingCuts

                else:
                    raise ValueError(f"Im the backward worker {bRank}" +
                                                    f", and Ive received a message with tag {tg}")

            else:
                raise ValueError(f"Im the backward worker {bRank}" +
                                        f", and Ive received a message from {src} with tag {tg}")

    return(redFlag, totaltime_communicating, totalAddingCuts)

def backwardStepPar(params, thermals, it, ub, lb, backwardInfo, objValRelax,
                    optModels, optModelsRelax, lbda,
                    fixedVars, couplVars, couplVarsRelax, couplConstrsRelax,
                    alphaRelax, betaRelax, beta, alpha, redFlag,
                    bufferBackward, W_RANK, bComm, bRank):
    """Backward step in the DDiP"""

    bwPackageRecv = np.array([0, 0, 0, 0, 0, 0], dtype = 'd')
    event_tracker = []

    iniCheck = dt()

    status = MPI.Status()

    if not(params.SOLVE_ONLY_1st_SUBH):
        # if this backward process is not going to solve only its first subhorizon problem,
        # then it needs a primal solution to start performing a backward pass
        redFlag, totaltime_communicating, totalAddingCuts = recvMessageFromGenCoord(
                                        params, it, redFlag, status, fixedVars,
                                        optModelsRelax, couplVarsRelax, betaRelax, bComm, bRank,
                                        event_tracker)

    time_communicating, timeAddingCuts = 0, 0
    msgRecv = True
    while (redFlag != 1) and msgRecv:
        if (bComm.Iprobe(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)):

            time_communicating, timeAddingCuts = recvSubgradFromAnotherBackwardWorker(
                                            params, it, optModelsRelax,
                                            couplVarsRelax, betaRelax, bwPackageRecv, bComm, bRank,
                                            event_tracker)
        else:
            if not(params.SOLVE_ONLY_1st_SUBH) or (redFlag != 0):
                msgRecv = False
            else:
                if timeAddingCuts == 0:
                    sleep(1) # timeAddingCuts is zero if nothing has been received
                else:
                    msgRecv = False

    while redFlag != 1:

        redFlag, ub, lb, lbda = backwardStep(params, thermals, it, ub, lb,
                                            backwardInfo, objValRelax, optModels, optModelsRelax,
                                            lbda, fixedVars, couplVars,
                                            couplVarsRelax, couplConstrsRelax,
                                            alphaRelax, betaRelax, beta, alpha, redFlag,
                                            bufferBackward,
                                            bComm, bRank, status,
                                            event_tracker)

        if (redFlag != 1):
            b = 0 if params.SOLVE_ONLY_1st_SUBH else 1
            bufferBackward[('package',it,b)]=[np.array([redFlag, ub, lb, it, b], dtype = 'd'),None]
            bufferBackward[('subgrad', it, b)] = [deepcopy(lbda), None]
            bufferBackward[('objVals', it, b)] = [deepcopy(objValRelax), None]

            bufferBackward[('package', it, b)][1] = bComm.Isend(
                                                            [bufferBackward[('package', it, b)][0],
                                                                MPI.DOUBLE], dest = 0, tag = 38)
            bufferBackward[('subgrad', it, b)][1] = bComm.Isend(
                                                            [bufferBackward[('subgrad', it, b)][0],
                                                                MPI.DOUBLE], dest = 0, tag = 39)
            bufferBackward[('objVals', it, b)][1] = bComm.Isend(
                                                            [bufferBackward[('objVals', it, b)][0],
                                                                MPI.DOUBLE], dest = 0, tag = 40)
            if b == 0:
                event_tracker.append(('completeDualSolutionSent', '130', ' ', lb,
                                                        bRank, ' ', ' ', ' ',
                                                            it, b, ' ', ' ', dt() - params.START))
            else:
                event_tracker.append(('completeDualSolutionSent', '134', ' ', ' ',
                                                        bRank, ' ', ' ', ' ',
                                                            it, b, ' ', ' ', dt() - params.START))

        if (redFlag != 1):
            if not(params.SOLVE_ONLY_1st_SUBH):
                redFlag, totaltime_communicating, totalAddingCuts = recvMessageFromGenCoord(
                                        params, it, redFlag, status, fixedVars,
                                        optModelsRelax, couplVarsRelax, betaRelax, bComm, bRank,
                                        event_tracker)
                backwardInfo[0]['communication'][-1] += totaltime_communicating
                backwardInfo[0]['timeToAddCuts'][-1] += totalAddingCuts

            totaltime_communicating, totalAddingCuts = 0, 0

            msgRecv = True
            while (redFlag != 1) and msgRecv:
                if (bComm.Iprobe(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)):
                    time_communicating, timeAddingCuts = recvSubgradFromAnotherBackwardWorker(
                                            params, it, optModelsRelax,
                                            couplVarsRelax, betaRelax, bwPackageRecv, bComm, bRank,
                                            event_tracker)

                    totaltime_communicating += time_communicating
                    totalAddingCuts += timeAddingCuts
                else:
                    if redFlag!=0 or (not(params.SOLVE_ONLY_1st_SUBH) and params.ASYNCHRONOUS):
                        break # if the redFlag has changed to 1, then exit and while-loop
                    if not(params.ASYNCHRONOUS):
                        # synchronous
                        if (params.COUNT_CUTS_RECVD[-1][it] ==
                                                    params.CUTS_TO_BE_RECVD_AT_SYNCH_POINTS[-1]):
                            # if the current backward worker does not solve only
                            # its first subhorizon and the optimization is asynchronous,
                            # or, if the optimization is synchronous, only exit the while-loop if
                            # all cuts have been received
                            msgRecv = False
                        else:
                            sleep(1)
                    else:
                        # asynchronous
                        if params.SOLVE_ONLY_1st_SUBH and (totalAddingCuts == 0):
                            sleep(1) # totalAddingCuts is zero if nothing has been received
                        else:
                            msgRecv = False

            backwardInfo[0]['communication'][-1] += totaltime_communicating
            backwardInfo[0]['timeToAddCuts'][-1] += totalAddingCuts

        cutsAdded = 0
        if (redFlag != 1) and not(params.ASYNCHRONOUS):
            for subh in params.DELAYED_CUTS[it]:
                for bRankOrigin in params.DELAYED_CUTS[it][subh]:
                    for subhOrigin in [subhOrigin for subhOrigin in
                                params.DELAYED_CUTS[it][subh][bRankOrigin]
                            if params.DELAYED_CUTS[it][subh][bRankOrigin][subhOrigin] is not None
                                and params.DELAYED_CUTS[it][subh][bRankOrigin][subhOrigin][3] ==it]:
                        optCut = params.DELAYED_CUTS[it][subh][bRankOrigin][subhOrigin]
                        optModelsRelax[subh].add_constr(optCut[0], name = optCut[1])
                        event_tracker.append(('cutAdded', '188', ' ', ' ',
                                            optCut[2], bRank, ' ', ' ',
                                            optCut[3], optCut[4], it, subh, dt() - params.START))
                        del params.DELAYED_CUTS[it][subh][bRankOrigin][subhOrigin]
                        params.DELAYED_CUTS[it][subh][bRankOrigin][subhOrigin] = None

                        cutsAdded += 1

            assert cutsAdded == params.CUTS_TO_BE_RECVD_AT_SYNCH_POINTS[-1],\
                                        (f'Im bRank {bRank}. '+
                                        f'{cutsAdded} cuts were'+
                                        ' added but the number of cuts added ' +
                                        f'should be {params.CUTS_TO_BE_RECVD_AT_SYNCH_POINTS[-1]}')

        if (dt() - iniCheck) >= 30:
            # delete buffers that were already sent

            keys_to_delete = []

            for (k, _) in [(k, item) for k, item in bufferBackward.items() if item[1].Test()]:
                keys_to_delete.append(k)

            for k in keys_to_delete:
                del bufferBackward[k]

            iniCheck = dt()

        if not(params.ASYNCHRONOUS):
            event_tracker.append(('EndOfIteration', '209', ' ', ' ',
                                    ' ', ' ', ' ', ' ',
                                        it, ' ', ' ', ' ', dt() - params.START))

        it += 1

    write_event_tracker(params, event_tracker, W_RANK)

    return (ub, lb, (ub - lb)/ub, redFlag)
