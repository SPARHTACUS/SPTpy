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
from solver.write import writeEventTracker

def recvMessageFromGenCoord(params, it, redFlag, status, fixedVars,\
                            optModelsRelax, couplVarsRelax, betaRelax,\
                            bComm, bRank, eventTracker):
    '''Receive messages from the general coordinator'''

    fixedVarsRecv = False
    bwPackageRecv = np.array([0, 0, 0, 0, 0, 0], dtype = 'd')

    totalTimeCommunicating, totalAddingCuts = 0, 0

    while (redFlag != 1) and not(fixedVarsRecv):
        while (redFlag != 1) and\
                    not(bComm.Iprobe(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)):
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
                    eventTracker.append(('primalSolRecv', '40', ' ', ' ',\
                                        ' ', bRank, bwPackageRecv[1], ' ',
                                            bwPackageRecv[2], bwPackageRecv[3],\
                                                it, ' ', dt() - params.start))

                elif tg == 31:
                    timeCommunicating, timeAddingCuts = recvSubgradFromAnotherBackwardWorker(\
                                                        params, it, optModelsRelax, couplVarsRelax,\
                                                        betaRelax, bwPackageRecv, bComm, bRank,\
                                                        eventTracker)
                    totalTimeCommunicating += timeCommunicating
                    totalAddingCuts += timeAddingCuts

                else:
                    raise Exception('Im the backward worker ' + str(bRank) +\
                                                    ', and Ive received a message with tag'+str(tg))

            else:
                raise Exception('Im the backward worker ' + str(bRank) +\
                            ', and Ive received a message from ' + str(src) + ' with tag'+str(tg))

    return(redFlag, totalTimeCommunicating, totalAddingCuts)

def backwardStepPar(params, thermals, it, ub, lb, backwardInfo, objValRelax,\
                    optModels, optModelsRelax, lbda,\
                    fixedVars, couplVars, couplVarsRelax, couplConstrsRelax,\
                    alphaRelax, betaRelax, beta, alpha, redFlag,\
                    bufferBackward, wRank, bComm, bRank):
    '''Backward step in the DDiP'''

    bwPackageRecv = np.array([0, 0, 0, 0, 0, 0], dtype = 'd')
    eventTracker = []

    iniCheck = dt()

    status = MPI.Status()

    if not(params.solveOnlyFirstSubhorizon):
        # if this backward process is not going to solve only its first subhorizon problem,
        # then it needs a primal solution to start performing a backward pass
        redFlag, totalTimeCommunicating, totalAddingCuts = recvMessageFromGenCoord(\
                                        params, it, redFlag, status, fixedVars,\
                                        optModelsRelax, couplVarsRelax, betaRelax, bComm, bRank,\
                                        eventTracker)

    timeCommunicating, timeAddingCuts = 0, 0
    msgRecv = True
    while (redFlag != 1) and msgRecv:
        if (bComm.Iprobe(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)):

            timeCommunicating, timeAddingCuts = recvSubgradFromAnotherBackwardWorker(\
                                            params, it, optModelsRelax,\
                                            couplVarsRelax, betaRelax, bwPackageRecv, bComm, bRank,\
                                            eventTracker)
        else:
            if not(params.solveOnlyFirstSubhorizon) or (redFlag != 0):
                msgRecv = False
            else:
                if timeAddingCuts == 0:
                    sleep(1) # timeAddingCuts is zero if nothing has been received
                else:
                    msgRecv = False

    while redFlag != 1:

        redFlag, ub, lb, lbda = backwardStep(params, thermals, it, ub, lb,\
                                            backwardInfo, objValRelax, optModels, optModelsRelax,\
                                            lbda, fixedVars, couplVars,\
                                            couplVarsRelax, couplConstrsRelax,\
                                            alphaRelax, betaRelax, beta, alpha, redFlag,\
                                            bufferBackward,\
                                            bComm, bRank, status,\
                                            eventTracker)

        if (redFlag != 1):
            b = 0 if params.solveOnlyFirstSubhorizon else 1
            bufferBackward[('package',it,b)]=[np.array([redFlag, ub, lb, it, b], dtype = 'd'),None]
            bufferBackward[('subgrad', it, b)] = [deepcopy(lbda), None]
            bufferBackward[('objVals', it, b)] = [deepcopy(objValRelax), None]

            bufferBackward[('package', it, b)][1] = bComm.Isend(\
                                                            [bufferBackward[('package', it, b)][0],\
                                                                MPI.DOUBLE], dest = 0, tag = 38)
            bufferBackward[('subgrad', it, b)][1] = bComm.Isend(\
                                                            [bufferBackward[('subgrad', it, b)][0],\
                                                                MPI.DOUBLE], dest = 0, tag = 39)
            bufferBackward[('objVals', it, b)][1] = bComm.Isend(\
                                                            [bufferBackward[('objVals', it, b)][0],\
                                                                MPI.DOUBLE], dest = 0, tag = 40)
            if b == 0:
                eventTracker.append(('completeDualSolutionSent', '130', ' ', lb, \
                                                        bRank, ' ', ' ', ' ',\
                                                            it, b, ' ', ' ', dt() - params.start))
            else:
                eventTracker.append(('completeDualSolutionSent', '134', ' ', ' ', \
                                                        bRank, ' ', ' ', ' ',\
                                                            it, b, ' ', ' ', dt() - params.start))

        if (redFlag != 1):
            if not(params.solveOnlyFirstSubhorizon):
                redFlag, totalTimeCommunicating, totalAddingCuts = recvMessageFromGenCoord(\
                                        params, it, redFlag, status, fixedVars,\
                                        optModelsRelax, couplVarsRelax, betaRelax, bComm, bRank,\
                                        eventTracker)
                backwardInfo[0]['communication'][-1] += totalTimeCommunicating
                backwardInfo[0]['timeToAddCuts'][-1] += totalAddingCuts

            totalTimeCommunicating, totalAddingCuts = 0, 0

            msgRecv = True
            while (redFlag != 1) and msgRecv:
                if (bComm.Iprobe(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)):
                    timeCommunicating, timeAddingCuts = recvSubgradFromAnotherBackwardWorker(\
                                            params, it, optModelsRelax,\
                                            couplVarsRelax, betaRelax, bwPackageRecv, bComm, bRank,\
                                            eventTracker)

                    totalTimeCommunicating += timeCommunicating
                    totalAddingCuts += timeAddingCuts
                else:
                    if redFlag!=0 or (not(params.solveOnlyFirstSubhorizon) and params.asynchronous):
                        break # if the redFlag has changed to 1, then exit and while-loop
                    if not(params.asynchronous):
                        # synchronous
                        if params.countCutsReceived[-1][it] ==\
                                                        params.cutsToBeReceivedAtSynchPoints[-1]:
                            # if the current backward worker does not solve only
                            # its first subhorizon and the optimization is asynchronous,
                            # or, if the optimization is synchronous, only exit the while-loop if
                            # all cuts have been received
                            msgRecv = False
                        else:
                            sleep(1)
                    else:
                        # asynchronous
                        if params.solveOnlyFirstSubhorizon and (totalAddingCuts == 0):
                            sleep(1) # totalAddingCuts is zero if nothing has been received
                        else:
                            msgRecv = False

            backwardInfo[0]['communication'][-1] += totalTimeCommunicating
            backwardInfo[0]['timeToAddCuts'][-1] += totalAddingCuts

        cutsAdded = 0
        if (redFlag != 1) and not(params.asynchronous):
            for subh in params.delayedCuts[it]:
                for bRankOrigin in params.delayedCuts[it][subh]:
                    for subhOrigin in [subhOrigin for subhOrigin in\
                                params.delayedCuts[it][subh][bRankOrigin]\
                            if params.delayedCuts[it][subh][bRankOrigin][subhOrigin] is not None\
                                and params.delayedCuts[it][subh][bRankOrigin][subhOrigin][3] == it]:
                        optCut = params.delayedCuts[it][subh][bRankOrigin][subhOrigin]
                        optModelsRelax[subh].add_constr(optCut[0], name = optCut[1])
                        eventTracker.append(('cutAdded', '188', ' ', ' ',\
                                            optCut[2], bRank, ' ', ' ',\
                                            optCut[3], optCut[4], it, subh, dt() - params.start))
                        del params.delayedCuts[it][subh][bRankOrigin][subhOrigin]
                        params.delayedCuts[it][subh][bRankOrigin][subhOrigin] = None

                        cutsAdded += 1

            assert cutsAdded == params.cutsToBeReceivedAtSynchPoints[-1],\
                                            f'Im bRank {bRank}. '+\
                                            f'{cutsAdded} cuts were'+\
                                            ' added but the number of cuts added ' +\
                                            f'should be {params.cutsToBeReceivedAtSynchPoints[-1]}'

        if (dt() - iniCheck) >= 30:
            # delete buffers that were already sent

            keysToDeleteBackward = []

            for k, item in bufferBackward.items():
                if item[1].Test():
                    keysToDeleteBackward.append(k)

            for k in keysToDeleteBackward:
                del bufferBackward[k]

            iniCheck = dt()

        if not(params.asynchronous):
            eventTracker.append(('EndOfIteration', '209', ' ', ' ',\
                                    ' ', ' ', ' ', ' ',\
                                        it, ' ', ' ', ' ', dt() - params.start))

        it += 1

    writeEventTracker(params, eventTracker, wRank)

    return (ub, lb, (ub - lb)/ub, redFlag)
