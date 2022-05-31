# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""

from copy import deepcopy
from time import sleep
from timeit import default_timer as dt

import numpy as np
from mpi4py import MPI

from mip import GUROBI, CONTINUOUS, LP_Method, xsum
from mip import OptimizationStatus as OptS

from solver.forward import addVIBasedOnRamp

def recvSubgradFromAnotherBackwardWorker(params, it, optModelsRelax, couplVarsRelax, betaRelax,\
                                        bwPackageRecv, bComm, bRank, eventTracker):
    '''Receive a subgradient from another backward worker'''

    timeCommunicating, timeAddingCuts = dt(), 0

    bComm.Recv([bwPackageRecv, MPI.DOUBLE], source = 0, tag = 31)
    backwardSrc = int(bwPackageRecv[0])

    objValuesRelaxs = np.zeros(params.periodsOfBackwardWs[backwardSrc].shape[0], dtype = 'd')
    subgrads = np.zeros((params.periodsOfBackwardWs[backwardSrc].shape[0],\
                                                                    params.nComplVars), dtype = 'd')
    evaluatedSol = np.zeros(params.nComplVars, dtype = 'd')

    bComm.Recv([objValuesRelaxs, MPI.DOUBLE], source = 0, tag = 32)
    bComm.Recv([subgrads, MPI.DOUBLE], source = 0, tag = 33)
    bComm.Recv([evaluatedSol, MPI.DOUBLE], source = 0, tag = 34)

    timeCommunicating = dt() - timeCommunicating

    # The subhorizon from which the subgradient was taken
    b = int(bwPackageRecv[5])
    itFromSource = int(bwPackageRecv[4])

    firstPeriodInSubh = min(params.periodsOfBackwardWs[backwardSrc][b])
    # Now, simply find the subhorizon in the receiving process whose
    # last period comes immediately before firstPeriodInSubh
    matchingSubh = 1e6
    for bb in range(params.nSubhorizons):
        if (firstPeriodInSubh - max(params.periodsPerSubhorizon[bb])) == 1:
            matchingSubh = bb
            break

    eventTracker.append(('partialDualSolutionRecv', '49', ' ', ' ',\
                            backwardSrc, bRank, ' ', ' ',\
                                itFromSource, b, it, matchingSubh, dt() - params.start))

    if matchingSubh < 1e6:
        ini = dt()

        nonZeros = np.where(np.abs(subgrads[b]) > 0)[0]
        constTerm = np.inner(subgrads[b][nonZeros], evaluatedSol[nonZeros])

        lhs = xsum(subgrads[b][i]*couplVarsRelax[matchingSubh][i] for i in nonZeros)

        if params.asynchronous:
            # if the optimization is asynchronous, or if it is synchronous but the sending
            # backward worker has a different number of subhorizons than the received backward
            # worker, then add the cut immediatelly
            optModelsRelax[matchingSubh].add_constr(betaRelax[matchingSubh]\
                                                        >= objValuesRelaxs[b] + lhs - constTerm,\
                        name = 'OptCutfrombRank' + str(backwardSrc) + '_subhFromSource' + str(b)\
                                                            + '_itFromSource' + str(itFromSource))
            eventTracker.append(('cutAdded', '71', ' ', ' ',\
                                backwardSrc, bRank, ' ', ' ',\
                                    itFromSource, b, it, matchingSubh, dt() - params.start))
        else:
            # if it is synchronous and both sending and receiving backward workers have the same
            # number of subhorizons, then add the cuts latter
            params.delayedCuts[itFromSource][matchingSubh][backwardSrc][b] =\
                                (betaRelax[matchingSubh] >= objValuesRelaxs[b] + lhs - constTerm,\
                                'OptCutfrombRank' + str(backwardSrc) + '_subhFromSource' + str(b)\
                                                            + '_itFromSource' + str(itFromSource),\
                                                                backwardSrc, itFromSource, b)

        if not(params.asynchronous):
            if not(params.solveOnlyFirstSubhorizon):
                params.countCutsReceived[-1][itFromSource] += 1
            else:
                params.countCutsReceived[matchingSubh][itFromSource] += 1

        timeAddingCuts += dt() - ini

    return(timeCommunicating, timeAddingCuts)

def backwardStep(params, thermals, it, ub, lb, backwardInfo, objValRelax, optModels,optModelsRelax,\
                lbda, fixedVars, couplVars, couplVarsRelax, couplConstrsRelax,\
                alphaRelax, betaRelax, beta, alpha, redFlag,\
                bufferBackward,\
                bComm, bRank, status, eventTracker = None):
    '''Backward step in the DDiP'''

    bwPackageRecv = np.array([0, 0, 0, 0, 0, 0], dtype = 'd')

    firstSubhToSolve = params.nSubhorizons - 1 if not(params.solveOnlyFirstSubhorizon) else 0
    lastSubhToSolve = 0 if params.solveOnlyFirstSubhorizon else 1

    if it == 0:
        if params.I_am_a_forwardWorker:
            for b in range(firstSubhToSolve, lastSubhToSolve - 1, -1):
                if int(alpha[b].name[alpha[b].name.rfind('_') + 1:]) < (params.T - 1):
                    alpha[b].obj = 0

        for b in range(firstSubhToSolve, lastSubhToSolve - 1, -1):
            if int(alphaRelax[b].name[alphaRelax[b].name.rfind('_') + 1:]) < (params.T - 1):
                alphaRelax[b].obj = 0

    for b in range(params.nSubhorizons):
        if params.BDBackwardProb:
            optModelsRelax[b].outerIteration = it
        for k in backwardInfo[b].keys():
            backwardInfo[b][k].append(0)

    for b in range(firstSubhToSolve, lastSubhToSolve - 1, -1):
        totalTimeCommunicating, totalAddingCuts = 0, 0

        if b in params.synchronizationPoints and params.solveOnlyFirstSubhorizon and\
                        (params.countCutsReceived[b][it] < params.cutsToBeReceivedAtSynchPoints[b]):
            msgRecv = False
            while (redFlag == 0) and not(msgRecv):
                if (bComm.Iprobe(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)):
                    timeCommunicating, timeAddingCuts = recvSubgradFromAnotherBackwardWorker(\
                                                    params, it, optModelsRelax,\
                                                    couplVarsRelax, betaRelax, bwPackageRecv,\
                                                    bComm, bRank, eventTracker)
                    totalTimeCommunicating += timeCommunicating
                    totalAddingCuts += timeAddingCuts
                    if (params.countCutsReceived[b][it] == params.cutsToBeReceivedAtSynchPoints[b]):
                        msgRecv = True
                    elif (params.countCutsReceived[b][it] >params.cutsToBeReceivedAtSynchPoints[b]):
                        raise Exception(f'There is something wrong! Im bRank {bRank} and Ive got '+\
                                f'more messages than I should had received for my subhorizon {b}')
                else:
                    sleep(0.1)

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
                            eventTracker.append(('cutAdded', '153', ' ', ' ',\
                                                optCut[2], bRank, ' ', ' ',\
                                                optCut[3], optCut[4], it, subh, dt() -params.start))
                            del params.delayedCuts[it][subh][bRankOrigin][subhOrigin]
                            params.delayedCuts[it][subh][bRankOrigin][subhOrigin] = None

                            cutsAdded += 1

                assert cutsAdded == params.cutsToBeReceivedAtSynchPoints[b],\
                                            f'Im bRank {bRank}. '+\
                                            f'{cutsAdded} cuts were'+\
                                            ' added but the number of cuts added ' +\
                                            f'should be {params.cutsToBeReceivedAtSynchPoints[b]}'

        if redFlag != 0:
            break

        ini = dt()
        if params.BDSubhorizonProb or params.BDnetworkSubhorizon or\
                                                                not(params.I_am_a_forwardWorker):
            for t in [t for t in range(params.T) if t < min(params.periodsPerSubhorizon[b])]:
                for i in set(params.varsPerPeriod[t]):
                    couplConstrsRelax[b][i].rhs = fixedVars[i]

        if not(params.BDSubhorizonProb or params.BDnetworkSubhorizon)\
                                                                and params.I_am_a_forwardWorker:
            for i in params.binVarsPerSubh[b]:
                couplVars[b][i].var_type = CONTINUOUS

        if b > 0 and not(params.BDBackwardProb):
            tempFeasConstrs = addVIBasedOnRamp(params, thermals, b,\
                            optModelsRelax[b], couplConstrsRelax[b], couplVarsRelax[b], fixedVars)
        else:
            tempFeasConstrs = []

        optModelsRelax[b].reset()

        optModelsRelax[b].preprocess = params.preprocess

        if (params.solver == GUROBI):
            optModelsRelax[b].solver.set_int_param("ScaleFlag", 3)
            optModelsRelax[b].solver.set_dbl_param("BarConvTol", 1e-16)

        optModelsRelax[b].lp_method = LP_Method.BARRIER

        if params.solveOnlyFirstSubhorizon and b == 0:
            for i in params.binVarsPerSubh[b]:
                couplVarsRelax[b][i].var_type = CONTINUOUS

            msStatus = optModelsRelax[b].optimize(max_seconds = max(params.lastTime - dt(), 0))

        else:
            msStatus = optModelsRelax[b].optimize(max_seconds = max(params.lastTime - dt(), 0))

        end = dt()

        if params.BDBackwardProb:
            redFlag = optModelsRelax[b].redFlag

        backwardInfo[b]['gap'][-1] = 0
        backwardInfo[b]['time'][-1] = end - ini

        if msStatus in (OptS.OPTIMAL, OptS.FEASIBLE) or ((b==0) and params.solveOnlyFirstSubhorizon\
                    and (msStatus == OptS.NO_SOLUTION_FOUND)):

            f =open(params.outputFolder + "bRank" + str(bRank) + "_subhorizon" + str(b)+".txt",'a',\
                                                                        encoding='utf-8')
            f.write('\n\nThe total elapsed time is (sec): ' + str(dt() - params.start) + '\n\n\n')
            f.close()

            objBound = optModelsRelax[b].objective_bound

            if b == 0:
                lb = max(objBound, lb)

            backwardInfo[b]['lb'][-1] = objBound
            backwardInfo[b]['ub'][-1] = optModelsRelax[b].objective_value

            objValRelax[b] = objBound

            if b > 0:
                for t in [t for t in range(params.T) if t < min(params.periodsPerSubhorizon[b])]:
                    for i in set(params.varsPerPeriod[t]):
                        lbda[b][i] = couplConstrsRelax[b][i].pi

            if b in params.backwardSendPoints:
                bufferBackward[('package', it, b)] = [np.array([redFlag,ub,max(objBound,lb),it,b],\
                                                                                dtype = 'd'),None]
                bufferBackward[('subgrad', it, b)] = [deepcopy(lbda), None]
                bufferBackward[('objVals', it, b)] = [deepcopy(objValRelax), None]

                bufferBackward[('package', it, b)][1] = bComm.Isend(\
                                                            [bufferBackward[('package', it, b)][0],\
                                                                MPI.DOUBLE], dest = 0, tag = 35)
                bufferBackward[('subgrad', it, b)][1] = bComm.Isend(\
                                                            [bufferBackward[('subgrad', it, b)][0],\
                                                                MPI.DOUBLE], dest = 0, tag = 36)
                bufferBackward[('objVals', it, b)][1] = bComm.Isend(\
                                                            [bufferBackward[('objVals', it, b)][0],\
                                                                MPI.DOUBLE], dest = 0, tag = 37)

                eventTracker.append(('partialDualSolutionSent', '225', ' ', ' ',\
                                        bRank, ' ', ' ', ' ',\
                                            it, b, ' ', ' ',\
                                                dt()-params.start))

            if b > 0 and params.asynchronous:
                msgRecv = True
                while msgRecv:
                    if (bComm.Iprobe(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)):
                        timeCommunicating, timeAddingCuts = recvSubgradFromAnotherBackwardWorker(\
                                                    params, it, optModelsRelax,\
                                                    couplVarsRelax, betaRelax, bwPackageRecv,\
                                                    bComm, bRank, eventTracker)
                        totalTimeCommunicating += timeCommunicating
                        totalAddingCuts += timeAddingCuts
                    else:
                        msgRecv = False

            backwardInfo[b]['communication'][-1] += totalTimeCommunicating
            backwardInfo[b]['timeToAddCuts'][-1] += totalAddingCuts
            backwardInfo[b]['timeStamp'][-1] = dt() - params.start

            ini = dt()

            if b >= 1 and (bRank == 0 or ((bRank >= 1) and ((b - 1) >= lastSubhToSolve))):
                nonZeros = np.where(np.abs(lbda[b]) > 0)[0]
                constTerm = np.inner(lbda[b][nonZeros], fixedVars[nonZeros])

                if ((params.BDSubhorizonProb or params.BDnetworkSubhorizon) or\
                                not(params.I_am_a_forwardWorker)) and ((b - 1) >= lastSubhToSolve):
                    lhs = xsum(lbda[b][i]*couplVarsRelax[b-1][i] for i in nonZeros)

                    optModelsRelax[b - 1].add_constr(betaRelax[b - 1]\
                                    >= objValRelax[b] + lhs - constTerm,\
                                        name = 'OptCutfrombRank' + str(bRank) + '_subh' + str(b)\
                                            + '_it' + str(it))

                if params.I_am_a_forwardWorker:
                    lhs = xsum(lbda[b][i]*couplVars[b - 1][i] for i in nonZeros)

                    optModels[b - 1].add_constr(beta[b - 1] >= objValRelax[b] + lhs - constTerm,\
                                        name = 'OptCutfrombRank' + str(bRank) + '_subh' + str(b)\
                                            + '_it' + str(it))

                    if params.BDSubhorizonProb:
                        # get the indices corresponding to continuous variables whose coefficients
                        # in this cut are negative
                        indsContNeg = np.intersect1d(np.where(lbda[b] < 0)[0],\
                                                    params.contVarsInPreviousAndCurrentSubh[b - 1])
                        indsContPos = np.intersect1d(np.where(lbda[b] > 0)[0],\
                                                    params.contVarsInPreviousAndCurrentSubh[b - 1])
                        # multiply them by their respective upper bounds to get a
                        # 'maximum negative term', i.e., a lower bound on this term. Do the
                        # same for the positive coefficients, but then use the lower bounds
                        constTermMP = np.inner(lbda[b][indsContNeg],\
                                                                params.ubOnCouplVars[indsContNeg])\
                                        + np.inner(lbda[b][indsContPos],\
                                                                params.lbOnCouplVars[indsContPos])
                        # now get the nonzero coefficients of the binary variables
                        indsOfBins = np.intersect1d(nonZeros,\
                                                    params.binVarsInPreviousAndCurrentSubh[b-1])
                        lhsMP = xsum(lbda[b][i]*optModels[b-1].copyVars[i] for i in indsOfBins)

                        optModels[b - 1].add_constr_MP(optModels[b-1].alphaVarMP >= objValRelax[b]+\
                                                    lhsMP + constTermMP - constTerm,\
                                        name = 'OptCutfrombRank' + str(bRank) + '_subh' + str(b)\
                                            + '_it' + str(it))

            if b > 0 and not(params.BDBackwardProb):
                optModelsRelax[b].remove(tempFeasConstrs)

            backwardInfo[b]['timeToAddCuts'][-1] += dt() - ini

        elif msStatus in (OptS.OTHER, OptS.NO_SOLUTION_FOUND):
            # Time limit reached
            redFlag = np.array(1, dtype = 'int')
            f =open(params.outputFolder + "bRank" + str(bRank) + "_subhorizon" + str(b)+".txt",'a',\
                                                                        encoding='utf-8')
            f.write('\n\nThe total elapsed time is (sec): ' + str(dt() - params.start) + '\n\n')
            f.close()

        else:
            f =open(params.outputFolder + "bRank" + str(bRank) + "_subhorizon" + str(b)+".txt",'a',\
                                                                        encoding='utf-8')
            f.write('\n\nThe total elapsed time is (sec): ' + str(dt() - params.start) + '\n\n')
            f.close()

            optModelsRelax[b].write('178rank' + str(bRank) + 'BackwardProblem' + str(b) + '.lp')
            optModelsRelax[b].write('178rank' + str(bRank) + 'BackwardProblem' + str(b) + '.mps')

            raise Exception(\
                f'Im rank {bRank}. The status of my backward subhorizon problem {b} is {msStatus}')

    return(redFlag, ub, lb, lbda)
