# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""

import locale
from copy import deepcopy
from math import ceil
from timeit import default_timer as dt
from mip import BINARY, LP_Method, SearchEmphasis, xsum, GUROBI
from mip import OptimizationStatus as OptS
import numpy as np
from mpi4py import MPI

locale.setlocale(locale.LC_ALL, 'en_US.utf-8')

def addVIBasedOnRamp(params, thermals, b, m, couplConstrs, couplVars, fixedVars):
    '''The thermal generating unit's previous state plus its ramping down limit may prevent it
    from being shut down. Thus, valid inequalities can be generated'''
    tempFeasConstrs = []    # list of valid inequalities.

    precedingTime = -1
    for t in [t for t in (set(range(params.T)) - params.periodsPerSubhorizon[b])\
                                                    if t <= max(params.periodsPerSubhorizon[b])]:
        precedingTime = max(t, precedingTime)

    listOfT = list(params.periodsPerSubhorizon[b])
    listOfT.sort()

    if precedingTime > -1:
        for g in range(len(thermals.id)):
            periodsInDisp = []

            previousGenValue = fixedVars[params.map['DpGenTG'][g, precedingTime]]

            if previousGenValue > 0:
                pDecrease = 0

                lastT = max(listOfT)
                for t in listOfT:
                    pDecrease += thermals.rampDown[g]
                    if (previousGenValue - pDecrease) <= 0:
                        # The unit reaches the minimum at t
                        # and can be turned off at t + len(thermals.shutDownTraj[g]) + 1
                        lastT = t
                        break
                periodsInDisp = list(range(min(listOfT), lastT + 1,1))

                if len(periodsInDisp) > 0:
                    tempFeasConstrs.append(m.add_constr(xsum(couplVars[params.map['DpTG'][g, t]]\
                                                                for t in periodsInDisp) >=\
                                                                    len(periodsInDisp),\
                                                                    name = f'tempVI_disp_{g}'))
                    tempFeasConstrs.append(m.add_constr(xsum(couplVars[params.map['stDwTG'][g, t]]\
                                                            + couplVars[params.map['stUpTG'][g, t]]\
                                                                for t in periodsInDisp) <= 0,\
                                                        name = f'tempVI_shutDownAndStartUp_{g}'))
    return (tempFeasConstrs)

def recvAndAddCuts(params, it, optModels, couplVars, beta,\
                        objValuesRelaxs, subgrads, evaluatedSol, fwPackageRecv, fRank, fComm,\
                        eventTracker, counter, recvAllBackwards):
    '''Receive cuts from the general coordinator and add them'''

    timeCommunicating = dt()

    fComm.Recv([fwPackageRecv, MPI.DOUBLE], source = 0, tag = 10)

    backwardSrc = int(fwPackageRecv[0])
    itFromSource = int(fwPackageRecv[4])

    fComm.Recv([objValuesRelaxs[backwardSrc], MPI.DOUBLE], source = 0, tag = 11)
    fComm.Recv([subgrads[backwardSrc], MPI.DOUBLE], source = 0, tag = 12)
    fComm.Recv([evaluatedSol, MPI.DOUBLE], source = 0, tag = 13)

    timeCommunicating = dt() - timeCommunicating

    eventTracker.append(('dualSolRecv', '72', ' ', ' ', backwardSrc, ' ', ' ', fRank,\
                                itFromSource, ' ', it, ' ', dt() - params.start))

    counter += 1

    if (counter == len(params.backwardWorkers)) and params.asynchronous:
        recvAllBackwards = True

    timeAddingCuts = 0

    for backwardSubh in range(params.periodsOfBackwardWs[backwardSrc].shape[0] - 1, 0, -1):
        # Find the corresponding subhorizon in the forward process that comes immediately before
        # the beginning of subhorizon b. To that end, find the first period in subhorizon b
        firstPeriodInSubh = min(params.periodsOfBackwardWs[backwardSrc][backwardSubh])
        # Now, simply find the subhorizon in the forward process whose
        # last period comes immediately before firstPeriodInSubh
        forwSubh = 1e6
        for fb in range(params.nSubhorizons):
            if (firstPeriodInSubh - max(params.periodsPerSubhorizon[fb])) == 1:
                forwSubh = fb
                break

        if forwSubh < 1e6:
            # then the newly received subgradient can be added to one of the forward's subhorizons.
            # specifically, subhorizon forwSubh
            eventTracker.append(('cutAdded', '97', ' ', ' ',\
                                backwardSrc, ' ', ' ', fRank,\
                                    itFromSource, backwardSubh, it, forwSubh, dt() - params.start))

            ini = dt()

            nonZeros = np.where(np.abs(subgrads[backwardSrc][backwardSubh]) > 0)[0]
            constTerm = np.inner(subgrads[backwardSrc][backwardSubh][nonZeros],\
                                                                            evaluatedSol[nonZeros])

            lhs = xsum(subgrads[backwardSrc][backwardSubh][i]*couplVars[forwSubh][i]\
                                                                        for i in nonZeros)

            optModels[forwSubh].add_constr(beta[forwSubh] >=\
                                    objValuesRelaxs[backwardSrc][backwardSubh] + lhs - constTerm,\
                                                    name = 'OptCutfrombRank' + str(backwardSrc) +\
                                                        '_subhFromSource' + str(backwardSubh)\
                                                            + '_itFromSource' + str(itFromSource))

            if params.BDSubhorizonProb:
                # get the indices corresponding to continuous variables whose coefficients
                # in this cut are negative
                indsContNeg = np.intersect1d(np.where(subgrads[backwardSrc][backwardSubh] < 0)[0],\
                                                params.contVarsInPreviousAndCurrentSubh[forwSubh])
                indsContPos = np.intersect1d(np.where(subgrads[backwardSrc][backwardSubh] > 0)[0],\
                                                params.contVarsInPreviousAndCurrentSubh[forwSubh])
                # multiply them by their respective upper bounds to get a
                # 'maximum negative term', i.e., a lower bound on this term
                constTermMP = np.inner(subgrads[backwardSrc][backwardSubh][indsContNeg],\
                                                        params.ubOnCouplVars[indsContNeg]) +\
                                np.inner(subgrads[backwardSrc][backwardSubh][indsContPos],\
                                                        params.lbOnCouplVars[indsContPos])
                # now get the nonzero coefficients of the binary variables
                indsOfBins=np.intersect1d(nonZeros,params.binVarsInPreviousAndCurrentSubh[forwSubh])
                lhsMP = xsum(subgrads[backwardSrc][backwardSubh][i]*\
                                                optModels[forwSubh].copyVars[i] for i in indsOfBins)

                optModels[forwSubh].add_constr_MP(optModels[forwSubh].alphaVarMP >=\
                                        objValuesRelaxs[backwardSrc][backwardSubh]+\
                                            lhsMP + constTermMP - constTerm,\
                                                    name = 'OptCutfrombRank' + str(backwardSrc) +\
                                                        '_subhFromSource' + str(backwardSubh)\
                                                            + '_itFromSource' + str(itFromSource))

            timeAddingCuts += dt() - ini

    return(recvAllBackwards, backwardSrc, counter, timeCommunicating, timeAddingCuts)

def solveSubhorizonProblem(params, b, it, ub, lb, gap, redFlag, optModels, fixedVars,\
        bestSol, previousSol, couplVars, presentCosts, futureCosts,  beta, alpha, fRank):
    '''Solve the subhorizon problem'''

    totalRunTime, dist, distBin, distStatusBin, distStatusBinBestSol = 0, 0, 0, 0, 0

    ini = dt()

    optModels[b].reset()
    if params.solver == GUROBI:
        optModels[b].solver.set_int_param("Presolve", 2)
        optModels[b].solver.set_int_param("ScaleFlag", 3)
        optModels[b].solver.set_dbl_param("BarConvTol", 1e-16)
        if b == 0 and it > 0:
            optModels[b].solver.set_dbl_param("BestBdStop", ub - params.relGapDDiP*ub)

    msStatus = optModels[b].optimize(max_seconds = max(params.lastTime - dt(), 0))

    totalRunTime += (dt() - ini)

    f = open(params.outputFolder + "fRank" + str(fRank) + "_subhorizon" + str(b)+".txt",'a',\
                                                                            encoding="utf-8")
    f.write('\n\nThe total elapsed time is (sec): ' + str(dt() - params.start) + '\n\n\n')
    f.close()

    if msStatus in (OptS.OPTIMAL, OptS.FEASIBLE):

        redFlag, dist, distBin, distStatusBin, distStatusBinBestSol, previousSol, ub, lb, gap =\
                                        checkConvergenceAndGetSolution(\
                                            params, b, it, ub, lb, redFlag, optModels, bestSol,\
                                                fixedVars, previousSol, presentCosts, futureCosts,\
                                                    beta, alpha, couplVars)

    elif msStatus in (OptS.OTHER, OptS.NO_SOLUTION_FOUND):
        redFlag = np.array(1, dtype = 'int')

    else:
        print(f'Im fRank {fRank}. My msStatus is {msStatus}', flush = True)
        for b2 in range(0, b + 1, 1):
            optModels[b2].write('fRank' + str(fRank) + '-Problem' + str(b2) + '.lp')
            optModels[b2].write('fRank' + str(fRank) + '-Problem' + str(b2) + '.mps')
        raise Exception('Im fRank ' + str(fRank) + '. Problem ' + str(b)\
                                                            + ' is not optimal: ' + str(msStatus))

    return(redFlag, totalRunTime,\
            dist, distBin, distStatusBin, distStatusBinBestSol, previousSol, msStatus, ub, lb, gap)

def prepareTheSubhProblem(params, thermals, b, it, optModels, couplVars,\
                        fixedVars, ub, lb, bestSol, couplConstrs, futureCosts):
    '''Change nature of variables, if necessary
        Change the rhs of the appropriate cosntraints to the latest solution'''

    tempFeasConstrs = []

    if it >= 1:
        if not(params.BDSubhorizonProb or params.BDnetworkSubhorizon):
            for i in [i for i in params.binVarsPerSubh[b]\
                                    if (couplVars[b][i].ub == 1) and (couplVars[b][i].lb == 0)]:
                couplVars[b][i].var_type = BINARY

    # Fix the decisions of previous subhorizons
    for t in [t for t in range(params.T) if t < min(params.periodsPerSubhorizon[b])]:
        for i in set(params.varsPerPeriod[t]):
            couplConstrs[b][i].rhs = fixedVars[i]

    if not(params.BDSubhorizonProb or params.BDnetworkSubhorizon):
        tempFeasConstrs = addVIBasedOnRamp(params, thermals, b,\
                                            optModels[b], couplConstrs[b], couplVars[b], fixedVars)

    optModels[b].lp_method = LP_Method.BARRIER
    optModels[b].SearchEmphasis = SearchEmphasis.OPTIMALITY

    if params.BDSubhorizonProb or params.BDnetworkSubhorizon:
        optModels[b].fixedVars = fixedVars
        optModels[b].bestSolUB = bestSol
        optModels[b].outerIteration = it
        if params.I_am_a_backwardWorker and (it >= 1) and (b >= 1):
            optModels[b].iniLB = np.array(futureCosts[b - 1], dtype = 'd')
        elif params.I_am_a_backwardWorker and (it >= 1) and (b == 0):
            optModels[b].iniLB = lb

    return(tempFeasConstrs)

def checkConvergenceAndGetSolution(params, b, it, ub, lb, redFlag, optModels, bestSol, fixedVars,\
                                    previousSol, presentCosts, futureCosts, beta, alpha, couplVars):
    '''Check convergence and get the solution from subhorizon b'''

    if ((b == 0 and it != 0)) or (params.nSubhorizons == 1):
        lb = max(optModels[0].objective_bound, lb)

    gap = (ub - lb)/ub

    if gap<=params.relGapDDiP and len(params.forwardWorkers)==1 and len(params.backwardWorkers)==1:
        redFlag = np.array(1, dtype = 'int')

    if not(params.BDSubhorizonProb or params.BDnetworkSubhorizon):
        sol = np.zeros((params.nComplVars, ), dtype = 'd')
        sol[params.varsInPreviousSubhs[b]] = fixedVars[params.varsInPreviousSubhs[b]]
        sol[params.varsPerSubh[b]] = np.array([couplVars[b][i].x\
                                                        for i in params.varsPerSubh[b]],dtype='d')
    else:
        sol = optModels[b].bestSol

    if it >= 1:
        dist = np.linalg.norm(previousSol[params.varsPerSubh[b]] - sol[params.varsPerSubh[b]])
        distBin = np.linalg.norm(previousSol[params.binVarsPerSubh[b]]\
                                                                - sol[params.binVarsPerSubh[b]])
        distStatusBin = np.linalg.norm(previousSol[params.binDispVarsPerSubh[b]]\
                                                            - sol[params.binDispVarsPerSubh[b]])
        distStatusBinBestSol = np.linalg.norm(bestSol[params.binDispVarsPerSubh[b]]\
                                                            - sol[params.binDispVarsPerSubh[b]])
    else:
        dist, distBin, distStatusBin, distStatusBinBestSol = 0.00, 0.00, 0.00, 0.00

    if not(params.BDSubhorizonProb or params.BDnetworkSubhorizon):
        if it == 0 and (b != (params.nSubhorizons - 1)):
            presentCosts[b] = optModels[b].objective_value - alpha[b].x
            futureCosts[b] = alpha[b].x if alpha[b].obj == 1 else beta[b].x

        else:
            if b != (params.nSubhorizons - 1):
                presentCosts[b] = optModels[b].objective_value - beta[b].x
                futureCosts[b] = beta[b].x
            else:
                presentCosts[b] = optModels[b].objective_value - alpha[b].x
                futureCosts[b] = alpha[b].x
    else:
        if (it == 0) or (b == (params.nSubhorizons - 1)):
            presentCosts[b] = optModels[b].objective_value - optModels[b].alpha
            if (b == (params.nSubhorizons - 1)):
                futureCosts[b] = optModels[b].alpha
            else:
                futureCosts[b] = optModels[b].alpha if it == 0 else optModels[b].beta

        else:
            presentCosts[b] = optModels[b].objective_value - optModels[b].beta
            futureCosts[b] = optModels[b].beta

    #### Round values to avoid numeric problems
    fixedVars[params.varsPerSubh[b]] = sol[params.varsPerSubh[b]]

    binaries = sol[params.binVarsPerSubh[b]]

    binaries[np.where(binaries <= 0.5)[0]] = 0
    binaries[np.where(binaries > 0.5)[0]] = 1

    fixedVars[params.binVarsPerSubh[b]] = binaries

    fixedVars[params.varsPerSubh[b]] = np.maximum(fixedVars[params.varsPerSubh[b]],\
                                                    params.lbOnCouplVars[params.varsPerSubh[b]])

    fixedVars[params.varsPerSubh[b]] = np.minimum(fixedVars[params.varsPerSubh[b]],\
                                                    params.ubOnCouplVars[params.varsPerSubh[b]])

    dispGen = fixedVars[params.dispGenVarsPerSubh[b]]
    dispGen[np.where(dispGen <= 1e-3)[0]] = 0
    fixedVars[params.dispGenVarsPerSubh[b]] = dispGen

    #### Store the current solution to keep track of how it changes over the iterations
    previousSol[params.varsPerSubh[b]] = sol[params.varsPerSubh[b]]\
                                    if not(params.BDSubhorizonProb or params.BDnetworkSubhorizon)\
                                        else optModels[b].bestSol[params.varsPerSubh[b]]

    return(redFlag, dist, distBin, distStatusBin, distStatusBinBestSol, previousSol, ub, lb, gap)

def printMetricsForCurrentSubhorizon(params, b: int, dist: float,\
                                distStatusBin: float, distStatusBinBestSol: float,\
                                presentCost: float, futureCost: float,\
                                totalRunTime: float, objVal: float, objBound: float) -> None:
    '''Print distances, costs, time, and gap for subhorizon b'''

    s = "\n\n" + f"Subhorizon: {b}" + "\t"

    s = s + "Dist. (cont, stsPrev, stsBest): " +\
                    f"({dist:.1f}, {distStatusBin**2:.1f}, {distStatusBinBestSol**2:.1f})" + "\t"

    s = s + "Present cost: " + locale.currency(presentCost/params.scalObjF, grouping = True) +\
            "\tFuture cost: " + locale.currency(futureCost/params.scalObjF, grouping = True)+"\t"
    s = s + "Total: " + locale.currency((presentCost + futureCost)/params.scalObjF, grouping=True)+\
                        "\t" + f"Runtime: {totalRunTime:.1f}" + "\t"
    s = s + f"Gap (%): {100*((objVal - objBound)/objVal):.4f}"

    print(s, flush = True)

    return()

def forwardStep(params, thermals,\
                it, subhorizonInfo, couplVars, fixedVars, previousSol, couplConstrs,\
                optModels, bestSol,\
                presentCosts, futureCosts,\
                alpha, beta, redFlag, ub, lb, gap, bufferForward,\
                status, objValuesRelaxs, subgrads, evaluatedSol, fwPackageRecv,\
                fComm, fRank, fSize, eventTracker = None):
    '''Forward step of the DDiP'''

    dualInfoFromMatchRecv = False   # True if the dual information from a matching backward process
                                    # was received while the forward process was solving the problem

    subhsDone = [0 for i in range(params.nSubhorizons)]# 1 if the subhorizon problem has been solved

    for b in range(params.nSubhorizons):
        for k in subhorizonInfo[b].keys():
            subhorizonInfo[b][k].append(0)

    for b in range(params.nSubhorizons):

        if redFlag != 0:
            break

        tempFeasConstrs = prepareTheSubhProblem(params, thermals, b, it,\
                                                        optModels, couplVars,\
                                                            fixedVars, ub, lb, bestSol,\
                                                                couplConstrs, futureCosts)

        redFlag, totalRunTime, dist, distBin, distStatusBin, distStatusBinBestSol,\
                    previousSol, msStatus, ub, lb, gap = solveSubhorizonProblem(params, b, it,\
                                                            ub, lb,\
                                                            gap, redFlag, optModels, fixedVars,\
                                                            bestSol, previousSol, couplVars,\
                                                            presentCosts, futureCosts,\
                                                            beta, alpha, fRank)

        if (redFlag == 0) and (msStatus in (OptS.OPTIMAL, OptS.FEASIBLE)):

            if (it != 0) and (b == 0) and (fSize == 1):
                lb = max(optModels[b].objective_bound, lb)
                gap = (ub - lb)/ub
                if gap <= params.relGapDDiP:
                    redFlag = np.array(1, dtype = 'int')

            if (optModels[b].objective_bound > lb) and (it > 0) and (b == 0) and (fSize > 1):
                lb = max(optModels[b].objective_bound, lb)
                gap = (ub - lb)/ub
                bufferForward[('lb', it, b)] = [np.array([lb, it, b], dtype = 'd'), None]
                bufferForward[('lb', it, b)][1] = fComm.Isend([bufferForward[('lb', it,b)][0],\
                                                                MPI.DOUBLE], dest = 0, tag = 29)
                eventTracker.append(('LBSent', '379', ' ', lb,\
                                        ' ', ' ', fRank, ' ',\
                                            it, b, ' ', ' ', dt() - params.start))

            if gap < 0:
                optModels[b].write('crazyLBfrom_fRank ' + str(fRank) + '.lp')
                optModels[b].write('crazyLBfrom_fRank ' + str(fRank) + '.mps')

                optModels[b].write(params.outputFolder +'crazyLBfrom_fRank ' + str(fRank) + '.lp')
                optModels[b].write(params.outputFolder +'crazyLBfrom_fRank ' + str(fRank) + '.mps')

            if fSize == 1:
                printMetricsForCurrentSubhorizon(params, b, dist, distStatusBin,\
                            distStatusBinBestSol, presentCosts[b], futureCosts[b], totalRunTime,\
                            optModels[b].objective_value, optModels[b].objective_bound)

            if (b < (params.nSubhorizons - 1)) and b in params.forwardSendPoints:
                # if the subhorizon's problem just solved is in params.forwardSendPoints, then
                # send the primal solution and the package to the general coordinator
                bufferForward[('package', it, b)] = [np.array([redFlag, ub, lb, it, b,\
                                                    sum(presentCosts[0:b + 1]), futureCosts[b]],\
                                                        dtype = 'd'), None]
                bufferForward[('sol', it, b)] = [deepcopy(fixedVars), None]
                bufferForward[('package', it, b)][1] = fComm.Isend([\
                                                            bufferForward[('package', it,b)][0],\
                                                                    MPI.DOUBLE], dest = 0, tag = 21)
                bufferForward[('sol', it, b)][1] = fComm.Isend([bufferForward[('sol', it,b)][0],\
                                                                    MPI.DOUBLE], dest = 0, tag = 22)
                eventTracker.append(('primalSolSent', '479', ' ', lb,\
                                        ' ', ' ', fRank, ' ',\
                                            it, b, ' ', ' ', dt() - params.start))

        elif msStatus in (OptS.OTHER, OptS.NO_SOLUTION_FOUND):
            if (optModels[b].objective_bound > lb) and (it > 0) and (b == 0):
                lb = max(optModels[b].objective_bound, lb)
                gap = (ub - lb)/ub
                if (fSize > 1):
                    bufferForward[('lb', it, b)] = [np.array([lb, it, b], dtype = 'd'), None]
                    bufferForward[('lb', it, b)][1] = fComm.Isend([bufferForward[('lb', it,b)][0],\
                                                                MPI.DOUBLE], dest = 0, tag = 29)
                    eventTracker.append(('LBSent', '492', ' ', lb,\
                                            ' ', ' ', fRank, ' ',\
                                                it, b,' ', ' ', dt() - params.start))


        if (redFlag == 0) and not(params.BDSubhorizonProb or params.BDnetworkSubhorizon):
            optModels[b].remove(tempFeasConstrs)

        subhorizonInfo[b]['time'][-1] = totalRunTime
        if params.BDSubhorizonProb or params.BDnetworkSubhorizon:
            subhorizonInfo[b]['iterations'][-1] = optModels[b].it
        subhorizonInfo[b]['gap'][-1] = optModels[b].gap
        subhorizonInfo[b]['optStatus'][-1] = msStatus

        if (redFlag == 0) or (gap <= params.relGapDDiP):
            subhsDone[b] = 1
            subhorizonInfo[b]['presentCots'][-1] = presentCosts[b]
            subhorizonInfo[b]['futureCosts'][-1] = futureCosts[b]
            subhorizonInfo[b]['distanceFromPreviousSol'][-1] = dist
            subhorizonInfo[b]['distBinVars'][-1] = distBin
            subhorizonInfo[b]['distStatusBinVars'][-1] = distStatusBin
            subhorizonInfo[b]['distStatusBinBestSol'][-1] = distStatusBinBestSol

        if redFlag != 0:
            break

        totalTimeCommunicating, totalTimeAddingCuts = 0, 0

        if (it > 0) and (fSize > 1) and (b < (params.nSubhorizons - 2)) and params.asynchronous:
            msgRecv = True
            while msgRecv:
                if (fComm.Iprobe(source = 0, tag = 10, status = status)):
                    _0, backwardSrc, _1, timeCommunicating, timeAddingCuts = recvAndAddCuts(\
                                        params, it, optModels, couplVars, beta,\
                                            objValuesRelaxs, subgrads, evaluatedSol,\
                                                fwPackageRecv, fRank, fComm, eventTracker, 0, False)

                    totalTimeCommunicating += timeCommunicating
                    totalTimeAddingCuts += timeAddingCuts

                    if params.periodsOfBackwardWs[backwardSrc].shape[0] == params.nSubhorizons:
                        # If the current forward worker has received subgradients from a
                        # backward worker with the same aggregation, then, there is enough
                        # information to continue without more cuts
                        dualInfoFromMatchRecv = True
                else:
                    msgRecv = False

        subhorizonInfo[b]['communication'][-1] += totalTimeCommunicating
        subhorizonInfo[b]['timeToAddCuts'][-1] += totalTimeAddingCuts
        subhorizonInfo[b]['timeStamp'][-1] = dt() - params.start

    if it == 0:
        for b in range(params.nSubhorizons - 1):
            if int(alpha[b].name[alpha[b].name.rfind('_') + 1:]) < (params.T - 1):
                alpha[b].obj = 0
            beta[b].obj = 1

    return (ub, lb, gap, redFlag, subhsDone, previousSol, dualInfoFromMatchRecv)
