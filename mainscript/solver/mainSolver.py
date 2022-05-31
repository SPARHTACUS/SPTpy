# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""

import locale
from timeit import default_timer as dt
from copy import deepcopy
from mpi4py import MPI
from mip import Model, GUROBI, BINARY, CONTINUOUS
import numpy as np

from solver.BD import BD
from solver.BDnetwork import BD_Network
from addCompToModel.addAllComponents import addAllComp
from solver.forwardParallel import forwardStepPar, forwardStep
from solver.backwardParallel import backwardStepPar, backwardStep
from solver.generalCoordinator import genCoord

locale.setlocale(locale.LC_ALL, 'en_US.utf-8')

def buildModels(params, redFlag, thermals, hydros, network, fixedVars,\
                fRank, bRank, wSize, buildingTimes):
    '''Build the optimization models'''

    # used to estimate subsequent subhorizons in forward
    beta = {b: None for b in range(params.nSubhorizons)}
    # used to estimate subsequent subhorizons in backward
    betaRelax={b:None for b in range(params.nSubhorizons)}
    alpha = {b: None for b in range(params.nSubhorizons)} # var of the cost-to-go func in forward
    # var of the cost-to-go func in backward
    alphaRelax = {b: None for b in range(params.nSubhorizons)}

    optModels = [None for b in range(params.nSubhorizons)]
    optModelsRelax = [None for b in range(params.nSubhorizons)]

    if params.BDSubhorizonProb or params.BDnetworkSubhorizon or params.BDBackwardProb:
        if params.I_am_a_forwardWorker:
            if params.BDSubhorizonProb:
                optModels = [BD(params, redFlag, thermals, hydros, b,\
                                'forwardSubhorizon'+str(b),\
                                solver_name = params.solver) for b in range(params.nSubhorizons)]
            if params.BDnetworkSubhorizon and not(params.BDSubhorizonProb):
                optModels = [BD_Network(params, redFlag, b, 'forwardSubhorizon'+str(b),\
                                solver_name = params.solver) for b in range(params.nSubhorizons)]
        else:
            optModels = [None for i in range(params.nSubhorizons)]

        if params.I_am_a_backwardWorker:
            if params.BDBackwardProb:
                optModelsRelax = [BD_Network(params, redFlag, b, 'backwardSubhorizon' + str(b),\
                                solver_name = params.solver) for b in range(params.nSubhorizons)]
            else:
                optModelsRelax = [Model('backwardSubhorizon' + str(b),\
                                solver_name = params.solver) for b in range(params.nSubhorizons)]
    else:
        # If no further decomposition is used, then the forward and backward models are the same
        optModels = [Model('dayAhead', solver_name = params.solver)\
                                                                for b in range(params.nSubhorizons)]
        optModelsRelax = [optModels[b] for b in range(params.nSubhorizons)]

    if params.I_am_a_forwardWorker:
        for b in range(params.nSubhorizons):
            if params.solver == GUROBI:
                optModels[b].solver.set_int_param("LogToConsole", params.verbose)
                optModels[b].solver.set_str_param("LogFile", params.outputFolder +\
                                            "fRank" + str(fRank) + "_subhorizon" + str(b) + ".txt")
            else:
                optModels[b].verbose = params.verbose

    if params.I_am_a_backwardWorker:
        firstSubhToSolve = params.nSubhorizons - 1 if not(params.solveOnlyFirstSubhorizon) else 0
        lastSubhToSolve = 0 if params.solveOnlyFirstSubhorizon or params.I_am_a_forwardWorker else 1

        for b in range(firstSubhToSolve, lastSubhToSolve - 1, -1):
            if params.solver == GUROBI:
                optModelsRelax[b].solver.set_int_param("LogToConsole", params.verbose)
                optModelsRelax[b].solver.set_str_param("LogFile", params.outputFolder +\
                                            "bRank" + str(bRank) + "_subhorizon" + str(b) + ".txt")
            else:
                optModelsRelax[b].verbose = params.verbose

    # coupling constraints in the subhorizons of the forward step. These are the equality
    # constraints that force that time-coupling decisions taken in previous subhorizons are
    # respected
    couplConstrs = {b: [] for b in range(params.nSubhorizons)}
    # coupling variables in the subhorizons of the forward step
    couplVars = {b: [] for b in range(params.nSubhorizons)}

    # the same as the above but now for the backward step. If no further decomposition is used,
    # then couplConstrsRelax is the same as couplConstrs, and couplVarsRelax == couplVars
    couplConstrsRelax = {b: [] for b in range(params.nSubhorizons)}
    couplVarsRelax = {b: [] for b in range(params.nSubhorizons)}

    if params.I_am_a_forwardWorker:
        for b in range(params.nSubhorizons):
            optModels[b].threads = params.threads
            optModels[b].max_mip_gap = params.gapMILPSubh
            optModels[b].preprocess = params.preprocess

            if params.solver == GUROBI:
                optModels[b].solver.set_int_param("ScaleFlag", 3)
                optModels[b].solver.set_dbl_param('NoRelHeurTime', 0)
                optModels[b].solver.set_dbl_param('NoRelHeurWork', 0)
                optModels[b].solver.set_dbl_param('Heuristics', params.heuristics)
            else:
                optModels[b].solver.set_str_param('heur', params.heuristics)

    if params.I_am_a_backwardWorker:
        for b in range(params.nSubhorizons):
            optModelsRelax[b].threads = params.threads
            optModelsRelax[b].preprocess = params.preprocess

    firstSubhToSolve = 0 if params.solveOnlyFirstSubhorizon or params.I_am_a_forwardWorker else 1
    lastSubhToSolve = params.nSubhorizons - 1 if not(params.solveOnlyFirstSubhorizon) else 0
    # note that, if params.solveOnlyFirstSubhorizon == True, then lastSubhToSolve is 0 and
    # firstSubhToSolve == 0
    # range(firstSubhToSolve, lastSubhToSolve + 1, 1) contains only 0. In other words,
    # only the first subhorizon problem is built
    for b in range(firstSubhToSolve, lastSubhToSolve + 1, 1):
        ini = dt()
        if params.BDSubhorizonProb or params.BDnetworkSubhorizon or params.BDBackwardProb:
            if params.I_am_a_forwardWorker:
                if params.BDSubhorizonProb:
                    couplConstrs[b], couplVars[b], alpha_, beta_ = optModels[b].addAllComp(\
                                                        params, hydros, thermals,\
                                                        network, fixedVars, b,\
                                                        binVars = BINARY)

                if params.BDnetworkSubhorizon and not(params.BDSubhorizonProb):
                    couplConstrs[b], couplVars[b], alpha_, beta_ = optModels[b].addAllComp(\
                                                        params, hydros, thermals,\
                                                        network, fixedVars, b,\
                                                        binVars = BINARY)
            else:
                alpha_, beta_ = None, None

            if params.I_am_a_backwardWorker:
                if params.BDBackwardProb:
                    varNature = BINARY if params.solveOnlyFirstSubhorizon and b == 0 else CONTINUOUS
                    couplConstrsRelax[b], couplVarsRelax[b], alph_, bet_ =\
                                                                optModelsRelax[b].addAllComp(\
                                                                params, hydros, thermals,\
                                                                network, fixedVars, b,\
                                                                binVars = varNature)
                else:
                    couplConstrsRelax[b], couplVarsRelax[b], alph_, bet_,\
                                            _0, _1, _2, _3, _4, _5, _6, _7 = addAllComp(params,\
                                            hydros, thermals, network,\
                                            optModelsRelax[b], optModelsRelax[b], optModelsRelax[b],
                                            b, fixedVars, BDbinaries = False,\
                                            BDnetwork = False, binVars = CONTINUOUS)
            else:
                alph_, bet_ = None, None

            betaRelax[b] = bet_
            alphaRelax[b] = alph_
        else:
            varNature = BINARY if (params.I_am_a_forwardWorker or\
                                (params.solveOnlyFirstSubhorizon and b == 0)) else CONTINUOUS
            # varNature is BINARY for forward processes, and for backward processes who will
            # only solve their respective first subhorizons
            couplConstrs[b], couplVars[b], alpha_, beta_, _0, _1, _2, _3, _4, _5, _6, _7 =\
                                                    addAllComp(params,\
                                                    hydros, thermals, network,\
                                                    optModels[b], optModels[b], optModels[b],\
                                                    b, fixedVars, BDbinaries = False,\
                                                    BDnetwork = False, binVars = varNature)

            couplConstrsRelax[b] = couplConstrs[b]
            couplVarsRelax[b] = couplVars[b]

            betaRelax[b] = beta_
            alphaRelax[b] = alpha_

        end = dt()

        buildingTimes[b] = end - ini

        beta[b] = beta_
        alpha[b] = alpha_

    return(beta, betaRelax, alpha, alphaRelax,\
            optModels, optModelsRelax, couplConstrs, couplVars, couplConstrsRelax, couplVarsRelax)

def runSolver(params, hydros, thermals, network,\
                wComm, wRank, wSize, fComm, fRank, fSize, bComm, bRank, bSize):
    '''Run the DDiP
    params:             an instance of OptOptions (optoptions.py) that contains the
                            parameters for the problem and the algorithm
    hydros:             an instance of Hydros (network.py) with all hydro data
    thermals:           an instance of Thermals (network.py) with all thermal data
    network:            an instance of Network (network.py) with all network data
    Comm, Rank, Size:   communicator, rank and size: w stands for the global parameters, i.e.,
                            wRank is the process' rank in the world communicator.
                            f stands for the forward communicator,
                            b stands for the backward communicator
    '''
    ub, lb = np.array(1e12, dtype = 'd'), np.array(0, dtype = 'd')

    redFlag = np.array(0, dtype = 'int')# use for broadcasting that either convergence or the time
                                        # limit has been reached

    # the following windows will be used by the general coordinator to push the updated values of
    # the redFlag, upper bound and lower bound. Only the general coordinator puts values in these
    # windows, while the workers can only retrieve values from them.
    if wSize > 1:
        params.winRedFlag = MPI.Win.Create(redFlag, 1, info = MPI.INFO_NULL, comm = wComm)# Red flag
        params.winUB = MPI.Win.Create(ub, 1, info = MPI.INFO_NULL, comm = wComm) # Upper bound
        params.winLB = MPI.Win.Create(lb, 1, info = MPI.INFO_NULL, comm = wComm) # Lower bound

    # present and future costs for each subhorizon
    presentCosts, futureCosts = [0 for i in range(params.nSubhorizons)],\
                                                        [0 for i in range(params.nSubhorizons)]

    # dictionary for storing data from the solution process
    pLog = {'lb': [], 'ub': [], 'gap': [], 'runTimeForward': [], 'runTimeBackward': []}

    # log for the forward subhorizon problems
    subhorizonInfo = {b: {'presentCots': [], 'futureCosts': [], 'time': [],\
                    'iterations': [], 'gap': [], 'distanceFromPreviousSol': [],\
                    'distBinVars': [], 'distStatusBinVars': [], 'distStatusBinBestSol': [],\
                    'optStatus': [], 'communication': [], 'timeToAddCuts': [], 'timeStamp': []}\
                    for b in range(params.nSubhorizons)}

    # log for the backward subhorizon problems
    backwardInfo = {b: {'lb': [], 'ub': [], 'gap': [], 'time': [], 'optStatus': [],\
                    'communication': [], 'timeToAddCuts': [], 'timeStamp': []}\
                    for b in range(params.nSubhorizons)}

    # dual variables' values associated with the time-coupling equality constraints
    lbda = np.zeros((params.nSubhorizons, params.nComplVars), dtype = 'd')
    # objective functions' values of the backward subhorizon problems
    objValRelax = np.zeros(params.nSubhorizons, dtype = 'd')

    gap, it = 1e12, 0  # relative gap, and iteration counter

    bestSol = np.zeros(params.nComplVars, dtype = 'd')          # best solution found so far
    fixedVars = np.zeros(params.nComplVars, dtype = 'd')        # current solution
    previousSol = np.zeros(params.nComplVars, dtype = 'd')      # solution from previous iteration

    buildingTimes = [0 for b in range(params.nSubhorizons)]     # times taken to build the
                                                                # subhorizon models

    beta, betaRelax, alpha, alphaRelax, optModels, optModelsRelax,\
            couplConstrs, couplVars, couplConstrsRelax, couplVarsRelax = buildModels(\
                                                        params, redFlag,\
                                                            thermals, hydros, network, fixedVars,\
                                                                fRank, bRank, wSize, buildingTimes)

    if params.I_am_a_backwardWorker:
        bufferBackward = {}

    if wSize == 1:
        print(f'{sum(buildingTimes):.2f} seconds building the subhorizon problems.')
        print(f'Max, min, and average (sec): ({max(buildingTimes):.2f}, {min(buildingTimes):.2f},'+\
                                    f' {sum(buildingTimes)/len(buildingTimes):.2f})', flush=True)
    if wSize > 1:
        if wRank == 0:
            bestSol, ub, lb, gap, redFlag, _666, _6661 =\
                                            genCoord(params, redFlag, ub, lb, it, bestSol, pLog,\
                                                        wRank, wSize, fComm, fSize, bComm, bSize)

        if params.I_am_a_forwardWorker:
            ub, lb, gap, redFlag = forwardStepPar(params, thermals,\
                                            it, subhorizonInfo, couplVars, fixedVars,\
                                            previousSol, couplConstrs, optModels, bestSol,\
                                            presentCosts, futureCosts,\
                                            alpha, beta, redFlag, ub, lb, gap,\
                                            wRank, fComm, fRank, fSize)

        if params.I_am_a_backwardWorker:
            ub, lb, gap, redFlag =  backwardStepPar(params, thermals,\
                                            it, ub, lb, backwardInfo, objValRelax,\
                                            optModels, optModelsRelax, lbda,\
                                            fixedVars, couplVars, couplVarsRelax,couplConstrsRelax,\
                                            alphaRelax, betaRelax, beta, alpha,\
                                            redFlag, bufferBackward,\
                                            wRank, bComm, bRank)

    while redFlag != 1:
        print('\n'+f'Iteration {it}.\tRemaining time (sec): {max(params.lastTime-dt(),0):.4f}',\
                                                                                        flush=True)

        totalRunTimeForward = dt()
        ub, lb, gap, redFlag, subhsDone, previousSol, _0 = forwardStep(params, thermals, it,\
                                        subhorizonInfo, couplVars, fixedVars, previousSol,\
                                            couplConstrs,\
                                            optModels, bestSol,\
                                                presentCosts, futureCosts,\
                                                alpha, beta, redFlag, ub, lb, gap,\
                                                    {}, {}, {}, {}, {}, {}, fComm, fRank, fSize, [])

        totalRunTimeForward = dt() - totalRunTimeForward

        if (redFlag == 0) and (sum(subhsDone) == params.nSubhorizons) and\
                                    (sum(presentCosts) + futureCosts[params.nSubhorizons - 1]) < ub:
            bestSol = deepcopy(fixedVars)
            ub = np.array(sum(presentCosts) + futureCosts[params.nSubhorizons - 1], dtype = 'd')

        gap = (ub - lb)/ub
        s = f"\n\nIteration: {it}\t\t"

        if (redFlag == 0) and (sum(subhsDone) == params.nSubhorizons):
            s = s + "Cost of current solution is: " + locale.currency((sum(presentCosts)\
                    + futureCosts[params.nSubhorizons - 1])/params.scalObjF, grouping = True)+"\t\t"

        s = s + "LB: " + locale.currency(lb/params.scalObjF, grouping = True) +\
                                "\t\tUB: " + locale.currency(ub/params.scalObjF, grouping = True) +\
                                    f"\t\tGap(%): {100*gap:.4f}\t\t"\
                        + f"Remaining time (sec): {max(params.lastTime - dt(), 0):.4f}"
        print(s, flush = True)

        if (redFlag != 1) and (gap <= params.relGapDDiP) or (it >= params.maxItDDiP):
            redFlag = np.array(1, dtype = 'int')

        lbda, objValRelax = np.zeros((params.nSubhorizons, params.nComplVars), dtype = 'd'),\
                                np.zeros(params.nSubhorizons, dtype = 'd')

        totalRunTimeBackward = dt()
        if redFlag != 1:
            redFlag, ub, lb, lbda = backwardStep(params, thermals,\
                                            it, ub, lb, backwardInfo, objValRelax,\
                                            optModels, optModelsRelax, lbda,\
                                            fixedVars, couplVars, couplVarsRelax,couplConstrsRelax,\
                                            alphaRelax, betaRelax, beta, alpha, redFlag,\
                                            {}, bComm, bRank, None)
        totalRunTimeBackward = dt() - totalRunTimeBackward

        gap = (ub - lb)/ub

        if wRank == 0:
            pLog['lb'].append(lb)
            pLog['ub'].append(ub)
            pLog['gap'].append(gap)
            pLog['runTimeForward'].append(totalRunTimeForward)
            pLog['runTimeBackward'].append(totalRunTimeBackward)

        it += 1

        if (redFlag != 1) and (gap <= params.relGapDDiP) or (it >= params.maxItDDiP):
            redFlag = np.array(1, dtype = 'int')

    wComm.Barrier()

    if params.I_am_a_forwardWorker:
        if params.BDSubhorizonProb:
            for b in range(params.nSubhorizons):
                f = open(params.outputFolder+'forwardBD - subhorizon '+str(b) +\
                        ' - fRank '+str(fRank)+' - ' + params.ps + ' - case ' + str(params.case) +\
                            '.csv','w',encoding='ISO-8859-1')
                for key in optModels[b].log.keys():
                    f.write(key + ';')
                f.write('\n')
                for i in range(len(optModels[b].log['DDiPIt'])):
                    for key in optModels[b].log.keys():
                        f.write(str(optModels[b].log[key][i]) + ';')
                    f.write('\n')
                f.close()

        if params.BDnetworkSubhorizon:
            for b in range(params.nSubhorizons):
                f = open(params.outputFolder + 'forwardNetworkBD - subhorizon ' + str(b) +\
                        ' - fRank '+str(fRank)+ ' - ' + \
                        params.ps + ' - case ' + str(params.case) +'.csv','w',encoding='ISO-8859-1')
                for key in optModels[b].log.keys():
                    f.write(key + ';')
                f.write('\n')
                for i in range(len(optModels[b].log['DDiPIt'])):
                    for key in optModels[b].log.keys():
                        f.write(str(optModels[b].log[key][i]) + ';')
                    f.write('\n')
                f.close()

    if params.I_am_a_backwardWorker and params.BDBackwardProb:
        for b in range(params.nSubhorizons):
            f = open(params.outputFolder + 'backwardNetworkBD - subhorizon '+ str(b) +\
                    ' - bRank '+str(bRank)+ ' - ' + \
                    params.ps + ' - case ' + str(params.case) +'.csv','w',encoding='ISO-8859-1')
            for key in optModelsRelax[b].log.keys():
                f.write(key + ';')
            f.write('\n')
            for i in range(len(optModelsRelax[b].log['DDiPIt'])):
                for key in optModelsRelax[b].log.keys():
                    f.write(str(optModelsRelax[b].log[key][i]) + ';')
                f.write('\n')
            f.close()

    if wRank == 0 and ub < 1e12:
        np.savetxt(params.outputFolder + 'bestSolutionFound - '+\
                                    params.ps + ' - case ' + str(params.case) + '.csv',\
                                    bestSol, fmt = '%.12f', delimiter=';')

    return(bestSol, ub, pLog, subhorizonInfo, backwardInfo)
