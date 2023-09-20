# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""

from copy import deepcopy
from time import sleep
from timeit import default_timer as dt

import numpy as np
from mpi4py import MPI

from optoptions import Coupling

from solver_interface.opt_model import LP_Method, OptimizationStatus as OptS

def recvSubgradFromAnotherBackwardWorker(params, it, optModelsRelax, couplVarsRelax, betaRelax,
                                                        bwPackageRecv, bComm, bRank, event_tracker):
    """Receive a subgradient from another backward worker"""

    time_communicating, timeAddingCuts = dt(), 0

    bComm.Recv([bwPackageRecv, MPI.DOUBLE], source = 0, tag = 31)
    backwardSrc = int(bwPackageRecv[0])

    objValuesRelaxs = np.zeros(params.PERIODS_OF_BACKWARD_WORKERS[backwardSrc].shape[0],
                                                                dtype = 'd')
    subgrads = np.zeros((params.PERIODS_OF_BACKWARD_WORKERS[backwardSrc].shape[0],
                                                                params.N_COMPL_VARS), dtype = 'd')
    evaluatedSol = np.zeros(params.N_COMPL_VARS, dtype = 'd')

    bComm.Recv([objValuesRelaxs, MPI.DOUBLE], source = 0, tag = 32)
    bComm.Recv([subgrads, MPI.DOUBLE], source = 0, tag = 33)
    bComm.Recv([evaluatedSol, MPI.DOUBLE], source = 0, tag = 34)

    time_communicating = dt() - time_communicating

    # The subhorizon from which the subgradient was taken
    itFromSource, b = int(bwPackageRecv[4]), int(bwPackageRecv[5])

    firstPeriodInSubh = min(params.PERIODS_OF_BACKWARD_WORKERS[backwardSrc][b])
    # Now, simply find the subhorizon in the receiving process whose
    # last period comes immediately before firstPeriodInSubh
    matchingSubh = 1e6
    for bb in range(params.N_SUBHORIZONS):
        if (firstPeriodInSubh - max(params.PERIODS_PER_SUBH[bb])) == 1:
            matchingSubh = bb
            break

    event_tracker.append(('partialDualSolutionRecv', '49', ' ', ' ',
                            backwardSrc, bRank, ' ', ' ',
                                itFromSource, b, it, matchingSubh, dt() - params.START))

    if matchingSubh < 1e6:
        ini = dt()

        nonZeros = np.where(np.abs(subgrads[b]) > 0)[0]
        constTerm = np.inner(subgrads[b][nonZeros], evaluatedSol[nonZeros])

        lhs = m.xsum(subgrads[b][i]*couplVarsRelax[matchingSubh][i] for i in nonZeros)

        if params.ASYNCHRONOUS:
            # if the optimization is asynchronous, or if it is synchronous but the sending
            # backward worker has a different number of subhorizons than the received backward
            # worker, then add the cut immediatelly
            optModelsRelax[matchingSubh].add_constr(betaRelax[matchingSubh]
                                                        >= objValuesRelaxs[b] + lhs - constTerm,
                        name = 'OptCutfrombRank' + str(backwardSrc) + '_subhFromSource' + str(b)
                                                            + '_itFromSource' + str(itFromSource))
            event_tracker.append(('cutAdded', '71', ' ', ' ',\
                                backwardSrc, bRank, ' ', ' ',\
                                    itFromSource, b, it, matchingSubh, dt() - params.START))
        else:
            # if it is synchronous and both sending and receiving backward workers have the same
            # number of subhorizons, then add the cuts latter
            params.DELAYED_CUTS[itFromSource][matchingSubh][backwardSrc][b] =\
                                (betaRelax[matchingSubh] >= objValuesRelaxs[b] + lhs - constTerm,
                                'OptCutfrombRank' + str(backwardSrc) + '_subhFromSource' + str(b)
                                                            + '_itFromSource' + str(itFromSource),
                                                                backwardSrc, itFromSource, b)

        if not(params.ASYNCHRONOUS):
            if not(params.SOLVE_ONLY_1st_SUBH):
                params.COUNT_CUTS_RECVD[-1][itFromSource] += 1
            else:
                params.COUNT_CUTS_RECVD[matchingSubh][itFromSource] += 1

        timeAddingCuts += dt() - ini

    return(time_communicating, timeAddingCuts)

def backwardStep(params, thermals, it, ub, lb, backwardInfo, objValRelax, optModels,optModelsRelax,
                lbda, fixedVars, couplVars, couplVarsRelax, couplConstrsRelax,
                alphaRelax, betaRelax, beta, alpha, redFlag,
                bufferBackward,
                bComm, bRank, status, event_tracker = None):
    """Backward step in the DDiP"""

    bwPackageRecv = np.array([0, 0, 0, 0, 0, 0], dtype = 'd')

    firstSubhToSolve = params.N_SUBHORIZONS - 1 if not(params.SOLVE_ONLY_1st_SUBH) else 0
    lastSubhToSolve = 0 if params.SOLVE_ONLY_1st_SUBH else 1

    if it == 0:
        if params.I_AM_A_FORWARD_WORKER:
            for b in range(firstSubhToSolve, lastSubhToSolve - 1, -1):
                if (int(optModelsRelax[b].get_name(alpha[b])
                            [optModelsRelax[b].get_name(alpha[b]).rfind('_')+ 1:])
                                    < (params.T - 1)):
                    optModelsRelax[b].set_obj_coeff(alpha[b], 0)

        for b in range(firstSubhToSolve, lastSubhToSolve - 1, -1):
            if (int(optModelsRelax[b].get_name(alphaRelax[b])
                        [optModelsRelax[b].get_name(alphaRelax[b]).rfind('_')+ 1:])
                                < (params.T - 1)):
                optModelsRelax[b].set_obj_coeff(alphaRelax[b], 0)

    for b in range(params.N_SUBHORIZONS):
        if params.BD_BACKWARD_PROB:
            optModelsRelax[b].outerIteration = it
        for k in backwardInfo[b].keys():
            backwardInfo[b][k].append(0)

    for b in range(firstSubhToSolve, lastSubhToSolve - 1, -1):
        totaltime_communicating, totalAddingCuts = 0, 0

        if b in params.SYNCH_POINTS and params.SOLVE_ONLY_1st_SUBH and\
                    (params.COUNT_CUTS_RECVD[b][it] < params.CUTS_TO_BE_RECVD_AT_SYNCH_POINTS[b]):
            msgRecv = False
            while (redFlag == 0) and not(msgRecv):
                if (bComm.Iprobe(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)):
                    time_communicating, timeAddingCuts = recvSubgradFromAnotherBackwardWorker(
                                                    params, it, optModelsRelax,
                                                    couplVarsRelax, betaRelax, bwPackageRecv,
                                                    bComm, bRank, event_tracker)
                    totaltime_communicating += time_communicating
                    totalAddingCuts += timeAddingCuts
                    if (params.COUNT_CUTS_RECVD[b][it]
                            == params.CUTS_TO_BE_RECVD_AT_SYNCH_POINTS[b]):
                        msgRecv = True
                    elif (params.COUNT_CUTS_RECVD[b][it]
                            > params.CUTS_TO_BE_RECVD_AT_SYNCH_POINTS[b]):
                        raise ValueError(f'There is something wrong! Im bRank {bRank} and Ive got '+
                                f'more messages than I should had received for my subhorizon {b}')
                else:
                    sleep(0.1)

            cutsAdded = 0
            if (redFlag != 1) and not(params.ASYNCHRONOUS):
                for subh in params.DELAYED_CUTS[it]:
                    for bRankOrigin in params.DELAYED_CUTS[it][subh]:
                        for subhOrigin in [subhOrigin for subhOrigin in
                                    params.DELAYED_CUTS[it][subh][bRankOrigin]
                            if params.DELAYED_CUTS[it][subh][bRankOrigin][subhOrigin] is not None
                                and params.DELAYED_CUTS[it][subh][bRankOrigin][subhOrigin][3]==it]:
                            optCut = params.DELAYED_CUTS[it][subh][bRankOrigin][subhOrigin]
                            optModelsRelax[subh].add_constr(optCut[0], name = optCut[1])
                            event_tracker.append(('cutAdded', '153', ' ', ' ',
                                                optCut[2], bRank, ' ', ' ',
                                                optCut[3], optCut[4], it, subh, dt() -params.START))
                            del params.DELAYED_CUTS[it][subh][bRankOrigin][subhOrigin]
                            params.DELAYED_CUTS[it][subh][bRankOrigin][subhOrigin] = None

                            cutsAdded += 1

                assert cutsAdded == params.CUTS_TO_BE_RECVD_AT_SYNCH_POINTS[b],\
                                        (f'Im bRank {bRank}. '+
                                        f'{cutsAdded} cuts were'+
                                        ' added but the number of cuts added ' +
                                        f'should be {params.CUTS_TO_BE_RECVD_AT_SYNCH_POINTS[b]}')

        if redFlag != 0:
            break

        ini = dt()
        if (params.BD_SUBHORIZON_PROB or params.BD_NETWORK_SUBHORIZON or
                                                                not(params.I_AM_A_FORWARD_WORKER)):

            if params.COUPLING == Coupling.CONSTRS:
                for t in [t for t in range(params.T) if t < min(params.PERIODS_PER_SUBH[b])]:
                    for i in set(params.VARS_PER_PERIOD[t]):
                        couplConstrsRelax[b][i].rhs = fixedVars[i]
            else:
                for t in [t for t in range(params.T) if t < min(params.PERIODS_PER_SUBH[b])]:
                    for i in set(params.VARS_PER_PERIOD[t]):
                        optModelsRelax[b].set_lb(couplConstrsRelax[b][i], fixedVars[i])
                        optModelsRelax[b].set_ub(couplConstrsRelax[b][i], fixedVars[i])

        if (not(params.BD_SUBHORIZON_PROB or params.BD_NETWORK_SUBHORIZON)
                                                                and params.I_AM_A_FORWARD_WORKER):
            for i in params.BIN_VARS_PER_SUBH[b]:
                optModelsRelax[b].set_var_type(couplVars[b][i], 'C')

        optModelsRelax[b].reset()

        optModelsRelax[b].lp_method = LP_Method.BARRIER

        if params.SOLVE_ONLY_1st_SUBH and b == 0:
            for i in params.BIN_VARS_PER_SUBH[b]:
                optModelsRelax[b].set_var_type(couplVarsRelax[b][i], 'C')

            msStatus = optModelsRelax[b].optimize(max_seconds = max(params.LAST_TIME - dt(), 0))

        else:

            msStatus = optModelsRelax[b].optimize(max_seconds = max(params.LAST_TIME - dt(), 0))

        end = dt()

        if params.BD_BACKWARD_PROB:
            redFlag = optModelsRelax[b].redFlag

        backwardInfo[b]['gap'][-1] = 0
        backwardInfo[b]['time'][-1] = end - ini

        if msStatus in (OptS.OPTIMAL, OptS.FEASIBLE) or ((b==0) and params.SOLVE_ONLY_1st_SUBH
                    and (msStatus == OptS.NO_SOLUTION_FOUND)):

            f =open(params.OUT_DIR + "bRank" + str(bRank) + "_subhorizon" + str(b)+".txt",'a',
                                                                        encoding='utf-8')
            f.write('\n\nThe total elapsed time is (sec): ' + str(dt() - params.START) + '\n\n\n')
            f.close()

            objBound = optModelsRelax[b].objective_bound

            if b == 0:
                lb = max(objBound, lb)

            backwardInfo[b]['lb'][-1] = objBound
            backwardInfo[b]['ub'][-1] = optModelsRelax[b].objective_value

            objValRelax[b] = objBound

            if b > 0:
                if params.COUPLING == Coupling.CONSTRS:
                    for t in [t for t in range(params.T) if t < min(params.PERIODS_PER_SUBH[b])]:
                        for i in set(params.VARS_PER_PERIOD[t]):
                            lbda[b][i] = couplConstrsRelax[b][i].pi
                else:
                    for t in [t for t in range(params.T) if t < min(params.PERIODS_PER_SUBH[b])]:
                        for i in set(params.VARS_PER_PERIOD[t]):
                            lbda[b][i] = optModelsRelax[b].get_var_red_cost(couplConstrsRelax[b][i])

            if b in params.BACKWARD_SEND_POINTS:
                bufferBackward[('package', it, b)] = [np.array([redFlag,ub,max(objBound,lb),it,b],
                                                                                dtype = 'd'),None]
                bufferBackward[('subgrad', it, b)] = [deepcopy(lbda), None]
                bufferBackward[('objVals', it, b)] = [deepcopy(objValRelax), None]

                bufferBackward[('package', it, b)][1] = bComm.Isend(
                                                            [bufferBackward[('package', it, b)][0],
                                                                MPI.DOUBLE], dest = 0, tag = 35)
                bufferBackward[('subgrad', it, b)][1] = bComm.Isend(
                                                            [bufferBackward[('subgrad', it, b)][0],
                                                                MPI.DOUBLE], dest = 0, tag = 36)
                bufferBackward[('objVals', it, b)][1] = bComm.Isend(
                                                            [bufferBackward[('objVals', it, b)][0],
                                                                MPI.DOUBLE], dest = 0, tag = 37)

                event_tracker.append(('partialDualSolutionSent', '225', ' ', ' ',
                                        bRank, ' ', ' ', ' ',
                                            it, b, ' ', ' ',
                                                dt()-params.START))

            if b > 0 and params.ASYNCHRONOUS:
                msgRecv = True
                while msgRecv:
                    if (bComm.Iprobe(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)):
                        time_communicating, timeAddingCuts = recvSubgradFromAnotherBackwardWorker(
                                                    params, it, optModelsRelax,
                                                    couplVarsRelax, betaRelax, bwPackageRecv,
                                                    bComm, bRank, event_tracker)
                        totaltime_communicating += time_communicating
                        totalAddingCuts += timeAddingCuts
                    else:
                        msgRecv = False

            backwardInfo[b]['communication'][-1] += totaltime_communicating
            backwardInfo[b]['timeToAddCuts'][-1] += totalAddingCuts
            backwardInfo[b]['timeStamp'][-1] = dt() - params.START

            ini = dt()

            if b >= 1 and (bRank == 0 or ((bRank >= 1) and ((b - 1) >= lastSubhToSolve))):
                nonZeros = np.where(np.abs(lbda[b]) > 0)[0]
                constTerm = np.inner(lbda[b][nonZeros], fixedVars[nonZeros])

                if (((params.BD_SUBHORIZON_PROB or params.BD_NETWORK_SUBHORIZON) or
                            not(params.I_AM_A_FORWARD_WORKER)) and ((b - 1) >= lastSubhToSolve)):
                    lhs = optModelsRelax[b - 1].xsum(
                                                    lbda[b][i]*couplVarsRelax[b-1][i]
                                                        for i in nonZeros
                                                    )

                    optModelsRelax[b - 1].add_constr(
                                                        betaRelax[b - 1]
                                                            >= objValRelax[b] + lhs - constTerm,
                                                                name = f"opt_cut_from_bRank_{bRank}"
                                                                + f"_subh_{b}_it_{it}"
                                                    )

                if params.I_AM_A_FORWARD_WORKER:
                    lhs = optModels[b - 1].xsum(lbda[b][i]*couplVars[b - 1][i] for i in nonZeros)

                    optModels[b - 1].add_constr(
                                                    beta[b - 1] >= objValRelax[b] + lhs - constTerm,
                                                            name = f"opt_cut_from_bRank_{bRank}"
                                                                        + f"_subh_{b}_it_{it}"
                                                )

                    if params.BD_SUBHORIZON_PROB:
                        # get the indices corresponding to continuous variables whose coefficients
                        # in this cut are negative
                        indsContNeg = np.intersect1d(np.where(lbda[b] < 0)[0],
                                                params.CON_VARS_IN_PREVIOUS_AND_CURRENT_SUBH[b - 1])
                        indsContPos = np.intersect1d(np.where(lbda[b] > 0)[0],
                                                params.CON_VARS_IN_PREVIOUS_AND_CURRENT_SUBH[b - 1])
                        # multiply them by their respective upper bounds to get a
                        # 'maximum negative term', i.e., a lower bound on this term. Do the
                        # same for the positive coefficients, but then use the lower bounds
                        constTermMP = (np.inner(lbda[b][indsContNeg],
                                                            params.UB_ON_COUPL_VARS[indsContNeg])
                                        + np.inner(lbda[b][indsContPos],
                                                            params.LB_ON_COUPL_VARS[indsContPos]))
                        # now get the nonzero coefficients of the binary variables
                        indsOfBins = np.intersect1d(nonZeros,
                                                params.BIN_VARS_IN_PREVIOUS_AND_CURRENT_SUBH[b-1])
                        lhsMP = optModels[b - 1].xsum(
                                                        lbda[b][i]*optModels[b-1].copyVars[i]
                                                            for i in indsOfBins
                                                    )

                        optModels[b - 1].add_constr_MP(optModels[b-1].alphaVarMP >= objValRelax[b]+
                                                    lhsMP + constTermMP - constTerm,\
                                        name = 'OptCutfrombRank' + str(bRank) + '_subh' + str(b)
                                            + '_it' + str(it))

            backwardInfo[b]['timeToAddCuts'][-1] += dt() - ini

        elif msStatus in (OptS.OTHER, OptS.NO_SOLUTION_FOUND):
            # Time limit reached
            redFlag = np.array(1, dtype = 'int')
            f =open(params.OUT_DIR + "bRank" + str(bRank) + "_subhorizon" + str(b)+".txt",'a',
                                                                        encoding='utf-8')
            f.write('\n\nThe total elapsed time is (sec): ' + str(dt() - params.START) + '\n\n')
            f.close()

        else:
            f =open(params.OUT_DIR + "bRank" + str(bRank) + "_subhorizon" + str(b)+".txt",'a',
                                                                        encoding='utf-8')
            f.write('\n\nThe total elapsed time is (sec): ' + str(dt() - params.START) + '\n\n')
            f.close()

            optModelsRelax[b].write('178rank' + str(bRank) + 'BackwardProblem' + str(b) + '.lp')
            optModelsRelax[b].write('178rank' + str(bRank) + 'BackwardProblem' + str(b) + '.mps')

            raise ValueError(
                f'Im rank {bRank}. The status of my backward subhorizon problem {b} is {msStatus}')

    return(redFlag, ub, lb, lbda)
