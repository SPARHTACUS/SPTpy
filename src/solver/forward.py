# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""

import locale
from copy import deepcopy
from timeit import default_timer as dt
from optoptions import Coupling
from solver_interface.opt_model import LP_Method, OptimizationStatus as OptS
import numpy as np
from mpi4py import MPI

locale.setlocale(locale.LC_ALL, '')

def add_VI_based_on_ramp(params, thermals, b, m, couplVars, fixed_vars):
    """
        The thermal generating unit's previous state plus its ramping down limit may prevent it
        from being shut down. Thus, valid inequalities can be generated.
        Because the state and generation of the unit might change over the iterations, these
        valid inequalities are temporary. they are removed at the beginning of the next iteration
    """
    temp_feas_constrs = []              # list of valid inequalities.

    precedingTime = -1
    for t in [t for t in (set(range(params.T)) - params.PERIODS_PER_SUBH[b])
                                                    if t <= max(params.PERIODS_PER_SUBH[b])]:
        precedingTime = max(t, precedingTime)

    subhorizon_periods = list(params.PERIODS_PER_SUBH[b])
    subhorizon_periods.sort()

    if precedingTime > -1:
        for g in thermals.ID:
            periods_in_disp = []        # a list of periods in which the unit must be on
                                        # based on its previous state and its local constraints

            previous_generation = fixed_vars[params.MAP['DpGenTG'][g, precedingTime]]

            if previous_generation > 0 and fixed_vars[params.MAP['DpTG'][g, precedingTime]] > 0:
                # only if the unit is generation something above its minimum and if it is
                # in the dispatch phase.
                # the second term in this chain is necessary to avoid adding constraints for
                # close-to-zero generation like previous_generation = 1e-6 (which might happen
                # even if the dispatch status is zero)
                p_decrease = 0

                lastT = max(subhorizon_periods)
                for t in subhorizon_periods:
                    p_decrease += thermals.RAMP_DOWN[g]
                    if (previous_generation - p_decrease) <= 0:
                        # The unit reaches the minimum at t
                        # and can be turned off at t + len(thermals.STDW_TRAJ[g]) + 1
                        lastT = t
                        break
                periods_in_disp = list(range(min(subhorizon_periods), lastT + 1,1))

                if len(periods_in_disp) > 0:
                    temp_feas_constrs.append(
                                            m.add_constr(
                                                m.xsum(couplVars[params.MAP['DpTG'][g, t]]
                                                    for t in periods_in_disp)>=len(periods_in_disp),
                                                        name = f'temp_VI_force_dispatch_disp_{g}'
                                                        )
                                        )
                    temp_feas_constrs.append(
                                            m.add_constr(
                                                m.xsum(couplVars[params.MAP['stDwTG'][g, t]]
                                                            + couplVars[params.MAP['stUpTG'][g, t]]
                                                                for t in periods_in_disp) <= 0,
                                                name = f'temp_VI_prevent_shut_down_and_start_up_{g}'
                                            )
                                        )
    return temp_feas_constrs

def recv_and_add_cuts(params, it, optModels, couplVars, beta,
                        objValuesRelaxs, subgrads, evaluatedSol, fwPackageRecv, fRank, fComm,
                        event_tracker, counter, recvAllBackwards):
    """Receive cuts from the general coordinator and add them"""

    time_communicating = dt()

    fComm.Recv([fwPackageRecv, MPI.DOUBLE], source = 0, tag = 10)

    backward_src, it_from_source = int(fwPackageRecv[0]), int(fwPackageRecv[4])

    fComm.Recv([objValuesRelaxs[backward_src], MPI.DOUBLE], source = 0, tag = 11)
    fComm.Recv([subgrads[backward_src], MPI.DOUBLE], source = 0, tag = 12)
    fComm.Recv([evaluatedSol, MPI.DOUBLE], source = 0, tag = 13)

    time_communicating = dt() - time_communicating

    event_tracker.append(('dualSolRecv', '72', ' ', ' ', backward_src, ' ', ' ', fRank,
                                it_from_source, ' ', it, ' ', dt() - params.START))

    counter += 1

    timeAddingCuts = 0

    for backward_subh in range(params.PERIODS_OF_BACKWARD_WORKERS[backward_src].shape[0]-1,0,-1):
        # Find the corresponding subhorizon in the forward process that comes immediately before
        # the beginning of subhorizon b. To that end, find the first period in subhorizon b
        _first_t_in_subh = min(params.PERIODS_OF_BACKWARD_WORKERS[backward_src][backward_subh])
        # Now, simply find the subhorizon in the forward process whose
        # last period comes immediately before firstPeriodInSubh
        forw_subh = [fb for fb in range(params.N_SUBHORIZONS)
                                    if (_first_t_in_subh - max(params.PERIODS_PER_SUBH[fb])) == 1]

        if len(forw_subh) > 0:
            # if len(forw_subh) then there is no coupling between subhorizon backward_subh of
            # backward worker backward_src and this forward worker

            forw_subh = forw_subh[0]

            # then the newly received subgradient can be added to one of the forward's subhorizons.
            # specifically, subhorizon forw_subh
            event_tracker.append(('cutAdded', '97', ' ', ' ',
                                backward_src, ' ', ' ', fRank,
                                it_from_source, backward_subh, it, forw_subh, dt() - params.START))

            ini = dt()

            nonZeros = np.where(np.abs(subgrads[backward_src][backward_subh]) > 0)[0]
            constTerm = np.inner(subgrads[backward_src][backward_subh][nonZeros],
                                                                            evaluatedSol[nonZeros])

            lhs = optModels[forw_subh].xsum(
                                    subgrads[backward_src][backward_subh][i]*couplVars[forw_subh][i]
                                                                        for i in nonZeros)

            optModels[forw_subh].add_constr(
                                            beta[forw_subh] >=
                                                objValuesRelaxs[backward_src][backward_subh]
                                                    + lhs - constTerm,
                                                    name = f"opt_cut_from_bRank_{backward_src}"
                                                            + f"_subh_from_source_{backward_subh}"
                                                            + f"_it_from_source_{it_from_source}"
                                            )

            if params.BD_SUBHORIZON_PROB:
                # get the indices corresponding to continuous variables whose coefficients
                # in this cut are negative
                indsContNeg = np.intersect1d(np.where(subgrads[backward_src][backward_subh] < 0)[0],
                                            params.CON_VARS_IN_PREVIOUS_AND_CURRENT_SUBH[forw_subh])
                indsContPos = np.intersect1d(np.where(subgrads[backward_src][backward_subh] > 0)[0],
                                            params.CON_VARS_IN_PREVIOUS_AND_CURRENT_SUBH[forw_subh])
                # multiply them by their respective upper bounds to get a
                # 'maximum negative term', i.e., a lower bound on this term
                constTermMP = (np.inner(subgrads[backward_src][backward_subh][indsContNeg],
                                                        params.UB_ON_COUPL_VARS[indsContNeg]) +
                                np.inner(subgrads[backward_src][backward_subh][indsContPos],
                                                        params.LB_ON_COUPL_VARS[indsContPos]))
                # now get the nonzero coefficients of the binary variables
                indsOfBins = np.intersect1d(nonZeros,
                                            params.BIN_VARS_IN_PREVIOUS_AND_CURRENT_SUBH[forw_subh])
                lhsMP = optModels[forw_subh].xsum(subgrads[backward_src][backward_subh][i]*
                                            optModels[forw_subh].copyVars[i] for i in indsOfBins)

                optModels[forw_subh].add_constr_MP(
                                                optModels[forw_subh].alphaVarMP >=
                                                    objValuesRelaxs[backward_src][backward_subh]
                                                        + lhsMP + constTermMP - constTerm,
                                                        name = f"opt_cut_from_bRank_{backward_src}"
                                                            + f"_subh_from_source_{backward_subh}"
                                                            + f"_it_from_source_{it_from_source}"
                                                )

            timeAddingCuts += dt() - ini

    return(recvAllBackwards, backward_src, counter, time_communicating, timeAddingCuts)

def solveSubhorizonProblem(params,
                                b, it,
                                    ub, lb, gap,
                                        redFlag,
                                            optModels, fixedVars,
                                                bestSol, previousSol, couplVars,
                                                    presentCosts, futureCosts,
                                                        beta, alpha, fRank):
    """
        Solve the subhorizon problem in the forward step
    """

    (totalRunTime, dist, distBin, distStatusBin, distStatusBinBestSol) = (0, 0, 0, 0, 0)

    ini = dt()

    optModels[b].reset()

    msStatus = optModels[b].optimize(max_seconds = max(params.LAST_TIME - dt(), 0))

    totalRunTime += (dt() - ini)

    f = open(params.OUT_DIR + "fRank" + str(fRank) + "_subhorizon" + str(b)+".txt",'a',
                                                                            encoding="utf-8")
    f.write('\n\nThe total elapsed time is (sec): ' + str(dt() - params.START) + '\n\n\n')
    f.close()

    if msStatus in (OptS.OPTIMAL, OptS.FEASIBLE):

        (redFlag, ub, lb, gap) = check_convergence(params, b, it, ub, lb, redFlag, optModels)

        (dist, distBin, distStatusBin,
                        distStatusBinBestSol, previousSol)= get_solution(
                                                                params, b, it, optModels,
                                                                    bestSol, fixedVars, previousSol,
                                                                        presentCosts, futureCosts,
                                                                            beta, alpha, couplVars)

    elif msStatus in (OptS.OTHER, OptS.NO_SOLUTION_FOUND):
        redFlag = np.array(1, dtype = 'int')

    else:
        print(f'Im fRank {fRank}. My msStatus is {msStatus}', flush = True)
        for b2 in range(0, b + 1, 1):
            optModels[b2].write('fRank' + str(fRank) + '-Problem' + str(b2) + '.lp')
            optModels[b2].write('fRank' + str(fRank) + '-Problem' + str(b2) + '.mps')
        raise ValueError(f"Im fRank {fRank}. Problem {b} is not optimal: {msStatus}")

    return(redFlag, totalRunTime,
            dist, distBin, distStatusBin, distStatusBinBestSol, previousSol, msStatus, ub, lb, gap)

def prepareTheSubhProblem(params, thermals, b, it, optModels, couplVars,
                        fixedVars, ub, lb, bestSol, couplConstrs, futureCosts):
    """
        Change nature of variables, if necessary
        Change the rhs of the appropriate cosntraints to the latest solution
    """

    tempFeasConstrs = []

    if it >= 1:
        if not(params.BD_SUBHORIZON_PROB or params.BD_NETWORK_SUBHORIZON):
            for i in [i for i in params.BIN_VARS_PER_SUBH[b]
                                            if (optModels[b].get_ub(couplVars[b][i]) == 1)
                                                and (optModels[b].get_lb(couplVars[b][i]) == 0)]:
                optModels[b].set_var_type(couplVars[b][i], 'B')

    # Fix the decisions of previous subhorizons
    if params.COUPLING == Coupling.CONSTRS:
        for t in [t for t in range(params.T) if t < min(params.PERIODS_PER_SUBH[b])]:
            for i in set(params.VARS_PER_PERIOD[t]):
                couplConstrs[b][i].rhs = fixedVars[i]
    else:
        for t in [t for t in range(params.T) if t < min(params.PERIODS_PER_SUBH[b])]:
            for i in set(params.VARS_PER_PERIOD[t]):
                optModels[b].set_lb(couplConstrs[b][i], fixedVars[i])
                optModels[b].set_ub(couplConstrs[b][i], fixedVars[i])

    if not(params.BD_SUBHORIZON_PROB or params.BD_NETWORK_SUBHORIZON):
        tempFeasConstrs = add_VI_based_on_ramp(params, thermals, b,
                                                         optModels[b], couplVars[b], fixedVars)

    optModels[b].lp_method = LP_Method.BARRIER

    if params.BD_SUBHORIZON_PROB or params.BD_NETWORK_SUBHORIZON:
        optModels[b].fixedVars = fixedVars
        optModels[b].bestSolUB = bestSol
        optModels[b].outerIteration = it
        if params.I_AM_A_BACKWARD_WORKER and (it >= 1) and (b >= 1):
            optModels[b].iniLB = np.array(futureCosts[b - 1], dtype = 'd')
        elif params.I_AM_A_BACKWARD_WORKER and (it >= 1) and (b == 0):
            optModels[b].iniLB = lb

    return tempFeasConstrs

def check_convergence(params, b, it, ub, lb, redFlag, optModels):
    """Check convergence"""

    if ((b == 0 and it != 0)) or (params.N_SUBHORIZONS == 1):
        # only update the lower bound if the current subhorizon is the first
        # or if there is a single subhorizon
        lb = max(optModels[0].objective_bound, lb)

    gap = (ub - lb)/ub

    if (gap <= params.REL_GAP_DDiP and len(params.FORWARD_WORKERS) == 1
                                                            and len(params.BACKWARD_WORKERS) == 1):
        # this is only used in case there is a single process
        redFlag = np.array(1, dtype = 'int')

    return (redFlag, ub, lb, gap)

def get_solution(params, b, it, optModels, bestSol, fixedVars,
                                    previousSol, presentCosts, futureCosts, beta, alpha, couplVars):
    """update the solution arrays with the output from subhorizon b"""

    if not(params.BD_SUBHORIZON_PROB or params.BD_NETWORK_SUBHORIZON):
        sol = np.zeros((params.N_COMPL_VARS, ), dtype = 'd')
        sol[params.VARS_IN_PREVIOUS_SUBHS[b]] = fixedVars[params.VARS_IN_PREVIOUS_SUBHS[b]]
        sol[params.VARS_PER_SUBH[b]] = np.array([optModels[b].get_var_x(couplVars[b][i])
                                                        for i in params.VARS_PER_SUBH[b]],dtype='d')
    else:
        sol = optModels[b].bestSol

    # compute distances between the current solution of subhorizon b and its solution in the
    # previous iterations, previousSol, and the corresponding values of subhorizon b's variables
    # in the best solution found so far, bestSol
    # these distance measures are only illustrative
    if it >= 1:
        dist = np.linalg.norm(previousSol[params.VARS_PER_SUBH[b]] - sol[params.VARS_PER_SUBH[b]])
        distBin = np.linalg.norm(previousSol[params.BIN_VARS_PER_SUBH[b]]
                                                                - sol[params.BIN_VARS_PER_SUBH[b]])
        distStatusBin = np.linalg.norm(previousSol[params.BIN_DISP_VARS_PER_SUBH[b]]
                                                            - sol[params.BIN_DISP_VARS_PER_SUBH[b]])
        distStatusBinBestSol = np.linalg.norm(bestSol[params.BIN_DISP_VARS_PER_SUBH[b]]
                                                            - sol[params.BIN_DISP_VARS_PER_SUBH[b]])
    else:
        (dist, distBin, distStatusBin, distStatusBinBestSol) = (0.00, 0.00, 0.00, 0.00)

    if not(params.BD_SUBHORIZON_PROB or params.BD_NETWORK_SUBHORIZON):
        if it == 0 and (b != (params.N_SUBHORIZONS - 1)):
            presentCosts[b] = optModels[b].objective_value - optModels[b].get_var_x(alpha[b])
            futureCosts[b] = (optModels[b].get_var_x(alpha[b])
                                                    if optModels[b].get_obj_coeff(alpha[b]) == 1
                                                        else optModels[b].get_var_x(beta[b]))

        else:
            if b != (params.N_SUBHORIZONS - 1):
                presentCosts[b] = optModels[b].objective_value - optModels[b].get_var_x(beta[b])
                futureCosts[b] = optModels[b].get_var_x(beta[b])
            else:
                presentCosts[b] = optModels[b].objective_value - optModels[b].get_var_x(alpha[b])
                futureCosts[b] = optModels[b].get_var_x(alpha[b])
    else:
        if (it == 0) or (b == (params.N_SUBHORIZONS - 1)):
            presentCosts[b] = optModels[b].objective_value - optModels[b].alpha
            if (b == (params.N_SUBHORIZONS - 1)):
                futureCosts[b] = optModels[b].alpha
            else:
                futureCosts[b] = optModels[b].alpha if it == 0 else optModels[b].beta

        else:
            presentCosts[b] = optModels[b].objective_value - optModels[b].beta
            futureCosts[b] = optModels[b].beta

    #### Round values to avoid numeric problems
    fixedVars[params.VARS_PER_SUBH[b]] = sol[params.VARS_PER_SUBH[b]]

    binaries = sol[params.BIN_VARS_PER_SUBH[b]]

    binaries[np.where(binaries <= 0.5)[0]] = 0
    binaries[np.where(binaries > 0.5)[0]] = 1

    fixedVars[params.BIN_VARS_PER_SUBH[b]] = binaries

    fixedVars[params.VARS_PER_SUBH[b]] = np.maximum(fixedVars[params.VARS_PER_SUBH[b]],
                                                params.LB_ON_COUPL_VARS[params.VARS_PER_SUBH[b]])

    fixedVars[params.VARS_PER_SUBH[b]] = np.minimum(fixedVars[params.VARS_PER_SUBH[b]],
                                                params.UB_ON_COUPL_VARS[params.VARS_PER_SUBH[b]])

    dispGen = fixedVars[params.DISP_GEN_VARS_PER_SUBH[b]]
    dispGen[np.where(dispGen <= 1e-5)[0]] = 0
    fixedVars[params.DISP_GEN_VARS_PER_SUBH[b]] = dispGen

    #### Store the current solution to keep track of how it changes over the iterations
    previousSol[params.VARS_PER_SUBH[b]] = (
                                sol[params.VARS_PER_SUBH[b]]
                                if not(params.BD_SUBHORIZON_PROB or params.BD_NETWORK_SUBHORIZON)
                                        else optModels[b].bestSol[params.VARS_PER_SUBH[b]]
                                        )

    return (dist, distBin, distStatusBin, distStatusBinBestSol, previousSol)

def _print_metrics_for_current_subhorizon(obj_function_scale_factor:float, b: int, dist: float,
                                    distStatusBin: float, distStatusBinBestSol: float,
                                    presentCost: float, futureCost: float,
                                    totalRunTime: float, objVal: float, objBound: float) -> None:
    """Print distances, costs, time, and gap for subhorizon b"""

    s = "\n\n" + f"Subhorizon: {b}" + "\t"

    s = (s + "Dist. (cont, stsPrev, stsBest): " +
                    f"({dist:.1f}, {distStatusBin**2:.1f}, {distStatusBinBestSol**2:.1f})" + "\t")

    s = (s + "Present cost: " + locale.currency(presentCost/obj_function_scale_factor,
                                                grouping = True) +
            "\tFuture cost: " + locale.currency(futureCost/obj_function_scale_factor,
                                                grouping = True)+"\t")
    s = (s + "Total: " + locale.currency((presentCost + futureCost)/obj_function_scale_factor,
                                            grouping=True)+
                                "\t" + f"Runtime: {totalRunTime:.1f}" + "\t")
    s = s + f"Gap (%): {100*((objVal - objBound)/objVal):.4f}"

    print(s, flush = True)

def forwardStep(params, thermals,
                it, subhorizonInfo, couplVars, fixedVars, previousSol, couplConstrs,
                optModels, bestSol,
                presentCosts, futureCosts,
                alpha, beta, redFlag, ub, lb, gap, bufferForward,
                status, objValuesRelaxs, subgrads, evaluatedSol, fwPackageRecv,
                fComm, fRank, fSize, event_tracker = None):
    """Forward step of the DDiP"""

    dualInfoFromMatchRecv = False   # True if the dual information from a matching backward process
                                    # was received while the forward process was solving the problem

    # 1 if the subhorizon problem has been solved
    subhsDone = [0 for i in range(params.N_SUBHORIZONS)]

    for b in range(params.N_SUBHORIZONS):
        for k in subhorizonInfo[b].keys():
            subhorizonInfo[b][k].append(0)

    for b in range(params.N_SUBHORIZONS):

        if redFlag != 0:
            break

        tempFeasConstrs = prepareTheSubhProblem(params, thermals, b, it,
                                                        optModels, couplVars,
                                                            fixedVars, ub, lb, bestSol,
                                                                couplConstrs, futureCosts)

        (redFlag, totalRunTime, dist, distBin, distStatusBin, distStatusBinBestSol,
                    previousSol, msStatus, ub, lb, gap) = solveSubhorizonProblem(params, b, it,
                                                            ub, lb,
                                                            gap, redFlag, optModels, fixedVars,
                                                            bestSol, previousSol, couplVars,
                                                            presentCosts, futureCosts,
                                                            beta, alpha, fRank)

        if (redFlag == 0) and (msStatus in (OptS.OPTIMAL, OptS.FEASIBLE)):

            if (it != 0) and (b == 0) and (fSize == 1):
                lb = max(optModels[b].objective_bound, lb)
                gap = (ub - lb)/ub
                if gap <= params.REL_GAP_DDiP:
                    redFlag = np.array(1, dtype = 'int')

            if (optModels[b].objective_bound > lb) and (it > 0) and (b == 0) and (fSize > 1):
                lb = max(optModels[b].objective_bound, lb)
                gap = (ub - lb)/ub
                bufferForward[('lb', it, b)] = [np.array([lb, it, b], dtype = 'd'), None]
                bufferForward[('lb', it, b)][1] = fComm.Isend([bufferForward[('lb', it,b)][0],
                                                                MPI.DOUBLE], dest = 0, tag = 29)
                event_tracker.append(('LBSent', '379', ' ', lb,
                                        ' ', ' ', fRank, ' ',
                                            it, b, ' ', ' ', dt() - params.START))

            if gap < 0:
                optModels[b].write('crazyLBfrom_fRank ' + str(fRank) + '.lp')
                optModels[b].write('crazyLBfrom_fRank ' + str(fRank) + '.mps')

                optModels[b].write(params.OUT_DIR +'crazyLBfrom_fRank ' + str(fRank) + '.lp')
                optModels[b].write(params.OUT_DIR +'crazyLBfrom_fRank ' + str(fRank) + '.mps')

            if fSize == 1:
                _print_metrics_for_current_subhorizon(params.SCAL_OBJ_F,
                            b, dist, distStatusBin,
                            distStatusBinBestSol, presentCosts[b], futureCosts[b], totalRunTime,
                            optModels[b].objective_value, optModels[b].objective_bound)

            if (b < (params.N_SUBHORIZONS - 1)) and b in params.FORWARD_SEND_POINTS:
                # if the subhorizon's problem just solved is in params.FORWARD_SEND_POINTS, then
                # send the primal solution and the package to the general coordinator
                bufferForward[('package', it, b)] = [np.array([redFlag, ub, lb, it, b,
                                                    sum(presentCosts[0:b + 1]), futureCosts[b]],
                                                        dtype = 'd'), None]
                bufferForward[('sol', it, b)] = [deepcopy(fixedVars), None]
                bufferForward[('package', it, b)][1] = fComm.Isend([
                                                            bufferForward[('package', it,b)][0],
                                                                    MPI.DOUBLE], dest = 0, tag = 21)
                bufferForward[('sol', it, b)][1] = fComm.Isend([bufferForward[('sol', it,b)][0],
                                                                    MPI.DOUBLE], dest = 0, tag = 22)
                event_tracker.append(('primalSolSent', '479', ' ', lb,
                                        ' ', ' ', fRank, ' ',
                                            it, b, ' ', ' ', dt() - params.START))

        elif msStatus in (OptS.OTHER, OptS.NO_SOLUTION_FOUND):
            if (optModels[b].objective_bound > lb) and (it > 0) and (b == 0):
                lb = max(optModels[b].objective_bound, lb)
                gap = (ub - lb)/ub
                if (fSize > 1):
                    bufferForward[('lb', it, b)] = [np.array([lb, it, b], dtype = 'd'), None]
                    bufferForward[('lb', it, b)][1] = fComm.Isend([bufferForward[('lb', it,b)][0],
                                                                MPI.DOUBLE], dest = 0, tag = 29)
                    event_tracker.append(('LBSent', '492', ' ', lb,
                                            ' ', ' ', fRank, ' ',
                                                it, b,' ', ' ', dt() - params.START))

        subhorizonInfo[b]['time'][-1] = totalRunTime
        if params.BD_SUBHORIZON_PROB or params.BD_NETWORK_SUBHORIZON:
            subhorizonInfo[b]['iterations'][-1] = optModels[b].it
        subhorizonInfo[b]['gap'][-1] = optModels[b].gap
        subhorizonInfo[b]['optStatus'][-1] = msStatus

        if (redFlag == 0) or (gap <= params.REL_GAP_DDiP):
            subhsDone[b] = 1
            subhorizonInfo[b]['presentCots'][-1] = presentCosts[b]
            subhorizonInfo[b]['futureCosts'][-1] = futureCosts[b]
            subhorizonInfo[b]['distanceFromPreviousSol'][-1] = dist
            subhorizonInfo[b]['distBinVars'][-1] = distBin
            subhorizonInfo[b]['distStatusBinVars'][-1] = distStatusBin
            subhorizonInfo[b]['distStatusBinBestSol'][-1] = distStatusBinBestSol

        if (redFlag == 0) and not(params.BD_SUBHORIZON_PROB or params.BD_NETWORK_SUBHORIZON):
            optModels[b].remove(tempFeasConstrs)

        if redFlag != 0:
            break

        totaltime_communicating, totalTimeAddingCuts = 0, 0

        if (it > 0) and (fSize > 1) and (b < (params.N_SUBHORIZONS - 2)) and params.ASYNCHRONOUS:
            msgRecv = True
            while msgRecv:
                if (fComm.Iprobe(source = 0, tag = 10, status = status)):
                    _0, backwardSrc, _1, time_communicating, timeAddingCuts = recv_and_add_cuts(
                                    params, it, optModels, couplVars, beta,
                                        objValuesRelaxs, subgrads, evaluatedSol,
                                            fwPackageRecv, fRank, fComm, event_tracker, 0, False)

                    totaltime_communicating += time_communicating
                    totalTimeAddingCuts += timeAddingCuts

                    if (params.PERIODS_OF_BACKWARD_WORKERS[backwardSrc].shape[0]
                                                                        == params.N_SUBHORIZONS):
                        # If the current forward worker has received subgradients from a
                        # backward worker with the same aggregation, then, there is enough
                        # information to continue without more cuts
                        dualInfoFromMatchRecv = True
                else:
                    msgRecv = False

        subhorizonInfo[b]['communication'][-1] += totaltime_communicating
        subhorizonInfo[b]['timeToAddCuts'][-1] += totalTimeAddingCuts
        subhorizonInfo[b]['timeStamp'][-1] = dt() - params.START

    if it == 0:
        for b in range(params.N_SUBHORIZONS - 1):
            if (int(optModels[b].get_name(alpha[b])[optModels[b].get_name(alpha[b]).rfind('_')+ 1:])
                        < (params.T - 1)):
                optModels[b].set_obj_coeff(alpha[b], 0)
            optModels[b].set_obj_coeff(beta[b], 1)

    return (ub, lb, gap, redFlag, subhsDone, previousSol, dualInfoFromMatchRecv)
