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

from solver_interface.opt_model import Model
from solver.BD import BD
from solver.BDnetwork import BD_Network
from addCompToModel.addAllComponents import add_all_comp
from solver.forwardParallel import forwardStepPar, forwardStep
from solver.backwardParallel import backwardStepPar, backwardStep
from solver.generalCoordinator import genCoord

locale.setlocale(locale.LC_ALL, '')

def build_opt_models(params, red_flag, thermals, hydros, network, fixed_vars,
                fRank, bRank, W_SIZE, building_times):
    """Build the optimization models"""

    # used to estimate from below the optimal cost of subsequent subhorizons in forward
    beta = {b: None for b in range(params.N_SUBHORIZONS)}
    # used to estimate from below the optimal cost subsequent subhorizons in backward
    beta_relax = {b:None for b in range(params.N_SUBHORIZONS)}

    alpha = {b: None for b in range(params.N_SUBHORIZONS)} # var of the cost-to-go func in forward
    # var of the cost-to-go func in backward
    alpha_relax = {b: None for b in range(params.N_SUBHORIZONS)}

    opt_models = [None for b in range(params.N_SUBHORIZONS)]
    relax_opt_models = [None for b in range(params.N_SUBHORIZONS)]

    if params.BD_SUBHORIZON_PROB or params.BD_NETWORK_SUBHORIZON or params.BD_BACKWARD_PROB:
        if params.I_AM_A_FORWARD_WORKER:
            if params.BD_SUBHORIZON_PROB:
                opt_models = [BD(params, red_flag, thermals, hydros, b,
                                'forward_subhorizon'+str(b),
                                solver_name = params.SOLVER) for b in range(params.N_SUBHORIZONS)]
            if params.BD_NETWORK_SUBHORIZON and not(params.BD_SUBHORIZON_PROB):
                opt_models = [BD_Network(params, red_flag, b, 'forward_subhorizon'+str(b),
                                solver_name = params.SOLVER) for b in range(params.N_SUBHORIZONS)]
        else:
            opt_models = [None for i in range(params.N_SUBHORIZONS)]

        if params.I_AM_A_BACKWARD_WORKER:
            if params.BD_BACKWARD_PROB:
                relax_opt_models = [
                                        BD_Network(params, red_flag, b,
                                                    'backward_subhorizon' + str(b),
                                                        solver_name = params.SOLVER)
                                            for b in range(params.N_SUBHORIZONS)
                                    ]
            else:
                relax_opt_models = [
                                        Model('backward_subhorizon' + str(b),
                                                solver_name = params.SOLVER,
                                                    package = params.PACKAGE
                                                )
                                            for b in range(params.N_SUBHORIZONS)
                                    ]
    else:
        # If no further decomposition is used, then the forward and backward models are the same
        opt_models = [
                        Model('day_ahead',
                                solver_name = params.SOLVER,
                                    package = params.PACKAGE
                                )
                            for b in range(params.N_SUBHORIZONS)
                        ]
        relax_opt_models = [opt_models[b] for b in range(params.N_SUBHORIZONS)]

    if params.I_AM_A_FORWARD_WORKER:
        for b in range(params.N_SUBHORIZONS):
            opt_models[b].verbose = params.VERBOSE

    if params.I_AM_A_BACKWARD_WORKER:
        firstSubhToSolve = params.N_SUBHORIZONS - 1 if not(params.SOLVE_ONLY_1st_SUBH) else 0
        lastSubhToSolve = 0 if params.SOLVE_ONLY_1st_SUBH or params.I_AM_A_FORWARD_WORKER else 1

        for b in range(firstSubhToSolve, lastSubhToSolve - 1, -1):
            relax_opt_models[b].verbose = params.VERBOSE

    # coupling constraints in the subhorizons of the forward step. These are the equality
    # constraints that force that time-coupling decisions taken in previous subhorizons are
    # respected
    couplConstrs = {b: [] for b in range(params.N_SUBHORIZONS)}
    # coupling variables in the subhorizons of the forward step
    couplVars = {b: [] for b in range(params.N_SUBHORIZONS)}

    # the same as the above but now for the backward step. If no further decomposition is used,
    # then couplConstrsRelax is the same as couplConstrs, and couplVarsRelax == couplVars
    couplConstrsRelax = {b: [] for b in range(params.N_SUBHORIZONS)}
    couplVarsRelax = {b: [] for b in range(params.N_SUBHORIZONS)}

    if params.I_AM_A_FORWARD_WORKER:
        for b in range(params.N_SUBHORIZONS):
            opt_models[b].threads = params.THREADS
            opt_models[b].max_mip_gap = params.MILP_GAP_SUBH

    if params.I_AM_A_BACKWARD_WORKER:
        for b in range(params.N_SUBHORIZONS):
            relax_opt_models[b].threads = params.THREADS

    first_subh_to_solve = 0 if params.SOLVE_ONLY_1st_SUBH or params.I_AM_A_FORWARD_WORKER else 1
    last_subh_to_solve = params.N_SUBHORIZONS - 1 if not(params.SOLVE_ONLY_1st_SUBH) else 0
    # note that, if params.SOLVE_ONLY_1st_SUBH == True, then lastSubhToSolve is 0 and
    # firstSubhToSolve == 0
    # range(firstSubhToSolve, lastSubhToSolve + 1, 1) contains only 0. In other words,
    # only the first subhorizon problem is built
    for b in range(first_subh_to_solve, last_subh_to_solve + 1, 1):
        ini = dt()
        if params.BD_SUBHORIZON_PROB or params.BD_NETWORK_SUBHORIZON or params.BD_BACKWARD_PROB:
            if params.I_AM_A_FORWARD_WORKER:
                if params.BD_SUBHORIZON_PROB:
                    couplConstrs[b], couplVars[b], alpha_, beta_ = opt_models[b].add_all_comp(
                                                                    params, hydros, thermals,
                                                                    network, fixed_vars, b,
                                                                    binVars = BINARY)

                if params.BD_NETWORK_SUBHORIZON and not(params.BD_SUBHORIZON_PROB):
                    couplConstrs[b], couplVars[b], alpha_, beta_ = opt_models[b].add_all_comp(
                                                                    params, hydros, thermals,
                                                                    network, fixed_vars, b,
                                                                    binVars = BINARY)
            else:
                alpha_, beta_ = None, None

            if params.I_AM_A_BACKWARD_WORKER:
                if params.BD_BACKWARD_PROB:
                    varNature = BINARY if params.SOLVE_ONLY_1st_SUBH and b == 0 else CONTINUOUS
                    couplConstrsRelax[b], couplVarsRelax[b], alph_, bet_ =\
                                                                relax_opt_models[b].add_all_comp(
                                                                params, hydros, thermals,
                                                                network, fixed_vars, b,
                                                                binVars = varNature)
                else:
                    (couplConstrsRelax[b], couplVarsRelax[b], alph_, bet_,
                                            _0, _1, _2, _3, _4, _5, _6, _7) = add_all_comp(params,
                                                            hydros, thermals, network,
                                                            relax_opt_models[b],
                                                            relax_opt_models[b],
                                                            relax_opt_models[b],
                                                            b, fixed_vars, BDbinaries = False,
                                                            BDnetwork = False, binVars = CONTINUOUS)
            else:
                alph_, bet_ = None, None

            beta_relax[b] = bet_
            alpha_relax[b] = alph_
        else:
            var_nature = BINARY if (params.I_AM_A_FORWARD_WORKER or
                                (params.SOLVE_ONLY_1st_SUBH and b == 0)) else CONTINUOUS
            # var_nature is BINARY for forward processes, and for backward processes who will
            # only solve their respective first subhorizons
            (couplConstrs[b], couplVars[b], alpha_, beta_,
                                                    _0, _1, _2, _3, _4, _5, _6, _7) =\
                                                        add_all_comp(params,
                                                        hydros, thermals, network,
                                                        opt_models[b], opt_models[b], opt_models[b],
                                                        b, fixed_vars, BDbinaries = False,
                                                        BDnetwork = False, binVars = var_nature)

            couplConstrsRelax[b] = couplConstrs[b]
            couplVarsRelax[b] = couplVars[b]

            beta_relax[b] = beta_
            alpha_relax[b] = alpha_

        end = dt()

        building_times[b] = end - ini

        beta[b] = beta_
        alpha[b] = alpha_

    return(beta, beta_relax, alpha, alpha_relax,
            opt_models, relax_opt_models,
                couplConstrs, couplVars, couplConstrsRelax, couplVarsRelax)

def run_solver(params, hydros, thermals, network,
                wComm, W_RANK, W_SIZE, fComm, fRank, fSize, bComm, bRank, bSize):
    """Run the DDiP
    params:             an instance of OptOptions (optoptions.py) that contains the
                            parameters for the problem and the algorithm
    hydros:             an instance of Hydros (network.py) with all hydro data
    thermals:           an instance of Thermals (network.py) with all thermal data
    network:            an instance of Network (network.py) with all network data
    Comm, Rank, Size:   communicator, rank and size: w stands for the global parameters, i.e.,
                            W_RANK is the process' rank in the world communicator.
                            f stands for the forward communicator,
                            b stands for the backward communicator
    """
    ub, lb = np.array(1e12, dtype = 'd'), np.array(0, dtype = 'd')

    red_flag = np.array(0, dtype='int') # use for broadcasting that either convergence or the time
                                        # limit has been reached

    # the following windows will be used by the general coordinator to push the updated values of
    # the red_flag, upper bound and lower bound. Only the general coordinator puts values in these
    # windows, while the workers can only retrieve values from them.
    if W_SIZE > 1:
        params.win_red_flag = MPI.Win.Create(red_flag, 1, info = MPI.INFO_NULL, comm = wComm)
        params.winUB = MPI.Win.Create(ub, 1, info = MPI.INFO_NULL, comm = wComm) # Upper bound
        params.winLB = MPI.Win.Create(lb, 1, info = MPI.INFO_NULL, comm = wComm) # Lower bound

    # present and future costs for each subhorizon
    present_costs, future_costs = ([0 for i in range(params.N_SUBHORIZONS)],
                                                        [0 for i in range(params.N_SUBHORIZONS)])

    # dictionary for storing data from the solution process
    p_log = {'lb': [], 'ub': [], 'gap': [], 'runTimeForward': [], 'runTimeBackward': []}

    # log for the forward subhorizon problems
    subhorizon_info = {b: {'presentCots': [], 'futureCosts': [], 'time': [],
                    'iterations': [], 'gap': [], 'distanceFromPreviousSol': [],
                    'distBinVars': [], 'distStatusBinVars': [], 'distStatusBinBestSol': [],
                    'optStatus': [], 'communication': [], 'timeToAddCuts': [], 'timeStamp': []}
                    for b in range(params.N_SUBHORIZONS)}

    # log for the backward subhorizon problems
    backward_info = {b: {'lb': [], 'ub': [], 'gap': [], 'time': [], 'optStatus': [],
                    'communication': [], 'timeToAddCuts': [], 'timeStamp': []}
                    for b in range(params.N_SUBHORIZONS)}

    ## dual variables' values associated with the time-coupling equality constraints
    lbda = np.zeros((params.N_SUBHORIZONS, params.N_COMPL_VARS), dtype = 'd')
    ## objective functions' values of the backward subhorizon problems
    obj_val_relax = np.zeros(params.N_SUBHORIZONS, dtype = 'd')

    gap, it = 1e12, 0  # relative gap, and iteration counter

    best_sol = np.zeros(params.N_COMPL_VARS, dtype = 'd')       # best solution found so far
    fixed_vars = np.zeros(params.N_COMPL_VARS, dtype = 'd')     # current solution
    previous_sol = np.zeros(params.N_COMPL_VARS, dtype = 'd')   # solution from previous iteration

    building_times = [0 for b in range(params.N_SUBHORIZONS)]   # times taken to build the
                                                                # subhorizon models

    (beta, betaRelax, alpha, alphaRelax, optModels, optModelsRelax,
            couplConstrs, couplVars, couplConstrsRelax, couplVarsRelax) = build_opt_models(
                                                        params, red_flag,
                                                            thermals, hydros, network, fixed_vars,
                                                            fRank, bRank, W_SIZE, building_times)

    if params.I_AM_A_BACKWARD_WORKER:
        bufferBackward = {}

    if W_SIZE == 1:
        print(f'{sum(building_times):.2f} seconds building the subhorizon problems.')
        print(f'Max, min, and avrg (sec): ({max(building_times):.2f}, {min(building_times):.2f},'+
                                f' {sum(building_times)/len(building_times):.2f})', flush=True)
    if W_SIZE > 1:
        if W_RANK == 0:
            (best_sol, ub, lb, gap, red_flag, _666, _6661) =\
                                            genCoord(params, red_flag, ub, lb, it, best_sol, p_log,
                                                        W_RANK, W_SIZE, fComm, fSize, bComm, bSize)

        if params.I_AM_A_FORWARD_WORKER:
            ub, lb, gap, red_flag = forwardStepPar(params, thermals,
                                            it, subhorizon_info, couplVars, fixed_vars,
                                            previous_sol, couplConstrs, optModels, best_sol,
                                            present_costs, future_costs,
                                            alpha, beta, red_flag, ub, lb, gap,
                                            W_RANK, fComm, fRank, fSize)

        if params.I_AM_A_BACKWARD_WORKER:
            ub, lb, gap, red_flag =  backwardStepPar(params, thermals,
                                            it, ub, lb, backward_info, obj_val_relax,
                                            optModels, optModelsRelax, lbda,
                                            fixed_vars, couplVars, couplVarsRelax,couplConstrsRelax,
                                            alphaRelax, betaRelax, beta, alpha,
                                            red_flag, bufferBackward,
                                            W_RANK, bComm, bRank)

    while red_flag != 1:
        print('n'+f'Iteration {it}.\tRemaining time (sec): {max(params.LAST_TIME-dt(),0):.4f}',
                                                                                        flush=True)

        totalRunTimeForward = dt()
        ub, lb, gap, red_flag, subhsDone, previous_sol, _0 = forwardStep(params, thermals, it,
                                        subhorizon_info, couplVars, fixed_vars, previous_sol,
                                            couplConstrs,
                                            optModels, best_sol,
                                                present_costs, future_costs,
                                                alpha, beta, red_flag, ub, lb, gap,
                                                    {}, {}, {}, {}, {}, {}, fComm, fRank, fSize, [])

        totalRunTimeForward = dt() - totalRunTimeForward

        if ((red_flag == 0) and (sum(subhsDone) == params.N_SUBHORIZONS) and
                                (sum(present_costs) + future_costs[params.N_SUBHORIZONS - 1]) < ub):
            best_sol = deepcopy(fixed_vars)
            ub = np.array(sum(present_costs) + future_costs[params.N_SUBHORIZONS - 1], dtype = 'd')

        gap = (ub - lb)/ub
        s = f"\n\nIteration: {it}\t\t"

        if (red_flag == 0) and (sum(subhsDone) == params.N_SUBHORIZONS):
            s = s + "Cost of current solution is: " + locale.currency((sum(present_costs)
                + future_costs[params.N_SUBHORIZONS - 1])/params.SCAL_OBJ_F, grouping = True)+"\t\t"

        s = (s + "LB: " + locale.currency(lb/params.SCAL_OBJ_F, grouping = True) +
                            "\t\tUB: " + locale.currency(ub/params.SCAL_OBJ_F, grouping = True) +
                                f"\t\tGap(%): {100*gap:.4f}\t\t"
                        + f"Remaining time (sec): {max(params.LAST_TIME - dt(), 0):.4f}")
        print(s, flush = True)

        if (red_flag != 1) and (gap <= params.REL_GAP_DDiP) or (it >= params.MAX_IT_DDiP):
            red_flag = np.array(1, dtype = 'int')

        lbda, obj_val_relax = (np.zeros((params.N_SUBHORIZONS, params.N_COMPL_VARS), dtype = 'd'),
                                np.zeros(params.N_SUBHORIZONS, dtype = 'd'))

        totalRunTimeBackward = dt()
        if red_flag != 1:
            red_flag, ub, lb, lbda = backwardStep(params, thermals,
                                            it, ub, lb, backward_info, obj_val_relax,
                                            optModels, optModelsRelax, lbda,
                                            fixed_vars, couplVars, couplVarsRelax,couplConstrsRelax,
                                            alphaRelax, betaRelax, beta, alpha, red_flag,
                                            {}, bComm, bRank, None)
        totalRunTimeBackward = dt() - totalRunTimeBackward

        gap = (ub - lb)/ub

        if W_RANK == 0:
            p_log['lb'].append(lb)
            p_log['ub'].append(ub)
            p_log['gap'].append(gap)
            p_log['runTimeForward'].append(totalRunTimeForward)
            p_log['runTimeBackward'].append(totalRunTimeBackward)

        it += 1

        if (red_flag != 1) and (gap <= params.REL_GAP_DDiP) or (it >= params.MAX_IT_DDiP):
            red_flag = np.array(1, dtype = 'int')

    wComm.Barrier()

    if params.I_AM_A_FORWARD_WORKER:
        if params.BD_SUBHORIZON_PROB:
            for b in range(params.N_SUBHORIZONS):
                f = open(params.OUT_DIR+'forwardBD - subhorizon '+str(b) +
                        ' - fRank '+str(fRank)+' - ' + params.PS + ' - case ' + str(params.CASE) +
                            '.csv','w',encoding='ISO-8859-1')
                for key in optModels[b].log.keys():
                    f.write(key + ';')
                f.write('n')
                for i in range(len(optModels[b].log['DDiPIt'])):
                    for key in optModels[b].log.keys():
                        f.write(str(optModels[b].log[key][i]) + ';')
                    f.write('n')
                f.close()

        if params.BD_NETWORK_SUBHORIZON:
            for b in range(params.N_SUBHORIZONS):
                f = open(params.OUT_DIR + 'forwardNetworkBD - subhorizon ' + str(b) +
                        ' - fRank '+str(fRank)+ ' - ' +
                        params.PS + ' - case ' + str(params.CASE) +'.csv','w',encoding='ISO-8859-1')
                for key in optModels[b].log.keys():
                    f.write(key + ';')
                f.write('n')
                for i in range(len(optModels[b].log['DDiPIt'])):
                    for key in optModels[b].log.keys():
                        f.write(str(optModels[b].log[key][i]) + ';')
                    f.write('n')
                f.close()

    if params.I_AM_A_BACKWARD_WORKER and params.BD_BACKWARD_PROB:
        for b in range(params.N_SUBHORIZONS):
            f = open(params.OUT_DIR + 'backwardNetworkBD - subhorizon '+ str(b) +
                    ' - bRank '+str(bRank)+ ' - ' +
                    params.PS + ' - case ' + str(params.CASE) +'.csv','w',encoding='ISO-8859-1')
            for key in optModelsRelax[b].log.keys():
                f.write(key + ';')
            f.write('n')
            for i in range(len(optModelsRelax[b].log['DDiPIt'])):
                for key in optModelsRelax[b].log.keys():
                    f.write(str(optModelsRelax[b].log[key][i]) + ';')
                f.write('n')
            f.close()

    if W_RANK == 0 and ub < 1e12:
        np.savetxt(params.OUT_DIR + 'best_solutionFound - '+
                                    params.PS + ' - case ' + str(params.CASE) + '.csv',
                                    best_sol, fmt = '%.12f', delimiter=';')

    return(best_sol, ub, p_log, subhorizon_info, backward_info)
