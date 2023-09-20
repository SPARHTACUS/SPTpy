# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""

import math
import numpy as np

from optoptions import Coupling, Solver

def add_sec_constraints_only_on_thermals(m, params, thermals, network, t_g, periods = None):
    """
        Add security constraints to the model
    """

    if params.SOLVER == Solver.GRB:
        m.update()

    s_sec_constrs = {}

    # get a list of the buses whose injections appear in one or more security constraints
    constrs_w_buses = set()
    for t in periods:
        for key, constr in [(item[0], item[1]) for item in network.SEC_CONSTRS[t].items()
                    if (item[1]['LB'] > -(99999.00/params.POWER_BASE) or
                        item[1]['UB'] < (99999.00/params.POWER_BASE))]:
            constrs_w_buses.add((t, key))

    power_injections = {(constr_id, t): - network.SEC_CONSTRS[t][constr_id]['net load']
                                                    for t, constr_id in constrs_w_buses
                            if len(network.SEC_CONSTRS[t][constr_id]['participants']['hydros']) == 0
                        and len(network.SEC_CONSTRS[t][constr_id]['participants']['thermals']) >= 1}

    for k in power_injections.keys():
        (constr_id, t) = k
        power_injections[constr_id, t] = (power_injections[constr_id, t] +
                                            m.xsum(network.SEC_CONSTRS[t][constr_id]
                                                            ['participants_factors']['thermals'][g]*
                                                                t_g[g, t]
                            for g in network.SEC_CONSTRS[t][constr_id]['participants']['thermals']
                                                        )
                                        )

    for (constr_id, t) in power_injections.keys():
        constr = network.SEC_CONSTRS[t][constr_id]

        # get all thermal units that participate in this constraint
        all_thermals = network.SEC_CONSTRS[t][constr_id]['participants']['thermals']

        if (constr['LB'] != constr['UB']):
            if constr['LB'] > -(99999.00/params.POWER_BASE):
                if len(all_thermals) == 1:
                    g = all_thermals[0]
                    const = - network.SEC_CONSTRS[t][constr_id]['net_load']
                    (var, coeff) = (t_g[g, t], network.SEC_CONSTRS[t][constr_id]
                                                        ['participants_factors']['thermals'][g])
                    if coeff > 0:
                        m.set_lb(var, max((constr['LB'] - const)/coeff, m.get_lb(var)))
                    elif coeff < 0:
                        m.set_ub(var, min((constr['LB'] - const)/coeff, m.get_ub(var)))
                else:
                    m.add_constr(power_injections[constr_id, t] >= constr['LB'],
                                                    name = f"thermals_only_{constr['name']}_LB_{t}")
            if constr['UB'] < (99999.00/params.POWER_BASE):
                if len(all_thermals) == 1:
                    g = all_thermals[0]
                    const = - network.SEC_CONSTRS[t][constr_id]['net_load']
                    (var, coeff) = (t_g[g, t], network.SEC_CONSTRS[t][constr_id]
                                                        ['participants_factors']['thermals'][g])
                    if coeff > 0:
                        m.set_ub(var, min((constr['UB'] - const)/coeff, m.get_ub(var)))
                    elif coeff < 0:
                        m.set_lb(var, max((constr['UB'] - const)/coeff, m.get_lb(var)))
                else:
                    m.add_constr(power_injections[constr_id, t] <= constr['UB'],
                                                    name = f"thermals_only_{constr['name']}_UB_{t}")
        else:
            m.add_constr(power_injections[constr_id, t] == constr['LB'],
                                                    name = f"thermals_only_{constr['name']}_EQ_{t}")

        if params.SOLVER == Solver.GRB:
            m.update()

    return s_sec_constrs

def fromHoursToTindex2(params, numberOfHours, indexFrom):
    """ How many indices should it go back to account for numberOfHours?"""
    sumHours = 0
    t = 0
    while sumHours < numberOfHours:
        t += 1
        if indexFrom - t < 0:
            sumHours += 1
        else:
            sumHours += params.DISCRETIZATION
    return t

def previousStates(params, thermals, listOfUnits, periodsWithConstrs,
                    m,
                    stUpTG, stDwTG, dispStat, stUpTj, stDownTj):
    """Create auxiliary keys and set variables bounds according to the states
    previous to the optimization horizon"""

    # the previous statuses of thermal units and their respective minimum up and down times, as well
    # as the ramping limits, might prevent the unit from being shut-down during a certain portion
    # of the planning horizon
    sd_dec = {g: 0 for g in thermals.ID}
    for g in thermals.ID:
        if (thermals.STATE_0[g] == 1) and not(thermals.T_G_0[g] <= thermals.MIN_P[g]):
            sd_dec[g] = params.T
            p_decrease = 0
            for t in range(params.T):
                p_decrease += thermals.RAMP_DOWN[g]
                if (thermals.T_G_0[g] - p_decrease) <= thermals.MIN_P[g]:
                    # The unit reaches the minimum at t, and can be turned off at t + 1
                    sd_dec[g] = t + 1
                    # remember that the signal to shut down happens immediately after reaching the
                    # minimum, i.e., at t + 1
                    break
    ############################################################################

    for g in listOfUnits:
        for t in set(range(0, min(sd_dec[g], params.T), 1)) & periodsWithConstrs:
            m.set_ub(stDwTG[g, t], 0)
    ############################################################################

    #### Previous states
    for g in listOfUnits:

        nHours = math.ceil((thermals.MAX_P[g] - thermals.MIN_P[g])/thermals.RAMP_UP[g])\
                                                            if thermals.RAMP_UP[g] > 0 else 0

        for t in range(- 4*thermals.MIN_UP[g] - len(thermals.STUP_TRAJ[g]) - nHours, 0, 1):
            dispStat[g, t] = thermals.STATE_0[g]
            stUpTG[g, t] = 0
        for t in range(- max(thermals.MIN_DOWN[g] + len(thermals.STDW_TRAJ[g]),\
                            thermals.MIN_UP[g], len(thermals.STUP_TRAJ[g]) + nHours, 1), 0, 1):
            stDwTG[g, t] = 0

        for t in range(params.T, params.T + len(thermals.STDW_TRAJ[g]) + nHours + 1, 1):
            stDwTG[g, t] = 0

        for t in range(- nHours, 0, 1):
            dispStat[g, t] = thermals.STATE_0[g]

        if (thermals.STATE_0[g] == 1):
            # Either if it is currently in a start-up trajectory or it has already finished
            # the trajectory, the generator was
            # brought on at time - thermals.N_HOURS_IN_PREVIOUS_STATE[g].
            # However, if it is currently in the shut-down trajectory, then it will eventually
            # be shut-down during the planning horizon

            # If the unit is currently in the dispatch phase, then it means that, at some point,
            # the unit was started-up and it successfully completed its start-up trajectory.
            # Thus, the unit was started-up at period 0 minus the number of periods it has been
            # in the dispatch phase (thermals.N_HOURS_IN_PREVIOUS_STATE[g]) minus the number
            # of periods necessary to complete the start-up trajectory
            stUpTG[g, min(- len(thermals.STUP_TRAJ[g])-thermals.N_HOURS_IN_PREVIOUS_STATE[g], -1)]=1
            dispStat[g, -1] = 1
            stUpTj[g, - 1] = 0
            stDownTj[g, -1] = 0
        else:
            stDwTG[g, min(- thermals.N_HOURS_IN_PREVIOUS_STATE[g]-len(thermals.STDW_TRAJ[g]),-1)] =1
            dispStat[g, -1] = 0
            stUpTj[g, - 1] = 0
            stDownTj[g, -1] = 0

def valid_inequalities(params, thermals, network, hydros,
                        m, periodsWithConstrs, stUpTG, stDwTG, dispStat, slacks):
    """
        Add valid inequalities based on minimum and maximum-generation requirements
    """

    if len(hydros.ID) == 0:
        # assuming that there are only thermal units in this case. then, a valid inequality
        # can be added to guarantee that a minimum of thermal units will be in the dispatch phase
        for t in periodsWithConstrs:
            slack_min_cap = m.add_var(obj = params.DEFICIT_COST, name = f'slack_min_disp_cap_{t}')
            slack_max_cap = m.add_var(obj = params.DEFICIT_COST, name = f'slack_max_disp_cap_{t}')
            m.add_constr(
                            m.xsum(thermals.MAX_P[g]*dispStat[g, t] - thermals.MIN_P[g]*stUpTG[g, t]
                                    for g in thermals.ID)
                                        + slack_min_cap >= np.sum(network.NET_LOAD[:, t]),
                                                        name = f'VI_min_disp_cap_{t}'
                        )
            m.add_constr(
                            m.xsum(thermals.MIN_P[g]*dispStat[g, t] for g in thermals.ID)
                                    - slack_max_cap <= np.sum(network.NET_LOAD[:, t]),
                                                        name = f'VI_max_disp_cap_{t}'
                        )

        for t in [t for t in periodsWithConstrs if t < (params.T - 1)]:
            slack_min_cap = m.add_var(obj = params.DEFICIT_COST,
                                                name = f'slack_min_disp_cap_next_period_{t}')
            m.add_constr(
                            m.xsum(thermals.MAX_P[g]*dispStat[g, t] - thermals.MIN_P[g]*stUpTG[g, t]
                                    + thermals.MIN_P[g]*(1 - dispStat[g, t])
                                    for g in thermals.ID)
                                + slack_min_cap >= np.sum(network.NET_LOAD[:, t + 1]),
                                                        name = f'VI_min_disp_cap_next_period_{t}'
                        )

    # the previous states of thermal units might prevent than from being shut-down
    if 0 in periodsWithConstrs:
        listOfT = list(periodsWithConstrs)
        listOfT.sort()
        for g in thermals.ID:
            periodsInDisp = []

            if thermals.T_G_0[g] > thermals.MIN_P[g]:
                pDecrease = 0

                lastT = max(listOfT)
                for t in listOfT:
                    pDecrease += thermals.RAMP_DOWN[g]
                    if (thermals.T_G_0[g] - pDecrease) <= thermals.MIN_P[g]:
                        # The unit reaches the minimum at t
                        # and can be turned off at t + len(thermals.STDW_TRAJ[g]) + 1
                        lastT = t
                        break
                periodsInDisp = list(range(min(listOfT), lastT + 1,1))
                if len(periodsInDisp) > 0:
                    m.add_constr(m.xsum(dispStat[g, t] for t in periodsInDisp) >= len(periodsInDisp),
                                                    name = f'UpTimeDueToRampAndPreviousState_{g}')

    for eq in thermals.EQ_GEN_CONSTR:
        # if a group of thermal units has its combined generation fixed to a given level, then
        # there should be no start-ups and no shut-downs for the units in this group. and
        # the dispatch-phase variable is set to the value in the period immediately before the
        # optimization horizon
        for g in eq[0]:
            for t in periodsWithConstrs:
                m.set_lb(dispStat[g, t], thermals.STATE_0[g])
                m.set_ub(stDwTG[g, t], 0)
                m.set_ub(stUpTG[g, t], 0)

    #### Add valid inequalities based on the minimum-generation requirements
    countMinConstrs = 0
    for constr in thermals.ADDTNL_MIN_P:
        if len(constr[0]) == 1:
            g = constr[0][0]
            for t in set(constr[1]) & periodsWithConstrs:
                m.set_lb(dispStat[g, t], 1)
                m.set_ub(stDwTG[g, t], 0)
                m.set_ub(stUpTG[g, t], 0)

                t2 = t
                if constr[2] > thermals.MIN_P[g]:
                    pDecrease = 0
                    for t2 in set(range(t + 1, params.T, 1)) & periodsWithConstrs:
                        pDecrease += thermals.RAMP_DOWN[g]
                        m.set_ub(stDwTG[g, t2], 0)
                        if (constr[2] - pDecrease) <= thermals.MIN_P[g]:
                            break

                if len(thermals.STDW_TRAJ[g]) >= 1:
                    for t3 in set(range(t2 + 1, min(params.T,
                                    t2 + len(thermals.STDW_TRAJ[g])), 1)) & periodsWithConstrs:
                        m.set_ub(stDwTG[g, t3], 0)
        else:
            if constr[2] > (sum(thermals.MAX_P[g] for g in constr[0])\
                                            - min({thermals.MAX_P[g] for g in constr[0]})):
                for t in set(constr[1]) & periodsWithConstrs:
                    for g in constr[0]:
                        m.set_lb(dispStat[g, t], 1)
                        m.set_ub(stDwTG[g, t], 0)
                        m.set_ub(stUpTG[g, t], 0)

                        t2 = t
                        if thermals.MAX_P[g] > thermals.MIN_P[g]:
                            pDecrease = 0
                            for t2 in set(range(t + 1, params.T, 1)) & periodsWithConstrs:
                                pDecrease += thermals.RAMP_DOWN[g]
                                m.set_ub(stDwTG[g, t2], 0)
                                if (thermals.MAX_P[g] -pDecrease)<= thermals.MIN_P[g]:
                                    break

                        if len(thermals.STDW_TRAJ[g]) >= 1:
                            for t3 in set(range(t2 + 1, min(params.T,
                                t2 + len(thermals.STDW_TRAJ[g])),1)) & periodsWithConstrs:
                                m.set_ub(stDwTG[g, t3], 0)
            else:
                # check if all maximum powers of the units in this constraint are the same
                maxPs = {thermals.MAX_P[g] for g in constr[0]}
                minUnitsOn = math.ceil(constr[2]/min(maxPs))

                minTotalTimeOut = {g: 1 for g in constr[0]}
                for g in constr[0]:
                    if (int(1/params.BASE_TIME_STEP)*thermals.MIN_DOWN[g]
                                                                >= len(thermals.STDW_TRAJ[g])):
                        minTotalTimeOut[g] += int(int(1/params.BASE_TIME_STEP)*thermals.MIN_DOWN[g])
                    else:
                        minTotalTimeOut[g] += len(thermals.STDW_TRAJ[g])
                    minTotalTimeOut[g] += len(thermals.STUP_TRAJ[g])

                for t in [i for i in constr[1] if
                                            (len(set(range(i - minTotalTimeOut[g] + 1, i + 1, 1))\
                                                & periodsWithConstrs) > 0) or
                                    ((i - len(thermals.STUP_TRAJ[g])) in periodsWithConstrs)]:

                    auxVar = {(g): m.add_var(ub = 1,
                            name = f"auxVarFutMinGen_{g}_{countMinConstrs}_{t}") for g in constr[0]}
                    constrsAdded = False
                    for g in constr[0]:
                        for t2 in (set(range(t - minTotalTimeOut[g] + 1, t + 1, 1))
                                                                            & periodsWithConstrs):
                            m.add_constr(auxVar[g] >= stDwTG[g, t2],
                                name = f"auxConstrShutDownFutMinGen_{g}_{countMinConstrs}_{t2}_{t}")
                            constrsAdded = True

                        if (t - len(thermals.STUP_TRAJ[g])) in periodsWithConstrs:
                            # the unit must either start-up at this point (the last chance), or
                            # it needs to already be operating
                            constrsAdded = True
                            rhs = (- m.xsum(stUpTG[g, t - len(thermals.STUP_TRAJ[g]) - j]
                                            for j in range(max(len(thermals.STUP_TRAJ[g]), 1)))
                                            - dispStat[g, t - len(thermals.STUP_TRAJ[g])])
                            m.add_constr(auxVar[g] >= 1 + rhs,
                                name = f"auxConstrDispOrStartUpFutMinGen_{g}_{countMinConstrs}_{t}")

                    slack = (m.add_var(obj = 10*params.DEFICIT_COST,
                            name = f'slackVIMPminThermalGen_{constr[0][0]}_{countMinConstrs}_{t}')
                                if slacks else 0)

                    lhs = m.xsum((1 - auxVar[g]) for g in constr[0]) if len(maxPs) == 1 else\
                                        m.xsum((1 - auxVar[g])*thermals.MAX_P[g] for g in constr[0])

                    if len(lhs.expr) > 0 and constrsAdded:
                        if len(maxPs) == 1:
                            m.add_constr(lhs + slack >= minUnitsOn,
                                    name =f'VIMPminThermalGen_{constr[0][0]}_{countMinConstrs}_{t}')
                        else:
                            m.add_constr(lhs + slack >= constr[2],
                                    name =f'VIMPminThermalGen_{constr[0][0]}_{countMinConstrs}_{t}')

        countMinConstrs += 1

    #### Add valid inequalities to the maximum generation
    for constr in [ctr for ctr in thermals.ADDTNL_MAX_P if ctr[2] == 0]:
        # maximum generation is zero
        for g in constr[0]:
            for t in set(constr[1]) & periodsWithConstrs:
                m.set_ub(dispStat[g, t], 0)
                m.set_ub(stUpTG[g, t], 0)
                m.set_ub(stDwTG[g, t], 0)

    countMaxConstrs = 0

    for constr in [ctr for ctr in thermals.ADDTNL_MAX_P if ctr[2] > 0]:
        # maximum generation is strictly more than zero

        # check if all minimum powers of the units in this constraint are the same
        minPs = {thermals.MIN_P[g] for g in constr[0]}
        maxUnitsOn = math.floor(constr[2]/min(minPs))

        minTotalTimeOn = {g: 1 for g in constr[0]}
        for g in constr[0]:
            minTotalTimeOn[g] += (int(1/params.BASE_TIME_STEP)*thermals.MIN_UP[g]
                                                                + len(thermals.STUP_TRAJ[g]))

        for t in [i for i in set(constr[1]) if
                    ((len(set(range(i - minTotalTimeOn[g] + 1, i, 1)) & periodsWithConstrs) > 0)
                        or (i in periodsWithConstrs))]:
            auxVar = {g: m.add_var(ub = 1, name = f'auxVarfutVIMPmaxGen_{g}_{countMaxConstrs}_{t}')
                                                                            for g in constr[0]}
            for g in constr[0]:
                rhs = m.xsum(stUpTG[g, t2] for t2 in range(t - minTotalTimeOn[g] + 1, t + 1, 1))
                if len(rhs.expr) > 0:
                    m.add_constr(auxVar[g] >= rhs,
                                        name = f'auxStUpVIMPmaxGen_{g}_{countMaxConstrs}_{t}_{t2}')

                if (t in periodsWithConstrs):
                    m.add_constr(auxVar[g] >= dispStat[g, t],
                                            name=f'auxDispatchVIMPmaxGen_{g}_{countMaxConstrs}_{t}')

            lhs = (
                    m.xsum(auxVar[g] for g in constr[0]) if len(minPs) == 1 else
                                            m.xsum(auxVar[g]*thermals.MIN_P[g] for g in constr[0])
                    )

            slack = (
                    m.add_var(obj = 10*params.DEFICIT_COST,
                        name=f'slackVIMPmaxThermalGen_{constr[0][0]}_{countMaxConstrs}_{t}')
                                                                                if slacks else 0
                    )

            if not(isinstance(lhs, (int, float))):
                if len(minPs) == 1:
                    m.add_constr(lhs - slack <= maxUnitsOn,
                                name = f'VIMPmaxThermalGen_{constr[0][0]}_{countMaxConstrs}_{t}')
                else:
                    m.add_constr(lhs - slack <= constr[2],
                                name = f'VIMPmaxThermalGen_{constr[0][0]}_{countMaxConstrs}_{t}')

        countMaxConstrs += 1

def thermalBin(m, params, thermals, network, hydros,
                listOfUnits,
                constrStUpTG, constrStDwTG, constrDispStat,
                fixedVars, periodsWithConstrs,
                varNature = 'B',
                slacks: bool = True):
    """Add binary variables associated with the thermals units
    m:                  optimization model
    params:             an instance of OptOptions (optoptions.py) that contains the
                            parameters for the problem and the algorithm
    thermals:           an instance of Thermals (network.py) with all thermal data
    network:            an instance of Network (network.py) with all network data
    hydros:             an instance of Hydros (network.py) with all hydro data. this is only used
                        here to build a valid inequality
    listOfUnits:        a list with the indices of units, as in thermals.ID, for
                        which the models should be built
    constrX:            dictionary containing the equality constraints that enforce decisions
                        taken in previous subhorizons
                        (X = StUpTG for the start-up decisions, StDwTG for shut-down decisions,
                        DispStat for dispatch status)
    fixedVars:          numpy array holding all time-coupling decisions
    periodsWithConstrs: a set with the periods in the planning horizon for which the hydro model
                        constraints and variables are to be included. For instance, if model m
                        is the optimization model of subhorizon 3, then periodsWithConstrs
                        has all time periods of this subhorizon
    varNature:          'B' or 'C'. Nature of the dispatch status, start-up and shut-down
                            decisions
    slacks:             True if slack variables are to be used in the constraints of minimum
                            and maximum generation for groups of thermal units
    """

    # Set the variables for periods in listTwithVars to binary, if varNature is 'B'
    tl_gBin = [(g, t) for g in listOfUnits for t in periodsWithConstrs]

    listOfTimesWithConVars = [t for t in range(params.T) if t < min(periodsWithConstrs)]

    tl_gC = [(g, t) for g in listOfUnits for t in listOfTimesWithConVars]

    #### For times subsequent to the current subhorizon, use zeros
    listOfTimesWithZeros =set(range(params.T))-set(periodsWithConstrs)-set(listOfTimesWithConVars)

    tl_zeros = [(g, t) for g in listOfUnits for t in listOfTimesWithZeros]

    #### Start-up decision
    stUpTG = {(g, t): m.add_var(var_type = varNature,
                                lb = 0,
                                ub = 0 if (thermals.MIN_P[g] + thermals.CONST_COST[g]) == 0 else 1,
                                    obj = thermals.ST_UP_COST[g],
                                        name = f'stup_tg_{g}_{t}') for (g, t) in tl_gBin}

    #### Shut-down decision
    stDwTG = {(g, t): m.add_var(var_type = 'C'
                                        if ((thermals.MIN_P[g] + thermals.CONST_COST[g]) == 0
                                            or (thermals.GEN_COST[g] + thermals.CONST_COST[g]) == 0)
                                                else varNature,
                                lb = 0,
                                ub = 0
                                        if ((thermals.MIN_P[g] + thermals.CONST_COST[g]) == 0
                                            or (thermals.GEN_COST[g] + thermals.CONST_COST[g]) == 0)
                                            else 1,
                                        name = f'stdw_tg_{g}_{t}') for (g, t) in tl_gBin}

    #### Dispatch phase
    dispStat = {(g, t): m.add_var(var_type = varNature,
                                lb = 1
                                        if ((thermals.MIN_P[g] + thermals.CONST_COST[g]) == 0
                                            or (thermals.GEN_COST[g] + thermals.CONST_COST[g]) == 0)
                                            else 0,
                                    ub = 1,
                                        obj = thermals.CONST_COST[g],
                                            name = f'disp_status_{g}_{t}') for (g, t) in tl_gBin}

    #### Set the variables for all periods before those in listTwithVars to continuous
    #### Start-up decision
    stUpTG.update({(g, t): m.add_var(var_type = 'C', ub = 1, obj = 0,
                                            name = f'stup_tg_{g}_{t}') for (g, t) in tl_gC})

    #### Shut-down decision
    stDwTG.update({(g, t): m.add_var(var_type = 'C', ub = 1, obj = 0,
                                            name = f'stdw_tg_{g}_{t}') for (g, t) in tl_gC})

    #### Dispatch phase
    dispStat.update({(g, t): m.add_var(var_type = 'C', ub = 1, obj = 0,
                                            name = f'disp_status_{g}_{t}') for (g, t) in tl_gC})

    #### Fill up the remaining periods with zeros
    #### Start-up decision
    stUpTG.update({(g, t): 0 for (g, t) in tl_zeros})
    #### Shut-down decision
    stDwTG.update({(g, t): 0 for (g, t) in tl_zeros})
    #### Dispatch phase
    dispStat.update({(g, t): 0 for (g, t) in tl_zeros})

    #### Start-up trajectory
    stUpTj = {(g, t): 0 for t in range(params.T) for g in listOfUnits}

    #### Shut-down trajectory
    stDownTj = {(g, t): 0 for t in range(params.T) for g in listOfUnits}

    previousStates(params, thermals, listOfUnits,
                    periodsWithConstrs, m, stUpTG, stDwTG, dispStat, stUpTj, stDownTj)

    if set(listOfUnits) == set(thermals.ID):
        valid_inequalities(params, thermals, network, hydros,
                                m, periodsWithConstrs, stUpTG, stDwTG, dispStat, slacks)

    #### Minimum up time
    for g in [g for g in listOfUnits if thermals.MIN_UP[g] > 0]:
        stUp = len(thermals.STUP_TRAJ[g])
        for t in periodsWithConstrs:
            m.add_constr(m.xsum(stUpTG[g, t2] for t2 in range(t -
                    fromHoursToTindex2(params, thermals.MIN_UP[g], t) - stUp + 1, t - stUp + 1, 1))
                                    <= dispStat[g, t], name = f'min_up_{g}_{t}')

    #### Minimum down time
    for g in [g for g in listOfUnits if thermals.MIN_DOWN[g] > 0]:
        for t in periodsWithConstrs:
            m.add_constr(m.xsum(stDwTG[g, t2] for t2 in range(t -
                                fromHoursToTindex2(params, thermals.MIN_DOWN[g], t) + 1, t + 1, 1))
                                            <= (1 - dispStat[g, t]), name = f'min_down_{g}_{t}')

    #### Logical constraitns
    for g in [g for g in listOfUnits if (thermals.MIN_P[g] + thermals.CONST_COST[g]) > 0]:
        for t in periodsWithConstrs:
            m.add_constr((stUpTG[g, t - len(thermals.STUP_TRAJ[g])] - stDwTG[g, t]
                        - dispStat[g, t] + dispStat[g, t - 1] == 0), name = f'logical_{g}_{t}')

    #### Prevent start-ups and shut-downs from happening at the same time
    for g in listOfUnits:
        if len(thermals.STUP_TRAJ[g]) > 0:
            stUp = len(thermals.STUP_TRAJ[g])
            for t in periodsWithConstrs:
                m.add_constr(m.xsum(stUpTG[g, i] for i in range(t - stUp + 1, t + 1, 1))
                                <= (1 - dispStat[g, t] - m.xsum(stDwTG[g, t2] for t2 in range(t -
                            fromHoursToTindex2(params, thermals.MIN_DOWN[g], t) + 1, t + 1, 1))),
                                        name = f'StUpStDown_{g}_{t}')
        if len(thermals.STUP_TRAJ[g]) == 0 or ((thermals.MIN_UP[g] + thermals.MIN_DOWN[g]) == 0):
            for t in periodsWithConstrs:
                m.add_constr(stUpTG[g, t] + stDwTG[g, t] <= 1, name = f'StUpStDown_{g}_{t}')

    stUpTG = {(g, t): stUpTG[g, t] for g in listOfUnits for t in range(params.T)}
    stDwTG = {(g, t): stDwTG[g, t] for g in listOfUnits for t in range(params.T)}
    dispStat = {(g, t): dispStat[g, t] for g in listOfUnits for t in range(params.T)}

    previousPeriod = {t for t in range(params.T) if t < min(periodsWithConstrs)}

    if params.COUPLING == Coupling.CONSTRS:
        for g, t in [(g, t) for g in listOfUnits for t in previousPeriod]:
            constrStUpTG[g, t] = m.add_constr(
                                            stUpTG[g, t] == fixedVars[params.MAP['stUpTG'][g,t]],
                                                name = f'constr_stup_tg_{g}_{t}',
                                                lhs = stUpTG[g, t],
                                                    rhs = fixedVars[params.MAP['stUpTG'][g,t]],
                                                        sign = '==')
            constrStDwTG[g, t] = m.add_constr(
                                            stDwTG[g, t] == fixedVars[params.MAP['stDwTG'][g,t]],
                                                name = f'constr_stdw_tg_{g}_{t}',
                                                lhs = stDwTG[g, t],
                                                    rhs = fixedVars[params.MAP['stDwTG'][g,t]],
                                                        sign = '==')
            constrDispStat[g, t] = m.add_constr(
                                            dispStat[g, t] == fixedVars[params.MAP['DpTG'][g,t]],
                                                name = f'constr_disp_status_{g}_{t}',
                                                lhs = dispStat[g, t],
                                                    rhs = fixedVars[params.MAP['DpTG'][g,t]],
                                                        sign = '==')
    else:
        for g, t in [(g, t) for g in listOfUnits for t in previousPeriod]:
            m.set_lb(stUpTG[g, t], fixedVars[params.MAP['stUpTG'][g,t]])
            m.set_ub(stUpTG[g, t], fixedVars[params.MAP['stUpTG'][g,t]])

            m.set_lb(stDwTG[g, t], fixedVars[params.MAP['stDwTG'][g,t]])
            m.set_ub(stDwTG[g, t], fixedVars[params.MAP['stDwTG'][g,t]])

            m.set_lb(dispStat[g, t], fixedVars[params.MAP['DpTG'][g,t]])
            m.set_ub(dispStat[g, t], fixedVars[params.MAP['DpTG'][g,t]])

            constrStUpTG[g, t] = stUpTG[g, t]
            constrStDwTG[g, t] = stDwTG[g, t]
            constrDispStat[g, t] = dispStat[g, t]

    return (stUpTG, stDwTG, dispStat)

def thermalCont(m, params, thermals, network, listOfUnits, constrTgDisp, fixedVars,
                    periodsWithConstrs, stUpTG, stDwTG, dispStat, slacks: bool = True):
    """Add continuous variables and their constraints for the thermal model
    m:                  optimization model
    params:             an instance of OptOptions (optoptions.py) that contains the
                            parameters for the problem and the algorithm
    thermals:           an instance of Thermals (network.py) with all thermal data
    listOfUnits:        a list with the indices of units, as in thermals.ID, for
                        which the models should be built
    network:            an instance of Network (network.py) with all network data
    constrX:            dictionary containing the equality constraints that enforce decisions
                        taken in previous subhorizons
                        (X = TgDisp for the generation in the dispatch phase)
    fixedVars:          numpy array holding all time-coupling decisions
    periodsWithConstrs: a set with the periods in the planning horizon for which the hydro model
                        constraints and variables are to be included. For instance, if model m
                        is the optimization model of subhorizon 3, then periodsWithConstrs
                        has all time periods of this subhorizon
    stUpTG:             values of the start-up decisions
    stDwTG:             values of the shut-down decisions
    dispStat:           values of the dispatch-status decisions
    slacks:             True if slack variables are to be used in the constraints of minimum
                            and maximum generation for groups of thermal units
    """

    for g in listOfUnits:
        for t in range(- len(thermals.STUP_TRAJ[g]), 0, 1):
            stUpTG[g, t] = 0

    extraPeriodsWithVars = {t for t in range(params.T) if t < min(periodsWithConstrs)}
    #### For times after the current subhorizon, use zeros
    listOfTimesAfterSubhorizon = set(range(params.T)) - set(periodsWithConstrs)-extraPeriodsWithVars

    sub_set_thermal_units = [g for g in listOfUnits
                                            if thermals.MIN_P[g] > 0
                                                or (len(thermals.STUP_TRAJ[g]) > 0
                                                    or len(thermals.STDW_TRAJ[g]) > 0)]

    tg_disp_obj = {(g,t): 0 for g in listOfUnits for t in periodsWithConstrs | extraPeriodsWithVars}
    for g in set(listOfUnits) - set(sub_set_thermal_units):
        for t in periodsWithConstrs:
            tg_disp_obj[g, t] = thermals.GEN_COST[g]

    tgDisp = {}
    for g in listOfUnits:
        tgDisp.update({(g, t): m.add_var(var_type = 'C',
                                        obj = tg_disp_obj[g, t],
                                            name = f"tgDisp_{g}_{t}")
                                                for t in periodsWithConstrs | extraPeriodsWithVars})

        tgDisp.update({(g, t): 0 for t in listOfTimesAfterSubhorizon})

    tg = {(g, t): m.add_var(var_type = 'C', obj = thermals.GEN_COST[g],
                                        name = f'tg_{g}_{t}') for t in periodsWithConstrs
                                            for g in sub_set_thermal_units}

    tg.update({(g, t): tgDisp[g, t] for t in periodsWithConstrs
                                            for g in set(listOfUnits) - set(sub_set_thermal_units)})

    tg.update({(g, t): m.add_var(var_type = 'C', obj = 0,
                                        name = f'tg_{g}_{t}') for t in extraPeriodsWithVars
                                            for g in sub_set_thermal_units})

    tg.update({(g, t): tgDisp[g, t] for t in extraPeriodsWithVars
                                            for g in set(listOfUnits) - set(sub_set_thermal_units)})

    tg.update({(g, t): 0 for t in listOfTimesAfterSubhorizon for g in sub_set_thermal_units})

    #### Start-up trajectory
    tgUpTj = {(g, t): 0 for t in range(params.T) for g in listOfUnits}

    #### Shut-down trajectory
    tgDownTj = {(g, t): 0 for t in range(params.T) for g in listOfUnits}

    #### Start-up trajectory
    for g in listOfUnits:
        if len(thermals.STUP_TRAJ[g]) > 0:
            steps = len(thermals.STUP_TRAJ[g])

            for t in periodsWithConstrs:
                tgUpTj[g, t] = m.xsum(stUpTG[g, i]*thermals.STUP_TRAJ[g][t - steps - i]
                                                for i in range(max(t - steps + 1, 0), t + 1, 1))

    #### Shut-down trajectory
    #### Note that it is implicitly assumed that the power output of the
    #### thermal unit is zero when stDwTG = 1
    for g in [g for g in listOfUnits if len(thermals.STDW_TRAJ[g]) > 0]:
        steps = len(thermals.STDW_TRAJ[g])
        for t in periodsWithConstrs:
            tgDownTj[g, t] = m.xsum(stDwTG[g, t - i]*thermals.STDW_TRAJ[g][i]
                                                for i in [j for j in range(steps) if (t - j) >= 0])

    #### lower and upper operating limits of thermal units
    for g in listOfUnits:
        for t in periodsWithConstrs:
            m.add_constr(tgDisp[g, t] - (thermals.MAX_P[g] - thermals.MIN_P[g])*dispStat[g, t] <= 0,
                                                                            name = f'max_p_{g}_{t}')

    #### total generation
    for g in [g for g in listOfUnits if thermals.MIN_P[g] > 0
                                                or (len(thermals.STUP_TRAJ[g]) > 0
                                                        or len(thermals.STDW_TRAJ[g]) > 0)]:
        for t in periodsWithConstrs:
            m.add_constr(tg[g, t] - tgDisp[g, t] - thermals.MIN_P[g]*dispStat[g, t]
                                    - tgUpTj[g, t] - tgDownTj[g, t] == 0, name = f'gen_{g}_{t}')

    # new constraints
    if (0 in periodsWithConstrs):
        for g in listOfUnits:
            if thermals.STATE_0[g] == 0:
                m.add_constr(tgDisp[g, 0] <= 0, name = f'ramp_up_{g}_{0}')
            else:
                m.add_constr(tgDisp[g, 0] - (thermals.T_G_0[g] - thermals.MIN_P[g])
                                    <= thermals.RAMP_UP[g], name = f'ramp_up_{g}_{0}')
                m.add_constr(- tgDisp[g, 0] + (thermals.T_G_0[g] - thermals.MIN_P[g])
                                    <= thermals.RAMP_DOWN[g], name = f'ramp_down_{g}_{0}')

    for g in [g for g in listOfUnits if
                                (thermals.RAMP_UP[g] < (thermals.MAX_P[g] - thermals.MIN_P[g]))]:
        for t in set(periodsWithConstrs) - {0}:
            m.add_constr(tgDisp[g, t] - tgDisp[g, t - 1] <= thermals.RAMP_UP[g]*dispStat[g, t -1],
                                                name = f'ramp_up_{g}_{t}')
            m.add_constr(- tgDisp[g, t] + tgDisp[g, t - 1] <= thermals.RAMP_DOWN[g]*dispStat[g,t],
                                                name = f'ramp_down_{g}_{t}')
    for g in [g for g in listOfUnits if
                        (thermals.RAMP_UP[g] >= (thermals.MAX_P[g] - thermals.MIN_P[g]))]:
        for t in set(periodsWithConstrs) - {0}:
            m.add_constr(tgDisp[g, t] <= (thermals.MAX_P[g] - thermals.MIN_P[g])*dispStat[g, t -1],
                                                name = f'start_up_capability_{g}_{t}')
            m.add_constr(tgDisp[g, t - 1] <= (thermals.MAX_P[g] - thermals.MIN_P[g])*dispStat[g,t],
                                                name = f'shut_down_capability_{g}_{t}')

    if (0 in periodsWithConstrs):
        for g in listOfUnits:
            if ((thermals.STATE_0[g]==1)
                    and not(isinstance(stDwTG[g, 0], (int, float)))
                                        and (thermals.MAX_P[g] - thermals.MIN_P[g]) > 0):
                m.add_constr(thermals.T_G_0[g] - thermals.MAX_P[g] <=
                                -(thermals.MAX_P[g] - thermals.MIN_P[g])*stDwTG[g, 0],
                                                name = f'shut_down_capability_{g}_{0}')

    if len(network.SEC_CONSTRS) > 0:
        s_sec_constrs = add_sec_constraints_only_on_thermals(m, params, thermals, network,
                                                                tg, periods = periodsWithConstrs)

    previousPeriod = {t for t in range(params.T) if t < min(periodsWithConstrs)}

    if params.COUPLING == Coupling.CONSTRS:
        for g, t in [(g, t) for g in listOfUnits for t in previousPeriod]:
            constrTgDisp[g,t] = m.add_constr(tgDisp[g,t] == fixedVars[params.MAP['DpGenTG'][g,t]],
                                                        name = f'constrTgDisp_{g}_{t}',
                                                    lhs = tgDisp[g,t],
                                                        rhs = fixedVars[params.MAP['DpGenTG'][g,t]],
                                                            sign = '==')
    else:
        for g, t in [(g, t) for g in listOfUnits for t in previousPeriod]:
            m.set_lb(tgDisp[g, t], fixedVars[params.MAP['DpGenTG'][g,t]])
            m.set_ub(tgDisp[g, t], fixedVars[params.MAP['DpGenTG'][g,t]])

            constrTgDisp[g,t] = tgDisp[g,t]

    return (tg, tgDisp)
