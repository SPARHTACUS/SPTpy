# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""

import math
import numpy as np
from mip import BINARY, CONTINUOUS, xsum, entities

def fromHoursToTindex2(numberOfHours, indexFrom):
    ''' How many indices should it go back to account for numberOfHours?'''
    sumHours = 0
    t = 0
    while sumHours < numberOfHours:
        t += 1
        if indexFrom - t < 0:
            sumHours += 1
        else:
            sumHours += 0.5
    return(t)

def previousStates(params, thermals, listOfUnits, periodsWithConstrs,\
                    stUpTG, stDwTG, dispStat, stUpTj, stDownTj):
    '''Create auxiliary keys and set variables bounds according to the states
    previous to the optimization horizon'''

    for g in listOfUnits:
        for t in set(range(0, min(thermals.sdDec[g], params.T), 1)) & periodsWithConstrs:
            stDwTG[g, t].ub = 0
    ############################################################################

    #### Previous states
    for g in listOfUnits:

        nHours = math.ceil((thermals.maxP[g] - thermals.minP[g])/thermals.rampUp[g])\
                                                            if thermals.rampUp[g] > 0 else 0

        for t in range(- 4*thermals.minUp[g] - len(thermals.startUpTraj[g]) - nHours, 0, 1):
            dispStat[g, t] = thermals.state0[g]
            stUpTG[g, t] = 0
        for t in range(- max(thermals.minDown[g] + len(thermals.shutDownTraj[g]),\
                            thermals.minUp[g], len(thermals.startUpTraj[g]) + nHours, 1), 0, 1):
            stDwTG[g, t] = 0

        for t in range(params.T, params.T + len(thermals.shutDownTraj[g]) + nHours + 1, 1):
            stDwTG[g, t] = 0

        for t in range(- nHours, 0, 1):
            dispStat[g, t] = thermals.state0[g]

        if (thermals.state0[g] == 1):
            # Either if it is currently in a start-up trajectory or it has already finished
            # the trajectory, the generator was brought on at time - thermals.nHoursInPreState[g].
            # However, if it is currently in the shut-down trajectory, then it will eventually
            # be shut-down during the planning horizon
            if thermals.inStartUpTraj[g]:
                stUpTG[g, - thermals.nHoursInPreState[g]] = 1
                dispStat[g, -1] = 0
                stDownTj[g, -1] = 0
                for t in range(-thermals.nHoursInPreState[g], 0, 1):
                    stUpTj[g, t] = 1
            elif thermals.inShutDownTraj[g]:
                #### The generator will be effectively shut-down during the
                #### current planning horizon
                sdT = len(thermals.shutDownTraj[g]) - thermals.nHoursInPreState[g]
                if (sdT < params.T) and (sdT >= thermals.sdDec[g]) and (sdT in periodsWithConstrs):
                    stDwTG[g, sdT].lb = 1
                    stDwTG[g, sdT].ub = 1
                else:
                    stDwTG[g, min(sdT, -1)] = 1
                dispStat[g, -1] = 0
                stUpTj[g, - 1] = 0
                for t in range(-thermals.nHoursInPreState[g] - len(thermals.shutDownTraj[g]), 0, 1):
                    stDownTj[g, t] = 1
            else:
                # If the unit is currently in the dispatch phase, then it means that, at some point,
                # the unit was started-up and it successfully completed its start-up trajectory.
                # Thus, the unit was started-up at period 0 minus the number of periods it has been
                # in the dispatch phase (thermals.nHoursInPreState[g]) minus the number of periods
                # necessary to complete the start-up trajectory
                stUpTG[g, min(- len(thermals.startUpTraj[g]) - thermals.nHoursInPreState[g], -1)]=1
                dispStat[g, -1] = 1
                stUpTj[g, - 1] = 0
                stDownTj[g, -1] = 0
        else:
            stDwTG[g, min(- thermals.nHoursInPreState[g] - len(thermals.shutDownTraj[g]), -1)] = 1
            dispStat[g, -1] = 0
            stUpTj[g, - 1] = 0
            stDownTj[g, -1] = 0
    return()

def validIneqs(params, thermals, network, hydros,
        m, periodsWithConstrs, stUpTG, stDwTG, dispStat, slacks):
    '''Add valid inequalities based on minimum and maximum-generation requirements'''

    if len(hydros.id) == 0:
        # assuming that there are only thermal units in this case. then, a valid inequality
        # can be added to guarantee that a minimum of thermal units will be in the dispatch phase
        for t in periodsWithConstrs:
            slackVarMin = m.add_var(obj = 10*params.deficitCost, name = f'slackTotalMinBin_{t}')
            slackVarMax = m.add_var(obj = 10*params.deficitCost, name = f'slackTotalMaxBin_{t}')
            m.add_constr(xsum(thermals.maxP[g]*dispStat[g, t] for g in range(len(thermals.id)))\
                                + slackVarMin >= np.sum(network.load[t, :]),\
                                                        name = f'VI_minSetOfThermalUnitsOn_{t}')
            m.add_constr(xsum(thermals.minP[g]*dispStat[g, t] for g in range(len(thermals.id)))\
                                - slackVarMax <= np.sum(network.load[t, :]),\
                                                        name = f'VI_maxSetOfThermalUnitsOn_{t}')

    # the previous states of thermal units might prevent than from being shut-down
    if 0 in periodsWithConstrs:
        listOfT = list(periodsWithConstrs)
        listOfT.sort()
        for g in range(len(thermals.id)):
            periodsInDisp = []

            if thermals.tg0[g] > thermals.minP[g]:
                pDecrease = 0

                lastT = max(listOfT)
                for t in listOfT:
                    pDecrease += thermals.rampDown[g]
                    if (thermals.tg0[g] - pDecrease) <= thermals.minP[g]:
                        # The unit reaches the minimum at t
                        # and can be turned off at t + len(thermals.shutDownTraj[g]) + 1
                        lastT = t
                        break
                periodsInDisp = list(range(min(listOfT), lastT + 1,1))
                if len(periodsInDisp) > 0:
                    m.add_constr(xsum(dispStat[g, t] for t in periodsInDisp) >= len(periodsInDisp),\
                                                    name = f'UpTimeDueToRampAndPreviousState_{g}')

    for eq in thermals.equalityConstrs:
        # if a group of thermal units has its combined generation fixed to a given level, then
        # there should be no start-ups and no shut-downs for the units in this group. and
        # the dispatch-phase variable is set to the value in the period immediately before the
        # optimization horizon
        for g in eq[0]:
            for t in periodsWithConstrs:
                dispStat[g, t].lb = thermals.state0[g]
                stDwTG[g, t].ub = 0
                stUpTG[g, t].ub = 0

    #### Add valid inequalities based on the minimum-generation requirements
    countMinConstrs = 0
    for constr in thermals.minGen:
        if len(constr[0]) == 1:
            g = constr[0][0]
            for t in set(constr[1]) & periodsWithConstrs:
                dispStat[g, t].lb = 1
                stDwTG[g, t].ub = 0
                stUpTG[g, t].ub = 0

                t2 = t
                if constr[2] > thermals.minP[g]:
                    pDecrease = 0
                    for t2 in set(range(t + 1, params.T, 1)) & periodsWithConstrs:
                        pDecrease += thermals.rampDown[g]
                        stDwTG[g, t2].ub = 0
                        if (constr[2] - pDecrease) <= thermals.minP[g]:
                            break

                if len(thermals.shutDownTraj[g]) >= 1:
                    for t3 in set(range(t2 + 1, min(params.T,\
                                    t2 + len(thermals.shutDownTraj[g])), 1)) & periodsWithConstrs:
                        stDwTG[g, t3].ub = 0
        else:
            if constr[2] > (sum(thermals.maxP[g] for g in constr[0])\
                                            - min({thermals.maxP[g] for g in constr[0]})):
                for t in set(constr[1]) & periodsWithConstrs:
                    for g in constr[0]:
                        dispStat[g, t].lb = 1
                        stDwTG[g, t].ub = 0
                        stUpTG[g, t].ub = 0

                        t2 = t
                        if thermals.maxP[g] > thermals.minP[g]:
                            pDecrease = 0
                            for t2 in set(range(t + 1, params.T, 1)) & periodsWithConstrs:
                                pDecrease += thermals.rampDown[g]
                                stDwTG[g, t2].ub = 0
                                if (thermals.maxP[g] -pDecrease)<= thermals.minP[g]:
                                    break

                        if len(thermals.shutDownTraj[g]) >= 1:
                            for t3 in set(range(t2 + 1, min(params.T,\
                                t2 + len(thermals.shutDownTraj[g])),1)) & periodsWithConstrs:
                                stDwTG[g, t3].ub = 0
            else:
                # check if all maximum powers of the units in this constraint are the same
                maxPs = {thermals.maxP[g] for g in constr[0]}
                minUnitsOn = math.ceil(constr[2]/min(maxPs))

                minTotalTimeOut = {g: 1 for g in constr[0]}
                for g in constr[0]:
                    if int(1/params.baseTimeStep)*thermals.minDown[g]\
                                                                >= len(thermals.shutDownTraj[g]):
                        minTotalTimeOut[g] += int(int(1/params.baseTimeStep)*thermals.minDown[g])
                    else:
                        minTotalTimeOut[g] += len(thermals.shutDownTraj[g])
                    minTotalTimeOut[g] += len(thermals.startUpTraj[g])

                for t in [i for i in constr[1] if\
                                            (len(set(range(i - minTotalTimeOut[g] + 1, i + 1, 1))\
                                                & periodsWithConstrs) > 0) or\
                                        ((i - len(thermals.startUpTraj[g])) in periodsWithConstrs)]:

                    auxVar = {(g): m.add_var(ub = 1,\
                            name = f"auxVarFutMinGen_{g}_{countMinConstrs}_{t}") for g in constr[0]}
                    constrsAdded = False
                    for g in constr[0]:
                        for t2 in set(range(t - minTotalTimeOut[g] + 1, t + 1, 1))\
                                                                            & periodsWithConstrs:
                            m.add_constr(auxVar[g] >= stDwTG[g, t2],\
                                name = f"auxConstrShutDownFutMinGen_{g}_{countMinConstrs}_{t2}_{t}")
                            constrsAdded = True

                        if (t - len(thermals.startUpTraj[g])) in periodsWithConstrs:
                            # the unit must either start-up at this point (the last chance), or
                            # it needs to already be operating
                            constrsAdded = True
                            rhs = - xsum(stUpTG[g, t - len(thermals.startUpTraj[g]) - j]\
                                            for j in range(max(len(thermals.startUpTraj[g]), 1)))\
                                            - dispStat[g, t - len(thermals.startUpTraj[g])]
                            m.add_constr(auxVar[g] >= 1 + rhs,\
                                name = f"auxConstrDispOrStartUpFutMinGen_{g}_{countMinConstrs}_{t}")

                    slack = m.add_var(obj = 10*params.deficitCost,\
                            name = f'slackVIMPminThermalGen_{constr[0][0]}_{countMinConstrs}_{t}')\
                                if slacks else 0

                    lhs = xsum((1 - auxVar[g]) for g in constr[0]) if len(maxPs) == 1 else\
                                        xsum((1 - auxVar[g])*thermals.maxP[g] for g in constr[0])

                    if len(lhs.expr) > 0 and constrsAdded:
                        if len(maxPs) == 1:
                            m.add_constr(lhs + slack >= minUnitsOn,\
                                    name =f'VIMPminThermalGen_{constr[0][0]}_{countMinConstrs}_{t}')
                        else:
                            m.add_constr(lhs + slack >= constr[2],\
                                    name =f'VIMPminThermalGen_{constr[0][0]}_{countMinConstrs}_{t}')

        countMinConstrs += 1

    #### Add valid inequalities to the maximum generation
    for constr in [ctr for ctr in thermals.maxGen if ctr[2] == 0]:
        # maximum generation is zero
        for g in constr[0]:
            for t in set(constr[1]) & periodsWithConstrs:
                dispStat[g, t].ub = 0
                stUpTG[g, t].ub = 0
                stDwTG[g, t].ub = 0

    countMaxConstrs = 0

    for constr in [ctr for ctr in thermals.maxGen if ctr[2] > 0]:
        # maximum generation is strictly more than zero

        # check if all minimum powers of the units in this constraint are the same
        minPs = {thermals.minP[g] for g in constr[0]}
        maxUnitsOn = math.floor(constr[2]/min(minPs))

        minTotalTimeOn = {g: 1 for g in constr[0]}
        for g in constr[0]:
            minTotalTimeOn[g] += int(1/params.baseTimeStep)*thermals.minUp[g]\
                                                                    + len(thermals.startUpTraj[g])

        for t in [i for i in set(constr[1]) if\
                    ((len(set(range(i - minTotalTimeOn[g] + 1, i, 1)) & periodsWithConstrs) > 0)\
                        or (i in periodsWithConstrs))]:
            auxVar = {g: m.add_var(ub = 1, name = f'auxVarfutVIMPmaxGen_{g}_{countMaxConstrs}_{t}')\
                                                                            for g in constr[0]}
            for g in constr[0]:
                rhs = xsum(stUpTG[g, t2] for t2 in range(t - minTotalTimeOn[g] + 1, t + 1, 1))
                if len(rhs.expr) > 0:
                    m.add_constr(auxVar[g] >= rhs,\
                                        name = f'auxStUpVIMPmaxGen_{g}_{countMaxConstrs}_{t}_{t2}')

                if (t in periodsWithConstrs):
                    m.add_constr(auxVar[g] >= dispStat[g, t],\
                                            name=f'auxDispatchVIMPmaxGen_{g}_{countMaxConstrs}_{t}')

            lhs = xsum(auxVar[g] for g in constr[0]) if len(minPs) == 1 else\
                                                xsum(auxVar[g]*thermals.minP[g] for g in constr[0])

            slack = m.add_var(obj = 10*params.deficitCost,\
                        name=f'slackVIMPmaxThermalGen_{constr[0][0]}_{countMaxConstrs}_{t}')\
                                                                                    if slacks else 0

            if len(lhs.expr) > 0:
                if len(minPs) == 1:
                    m.add_constr(lhs - slack <= maxUnitsOn,\
                                name = f'VIMPmaxThermalGen_{constr[0][0]}_{countMaxConstrs}_{t}')
                else:
                    m.add_constr(lhs - slack <= constr[2],\
                                name = f'VIMPmaxThermalGen_{constr[0][0]}_{countMaxConstrs}_{t}')

        countMaxConstrs += 1

    return()

def thermalBin(m, params, thermals, network, hydros,\
                listOfUnits,\
                constrStUpTG, constrStDwTG, constrDispStat,\
                fixedVars, periodsWithConstrs,\
                varNature = BINARY,
                slacks: bool = True):
    '''Add binary variables associated with the thermals units
    m:                  optimization model
    params:             an instance of OptOptions (optoptions.py) that contains the
                            parameters for the problem and the algorithm
    thermals:           an instance of Thermals (network.py) with all thermal data
    network:            an instance of Network (network.py) with all network data
    hydros:             an instance of Hydros (network.py) with all hydro data. this is only used
                        here to build a valid inequality
    listOfUnits:        a list with the indices of units, as in range(len(thermals.id)), for
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
    varNature:          BINARY or CONTINUOUS. Nature of the dispatch status, start-up and shut-down
                            decisions
    slacks:             True if slack variables are to be used in the constraints of minimum
                            and maximum generation for groups of thermal units
    '''

    # Set the variables for periods in listTwithVars to binary, if varNature is BINARY
    tl_gBin = [(g, t) for g in listOfUnits for t in periodsWithConstrs]

    listOfTimesWithConVars = [t for t in range(params.T) if t < min(periodsWithConstrs)]

    tl_gC = [(g, t) for g in listOfUnits for t in listOfTimesWithConVars]

    #### For times subsequent to the current subhorizon, use zeros
    listOfTimesWithZeros =set(range(params.T))-set(periodsWithConstrs)-set(listOfTimesWithConVars)

    tl_zeros = [(g, t) for g in listOfUnits for t in listOfTimesWithZeros]

    #### Start-up decision
    stUpTG = {(g, t): m.add_var(var_type = varNature, ub = 1, obj = thermals.stUpCost[g],\
                                            name = f'stUpTG_{g}_{t}') for (g, t) in tl_gBin}

    #### Shut-down decision
    stDwTG = {(g, t): m.add_var(var_type = varNature, ub = 1, obj = thermals.stDwCost[g],\
                                            name = f'stDwTG_{g}_{t}') for (g, t) in tl_gBin}

    #### Dispatch phase
    dispStat = {(g, t): m.add_var(var_type = varNature, ub = 1, obj = thermals.constCost[g],\
                                            name = f'dispStat_{g}_{t}') for (g, t) in tl_gBin}

    #### Set the variables for all periods before those in listTwithVars to continuous
    #### Start-up decision
    stUpTG.update({(g, t): m.add_var(var_type = CONTINUOUS, ub = 1, obj = 0,\
                                            name = f'stUpTG_{g}_{t}') for (g, t) in tl_gC})

    #### Shut-down decision
    stDwTG.update({(g, t): m.add_var(var_type = CONTINUOUS, ub = 1, obj = 0,\
                                            name = f'stDwTG_{g}_{t}') for (g, t) in tl_gC})

    #### Dispatch phase
    dispStat.update({(g, t): m.add_var(var_type = CONTINUOUS, ub = 1, obj = 0,\
                                            name = f'dispStat_{g}_{t}') for (g, t) in tl_gC})

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

    previousStates(params, thermals, listOfUnits,\
                periodsWithConstrs, stUpTG, stDwTG, dispStat, stUpTj, stDownTj)

    if set(listOfUnits) == set(range(len(thermals.id))):
        validIneqs(params, thermals, network, hydros,\
                m, periodsWithConstrs, stUpTG, stDwTG, dispStat, slacks)

    #### Minimum up time
    for g in [g for g in listOfUnits if thermals.minUp[g] > 0]:
        stUp = len(thermals.startUpTraj[g])
        for t in periodsWithConstrs:
            m.add_constr(xsum(stUpTG[g, t2] for t2 in range(t -\
                            fromHoursToTindex2(thermals.minUp[g], t) - stUp + 1, t - stUp + 1, 1))\
                                    <= dispStat[g, t], name = f'minUp_{g}_{t}')

    #### Minimum down time
    for g in [g for g in listOfUnits if thermals.minDown[g] > 0]:
        for t in periodsWithConstrs:
            m.add_constr(xsum(stDwTG[g, t2] for t2 in range(t -\
                                        fromHoursToTindex2(thermals.minDown[g], t) + 1, t + 1, 1))\
                                            <= (1 - dispStat[g, t]), name = f'minDown_{g}_{t}')

    #### Logical constraitns
    for g in listOfUnits:
        for t in periodsWithConstrs:
            m.add_constr((stUpTG[g, t - len(thermals.startUpTraj[g])] - stDwTG[g, t]\
                        - dispStat[g, t] + dispStat[g, t - 1] == 0), name = f'logical_{g}_{t}')

    #### Prevent start-ups and shut-downs from happening at the same time
    for g in listOfUnits:
        if len(thermals.startUpTraj[g]) > 0:
            stUp = len(thermals.startUpTraj[g])
            for t in periodsWithConstrs:
                m.add_constr(xsum(stUpTG[g, i] for i in range(t - stUp + 1, t + 1, 1))\
                                <= (1 - dispStat[g, t] - xsum(stDwTG[g, t2] for t2 in range(t -\
                                    fromHoursToTindex2(thermals.minDown[g], t) + 1, t + 1, 1))),\
                                        name = f'StUpStDown_{g}_{t}')
        if len(thermals.startUpTraj[g]) == 0 or ((thermals.minUp[g] + thermals.minDown[g]) == 0):
            for t in periodsWithConstrs:
                m.add_constr(stUpTG[g, t] + stDwTG[g, t] <= 1, name = f'StUpStDown_{g}_{t}')

    stUpTG = {(g, t): stUpTG[g, t] for g in listOfUnits for t in range(params.T)}
    stDwTG = {(g, t): stDwTG[g, t] for g in listOfUnits for t in range(params.T)}
    dispStat = {(g, t): dispStat[g, t] for g in listOfUnits for t in range(params.T)}

    previousPeriod = {t for t in range(params.T) if t < min(periodsWithConstrs)}

    listOfTups = [(g, t) for g in listOfUnits for t in previousPeriod]
    for g, t in listOfTups:
        constrStUpTG[g, t] = m.add_constr(stUpTG[g, t] == fixedVars[params.map['stUpTG'][g,t]],\
                                                            name = f'constrStUpTG_{g}_{t}')
        constrStDwTG[g, t] = m.add_constr(stDwTG[g, t] == fixedVars[params.map['stDwTG'][g,t]],\
                                                            name = f'constrStDwTG_{g}_{t}')
        constrDispStat[g, t] = m.add_constr(dispStat[g, t] == fixedVars[params.map['DpTG'][g,t]],\
                                                            name = f'constrDispStat_{g}_{t}')

    return (stUpTG, stDwTG, dispStat)

def thermalCont(m, params, thermals, network, listOfUnits, constrTgDisp, fixedVars,\
                periodsWithConstrs, stUpTG, stDwTG, dispStat, slacks: bool = True):
    '''Add continuous variables and their constraints for the thermal model
    m:                  optimization model
    params:             an instance of OptOptions (optoptions.py) that contains the
                            parameters for the problem and the algorithm
    thermals:           an instance of Thermals (network.py) with all thermal data
    listOfUnits:        a list with the indices of units, as in range(len(thermals.id)), for
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
    '''

    for g in listOfUnits:
        for t in range(- len(thermals.startUpTraj[g]), 0, 1):
            stUpTG[g, t] = 0

    tg = {(g, t): m.add_var(var_type = CONTINUOUS, obj =thermals.genCost[g],\
                                        name = f'tg_{g}_{t}') for t in periodsWithConstrs\
                                            for g in listOfUnits}

    extraPeriodsWithVars = {t for t in range(params.T) if t < min(periodsWithConstrs)}

    tg.update({(g, t): m.add_var(var_type = CONTINUOUS, obj = 0,\
                                        name = f'tg_{g}_{t}') for t in extraPeriodsWithVars\
                                            for g in listOfUnits})

    #### For times after the current subhorizon, use zeros
    listOfTimesAfterSubhorizon = set(range(params.T)) - set(periodsWithConstrs)-extraPeriodsWithVars

    tg.update({(g, t): 0 for t in listOfTimesAfterSubhorizon for g in listOfUnits})

    tgDisp = {}
    for g in listOfUnits:
        tgDisp.update({(g, t): m.add_var(var_type = CONTINUOUS, name = f"tgDisp_{g}_{t}")\
                                            for t in periodsWithConstrs | extraPeriodsWithVars})

        tgDisp.update({(g, t): 0 for t in listOfTimesAfterSubhorizon})


    #### Start-up trajectory
    tgUpTj = {(g, t): 0 for t in range(params.T) for g in listOfUnits}

    #### Shut-down trajectory
    tgDownTj = {(g, t): 0 for t in range(params.T) for g in listOfUnits}

    #### Start-up trajectory
    for g in listOfUnits:
        if len(thermals.startUpTraj[g]) > 0:
            steps = len(thermals.startUpTraj[g])

            if thermals.inStartUpTraj[g]:

                for t in set(range(0, min(len(thermals.startUpTraj[g]) \
                                - thermals.nHoursInPreState[g], params.T), 1)) & periodsWithConstrs:
                    i = thermals.nHoursInPreState[g] + t
                    tgUpTj[g, t] = thermals.startUpTraj[g][i]

                for t in set(range(len(thermals.startUpTraj[g])\
                                - thermals.nHoursInPreState[g], params.T, 1)) & periodsWithConstrs:
                    tgUpTj[g, t] = xsum(stUpTG[g, i]*thermals.startUpTraj[g][t - steps - i]\
                                                    for i in range(max(t - steps + 1, 0), t + 1, 1))
            else:
                for t in periodsWithConstrs:
                    tgUpTj[g, t] = xsum(stUpTG[g, i]*thermals.startUpTraj[g][t - steps - i]\
                                                    for i in range(max(t - steps + 1, 0), t + 1, 1))

    #### Shut-down trajectory
    #### Note that it is implicitly assumed that the power output of the
    #### thermal unit is zero when stDwTG = 1
    for g in [g for g in listOfUnits if len(thermals.shutDownTraj[g]) > 0]:
        steps = len(thermals.shutDownTraj[g])
        for t in periodsWithConstrs:
            tgDownTj[g, t] = xsum(stDwTG[g, t - i]*thermals.shutDownTraj[g][i]\
                                                for i in [j for j in range(steps) if (t - j) >= 0])


    #### lower and upper operating limits of thermal units
    for g in listOfUnits:
        for t in periodsWithConstrs:
            m.add_constr(tgDisp[g, t] - (thermals.maxP[g] - thermals.minP[g])*dispStat[g, t] <= 0,\
                                                                            name=f'maxP_{g}_{t}')

    #### total generation
    for g in listOfUnits:
        for t in periodsWithConstrs:
            m.add_constr(tg[g, t] - tgDisp[g, t] - thermals.minP[g]*dispStat[g, t]\
                                    - tgUpTj[g, t] - tgDownTj[g, t] == 0, name = f'gen_{g}_{t}')

    # new constraints
    if (0 in periodsWithConstrs):
        for g in listOfUnits:
            if (thermals.state0[g]==0) or (thermals.inStartUpTraj[g] or thermals.inShutDownTraj[g]):
                m.add_constr(tgDisp[g, 0] <= 0, name = f'rampUp_{g}_{0}')
            else:
                m.add_constr(tgDisp[g, 0] - (thermals.tg0[g] - thermals.minP[g])\
                                    <= thermals.rampUp[g], name = f'rampUp_{g}_{0}')
                m.add_constr(- tgDisp[g, 0] + (thermals.tg0[g] - thermals.minP[g])\
                                    <= thermals.rampDown[g], name = f'rampDown_{g}_{0}')

    #### only add the ramp constraints if they are really necessary
    for g in [g for g in listOfUnits if (thermals.rampUp[g] < (thermals.maxP[g]-thermals.minP[g]))]:
        for t in set(range(1, params.T, 1)) & periodsWithConstrs:
            m.add_constr(tgDisp[g, t] - tgDisp[g, t - 1] <= thermals.rampUp[g],\
                                                            name = f'rampUp_{g}_{t}')
    for g in [g for g in listOfUnits if (thermals.rampDown[g]<(thermals.maxP[g]-thermals.minP[g]))]:
        for t in set(range(1, params.T, 1)) & periodsWithConstrs:
            m.add_constr(- tgDisp[g, t] + tgDisp[g, t - 1] <= thermals.rampDown[g],\
                                                                name = f'rampDown_{g}_{t}')

    #### start and shut down at minimum power
    for g in listOfUnits:
        for t in [t2 for t2 in periodsWithConstrs if t2 > 0]:
            m.add_constr(tgDisp[g, t] <=\
                            (thermals.maxP[g] - thermals.minP[g])*(dispStat[g, t]\
                                - stUpTG[g, t - len(thermals.startUpTraj[g])]),\
                                                    name = f'minGenAtFirstPeriod_startUP_{g}_{t}')
            m.add_constr(tgDisp[g, t - 1] <=\
                            (thermals.maxP[g] - thermals.minP[g])*(dispStat[g, t-1] - stDwTG[g,t]),\
                                                    name = f'minGenAtFirstPeriod_shutDown_{g}_{t}')
    if (0 in periodsWithConstrs):
        for g in listOfUnits:
            if (thermals.state0[g]==1) and isinstance(stDwTG[g, 0], entities.Var)\
                                                and (thermals.maxP[g] - thermals.minP[g]) > 0:
                m.add_constr(thermals.tg0[g] - thermals.maxP[g] <=\
                                -(thermals.maxP[g] - thermals.minP[g])*stDwTG[g, 0],\
                                                name = f'minGenAtFirstPeriod_shutDown_{g}_{0}')

    if set(listOfUnits) == set(range(len(thermals.id))):

        tl_sMin = []
        constrIndex = 0
        for constr in thermals.minGen:
            if len(constr[0]) > 1:
                for t in constr[1]:
                    tl_sMin.append((constr[0][0], constrIndex, t))
            constrIndex += 1

        tl_sMax = []
        constrIndex = 0
        for constr in thermals.maxGen:
            if len(constr[0]) > 1 and constr[2] > 0:
                for t in constr[1]:
                    tl_sMax.append((constr[0][0], constrIndex, t))
            constrIndex += 1

        if slacks:
            sMinGen = {k: m.add_var(var_type = CONTINUOUS, obj = 10*params.deficitCost,\
                                name = f'slackMinThermalGen_{k[0]}_{k[1]}_{k[2]}') for k in tl_sMin}
            sMaxGen = {k: m.add_var(var_type = CONTINUOUS, obj = 10*params.deficitCost,\
                                name = f'slackMaxThermalGen_{k[0]}_{k[1]}_{k[2]}') for k in tl_sMax}
        else:
            sMinGen = {k: 0 for k in tl_sMin}
            sMaxGen = {k: 0 for k in tl_sMax}

        constrIndex = 0
        for constr in thermals.minGen:
            if len(constr[0]) == 1:
                for t in set(constr[1]) & periodsWithConstrs:
                    tg[constr[0][0], t].lb = constr[2]
                    tgDisp[constr[0][0], t].lb = max(constr[2] - thermals.minP[constr[0][0]], 0)
            else:
                for t in set(constr[1]) & periodsWithConstrs:
                    m.add_constr(xsum(tg[g, t] for g in constr[0])\
                                            + sMinGen[constr[0][0], constrIndex, t] >= constr[2],\
                            name = f'minThermalGenCont_{thermals.plantsIDDESSEM[constr[0][0]]}_'+\
                                                                            f'{constrIndex}_{t}')
            constrIndex += 1

        constrIndex = 0
        for constr in thermals.maxGen:
            if len(constr[0]) == 1:
                for t in set(constr[1]) & periodsWithConstrs:
                    tg[constr[0][0], t].ub = constr[2]
                    tgDisp[constr[0][0], t].ub = max(constr[2] - thermals.minP[constr[0][0]], 0)
            else:
                if constr[2] == 0:
                    for g in constr[0]:
                        for t in set(constr[1]) & periodsWithConstrs:
                            tg[g, t].ub = 0
                            tgDisp[g, t].ub = 0
                else:
                    for t in set(constr[1]) & periodsWithConstrs:
                        m.add_constr(xsum(tg[g,t] for g in constr[0])\
                                            - sMaxGen[constr[0][0], constrIndex, t]\
                                                <= constr[2],\
                            name =f'maxThermalGenCont_{thermals.plantsIDDESSEM[constr[0][0]]}_'+\
                                                                            f'{constrIndex}_{t}')
            constrIndex += 1

        constrIndex = 0
        for eq in thermals.equalityConstrs:
            for t in periodsWithConstrs:
                m.add_constr(xsum(tg[g, t] for g in eq[0]) == eq[2], name =\
                                        f'eqThermalGenCont_{thermals.plantsIDDESSEM[eq[0][0]]}_'+\
                                            f'{constrIndex}_{t}')
        constrIndex += 1

    previousPeriod = {t for t in range(params.T) if t < min(periodsWithConstrs)}

    for g, t in [(g, t) for g in listOfUnits for t in previousPeriod]:
        constrTgDisp[g,t] = m.add_constr(tgDisp[g,t] == fixedVars[params.map['DpGenTG'][g,t]],\
                                                        name = f'constrTgDisp_{g}_{t}')

    return (tg, tgDisp)
