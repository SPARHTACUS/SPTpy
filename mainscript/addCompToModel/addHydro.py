# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""

from mip import LinExpr, xsum, CONTINUOUS

def boundsOnGenerationOfGroupsOfPlants(params, hydros, listOfHydros, hgEachBus,\
                                            m, periodsWithConstrs):
    '''Add bounds on the generation of groups of hydro plants'''

    for constr in [c for c in hydros.maxGen if set(c[0]) & set(listOfHydros) == set(c[0]) and\
                                                    len(set(c[2]) & periodsWithConstrs) > 0]:
        # then all hydros in this constraint are in listOfHydros and there are periods of this
        # constraint that are in periodsWithConstrs
        for t in set(constr[2]) & periodsWithConstrs:
            m.add_constr(xsum(xsum(hgEachBus[constr[0][i],\
                                hydros.busesOfEachGroup[constr[0][i]]\
                                    [hydros.groupsOfUnits[constr[0][i]].index(group)][0], t]\
                for group in constr[1][i])\
                    for i in range(len(constr[0]))) >= constr[3],\
                                                name = f'maxGen_{hydros.name[constr[0][0]]}_{t}')

    for constr in [c for c in hydros.minGen if set(c[0]) & set(listOfHydros) == set(c[0]) and\
                                                    len(set(c[2]) & periodsWithConstrs) > 0]:
        for t in set(constr[2]) & periodsWithConstrs:
            m.add_constr(xsum(xsum(hgEachBus[constr[0][i],\
                                hydros.busesOfEachGroup[constr[0][i]]\
                                    [hydros.groupsOfUnits[constr[0][i]].index(group)][0], t]\
                for group in constr[1][i])\
                    for i in range(len(constr[0]))) <= constr[3],\
                                                name = f'minGen_{hydros.name[constr[0][0]]}_{t}')

    for constr in [c for c in hydros.equalityConstrs\
                                            if set(c[0]) & set(listOfHydros) == set(c[0]) and\
                                                    len(set(c[2]) & periodsWithConstrs) > 0]:
        for t in set(constr[2]) & periodsWithConstrs:
            m.add_constr(xsum(xsum(hgEachBus[constr[0][i],\
                                hydros.busesOfEachGroup[constr[0][i]]\
                                    [hydros.groupsOfUnits[constr[0][i]].index(group)][0], t]\
                for group in constr[1][i])\
                    for i in range(len(constr[0]))) <= constr[3],\
                                                name = f'eqGen_{hydros.name[constr[0][0]]}_{t}')

    return()


def addHydroVars(params, hydros, m, periodsWithConstrs):
    '''Add the hydro vars
    First, add the state variables v, s, q, qBypass, and qPump, then
    add all other variables that do not couple the problem in time.
    For periods before the current subhorizon, i.e.,
    the subhorizon whose periods are in periodsWithConstrs, we create additional variables that
    will later be forced to take the values already decided for them in previous subhorizons
    through equality constraints'''

    #### Create the variables for the current subhorizon
    tl_h = [(h, t) for t in periodsWithConstrs for h in range(len(hydros.id))]

    # Reservoir volume in hm3
    v = {(h, t): m.add_var(var_type = CONTINUOUS, lb = hydros.minVol[h], ub = hydros.maxVol[h],\
                                                    name = f"v_{h}_{t}") for (h, t) in tl_h}

    # Spillage in m3/s
    s = {(h, t): m.add_var(var_type = CONTINUOUS, name = f"spil_{h}_{t}") for (h, t) in tl_h}

    for h in range(len(hydros.id)):
        for t in periodsWithConstrs:
            s[h, t].ub = hydros.maxSpil[h]

    # Discharge per plant in m3/s
    q = {(h, t): m.add_var(var_type = CONTINUOUS, name = f"q_{h}_{t}") for (h, t) in tl_h}

    # Water bypass in m3/s
    qBypass = {(h, t): m.add_var(var_type = CONTINUOUS,\
                                        name = f"qBypass_{h}_{t}") for h in range(len(hydros.id))\
                                        if len(hydros.downRiverTransferPlantID[h]) > 0\
                                            for t in periodsWithConstrs}

    qBypass.update({(h, t): 0 for h in range(len(hydros.id))\
                        if len(hydros.downRiverTransferPlantID[h])==0 for t in periodsWithConstrs})

    for h in [h for h in range(len(hydros.id)) if len(hydros.downRiverTransferPlantID[h]) > 0]:
        for t in periodsWithConstrs:
            qBypass[h, t].ub = hydros.maxTransfer[h]

    # pumped water in m3/s
    qPump = {(h, t): m.add_var(var_type = CONTINUOUS, name = f"qPump_{h}_{t}",\
                                ub = hydros.plantMaxTurbDisc[h]) for h in range(len(hydros.id)) if\
                                    hydros.turbOrPump[h] == 'Pump' for t in periodsWithConstrs}

    qPumpExp = {(h, t): LinExpr(0) for h in range(len(hydros.id)) for t in range(-168, params.T, 1)}

    #### Create the state variables for periods before this subhorizon
    listOfTimesBeforeSubh = [t for t in range(params.T) if t < min(periodsWithConstrs)]

    # Reservoir volume in hm3
    tl_v = [(h, t) for t in listOfTimesBeforeSubh for h in range(len(hydros.id))]
    v.update({k: m.add_var(var_type = CONTINUOUS, name = f"v_{k[0]}_{k[1]}") for k in tl_v})

    complement_v = {(h, t) for t in listOfTimesBeforeSubh for h in range(len(hydros.id))}- set(tl_v)
    v.update({k: None for k in complement_v})

    # Spillage in m3/s and turbine discharge in m3/s
    tl_sq = [(h, t) for h in range(len(hydros.id)) for t in listOfTimesBeforeSubh]

    complement_sq = {(h, t) for t in listOfTimesBeforeSubh for h in range(len(hydros.id))}\
                                                                                        - set(tl_sq)

    s.update({(h, t): m.add_var(var_type=CONTINUOUS, name = f"spil_{h}_{t}") for (h,t) in tl_sq})
    s.update({k: None for k in complement_sq})

    q.update({(h, t): m.add_var(var_type = CONTINUOUS, name = f"q_{h}_{t}") for (h,t) in tl_sq})
    q.update({k: None for k in complement_sq})

    # Water bypass in m3/s
    tl_qt = [(h, t) for h in [h for h in range(len(hydros.id))\
                    if len(hydros.downRiverTransferPlantID[h]) > 0] for t in listOfTimesBeforeSubh]

    complement_qt = {(h, t) for t in listOfTimesBeforeSubh\
                                for h in [h for h in range(len(hydros.id))\
                                    if len(hydros.downRiverTransferPlantID[h]) > 0]} - set(tl_qt)

    qBypass.update({(h, t): m.add_var(var_type = CONTINUOUS,\
                                        name = f"qBypass_{h}_{t}") for (h, t) in tl_qt})

    qBypass.update({(h, t): 0 for h in range(len(hydros.id))\
                                        if len(hydros.downRiverTransferPlantID[h])==0\
                                                                for t in listOfTimesBeforeSubh})

    qBypass.update({k: None for k in complement_qt})

    # pumped water in m3/s
    tl_qp = [(h, t) for h in range(len(hydros.id)) if hydros.turbOrPump[h] == 'Pump'\
                                                                for t in listOfTimesBeforeSubh]

    complement_qp = {(h,t) for t in listOfTimesBeforeSubh for h in range(len(hydros.id))}-set(tl_qp)

    qPump.update({(h, t): m.add_var(var_type = CONTINUOUS,\
                                                name = f"qPump_{h}_{t}") for (h, t) in tl_qp})

    qPump.update({k: None for k in complement_qp})

    #### Use a zero for everything in the future
    listOfTimesAfterSubh = {t for t in range(params.T) if t > max(periodsWithConstrs)}
    tl_prevTs = [(h, t) for t in listOfTimesAfterSubh for h in range(len(hydros.id))]
    v.update({k: 0 for k in tl_prevTs})                     # Reservoir volume in hm3
    s.update({k: 0 for k in tl_prevTs})                     # Spillage in m3/s
    q.update({(h, t): 0 for (h, t) in tl_prevTs})           # Discharge in m3/s
    qBypass.update({(h, t): 0 for (h, t) in tl_prevTs})     # Water bypass in m3/s
    qPump.update({(h, t): 0 for h in range(len(hydros.id)) if hydros.turbOrPump[h] == 'Pump'\
                                                            for t in listOfTimesAfterSubh}) # Pumps
    ############################################################################

    # Electric power output (p.u.): note that the variables in this dictionary are indexed by three
    # indices: the plant's id, the network bus, and the time period. The network bus is necessary
    # because some plants have generating units connected to different buses.
    hgEachBus = {(h, hydros.plantBuses[h][b], t): m.add_var(var_type = CONTINUOUS,\
                                    ub = hydros.plantBusesCap[h][b], name=f"hgEachBus_{h}_{b}_{t}")\
                                                for h in range(len(hydros.id))\
                                                    for b in range(len(hydros.plantBuses[h]))\
                                                        for t in periodsWithConstrs}

    # Expected future cost of water storage (cost-to-go function) in $
    alpha = m.add_var(var_type = CONTINUOUS, name = f"alpha_{max(periodsWithConstrs)}",\
                                    obj = 1 if max(periodsWithConstrs) == params.T - 1 else 1)

    return(v, s, q, qBypass, qPump, qPumpExp, hgEachBus, alpha)

def addHydropowerProductionFunction(m, v, q, s, hgEachBus,\
                                    hydros, subListOfHydrosWithTurbines, periodsWithConstrs):
    '''Add the hydropower function'''

    for h in subListOfHydrosWithTurbines:
        for i in range(len(hydros.A0[h])):
            for t in periodsWithConstrs:
                m.add_constr((xsum(hgEachBus[h, bus, t] for bus in hydros.plantBuses[h])\
                                    - hydros.A0[h][i]*q[h, t] - hydros.A1[h][i]*v[h, t]\
                                        - hydros.A2[h][i]*s[h, t] - hydros.A3[h][i] <= 0),\
                                            name = f"HPF_{h}_{i}_{t}")

    return()

def addMassBalance(m, v, q, qBypass, qPumpExp, s,\
                    params, hydros, listOfHydros, periodsWithConstrs,\
                    slackForVolumes = False):
    '''Add mass balance constraints to the reservoirs'''

    # Outflow of the hydro reservoirs in hm3
    outflow = {(h, t): m.add_var(var_type = CONTINUOUS, name = f"outflow_{h}_{t}", lb = 0)\
                        for (h, t) in [(h,t) for t in periodsWithConstrs for h in listOfHydros]}
    # Inflow to the hydro reservoirs in hm3
    inflow = {}

    # Total inflow in hm3. Water coming from other reservoirs as well as from incremental inflow
    for h in listOfHydros:
        for t in periodsWithConstrs:
            lhs = params.massBal*( - xsum(params.c_h*\
                            (s[UpR, t - hydros.travelTime[hydros.name[UpR], hydros.name[h], t]]\
                            + q[UpR, t - hydros.travelTime[hydros.name[UpR], hydros.name[h], t]])\
                                        for UpR in hydros.upRiverPlantIDs[h])
                            - xsum(params.c_h\
                                        * qBypass[UpR, t - hydros.transferTravelTime[UpR, t]]\
                                            for UpR in hydros.upRiverTransferPlantID[h])\
                            -xsum(qPumpExp[h, t - hydros.pumpageWaterTTime[hp, t]]\
                                for hp in hydros.downRiverPumps[h]))
            if len(lhs.expr) > 0:
                inflow.update({(h, t): m.add_var(var_type = CONTINUOUS,\
                                                                name = f"inflow_{h}_{t}")})
                lhs.add_term(term = inflow[h, t], coeff = params.massBal)
                m.add_constr((lhs == params.massBal*params.c_h*hydros.inflows[h, t]),\
                                                                name = f"totalInflow_{h}_{t}")
            else:
                # Recall that lhs is in the left-hand side of the equation, thus the minus sign
                inflow[h, t] = params.c_h*hydros.inflows[h, t] - lhs.const/params.massBal

    # Total outflow in hm3
    for h in listOfHydros:
        for t in periodsWithConstrs:
            m.add_constr((params.massBal*(outflow[h, t] - qPumpExp[h, t] - params.c_h*(\
                                                    s[h, t] + qBypass[h, t] + q[h, t])) == 0),\
                                                        name = f"totalOutflow_{h}_{t}")

    if slackForVolumes:
        slackOutflowV = {(h, t): m.add_var(var_type = CONTINUOUS,\
                                        name = f"slackOutflowV_{h}_{t}",\
                                            obj = params.costOfVolViolation, lb = 0.0, ub = 10)\
                                                for h in listOfHydros for t in periodsWithConstrs}
        slackInflowV = {(h, t): m.add_var(var_type = CONTINUOUS,\
                                        name = f"slackInflowV_{h}_{t}",\
                                            obj = params.costOfVolViolation, lb = 0.0, ub = 10)\
                                                for h in listOfHydros for t in periodsWithConstrs}
    else:
        slackOutflowV = {(h, t): 0 for h in listOfHydros for t in periodsWithConstrs}
        slackInflowV = {(h, t): 0 for h in listOfHydros for t in periodsWithConstrs}

    #### Reservoir volume balance
    for h in listOfHydros:
        if hydros.maxVol[h] > hydros.minVol[h]:
            for t in periodsWithConstrs:
                m.add_constr(((v[h, t] - v[h, t - 1]) + (outflow[h, t] - inflow[h, t])\
                                                    == - slackOutflowV[h, t] + slackInflowV[h, t]),\
                                                                name = f"MassBalance_{h}_{t}")
        else:
            for t in periodsWithConstrs:
                m.add_constr((outflow[h, t] - inflow[h, t]\
                                                    == - slackOutflowV[h, t] + slackInflowV[h, t]),\
                                                                name = f"MassBalance_{h}_{t}")

    return (inflow, outflow)

def addHydro(m, params, hydros,\
            constrV, constrQ, constrQbyPass, constrQPump, constrS, fixedVars,\
            periodsWithConstrs, slackForVolumes: bool = False):
    '''Add hydro variables and constraints to the model
    m:                  optimization model
    params:             an instance of OptOptions (optoptions.py) that contains the
                            parameters for the problem and the algorithm
    hydros:             an instance of Hydros (network.py) with all hydro data
    constrX:            dictionary containing the equality constraints that enforce decisions
                        taken in previous subhorizons
                        (X = V for volumes, Q for turbine discharge, QbyPass for water bypass,
                        QPump for pumped water, S for spillage)
    fixedVars:          numpy array holding all time-coupling decisions
    periodsWithConstrs: a set with the periods in the planning horizon for which the hydro model
                        constraints and variables are to be included. For instance, if model m
                        is the optimization model of subhorizon 3, then periodsWithConstrs
                        has all time periods of this subhorizon
    slackForVolumes:    True if slack variables are to be used in the water-balance constraints of
                        the hydro plants, and False otherwise
    '''

    assert len(periodsWithConstrs) > 0, "periodsWithConstrs is empty"

    subListOfHydrosWithTurbines = [h for h in range(len(hydros.id)) if hydros.plantMaxPower[h] > 0]

    v, s, q, qBypass, qPump, qPumpExp, hgEachBus, alpha = addHydroVars(params, hydros, m,\
                                                                        periodsWithConstrs)

    #### For the pumps, the water goes from the
    #### downriver reservoir to the upriver reservoir
    for h in [h for h in range(len(hydros.id)) if hydros.turbOrPump[h] == 'Pump']:
        for t in periodsWithConstrs:
            # Reservoir form which the water is coming
            qPumpExp[hydros.dnrOfPumpsID[h], t].add_term(term = qPump[h, t], coeff = params.c_h)

            # Reservoir where the water is going to
            qPumpExp[hydros.uprOfPumpsID[h], t + hydros.pumpageWaterTTime[h, t]].add_term(\
                                                        term = qPump[h, t], coeff = -params.c_h)

    #### Previous states
    for h in range(len(hydros.id)):
        v[h, -1] = hydros.V0[h] # reservoir volume in hm3 immediatelly before the beginning of the
                                # planning horizon

        for UpR in hydros.upRiverPlantIDs[h]:
            for t in range(-hydros.travelTime[hydros.name[UpR],hydros.name[h],0], 0, 1):
                s[UpR, t] = hydros.spil0[UpR, t]
                q[UpR, t] = hydros.q0[UpR]

        for UpR in hydros.upRiverTransferPlantID[h]:
            for t in range(-hydros.transferTravelTime[UpR,0], 0, 1):
                qBypass[UpR, t] = 0

        if hydros.turbOrPump[h] == 'Pump':
            for t in range(-hydros.pumpageWaterTTime[h, 0], 0, 1):
                qPump[h, t] = 0
    ############################################################################

    #### Maximum turbine discharge for the aggregation of units
    # No binary variables
    for h in [h for h in range(len(hydros.id)) if hydros.plantMaxPower[h] > 0]:
        for t in periodsWithConstrs:
            q[h, t].ub = hydros.plantMaxTurbDisc[h]

    for h in [h for h in range(len(hydros.id)) if hydros.plantMaxPower[h] == 0]:
        for t in periodsWithConstrs:
            q[h, t].ub = 0.0

    inflow, outflow = addMassBalance(m, v, q, qBypass, qPumpExp, s,\
                                    params, hydros, range(len(hydros.id)),\
                                    periodsWithConstrs, slackForVolumes)

    boundsOnGenerationOfGroupsOfPlants(params, hydros, range(len(hydros.id)), hgEachBus,\
                                                                            m, periodsWithConstrs)

    # Add piecewise-linear model of the hydropower function
    addHydropowerProductionFunction(m, v, q, s, hgEachBus,\
                                    hydros, subListOfHydrosWithTurbines, periodsWithConstrs)
    ############################################################################

    #### Pumps
    for h in [h for h in range(len(hydros.id)) if hydros.turbOrPump[h] == 'Pump']:
        #### For the periods with aggregated power with and without
        #### binaries
        for t in periodsWithConstrs:
            for b in range(len(hydros.plantBuses[h])):
                hgEachBus[h, hydros.plantBuses[h][b], t].lb = 0
                hgEachBus[h, hydros.plantBuses[h][b], t].ub = 1e3

                m.add_constr(hgEachBus[h, hydros.plantBuses[h][b], t] -\
                                                            qPump[h, t]*hydros.convMWm3s[h] == 0,\
                                                            name = f"convPump_{h}_{b}_{t}")

    #### Cost-to-go function
    tctf = max(periodsWithConstrs)
    for c in range(len(hydros.CTFrhs)):
        m.add_constr(-1*(alpha + xsum(v[h, tctf]*hydros.CTF[c][h] for h in range(len(hydros.id))))\
                                                    <= -hydros.CTFrhs[c], name = f"CostToGo_{c}")

    # set of time periods immediatelly before this subhorizon, and the indices of hydro plants.
    # this set is used for setting the decisions taken in previous subhorizons.
    originalListOfTups = {(h, t) for h in range(len(hydros.id))\
                            for t in [t for t in range(params.T) if t < min(periodsWithConstrs)]}

    for h, t in originalListOfTups:
        constrV[h, t] = m.add_constr(v[h, t] == fixedVars[params.map['v'][h, t]],\
                                                                    name = f'constrV_{h}_{t}')

    for h, t in originalListOfTups:
        constrQ[h, t] = m.add_constr(q[h, t] == fixedVars[params.map['q'][h, t]],\
                                                                    name = f'constrQ_{h}_{t}')

    for h, t in originalListOfTups:
        constrS[h, t] = m.add_constr(s[h, t] == fixedVars[params.map['s'][h, t]],\
                                                                    name = f'constrS_{h}_{t}')

    for h, t in originalListOfTups:
        if hydros.turbOrPump[h] == 'Pump':
            constrQPump[h, t] = m.add_constr(qPump[h, t]== fixedVars[params.map['pump'][h, t]],\
                                                                    name = f'constrQPump_{h}_{t}')

    for h, t in originalListOfTups:
        if len(hydros.downRiverTransferPlantID[h]) > 0:
            constrQbyPass[h, t]=m.add_constr(qBypass[h, t]==fixedVars[params.map['QbyPass'][h, t]],\
                                                                    name = f'constrQbyPass_{h}_{t}')

    return (hgEachBus, v, q, qBypass, qPump, s, inflow, outflow, alpha)
