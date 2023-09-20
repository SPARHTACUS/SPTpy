# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""

from optoptions import Coupling

def boundsOnGenerationOfGroupsOfPlants(params, hydros, listOfHydros, hg,
                                            m, periods_with_constrs):
    """Add bounds on the generation of groups of hydro plants"""

    for constr in [c for c in hydros.maxGen if set(c[0]) & set(listOfHydros) == set(c[0])]:
        # then all hydros in this constraint are in listOfHydros and there are periods of this
        # constraint that are in periods_with_constrs
        for t in set(constr[2]) & periods_with_constrs:
            m.add_constr(
                        m.xsum(
                            m.xsum(
                                m.xsum(
                                        hg[constr[0][i], u, t]
                                        for u in hydros.UNIT_ID[constr[0][i]]
                                    if hydros.UNIT_GROUP[constr[0][i]][u] == group)
                                for group in constr[1][i])
                            for i in range(len(constr[0])))
                                >= constr[3],
                                                name = f'maxGen_{hydros.NAME[constr[0][0]]}_{t}')

    for constr in [c for c in hydros.minGen if set(c[0]) & set(listOfHydros) == set(c[0])]:
        for t in set(constr[2]) & periods_with_constrs:
            m.add_constr(
                        m.xsum(
                            m.xsum(
                                m.xsum(
                                        hg[constr[0][i], u, t]
                                        for u in hydros.UNIT_ID[constr[0][i]]
                                    if hydros.UNIT_GROUP[constr[0][i]][u] == group)
                                for group in constr[1][i])
                            for i in range(len(constr[0])))
                                    <= constr[3],
                                                name = f'minGen_{hydros.NAME[constr[0][0]]}_{t}')

    for constr in [c for c in hydros.equalityConstrs if set(c[0]) & set(listOfHydros) == set(c[0])]:
        for t in set(constr[2]) & periods_with_constrs:
            m.add_constr(
                        m.xsum(
                            m.xsum(
                                m.xsum(
                                        hg[constr[0][i], u, t]
                                        for u in hydros.UNIT_ID[constr[0][i]]
                                    if hydros.UNIT_GROUP[constr[0][i]][u] == group)
                                for group in constr[1][i])
                            for i in range(len(constr[0])))
                                    <= constr[3],
                                                name = f'eqGen_{hydros.NAME[constr[0][0]]}_{t}')

def add_hydro_vars(params, hydros, m, periods_with_constrs):
    """Add the hydro vars
    First, add the state variables v, s, q, qBypass, and qPump, then
    add all other variables that do not couple the problem in time.
    For periods before the current subhorizon, i.e.,
    the subhorizon whose periods are in periods_with_constrs, we create additional variables that
    will later be forced to take the values already decided for them in previous subhorizons
    through equality constraints"""

    #### Create the variables for the current subhorizon
    tl_h = [(h, t) for t in periods_with_constrs for h in hydros.ID]

    # Reservoir volume in hm3
    v = {(h, t): m.add_var(var_type = 'C', lb = hydros.MIN_VOL[h], ub = hydros.MAX_VOL[h],
                                                    name = f"v_{h}_{t}") for (h, t) in tl_h}

    # Spillage in m3/s
    s = {(h, t): m.add_var(var_type = 'C', name = f"spil_{h}_{t}") for (h, t) in tl_h}

    for h in hydros.ID:
        for t in periods_with_constrs:
            s[h, t].ub = hydros.MAX_SPIL[h]

    # Discharge per plant in m3/s
    q = {(h, t): m.add_var(var_type = 'C',
                           ub = (sum(hydros.UNIT_MAX_TURB_DISCH[h].values())
                                    if sum(hydros.UNIT_MAX_P[h].values()) > 0
                                        else 0),
                           name = f"q_{h}_{t}") for (h, t) in tl_h}

    # Water bypass in m3/s
    q_by_pass = {(h, t): m.add_var(var_type = 'C',
                                        name = f"q_by_pass_{h}_{t}") for h in hydros.ID
                                            if hydros.DOWN_RIVER_BY_PASS[h] is not None
                                                for t in periods_with_constrs}

    q_by_pass.update({(h, t): 0 for h in hydros.ID
                            if hydros.DOWN_RIVER_BY_PASS[h] is None for t in periods_with_constrs})

    for h in [h for h in hydros.ID if hydros.DOWN_RIVER_BY_PASS[h] is not None]:
        for t in periods_with_constrs:
            q_by_pass[h, t].ub = hydros.MAX_BY_PASS[h]

    # pumped water in m3/s
    q_pump = {(h, t): m.add_var(var_type = 'C', name = f"q_pump_{h}_{t}",
                                                ub = sum(hydros.UNIT_MAX_TURB_DISCH[h].values()))
                                                    for h in hydros.ID
                                                        if hydros.TURB_OR_PUMP[h] == 'Pump'
                                                            for t in periods_with_constrs}

    #### Create the state variables for periods before this subhorizon
    times_before = [t for t in range(params.T) if t < min(periods_with_constrs)]

    # Reservoir volume in hm3
    tl_v = [(h, t) for t in times_before for h in hydros.ID]
    v.update({k: m.add_var(var_type = 'C', name = f"v_{k[0]}_{k[1]}") for k in tl_v})

    complement_v = {(h, t) for t in times_before for h in hydros.ID}- set(tl_v)
    v.update({k: None for k in complement_v})

    # Spillage in m3/s and turbine discharge in m3/s
    tl_sq = [(h, t) for h in hydros.ID for t in times_before]

    complement_sq = ({(h, t) for t in times_before for h in hydros.ID} - set(tl_sq))

    s.update({(h, t): m.add_var(var_type='C', name = f"spil_{h}_{t}") for (h, t) in tl_sq})
    s.update({k: None for k in complement_sq})

    q.update({(h, t): m.add_var(var_type = 'C', name = f"q_{h}_{t}") for (h, t) in tl_sq})
    q.update({k: None for k in complement_sq})

    # Water bypass in m3/s
    tl_qt = [(h, t) for h in [h for h in hydros.ID
                                if hydros.DOWN_RIVER_BY_PASS[h] is not None] for t in times_before]

    complement_qt = {(h, t) for t in times_before
                                for h in [h for h in hydros.ID
                                    if hydros.DOWN_RIVER_BY_PASS[h] is not None]} - set(tl_qt)

    q_by_pass.update({(h, t): m.add_var(var_type = 'C',
                                        name = f"q_by_pass_{h}_{t}") for (h, t) in tl_qt})

    q_by_pass.update({(h, t): 0 for h in hydros.ID if hydros.DOWN_RIVER_BY_PASS[h] is None
                                                                for t in times_before})

    q_by_pass.update({k: None for k in complement_qt})

    # pumped water in m3/s
    tl_qp = [(h, t) for h in hydros.ID if hydros.TURB_OR_PUMP[h] == 'Pump' for t in times_before]

    complement_qp = {(h,t) for t in times_before for h in hydros.ID}-set(tl_qp)

    q_pump.update({(h, t): m.add_var(var_type = 'C',
                                                name = f"qPump_{h}_{t}") for (h, t) in tl_qp})

    q_pump.update({k: None for k in complement_qp})

    #### Use a zero for everything in the future
    times_after = {t for t in range(params.T) if t > max(periods_with_constrs)}
    tl_prevTs = [(h, t) for t in times_after for h in hydros.ID]
    v.update({k: 0 for k in tl_prevTs})                     # Reservoir volume in hm3
    s.update({k: 0 for k in tl_prevTs})                     # Spillage in m3/s
    q.update({(h, t): 0 for (h, t) in tl_prevTs})           # Discharge in m3/s
    q_by_pass.update({(h, t): 0 for (h, t) in tl_prevTs})     # Water bypass in m3/s
    q_pump.update({(h, t): 0 for h in hydros.ID if hydros.TURB_OR_PUMP[h] == 'Pump'
                                                            for t in times_after}) # Pumps
    ############################################################################

    # Electric power output (p.u.): note that the variables in this dictionary are indexed by three
    # indices: the plant's id, the network bus, and the time period. The network bus is necessary
    # because some plants have generating units connected to different buses.
    hg = {(h, u, t): m.add_var(var_type = 'C', ub = hydros.UNIT_MAX_P[h][u],
                                                    name=f"hg_{h}_{u}_{t}")
                                                        for h in hydros.ID
                                                            for u in hydros.UNIT_ID[h]
                                                                for t in periods_with_constrs}

    # Expected future cost of water storage (cost-to-go function) in $
    alpha = m.add_var(var_type = 'C', name = f"alpha_{max(periods_with_constrs)}",
                                    obj = 1 if max(periods_with_constrs) == params.T - 1 else 1)

    return(v, s, q, q_by_pass, q_pump, hg, alpha)

def add_hydropower_function(m, v, q, s, hg, hydros, periods_with_constrs):
    """
        Add the convex piecewise-linear hydropower function model
    """

    for h in [h for h in hydros.ID if sum(hydros.UNIT_MAX_P[h].values()) > 0]:
        for i in range(len(hydros.A0[h])):
            for t in periods_with_constrs:
                m.add_constr((m.xsum(hg[h, u, t] for u in hydros.UNIT_ID[h])
                                    - hydros.A0[h][i]*q[h, t] - hydros.A1[h][i]*v[h, t]
                                        - hydros.A2[h][i]*s[h, t] - hydros.A3[h][i] <= 0),
                                            name = f"HPF_{h}_{i}_{t}")

def add_mass_balance(m, v, q, q_by_pass, q_pump, s,
                    params, hydros, listOfHydros, periods_with_constrs):
    """
        Add mass balance constraints to the reservoirs
    """

    # a scaling factor and a conversion factor from flow in m3/s to volume in hm3
    MASS_BALANCE_SCALING, C_H = (1e2, params.DISCRETIZATION*(3600*1e-6))

    # Outflow of the hydro reservoirs in hm3
    outflow = {(h, t): m.add_var(var_type = 'C',
                                    name = f"outflow_{h}_{t}", lb = 0)
                                        for (h, t) in [(h,t) for t in periods_with_constrs
                                            for h in listOfHydros]}
    # Inflow to the hydro reservoirs in hm3
    inflow = {}

    # Total inflow in hm3. Water coming from other reservoirs as well as from incremental inflow
    for h in listOfHydros:
        upriver_reservoirs = [h_up for h_up in listOfHydros
                                            if hydros.DOWN_RIVER_RESERVOIR[h_up] == hydros.NAME[h]]
        upriver_reservoirs_by_pass = [h_up for h_up in listOfHydros
                                            if hydros.DOWN_RIVER_BY_PASS[h_up] == hydros.NAME[h]]
        downriver_pumps = [h_dn for h_dn in listOfHydros
                                            if hydros.DOWN_RIVER_PUMP[h_dn] == hydros.NAME[h]]
        for t in periods_with_constrs:
            lhs = MASS_BALANCE_SCALING*(
                        - C_H*m.xsum(
                        (s[UpR, t - hydros.WATER_TRAVEL_TIME[hydros.NAME[UpR], hydros.NAME[h], t]]
                        + q[UpR, t - hydros.WATER_TRAVEL_TIME[hydros.NAME[UpR], hydros.NAME[h], t]])
                                    for UpR in upriver_reservoirs)
                            - C_H*m.xsum(q_by_pass[UpR, t - hydros.BY_PASS_TRAVEL_TIME[UpR, t]]
                                            for UpR in upriver_reservoirs_by_pass)
                            - C_H*m.xsum(q_pump[hp, t - hydros.PUMP_TRAVEL_TIME[hp, t]]
                                            for hp in downriver_pumps))
            if not(isinstance(lhs, (int, float))):
                inflow.update({(h, t): m.add_var(var_type = 'C', name = f"inflow_{h}_{t}")})
                lhs += MASS_BALANCE_SCALING*inflow[h, t]
                m.add_constr((lhs == MASS_BALANCE_SCALING*C_H*hydros.INFLOWS[h][t]),
                                                            name = f"total_inflow_{h}_{t}")
            else:
                # Recall that lhs is in the left-hand side of the equation, thus the minus sign
                inflow[h, t] = C_H*hydros.INFLOWS[h][t] - lhs/MASS_BALANCE_SCALING

    # Total outflow in hm3
    for h in listOfHydros:
        qp = (q_pump
                    if hydros.DOWN_RIVER_PUMP[h] is not None
                        else {(h, t): 0 for t in periods_with_constrs}
                )

        qbp = (q_by_pass
                    if hydros.DOWN_RIVER_BY_PASS[h] is not None
                        else {(h, t): 0 for t in periods_with_constrs}
                )

        for t in periods_with_constrs:
            m.add_constr((MASS_BALANCE_SCALING*(outflow[h, t] - C_H*(
                                                    s[h, t] + q[h, t] + qbp[h, t])
                                                    + qp[h,t]) == 0),
                                                        name = f"total_outflow_{h}_{t}")

    # slack variables used in the water-balance constraints to compensate for possible
    # infeasibilities due to decisions taken in previous subhorizons
    slack_outflow_V = {(h, t): m.add_var(var_type = 'C',
                                    name = f"slack_outflow_{h}_{t}",
                                        obj = params.COST_OF_VOL_VIOL, lb = 0.0, ub = 10)
                                            for h in listOfHydros for t in periods_with_constrs}
    slack_inflow_V = {(h, t): m.add_var(var_type = 'C',
                                    name = f"slack_inflow_{h}_{t}",
                                        obj = params.COST_OF_VOL_VIOL, lb = 0.0, ub = 10)
                                            for h in listOfHydros for t in periods_with_constrs}

    #### Reservoir volume balance
    for h in listOfHydros:
        for t in periods_with_constrs:
            m.add_constr(((v[h, t] - v[h, t - 1]) + (outflow[h, t] - inflow[h, t])
                                            == - slack_outflow_V[h, t] + slack_inflow_V[h, t]),
                                                            name = f"mass_balance_{h}_{t}")
    return (inflow, outflow)

def addHydro(m, params, hydros,
            constrV, constrQ, constrQbyPass, constrQPump, constrS, fixedVars,
            periods_with_constrs):
    """Add hydro variables and constraints to the model
    m:                  optimization model
    params:             an instance of OptOptions (optoptions.py) that contains the
                            parameters for the problem and the algorithm
    hydros:             an instance of Hydros (network.py) with all hydro data
    constrX:            dictionary containing the equality constraints that enforce decisions
                        taken in previous subhorizons
                        (X = V for volumes, Q for turbine discharge, QbyPass for water bypass,
                        QPump for pumped water, S for spillage)
    fixedVars:          numpy array holding all time-coupling decisions
    periods_with_constrs: a set with the periods in the planning horizon for which the hydro model
                        constraints and variables are to be included. For instance, if model m
                        is the optimization model of subhorizon 3, then periods_with_constrs
                        has all time periods of this subhorizon
    """

    assert len(periods_with_constrs) > 0, "periods_with_constrs is empty"

    (v, s, q, q_by_pass, q_pump, hg, alpha) = add_hydro_vars(params, hydros, m,
                                                                        periods_with_constrs)

    #### Previous states
    for h in hydros.ID:
        v[h, -1] = hydros.V_0[h]# reservoir volume in hm3 immediatelly before the beginning of the
                                # planning horizon

        for UpR in [h_up for h_up in hydros.ID
                                            if hydros.DOWN_RIVER_RESERVOIR[h_up] == hydros.NAME[h]]:
            for t in range(-hydros.WATER_TRAVEL_TIME[hydros.NAME[UpR],hydros.NAME[h],0], 0, 1):
                s[UpR, t] = hydros.SPIL_0[UpR, t]
                q[UpR, t] = hydros.Q_0[UpR]

        for UpR in [h_up for h_up in hydros.ID if hydros.DOWN_RIVER_BY_PASS[h_up] ==hydros.NAME[h]]:
            for t in range(-hydros.BY_PASS_TRAVEL_TIME[UpR,0], 0, 1):
                q_by_pass[UpR, t] = 0

        if hydros.TURB_OR_PUMP[h] == 'Pump':
            for t in range(-hydros.PUMP_TRAVEL_TIME[h, 0], 0, 1):
                q_pump[h, t] = 0
    ############################################################################

    inflow, outflow = add_mass_balance(m, v, q, q_by_pass, q_pump, s,
                                        params, hydros, hydros.ID,
                                            periods_with_constrs)

    # Add piecewise-linear model of the hydropower function
    add_hydropower_function(m, v, q, s, hg, hydros, periods_with_constrs)
    ############################################################################

    #### Pumps
    for h in [h for h in hydros.ID if hydros.TURB_OR_PUMP[h] == 'Pump']:
        for t in periods_with_constrs:
            for u in hydros.UNIT_ID[h]:
                m.set_lb(hg[h, u, t], 0)
                m.set_ub(hg[h, u, t], 1e100)
                m.add_constr(hg[h, u, t] - q_pump[h, t]*hydros.PUMP_CONVERSION_FACTOR[h] == 0,
                                                            name = f"conv_pump_{h}_{u}_{t}")

    #### Cost-to-go function
    tctf = max(periods_with_constrs)
    for c in hydros.CTFrhs.keys():
        m.add_constr(-1*(alpha + m.xsum(v[h, tctf]*hydros.CTF[h][c] for h in hydros.ID))
                                                    <= -hydros.CTFrhs[c], name = f"cost_to_go_{c}")

    # set of time periods immediatelly before this subhorizon, and the indices of hydro plants.
    # this set is used for setting the decisions taken in previous subhorizons.
    coupling_decisions = {(h, t) for h in hydros.ID
                            for t in [t for t in range(params.T) if t < min(periods_with_constrs)]}

    if params.COUPLING == Coupling.CONSTRS:
        for h, t in coupling_decisions:
            constrV[h, t] = m.add_constr(v[h, t] == fixedVars[params.MAP['v'][h, t]],
                                                                    name = f'constrV_{h}_{t}')

        for h, t in coupling_decisions:
            constrQ[h, t] = m.add_constr(q[h, t] == fixedVars[params.MAP['q'][h, t]],
                                                                    name = f'constrQ_{h}_{t}')

        for h, t in coupling_decisions:
            constrS[h, t] = m.add_constr(s[h, t] == fixedVars[params.MAP['s'][h, t]],
                                                                    name = f'constrS_{h}_{t}')

        for h, t in [(h, t) for h, t in coupling_decisions if hydros.TURB_OR_PUMP[h] == 'Pump']:
            constrQPump[h, t] = m.add_constr(q_pump[h, t] == fixedVars[params.MAP['pump'][h, t]],
                                                                    name = f'constrQPump_{h}_{t}')

        for h, t in [(h, t) for h, t in coupling_decisions
                                                    if hydros.DOWN_RIVER_BY_PASS[h] is not None]:
            constrQbyPass[h, t] = m.add_constr(
                                        q_by_pass[h, t] == fixedVars[params.MAP['QbyPass'][h,t]],
                                                                    name = f'constrQbyPass_{h}_{t}')
    else:
        for h, t in coupling_decisions:
            m.set_lb(v[h, t], fixedVars[params.MAP['v'][h, t]])
            m.set_ub(v[h, t], fixedVars[params.MAP['v'][h, t]])
            constrV[h, t] = v[h, t]

        for h, t in coupling_decisions:
            m.set_lb(q[h, t], fixedVars[params.MAP['q'][h, t]])
            m.set_ub(q[h, t], fixedVars[params.MAP['q'][h, t]])
            constrQ[h, t] = q[h, t]

        for h, t in coupling_decisions:
            m.set_lb(s[h, t], fixedVars[params.MAP['s'][h, t]])
            m.set_ub(s[h, t], fixedVars[params.MAP['s'][h, t]])
            constrS[h, t] = s[h, t]

        for h, t in [(h, t) for h, t in coupling_decisions if hydros.TURB_OR_PUMP[h] == 'Pump']:
            m.set_lb(q_pump[h, t], fixedVars[params.MAP['pump'][h, t]])
            m.set_ub(q_pump[h, t], fixedVars[params.MAP['pump'][h, t]])
            constrQPump[h, t] = q_pump[h, t]

        for h, t in [(h, t) for h, t in coupling_decisions
                                                    if hydros.DOWN_RIVER_BY_PASS[h] is not None]:
            m.set_lb(q_by_pass[h, t], fixedVars[params.MAP['QbyPass'][h, t]])
            m.set_ub(q_by_pass[h, t], fixedVars[params.MAP['QbyPass'][h, t]])
            constrQbyPass[h, t] = q_by_pass[h, t]

    return (hg, v, q, q_by_pass, q_pump, s, inflow, outflow, alpha)
