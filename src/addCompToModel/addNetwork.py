# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""
from timeit import default_timer as dt
import numpy as np
from optoptions import NetworkModel

def add_sec_constraints(m, params, thermals, hydros, network,
                            t_g, h_g,
                                flow_AC,
                                    periods = None):
    """
        Add security constraints to the model
    """

    s_sec_constrs = {}

    # get a list of the buses whose injections appear in one or more security constraints
    constrs_w_buses = set()
    for t in periods:
        for key, constr in [(item[0], item[1]) for item in network.SEC_CONSTRS[t].items()
                    if (item[1]['LB'] > -(99999.00/params.POWER_BASE) or
                        item[1]['UB'] < (99999.00/params.POWER_BASE))]:
            constrs_w_buses.add((t, key))

    power_injections = {}

    for t, constr_id in constrs_w_buses:
        power_injections[constr_id, t] = ( - network.SEC_CONSTRS[t][constr_id]['net load']
                            + m.xsum(
                                        network.SEC_CONSTRS[t][constr_id]
                                                        ['participants_factors']['thermals'][g]*
                                                            t_g[g, t]
                            for g in network.SEC_CONSTRS[t][constr_id]['participants']['thermals'])
                            + m.xsum(
                                        network.SEC_CONSTRS[t][constr_id]
                                                        ['participants_factors']['hydros'][h ,u]*
                                                            h_g[h, u, t]
                        for (h,u) in network.SEC_CONSTRS[t][constr_id]['participants']['hydros']
                            if hydros.TURB_OR_PUMP[h] != "Pump")
                            - m.xsum(
                                        network.SEC_CONSTRS[t][constr_id]
                                                        ['participants_factors']['hydros'][h ,u]*
                                                            h_g[h, u, t]
                        for (h,u) in network.SEC_CONSTRS[t][constr_id]['participants']['hydros']
                            if hydros.TURB_OR_PUMP[h] == "Pump")
                                        )

    for (constr_id, t) in power_injections.keys():
        constr = network.SEC_CONSTRS[t][constr_id]

        if isinstance(power_injections[constr_id, t], (int, float)):
            continue

        if (constr['LB'] != constr['UB']):
            if constr['LB'] > -(99999.00/params.POWER_BASE):
                m.add_constr(power_injections[constr_id, t] >= constr['LB'],
                                                                name = f"{constr['name']}_LB_{t}")
            if constr['UB'] < (99999.00/params.POWER_BASE):
                m.add_constr(power_injections[constr_id, t] <= constr['UB'],
                                                                name = f"{constr['name']}_UB_{t}")
        else:
            m.add_constr(power_injections[constr_id, t] == constr['LB'],
                                                                name = f"{constr['name']}_EQ_{t}")

    return s_sec_constrs

def PTDF_formulation(m, params, thermals, network, hydros,
                            hg, tg,
                                t_ang,
                                    s_gen, s_load, s_Renewable,
                                        flow_AC, flow_DC,
                                            periods):
    """
        Use the PTDF formulation to represent the DC model
    """

    time_0 = dt()

    exp = get_bus_injection_expr(m, hydros, thermals, network,
                                    tg, hg,
                                        flow_AC, flow_DC,
                                            s_gen, s_load, s_Renewable,
                                                network.BUS_ID, t_ang,
                                                    include_AC_flows = False)

    # get a set of buses for which at least in one period there might be a power injection
    # these are the buses to which either a load or a generating unit is connected to at some
    # point in time
    buses_with_injections_idxs = np.array([b for b, bus in enumerate(network.BUS_ID)
                                if any((len(exp[bus, t]) >= 1 or abs(exp[bus,t].const) != 0
                                    for t in periods))], dtype = 'int64')

    _PTDF = network.PTDF[:]
    _PTDF[np.where(abs(_PTDF) < 1e-4)] = 0

    possibly_active_bounds = [l for l in network.LINE_ID if network.ACTIVE_BOUNDS[l]]
    possibly_active_bounds_idxs = [l_idx for l_idx, l in enumerate(network.LINE_ID)
                                                                        if network.ACTIVE_BOUNDS[l]]
    map_idx = {l: l_sub_idx for l_sub_idx, l in enumerate(possibly_active_bounds)}

    constrs = []

    count_constrs_added = 0

    sub_PTDF_only_act_lines = _PTDF[possibly_active_bounds_idxs, :]

    non_zeros = np.intersect1d(np.where(abs(sub_PTDF_only_act_lines) > 0)[1],
                                                buses_with_injections_idxs)
    buses_of_interest = [network.BUS_ID[b] for b in non_zeros]

    all_flows = (sub_PTDF_only_act_lines[:,non_zeros]
                            @ [[exp[bus, t] for t in periods] for bus in buses_of_interest])

    for l in possibly_active_bounds:
        l_idx = map_idx[l]

        ts = [t for t in periods if network.ACTIVE_UB_PER_PERIOD[l][t]
                                                            or network.ACTIVE_LB_PER_PERIOD[l][t]]

        flows = all_flows[l_idx, :]

        for t in ts:

            flow_exp = flows[ts.index(t)]

            if len(flow_exp) >= 1:
                if network.ACTIVE_UB_PER_PERIOD[l][t]:
                    constrs.append(m.add_constr(
                                            flow_exp <= network.LINE_FLOW_UB[l][t],
                                                name = f"ptdf_UB_{network.LINE_F_T[l][0]}_" +
                                                            f"{network.LINE_F_T[l][1]}_{l}_{t}"
                                            )
                                    )
                    count_constrs_added += 1

                if network.ACTIVE_LB_PER_PERIOD[l][t]:
                    constrs.append(m.add_constr(
                                            flow_exp >= network.LINE_FLOW_LB[l][t],
                                                name = f"ptdf_LB_{network.LINE_F_T[l][0]}_" +
                                                            f"{network.LINE_F_T[l][1]}_{l}_{t}"
                                            )
                                    )
                    count_constrs_added += 1

    buses_in_system = []

    for disj_subs, sub_sys in network.DISJOINT_AC_SUBS.items():
        buses_in_system += list(sub_sys['nodes'])
        for t in periods:
            m.add_constr(m.xsum(exp[bus, t] for bus in sub_sys['nodes']) == 0,
                                                            name = f"power_balance_{disj_subs}_{t}")

    #### some buses might not be in any subsystem because they are isolated
    for bus in [bus for bus in network.BUS_ID if bus not in buses_in_system]:
        for t in [t for t in periods if len(exp[bus, t]) >= 1]:
            m.add_constr(exp[bus, t] == 0, name = f"power_balance_{bus}_{t}")

    time_end = dt()

    print(f"\nIt took {time_end-time_0:,.4f} sec to add the PTDF constraints", flush = True)

def _single_bus(m, network, thermals, hydros, hg, tg, listT):
    """Add single-bus power balances for the periods in listT"""

    for t in listT:
        m.add_constr(m.xsum(tg[g, t] for g in thermals.ID) +
                    m.xsum(hg[k] for k in hg.keys()
                                        if hydros.TURB_OR_PUMP[k[0]] != 'Pump' and k[2] in listT)
                    - m.xsum(hg[k] for k in hg.keys()
                                        if hydros.TURB_OR_PUMP[k[0]] == 'Pump' and k[2] in listT)
                                                    == np.sum(network.NET_LOAD[:, t]),
                                                        name = f"single_bus_power_balance_{t}")

def get_bus_injection_expr(m, hydros, thermals, network,
                            tg, hg,
                                flow_AC, flow_DC,
                                    s_gen, s_load, s_Renewable,
                                        buses, periods,
                                            include_AC_flows:bool = True):
    """
        Get a linear expression of the power injection at bus bus and time t
    """

    t_0 = min(periods)
    thermals_per_bus = {bus: [g for g in thermals.ID if bus in thermals.BUS[g]] for bus in buses}
    turb_per_bus = {bus: [(h, u) for h in hydros.ID for u in hydros.UNIT_ID[h]
                            if bus in hydros.UNIT_BUS[h][u] and hydros.TURB_OR_PUMP[h] != 'Pump']
                                for bus in buses}
    pump_per_bus = {bus: [(h, u) for h in hydros.ID for u in hydros.UNIT_ID[h]
                            if bus in hydros.UNIT_BUS[h][u] and hydros.TURB_OR_PUMP[h] == 'Pump']
                                for bus in buses}
    s_gen_per_bus = {bus: [bus] if (bus, t_0) in s_gen.keys() else [] for bus in buses}
    s_load_per_bus = {bus: [bus] if (bus, t_0) in s_load.keys() else [] for bus in buses}
    s_ren_per_bus = {bus: [bus] if (bus, t_0) in s_Renewable.keys() else [] for bus in buses}

    #### Active power balance
    exp = {(bus, t): 0 for t in periods for bus in buses}
    if include_AC_flows:
        for bus in buses:
            for t in periods:
                exp[bus, t] = (
                        - m.xsum(flow_AC[network.LINE_F_T[l][0],
                                                        network.LINE_F_T[l][1], l, t]
                                                        for l in network.LINES_FROM_BUS[bus])
                                    + m.xsum(flow_AC[network.LINE_F_T[l][0],
                                                        network.LINE_F_T[l][1], l, t]
                                                        for l in network.LINES_TO_BUS[bus])
                            - network.NET_LOAD[network.BUS_HEADER[bus]][t]
                        +m.xsum(thermals.BUS_COEFF[g][bus]*tg[g,t] for g in thermals_per_bus[bus])
                        +m.xsum(hydros.UNIT_BUS_COEFF[k[0]][k[1]][bus]*hg[k[0], k[1],t]
                                                                    for k in turb_per_bus[bus])
                        -m.xsum(hydros.UNIT_BUS_COEFF[k[0]][k[1]][bus]*hg[k[0], k[1],t]
                                                                    for k in pump_per_bus[bus])
                        + m.xsum(s_gen[bus, t] for bus in s_gen_per_bus[bus])
                        - m.xsum(s_load[bus, t] for bus in s_load_per_bus[bus])
                        - m.xsum(s_Renewable[bus, t] for bus in s_ren_per_bus[bus])
                            )
    else:
        for bus in buses:
            for t in periods:
                exp[bus, t] = (
                        - network.NET_LOAD[network.BUS_HEADER[bus]][t]
                        +m.xsum(thermals.BUS_COEFF[g][bus]*tg[g,t] for g in thermals_per_bus[bus])
                        +m.xsum(hydros.UNIT_BUS_COEFF[k[0]][k[1]][bus]*hg[k[0], k[1],t]
                                                                    for k in turb_per_bus[bus])
                        -m.xsum(hydros.UNIT_BUS_COEFF[k[0]][k[1]][bus]*hg[k[0], k[1],t]
                                                                    for k in pump_per_bus[bus])
                        + m.xsum(s_gen[bus, t] for bus in s_gen_per_bus[bus])
                        - m.xsum(s_load[bus, t] for bus in s_load_per_bus[bus])
                        - m.xsum(s_Renewable[bus, t] for bus in s_ren_per_bus[bus])
                            )
    return exp

def B_theta_network_model(m, thermals, network, hydros,
                            hg, tg,
                                t_ang,
                                    s_gen, s_load, s_Renewable,
                                        flow_AC, flow_DC):
    """Add a DC representation of the network to the model"""

    # network buses' voltage angle in rad
    theta = {
                (bus, t):
                            m.add_var(var_type = 'C',
                                        lb = - network.THETA_BOUND,
                                        ub = network.THETA_BOUND,
                                            name = f'theta_{bus}_{t}')
                                                for t in t_ang
                                                    for bus in network.BUS_ID
            }

    #### Set the voltage angle reference
    for bus in set(network.REF_BUS_ID) & set(network.BUS_ID):
        for t in t_ang:
            m.set_lb(theta[bus, t], 0)
            m.set_ub(theta[bus, t], 0)

    exp = get_bus_injection_expr(m, hydros, thermals, network,
                                    tg, hg,
                                        flow_AC, flow_DC,
                                            s_gen, s_load, s_Renewable,
                                                network.BUS_ID, t_ang,
                                                    include_AC_flows = True)

    for bus in network.BUS_ID:
        for t in t_ang:
            m.add_constr(exp[bus, t] == 0, name = f"bus_{bus}_{t}")

    ## AC transmission limits
    for l in network.LINE_ID:
        ADMT = 1/network.LINE_X[l]
        if abs(ADMT) <= 1e-1:
            for t in t_ang:
                m.add_constr(1e2*flow_AC[network.LINE_F_T[l][0],
                                    network.LINE_F_T[l][1], l, t]
                                            == 1e2*ADMT*
                                                (theta[network.LINE_F_T[l][0], t] -
                                                theta[network.LINE_F_T[l][1], t]),
                                            name = f"ACflow_{network.LINE_F_T[l][0]}"+
                                                f"_{network.LINE_F_T[l][1]}_{l}_{t}")
        elif abs(ADMT) >= 1e3:
            for t in t_ang:
                m.add_constr(1e-2*flow_AC[network.LINE_F_T[l][0],
                                    network.LINE_F_T[l][1], l, t]
                                            == 1e-2*ADMT*
                                                (theta[network.LINE_F_T[l][0], t] -
                                                theta[network.LINE_F_T[l][1], t]),
                                            name = f"ACflow_{network.LINE_F_T[l][0]}"+
                                                f"_{network.LINE_F_T[l][1]}_{l}_{t}")
        else:
            for t in t_ang:
                m.add_constr(flow_AC[network.LINE_F_T[l][0],
                                    network.LINE_F_T[l][1], l, t]
                                            == ADMT*
                                                (theta[network.LINE_F_T[l][0], t] -
                                                theta[network.LINE_F_T[l][1], t]),
                                            name = f"ACflow_{network.LINE_F_T[l][0]}"+
                                                f"_{network.LINE_F_T[l][1]}_{l}_{t}")
    return theta

def addNetwork(m, params, thermals, network, hydros,
                    hg, tg,
                        t_ang, t_single_bus):
    """Add variables and constrains associated with the network
    m:                  optimization model
    params:             an instance of OptOptions (optoptions.py) that contains the
                            parameters for the problem and the algorithm
    network:            an instance of Network (network.py) with all network data
    thermals:           an instance of Thermals (network.py) with all thermal data
    hydros:             an instance of Hydros (network.py) with all hydro data
    hg:                 variables for the hydro generation of each hydro unit
    tg:                 variables for the thermal generation
    tAng:               set containing the periods for each the model for the network is DC
    tSingleBus:         set containing the period for each the model is a single bus
    """

    assert len(t_ang) + len(t_single_bus) > 0,\
                                "There must be at least one period in either t_ang or t_single_bus"

    theta, s_gen, s_load, s_renewable, flow_AC  = {}, {}, {}, {}, {}

    #### Flows in AC transmission lines
    if params.NETWORK_MODEL == NetworkModel.B_THETA:
        flow_AC = {
                    (network.LINE_F_T[l][0], network.LINE_F_T[l][1], l, t):
                            m.add_var(var_type = 'C',
                                        lb = network.LINE_FLOW_LB[l][t],
                                            ub = network.LINE_FLOW_UB[l][t],
                                                name = f"flowAC_{network.LINE_F_T[l][0]}_" +
                                                        f"{network.LINE_F_T[l][1]}" +
                                                        f'_{l}_{t}')
                                                    for t in t_ang
                                                        for l in network.LINE_F_T
                    }

    #### Flows in DC links
    flow_DC = {
                (network.LINK_F_T[l][0], network.LINK_F_T[l][1], l, t):
                        m.add_var(var_type = 'C',
                                    lb = -network.LINK_MAX_P[l],
                                        ub = network.LINK_MAX_P[l],
                                            name = f"flowDC_{network.LINK_F_T[l][0]}_"+
                                                    f"{network.LINK_F_T[l][1]}"+
                                                    f'_{l}_{t}')
                                                for t in t_ang
                                                    for l in network.LINK_F_T}

    if len(t_ang) > 0:

        renewable_gen_buses = network.get_renewable_gen_buses(thermals, hydros)
        s_renewable.update(
                            {(bus, t):
                                        m.add_var(var_type = 'C',
                                                  obj = network.DEFICIT_COST/4,
                                        ub = max(-1*network.NET_LOAD[network.BUS_HEADER[bus]][t],0),
                                                name = f'slack_renewable_{bus}_{t}')
                                                    for t in t_ang
                                                        for bus in renewable_gen_buses}
                            )

        load_buses = network.get_load_buses(thermals, hydros)
        s_gen.update(
                            {(bus, t):
                                        m.add_var(var_type = 'C',
                                            obj = network.DEFICIT_COST,
                                                name = f'slack_gen_{bus}_{t}')
                                                    for t in t_ang
                                                        for bus in load_buses}
                    )
        gen_buses = network.get_gen_buses(thermals, hydros)
        s_load.update(
                            {(bus, t):
                                        m.add_var(var_type = 'C',
                                            obj = network.DEFICIT_COST/4,
                                                name = f'slack_load_{bus}_{t}')
                                                    for t in t_ang
                                                        for bus in gen_buses - renewable_gen_buses}
                    )

        if params.NETWORK_MODEL == NetworkModel.PTDF:
            PTDF_formulation(m, params, thermals, network, hydros,
                                hg, tg,
                                    t_ang,
                                        s_gen, s_load, s_renewable,
                                            flow_AC, flow_DC,
                                                t_ang)

        elif params.NETWORK_MODEL == NetworkModel.B_THETA:
            theta = B_theta_network_model(m, thermals, network, hydros,
                                            hg,
                                                tg,
                                                    t_ang,
                                                        s_gen, s_load, s_renewable,
                                                            flow_AC, flow_DC)

    if len(t_single_bus) > 0 or params.NETWORK_MODEL == NetworkModel.SINGLE_BUS:
        _single_bus(m, network, thermals, hydros, hg, tg, list(set(t_single_bus) | set(t_ang)))

    if len(network.SEC_CONSTRS) > 0 and len(t_ang) > 0:
        s_sec_constrs = add_sec_constraints(m, params, thermals, hydros, network,
                                                                tg, hg, flow_AC, periods = t_ang)

    return (theta, flow_AC, flow_DC, s_gen, s_load, s_renewable)
