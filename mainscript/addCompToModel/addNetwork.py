# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""
from timeit import default_timer as dt
import numpy as np
from mip import xsum, CONTINUOUS, LinExpr

def singleBus(m, network, thermals, hydros, flowAC, hgEachBus, tg, listT):
    '''Add single-bus power balances for the periods in listT'''

    for subsys in network.subsystems:
        subSetOfThermals =set(network.listOfThermalsInSubsys[subsys]) & set(range(len(thermals.id)))
        subSetOfHydros = set(network.listOfHydrosInSubsys[subsys]) & set(range(len(hydros.id)))

        # Remember: k[0] is the 'from' subsystem, k[1] is the 'to' subsystem, and k[2] is an
        # unique identifier to differentiate multiple connections between subsystems
        listOfFlowsToSubS = [(k[0], k[1], k[2]) for k in network.flowsBetweenSubsystems.keys()\
                                                                                if k[1] == subsys]
        listOfFlowsFromSubS = [(k[0], k[1], k[2]) for k in network.flowsBetweenSubsystems.keys()\
                                                                                if k[0] == subsys]
        for t in listT:

            slackGen = m.add_var(var_type = CONTINUOUS, obj = network.deficitCost,\
                                            name = f'slackgenSubSys_{subsys}_{t}')

            s_Renewable = m.add_var(var_type = CONTINUOUS, obj = 0,\
                                            name = f's_RenewableSubSy_{subsys}_{t}')

            m.add_constr(slackGen - s_Renewable\
                        + xsum(tg[g, t] for g in subSetOfThermals) +\
                        xsum(hgEachBus[h, bus, t] for h in\
                                    [h for h in subSetOfHydros if hydros.turbOrPump[h] != 'Pump']\
                                                                for bus in hydros.plantBuses[h])\
                        - xsum(hgEachBus[h, bus, t] for h in\
                                    [h for h in subSetOfHydros if hydros.turbOrPump[h] == 'Pump']\
                                                                for bus in hydros.plantBuses[h]) +\
                        xsum(flowAC[k[0], subsys, k[2], t] for k in listOfFlowsToSubS) -\
                        xsum(flowAC[subsys, k[1], k[2], t] for k in listOfFlowsFromSubS)\
                                                        == network.subsysLoad[t, subsys],\
                                                            name = f"subsystem_{subsys}_{t}")

    return()

def getInjectionExprAtBus(exp, hydros, thermals, network, tg, hgEachBus, flowDC,\
                        s_gen, s_load, s_Renewable, bus, t, coeff):
    '''Get a linear expression of the power injection at bus bus and time t'''

    exp.add_const(-1*coeff*network.load[t][network.loadHeader[bus]])

    for l in network.DClinksFromBus[bus]:
        exp.add_term(term = flowDC[network.DClinkFromTo[l][0],\
                                            network.DClinkFromTo[l][1], l, t], coeff = -coeff)
    for l in network.DClinksToBus[bus]:
        exp.add_term(term = flowDC[network.DClinkFromTo[l][0],\
                                            network.DClinkFromTo[l][1], l, t], coeff = coeff)

    for g in network.thermalsAtBus[bus]:
        exp.add_term(term = tg[g, t], coeff = coeff)

    for h in [h for h in network.hydrosAtBus[bus] if hydros.turbOrPump[h] != 'Pump']:
        exp.add_term(term = hgEachBus[h, bus, t], coeff = coeff)

    for h in [h for h in network.hydrosAtBus[bus] if hydros.turbOrPump[h] == 'Pump']:
        exp.add_term(term = hgEachBus[h, bus, t], coeff = -coeff)

    if bus in network.loadBuses:
        exp.add_term(term = s_gen[bus, t], coeff = coeff)

    if bus in network.genBuses - network.renewableGenBuses:
        exp.add_term(term = s_load[bus, t], coeff = - coeff)

    if bus in network.renewableGenBuses:
        exp.add_term(term = s_Renewable[bus, t], coeff = - coeff)

    return()

def DCnetworkModel(m, thermals, network, hydros,\
                    hgEachBus, tg, listT, s_gen, s_load, s_Renewable, flowAC, flowDC):
    '''Add a DC representation of the network to the model'''

    # network buses' voltage angle in rad
    theta = {(bus, t): m.add_var(var_type = CONTINUOUS, lb = - network.thetaBound,\
                                        ub = network.thetaBound, name = f'theta_{bus}_{t}')\
                                            for t in listT for bus in network.busID}

    #### Set the voltage angle reference
    for bus in set(network.refBusesOfIslands.values()) & set(network.busID):
        for t in listT:
            theta[bus, t].lb = 0
            theta[bus, t].ub = 0

    #### Active power balance
    exp = {(bus, t): None for t in listT for bus in network.busID}
    for bus in network.busID:
        for t in listT:
            exp[bus, t] = - xsum(flowAC[network.AClineFromTo[l][0],\
                                                            network.AClineFromTo[l][1], l, t]\
                                                            for l in network.AClinesFromBus[bus]) \
                            + xsum(flowAC[network.AClineFromTo[l][0],\
                                                            network.AClineFromTo[l][1], l, t]\
                                                            for l in network.AClinesToBus[bus]) \
                            - network.load[t][network.loadHeader[bus]]

    for l in network.DClinkFromTo.keys():
        if network.DClinkFromTo[l][0] in network.busID:
            for t in listT:
                exp[network.DClinkFromTo[l][0], t].add_term(\
                                                    term = flowDC[network.DClinkFromTo[l][0],\
                                                    network.DClinkFromTo[l][1], l, t], coeff = -1.0)

        if network.DClinkFromTo[l][1] in network.busID:
            for t in listT:
                exp[network.DClinkFromTo[l][1], t].add_term(term = \
                                                    flowDC[network.DClinkFromTo[l][0],\
                                                    network.DClinkFromTo[l][1], l, t], coeff = 1)

    for g in range(len(thermals.id)):
        for t in listT:
            exp[thermals.bus[g], t].add_term(term = tg[g, t], coeff = 1)

    for h in [h for h in range(len(hydros.id)) if hydros.turbOrPump[h] != 'Pump']:
        for t in listT:
            for bus in hydros.plantBuses[h]:
                exp[bus, t].add_term(term = hgEachBus[h, bus, t], coeff = 1)

    for h in [h for h in range(len(hydros.id)) if hydros.turbOrPump[h] == 'Pump']:
        for t in listT:
            for bus in hydros.plantBuses[h]:
                exp[bus, t].add_term(term = hgEachBus[h, bus, t], coeff = -1)

    for k in s_gen:
        exp[k].add_term(term = s_gen[k], coeff = 1.0)

    for k in s_load:
        exp[k].add_term(term = s_load[k], coeff = - 1.0)

    for k in s_Renewable:
        exp[k].add_term(term = s_Renewable[k], coeff = - 1.0)

    for bus in network.busID:
        for t in listT:
            if len(exp[bus, t]) >= 1:
                m.add_constr(exp[bus, t] == 0, name = f"bus_{bus}_{t}")

    ## AC transmission limits
    for l in network.AClineFromTo.keys():
        if not(((len(network.AClinesFromBus[network.AClineFromTo[l][0]]) +\
                len(network.AClinesToBus[network.AClineFromTo[l][0]])) == 1)\
                    or\
                ((len(network.AClinesFromBus[network.AClineFromTo[l][1]]) +\
                len(network.AClinesToBus[network.AClineFromTo[l][1]])) == 1)):
            # Only add the constraints if none of the line's buses is a
            # end-of-line bus, i.e., a bus that is connected to a single line.
            # if the bus is connected to a single line, then the angle variable
            # associated with this bus can be anything and then the flow equality is not necessary

            if abs(network.AClineAdmt[l]) <= 1e-1:
                for t in listT:
                    m.add_constr(1e2*flowAC[network.AClineFromTo[l][0],\
                                            network.AClineFromTo[l][1], l, t]\
                                                    == 1e2*network.AClineAdmt[l]*\
                                                        (theta[network.AClineFromTo[l][0], t] -\
                                                        theta[network.AClineFromTo[l][1], t]), \
                                                    name = f"ACflow_{network.AClineFromTo[l][0]}"+\
                                                        f"_{network.AClineFromTo[l][1]}_{l}_{t}")
            elif abs(network.AClineAdmt[l]) >= 1e3:
                for t in listT:
                    m.add_constr(1e-2*flowAC[network.AClineFromTo[l][0],\
                                            network.AClineFromTo[l][1], l, t]\
                                                    == 1e-2*network.AClineAdmt[l]*\
                                                        (theta[network.AClineFromTo[l][0], t] -\
                                                        theta[network.AClineFromTo[l][1], t]), \
                                                    name = f"ACflow_{network.AClineFromTo[l][0]}"+\
                                                        f"_{network.AClineFromTo[l][1]}_{l}_{t}")
            else:
                for t in listT:
                    m.add_constr(flowAC[network.AClineFromTo[l][0],\
                                            network.AClineFromTo[l][1], l, t]\
                                                    == network.AClineAdmt[l]*\
                                                        (theta[network.AClineFromTo[l][0], t] -\
                                                        theta[network.AClineFromTo[l][1], t]), \
                                                    name = f"ACflow_{network.AClineFromTo[l][0]}"+\
                                                        f"_{network.AClineFromTo[l][1]}_{l}_{t}")
    return(theta)

def addNetwork( m, params, thermals, network, hydros, hgEachBus, tg, tAng, tSingleBus):
    '''Add variables and constrains associated with the network
    m:                  optimization model
    params:             an instance of OptOptions (optoptions.py) that contains the
                            parameters for the problem and the algorithm
    network:            an instance of Network (network.py) with all network data
    thermals:           an instance of Thermals (network.py) with all thermal data
    hydros:             an instance of Hydros (network.py) with all hydro data
    hgEachBus:          variables for the hydro generation
    tg:                 variables for the thermal generation
    tAng:               set containing the periods for each the model for the network is DC
    tSingleBus:         set containing the period for each the model is a single bus
    '''

    assert len(tAng) + len(tSingleBus) > 0,\
                                    "There must be at least one period in either tAng or tSingleBus"

    theta, s_gen, s_load, s_Renewable  = {}, {}, {}, {}

    #### Flows in AC transmission lines
    flowAC = {(network.AClineFromTo[l][0], network.AClineFromTo[l][1], l, t):\
                                        m.add_var(var_type = CONTINUOUS,\
                                        lb = network.AClineLBCap[l], ub = network.AClineUBCap[l],
                        name = f'flowAC_{network.AClineFromTo[l][0]}_{network.AClineFromTo[l][1]}'\
                                + f'_{l}_{t}') for t in tAng for l in network.AClineFromTo}
    #### Flows in DC links
    flowDC = {(network.DClinkFromTo[l][0], network.DClinkFromTo[l][1], l, t):
                        m.add_var(var_type = CONTINUOUS,\
                        lb = -network.DClinkCap[l], ub = network.DClinkCap[l],\
                        name = f'flowDC_{network.DClinkFromTo[l][0]}_{network.DClinkFromTo[l][1]}'+\
                                f'_{l}_{t}') for t in tAng for l in network.DClinkFromTo}

    if len(tAng) > 0:
        s_gen.update({(bus, t): m.add_var(var_type = CONTINUOUS, obj = network.deficitCost,\
                                            name = f'slackgen_{bus}_{t}') for t in tAng\
                                                for bus in network.loadBuses})

        s_load.update({(bus, t): m.add_var(var_type = CONTINUOUS, obj = network.deficitCost/4,\
                                    name = f'slackload_{bus}_{t}') for t in tAng\
                                        for bus in (network.genBuses - network.renewableGenBuses)})

        s_Renewable.update({(bus, t): m.add_var(var_type = CONTINUOUS, obj = 0,\
                                    name = f'slackRenewable_{bus}_{t}') for t in tAng\
                                        for bus in network.renewableGenBuses})

        for bus in network.renewableGenBuses:
            for t in tAng:
                # Make sure that, at each bus, the amount of generation shedding is not greater
                # than the amount of renewable energy forecasted to the respective buses
                s_Renewable[bus, t].ub = max(-1*network.load[t][network.loadHeader[bus]], 0)

        theta = DCnetworkModel(m, thermals, network, hydros, hgEachBus,\
                                tg, tAng, s_gen, s_load, s_Renewable, flowAC, flowDC)

    if len(tSingleBus) > 0:
        # Create the variables of flows between subsystems
        connecAC = [(k[0], k[1], k[2], t) for t in tSingleBus\
                                                            for k in network.flowsBetweenSubsystems]

        # Remember that network.flowsBetweenSubsystems[(k[0], k[1], k[2])] gives the maximum flow,
        # in p.u., for the connecting line k[2] from subsystem k[0] to subsystem k[1]

        #### Limits on flows in the AC transmission lines
        lbs = [-1*network.flowsBetweenSubsystems[k] for t in tSingleBus\
                                                            for k in network.flowsBetweenSubsystems]
        ubs = [network.flowsBetweenSubsystems[k] for t in tSingleBus\
                                                            for k in network.flowsBetweenSubsystems]

        #### Flows between subsystems
        flowAC.update({(connecAC[i][0], connecAC[i][1], connecAC[i][2], connecAC[i][3]):\
                                        m.add_var(var_type = CONTINUOUS, lb = lbs[i], ub = ubs[i],
                                            name = f'flowAC_{connecAC[i][0]}_{connecAC[i][1]}'+\
                                                    f'_{connecAC[i][2]}_{connecAC[i][3]}')\
                                                    for i in range(len(connecAC))})

        singleBus(m, network, thermals, hydros, flowAC, hgEachBus, tg, tSingleBus)

    return (theta, flowAC, flowDC, s_gen, s_load, s_Renewable)
