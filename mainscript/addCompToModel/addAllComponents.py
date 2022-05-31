# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""
from typing import Any, Tuple
from copy import deepcopy
from mip import BINARY, CONTINUOUS
from addCompToModel.addHydro import addHydro
from addCompToModel.addNetwork import addNetwork
from addCompToModel.addThermal import thermalBin, thermalCont

def couplingConstrsAndVars(thermals, hydros, stUpTG, stDwTG, dispStat,\
            tgDisp, v, q, s, qBypass, qPump, listOfPeriods) -> list:
    '''Store all coupling constraints in a list.
    This function receives dictionaries of either variables or constraints.
    The returned list is then a ordered arrangement of these elements'''

    coupl = []

    for t in listOfPeriods:
        # Start-up decision
        for g in range(len(thermals.id)):
            coupl.append(stUpTG[g, t])

        # Shut-down decision
        for g in range(len(thermals.id)):
            coupl.append(stDwTG[g, t])

        # Dispatch in the dispatch phase
        for g in range(len(thermals.id)):
            coupl.append(dispStat[g, t])

    for t in listOfPeriods:
        # Dispatch in the dispatch phase
        for g in range(len(thermals.id)):
            coupl.append(tgDisp[g, t])

        #### Hydro plants
        # Reservoir volume
        for h in range(len(hydros.id)):
            coupl.append(v[h, t])

        # Turbine discharge
        for h in range(len(hydros.id)):
            coupl.append(q[h, t])

        # Spillage
        for h in range(len(hydros.id)):
            coupl.append(s[h, t])

        # Water transfer
        for h in [h for h in range(len(hydros.id)) if len(hydros.downRiverTransferPlantID[h]) > 0]:
            coupl.append(qBypass[h, t])

        # Water transfer
        for h in [h for h in range(len(hydros.id)) if hydros.turbOrPump[h] == 'Pump']:
            coupl.append(qPump[h, t])

    return(coupl)

def addAllComp(params, hydros, thermals, network, MP, SP, SPnetwork, b: int,\
                fixedVars,\
                BDbinaries: bool = False, BDnetwork: bool = False,\
                binVars = BINARY) -> Tuple[list, list, Any, Any, Any, Any,\
                                            list, list, dict, dict, list, list, dict]:
    '''Add all components of the unit-commitment problem
    params:             an instance of OptOptions (optoptions.py) that contains the
                            parameters for the problem and the algorithm
    hydros:             an instance of Hydros (network.py) with all hydro data
    thermals:           an instance of Thermals (network.py) with all thermal data
    network:            an instance of Network (network.py) with all network data
    MP:                 optimization model for which the 'binary part' of the unit-commitment
                            problem is to be added to
    SP:                 optimization model for which the 'continuous part' of the unit-commitment
                            problem is to be added to
    SPnetwork:          optimization model for which the network model of the unit-commitment
                            problem is to be added to
    b:                  the index of the subhorizon (starts from 0)
    fixedVars:          numpy array holding all time-coupling decisions
    BDbinaries:         indicates whether the subhorizon problem is to be decomposed using the
                            classical Benders Decomposition: binary variables in the master problem
                            and the continuous part of the model in the subproblem
    BDnetwork:          can only be True if BDbinaries is True. In case it is True,
                            the Benders' subproblem is further decomposed into a generation problem
                            and a network problem
    binVars:            nature of the ON/OFF thermal decisions: either BINARY or CONTINUOUS

    Note that, if Benders decomposition is not used at all, then MP and SP are the same model, and
    SPnetwork is not used at all. In contrast, if Benders decomposition is used, then MP is
    the Benders' master problem with all binary variables and related constraints (in this case,
    minimum up and down times, logical constraints, and so on). The SP then contains everything else
    once the binary variables are fixed. Thus, the SP is then a continuous problem. Lastly,
    if BD is used and then the SP is further decomposed into a generation problem and a network
    problem, then SPnetwork is a different model from MP and SP.
    '''

    # The following dictionaries are used for storing the constraints that enforce that
    # decisions taken in previous subhorizons must not change
    constrStUpTG = {(g, t): None for g in range(len(thermals.id)) for t in range(params.T)}
    constrStDwTG = deepcopy(constrStUpTG)
    constrDispStat = deepcopy(constrStUpTG)
    constrTgDisp = deepcopy(constrStUpTG)

    constrV = {(h, t): None for h in range(len(hydros.id)) for t in range(params.T)}
    constrQ = deepcopy(constrV)
    constrQbyPass = deepcopy(constrV)
    constrQPump = deepcopy(constrV)
    constrS = deepcopy(constrV)

    # Variable used to estimate the total cost of subsequent subhorizons
    beta = SP.add_var(var_type=CONTINUOUS, name="beta", obj = 1\
                                                            if b <= params.nSubhorizons - 2 else 0)

    #### Add hydro variables and constraints to model SP
    hgEachBus, v, q, qBypass, qPump, s, _6, _7, alpha =\
                            addHydro(SP, params, hydros, constrV, constrQ,\
                            constrQbyPass, constrQPump, constrS, fixedVars,\
                            params.periodsPerSubhorizon[b], slackForVolumes = True)

    #### Add the thermal binary variables and related constraints to model MP
    stUpTG, stDwTG, dispStat = thermalBin(MP, params, thermals, network, hydros,\
                                    range(len(thermals.id)),\
                                    constrStUpTG, constrStDwTG, constrDispStat, fixedVars,\
                                    params.periodsPerSubhorizon[b], varNature = binVars)

    copyOfMPBinVars = []        # list of binary variables in the MP
    constrOfCopyOfMPBinVars = []# corresponding list of equality constraints in the SP that make
                                # SP's continuous variables equal to the decisions of MP's binaries

    if BDbinaries:
        # In case the subhorizon problem is decomposed using Benders into a binary MP and a
        # continuous SP, then create continuous variables (in the SP)
        # for the 0/1 decisions of the MP.
        # These continuous variables will latter be forced to assume the values chosen for their
        # binary counterparts in the MP.

        listOfPeriods = [t for t in range(params.T) if t <= max(params.periodsPerSubhorizon[b])]

        for t in listOfPeriods:
            for g in range(len(thermals.id)):
                copyOfMPBinVars.append(stUpTG[g, t])
            for g in range(len(thermals.id)):
                copyOfMPBinVars.append(stDwTG[g, t])
            for g in range(len(thermals.id)):
                copyOfMPBinVars.append(dispStat[g, t])

        tlgBin = [(g, t) for g in range(len(thermals.id)) for t in listOfPeriods]

        #### Copy of the startup decision in the SP
        stUpTG_SP = {(g, t): SP.add_var(var_type = CONTINUOUS, ub = 1, obj = 0,\
                                            name = f'stUpTG_SP_{g}_{t}') for (g, t) in tlgBin}
        #### Copy of the shutdown decision in the SP
        stDwTG_SP = {(g, t): SP.add_var(var_type = CONTINUOUS, ub = 1, obj = 0,\
                                            name = f'stDwTG_SP_{g}_{t}') for (g, t) in tlgBin}
        #### Copy of the dispatch status decision in the SP
        dispStat_SP = {(g, t): SP.add_var(var_type = CONTINUOUS, ub = 1, obj = 0,\
                                            name = f'dispStat_SP_{g}_{t}') for (g, t) in tlgBin}

        #### zeros for everything that comes after this subhorizon. these are only
        #### auxiliary entries to comply with the indices
        tlgBinZeros = [(g, t) for g in range(len(thermals.id))\
                                                for t in set(range(params.T)) - set(listOfPeriods)]
        stUpTG_SP.update({(g, t): 0 for (g, t) in tlgBinZeros})
        stDwTG_SP.update({(g, t): 0 for (g, t) in tlgBinZeros})
        dispStat_SP.update({(g, t): 0 for (g, t) in tlgBinZeros})

        for t in listOfPeriods:
            for g in range(len(thermals.id)):
                constrOfCopyOfMPBinVars.append(SP.add_constr(stUpTG_SP[g, t] == 0,\
                                                                name = f'EqCopyVarStUp_{g}_{t}'))
            for g in range(len(thermals.id)):
                constrOfCopyOfMPBinVars.append(SP.add_constr(stDwTG_SP[g, t] == 0,\
                                                                name = f'EqCopyVarStDw_{g}_{t}'))
            for g in range(len(thermals.id)):
                constrOfCopyOfMPBinVars.append(SP.add_constr(dispStat_SP[g, t] ==0,\
                                                                name = f'EqCopyVarDp_{g}_{t}'))
    else:
        stUpTG_SP = stUpTG
        stDwTG_SP = stDwTG
        dispStat_SP = dispStat

    #### Add the continuous part of the thermal units to the optimization model SP
    tg, tgDisp = thermalCont(SP, params, thermals, network, range(len(thermals.id)),\
                                                    constrTgDisp, fixedVars,\
                                                    params.periodsPerSubhorizon[b],\
                                                    stUpTG_SP, stDwTG_SP, dispStat_SP)

    couplConstrs = couplingConstrsAndVars(thermals, hydros,\
                        constrStUpTG, constrStDwTG, constrDispStat,\
                        constrTgDisp, constrV, constrQ, constrS, constrQbyPass, constrQPump,\
                        range(params.T))

    couplVars = couplingConstrsAndVars(thermals, hydros,\
                        stUpTG_SP, stDwTG_SP, dispStat_SP,\
                        tgDisp, v, q, s, qBypass, qPump,\
                        range(params.T))

    copyGenVars = []
    constrCopyGenVars = []

    alphaVarSPnetwork = {}

    if BDnetwork:
        # In this case, a network decomposition is also applied. The generation decisions are taken
        # and afterwatds evaluated in a network-check problem that basically checks the
        # feasiblity of the generation decisions with respect to the transmission lines' limits

        alphaVarSPnetwork = SP.add_var(var_type = CONTINUOUS, obj = 1, name = 'alpha')

        # Add the network constraints for the current subhorizon
        _0,_1,_2,_3,_4,_5 = addNetwork(SP, params, thermals, network,\
                                            hydros, hgEachBus, tg,\
                                            tAng = [], tSingleBus = params.periodsPerSubhorizon[b])


        # Create the thermal generation copy variables for the network subproblem
        tlg = [(g, t) for g in range(len(thermals.id)) for t in params.periodsPerSubhorizon[b]]
        tg_SP_network = {(g, t): SPnetwork[t].add_var(var_type = CONTINUOUS,\
                                                name = f'tg_SP_{g}_{t}') for (g, t) in tlg}

        # Create the hydro generation copy variables for the network subproblem
        tlh = [(h, hydros.plantBuses[h][busIndex], t) for h in range(len(hydros.id))\
                                                for busIndex in range(len(hydros.plantBuses[h]))\
                                                    for t in params.periodsPerSubhorizon[b]]
        hgEachBus_SP_network = {(h, bus, t): SPnetwork[t].add_var(var_type = CONTINUOUS,\
                                        name=f'hgEachBus_SP_{h}_{bus}_{t}') for (h, bus, t) in tlh}

        # Create a list of the generation variables
        i = 0 # index of variable in the list
        for k in tlg:
            copyGenVars.append(tg[k])
            if k[1] in params.periodsPerSubhorizon[b]:
                params.genConstrsPerPeriod[b][k[1]].append(i)
            i += 1
        for k in tlh:
            copyGenVars.append(hgEachBus[k])
            if k[2] in params.periodsPerSubhorizon[b]:
                params.genConstrsPerPeriod[b][k[2]].append(i)
            i += 1

        # Create constraints to ensure that the copy of generation variables take the values
        # define in the master problem
        for k in tlg:
            constrCopyGenVars.append(SPnetwork[k[1]].add_constr(tg_SP_network[k] == 0,\
                                                name = f'EqCopyVarTG_{k[0]}_{k[1]}'))

        for k in tlh:
            constrCopyGenVars.append(SPnetwork[k[2]].add_constr(hgEachBus_SP_network[k] == 0,\
                                                name = f'EqCopyVarHG_{k[0]}_{k[1]}_{k[2]}'))

        for k in SPnetwork.keys():
            # Add the DC network constraints for each period of the current subhorizon
            _0,_1,_2,_3,_4,_5 = addNetwork(SPnetwork[k], params, thermals, network,\
                                                hydros, hgEachBus_SP_network, tg_SP_network,\
                                                tAng = {k}, tSingleBus = {})
    else:
        # First, add the DC-network constraints to the period of the current subhorizon
        _0,_1,_2,_3,_4,_5 = addNetwork(SPnetwork,\
                                        params, thermals, network, hydros, hgEachBus, tg,\
                                        tAng = params.periodsPerSubhorizon[b], tSingleBus = {})


    alphaVarMP = MP.add_var(var_type = CONTINUOUS, obj = 1, name = 'alpha') if BDbinaries else None

    return(couplConstrs, couplVars, alpha, beta, alphaVarMP,\
            copyOfMPBinVars, constrOfCopyOfMPBinVars, dispStat, constrTgDisp,\
            copyGenVars, constrCopyGenVars, alphaVarSPnetwork)
