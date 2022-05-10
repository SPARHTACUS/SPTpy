# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""

import locale

from mip import Model, BINARY, OptimizationStatus, entities

from solver.write import writeReadableSol, writeReadableThermalSol, writeReadableHydroSol
from addCompToModel.addHydro import addHydro
from addCompToModel.addNetwork import addNetwork
from addCompToModel.addThermal import thermalBin, thermalCont

def postOptimization(params, thermals, hydros, network, fixedVars, ub):
    '''Optimize the model with the decision variables fixed and write the result to csv files'''

    locale.setlocale(locale.LC_ALL, 'en_US.utf-8')

    nPeriodsPerSubhorizon = 8
    if (params.T/nPeriodsPerSubhorizon) == int(params.T/nPeriodsPerSubhorizon):
        periodsPerSubhorizon = {b: set(range(b*nPeriodsPerSubhorizon,\
                                                (b + 1)*nPeriodsPerSubhorizon, 1))\
                                                for b in range(int(params.T/nPeriodsPerSubhorizon))}
    else:
        periodsPerSubhorizon = {b: set(range(b*6, (b + 1)*6, 1)) for b in range(int(params.T/6))}

    presentCosts, futureCosts = [0 for b in periodsPerSubhorizon.keys()],\
                                [0 for b in periodsPerSubhorizon.keys()]

    hgEachBus, v, q, qBypass, qPump, s, inflow, outflow = {}, {}, {}, {}, {}, {}, {}, {}
    stUpTG, stDwTG, dispStat, tg, tgDisp = {}, {}, {}, {}, {}
    s_gen, s_load, s_Renewable  = {}, {}, {}
    allVars = {}

    params.nSubhorizons = len(periodsPerSubhorizon.keys())

    for b in periodsPerSubhorizon.keys():

        m = Model('m', solver_name = params.solver)

        m.verbose = 0

        hgEachBusVar, vVar, qVar, qBypassVar, qPumpVar, sVar, inflowVar, outflowVar, alpha =\
                                    addHydro(m, params,\
                                    hydros, {}, {}, {}, {}, {},\
                                    fixedVars, periodsPerSubhorizon[b],\
                                    slackForVolumes = True)

        if b != max(list(periodsPerSubhorizon.keys())):
            alpha.obj = 0

        stUpTGVar, stDwTGVar, dispStatVar = thermalBin(m, params, thermals, network, hydros,\
                                            range(len(thermals.id)),\
                                            {}, {}, {}, fixedVars, periodsPerSubhorizon[b],\
                                            varNature = BINARY)

        tgVar, tgDispVar = thermalCont(m, params, thermals, network, range(len(thermals.id)),\
                                {}, fixedVars,\
                                        periodsPerSubhorizon[b], stUpTGVar, stDwTGVar, dispStatVar)

        _0, _1, _2, s_genVar, s_loadVar, s_RenewableVar = addNetwork(m,\
                                        params, thermals, network, hydros, hgEachBusVar,\
                                        tgVar, periodsPerSubhorizon[b], {})

        for g in range(len(thermals.id)):
            for t in periodsPerSubhorizon[b]:
                m.add_constr(stUpTGVar[g, t] == fixedVars[params.map['stUpTG'][g,t]],\
                                                                    name = f'constrStUpTG_{g}_{t}')
                m.add_constr(stDwTGVar[g, t] == fixedVars[params.map['stDwTG'][g,t]],\
                                                                    name = f'constrStDwTG_{g}_{t}')
                m.add_constr(dispStatVar[g, t] == fixedVars[params.map['DpTG'][g,t]],\
                                                                    name = f'constrDpTG_{g}_{t}')
                m.add_constr(tgDispVar[g,t] == fixedVars[params.map['DpGenTG'][g,t]],\
                                                                    name = f'constrDpGenTG_{g}_{t}')

        for h in range(len(hydros.id)):
            for t in periodsPerSubhorizon[b]:
                m.add_constr(vVar[h, t] == fixedVars[params.map['v'][h, t]],\
                                                                        name = f'constrV_{h}_{t}')

        for h in range(len(hydros.id)):
            for t in periodsPerSubhorizon[b]:
                if fixedVars[params.map['q'][h, t]] >= 1e-3:
                    m.add_constr(qVar[h, t] == fixedVars[params.map['q'][h, t]],\
                                                                        name = f'constrQ_{h}_{t}')
                else:
                    m.add_constr(qVar[h, t] == 0, name = f'constrQ_{h}_{t}')

                m.add_constr(sVar[h, t] == fixedVars[params.map['s'][h, t]],\
                                                                        name = f'constrS_{h}_{t}')
                if len(hydros.downRiverTransferPlantID[h]) > 0:
                    m.add_constr(qBypassVar[h, t] == fixedVars[params.map['QbyPass'][h,t]],\
                                                                    name = f'constrQbyPass_{h}_{t}')
                if hydros.turbOrPump[h] == 'Pump':
                    m.add_constr(qPumpVar[h, t] == fixedVars[params.map['pump'][h, t]],\
                                                                    name = f'constrQpump_{h}_{t}')

        m.max_mip_gap = 1e-6
        postOptStatus = m.optimize()

        if postOptStatus not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
            break

        print(f"Subhorizon {b}\t\t" +\
                f"Gap (%): {100*((m.objective_value - m.objective_bound)/m.objective_value):.4f}")

        presentCosts[b] = m.objective_value
        if b == max(list(periodsPerSubhorizon.keys())):
            presentCosts[b] = m.objective_value - alpha.x
            futureCosts[b] = alpha.x

        allVars.update({var.name: var.x for var in m.vars if int(var.name[var.name.rfind('_')+1:])\
                                                                >= min(periodsPerSubhorizon[b])})

        hgEachBus.update({k: hgEachBusVar[k].x for k in [k for k in hgEachBusVar.keys()\
                                                        if k[2] in periodsPerSubhorizon[b]]})
        v.update({k:vVar[k].x for k in [k for k in vVar.keys() if k[1] in periodsPerSubhorizon[b]]})
        q.update({k:qVar[k].x for k in [k for k in qVar.keys() if k[1] in periodsPerSubhorizon[b]]})
        qBypass.update({k: qBypassVar[k].x for k in [k for k in qBypassVar.keys()\
                                                if isinstance(qBypassVar[k], entities.Var) and\
                                                                k[1] in periodsPerSubhorizon[b]]})
        qPump.update({k: qPumpVar[k].x for k in [k for k in qPumpVar.keys()\
                                                    if isinstance(qPumpVar[k], entities.Var) and\
                                                                k[1] in periodsPerSubhorizon[b]]})
        s.update({k:sVar[k].x for k in [k for k in sVar.keys() if k[1] in periodsPerSubhorizon[b]]})
        inflow.update({k: inflowVar[k].x for k in [k for k in inflowVar.keys()\
                                                    if isinstance(inflowVar[k], entities.Var) and\
                                                                k[1] in periodsPerSubhorizon[b]]})
        inflow.update({k: inflowVar[k] for k in [k for k in inflowVar.keys()\
                                                if not(isinstance(inflowVar[k], entities.Var)) and\
                                                                k[1] in periodsPerSubhorizon[b]]})
        outflow.update({k: outflowVar[k].x for k in [k for k in outflowVar.keys()\
                                                            if k[1] in periodsPerSubhorizon[b]]})
        stUpTG.update({k: stUpTGVar[k].x for k in [k for k in stUpTGVar.keys()\
                                                            if k[1] in periodsPerSubhorizon[b]]})
        stDwTG.update({k: stDwTGVar[k].x for k in [k for k in stDwTGVar.keys()\
                                                            if k[1] in periodsPerSubhorizon[b]]})
        dispStat.update({k: dispStatVar[k].x for k in [k for k in dispStatVar.keys()\
                                                            if k[1] in periodsPerSubhorizon[b]]})
        tg.update({k: tgVar[k].x for k in [k for k in tgVar.keys()\
                                                            if k[1] in periodsPerSubhorizon[b]]})
        tgDisp.update({k: tgDispVar[k].x for k in [k for k in tgDispVar.keys()\
                                                            if k[1] in periodsPerSubhorizon[b]]})

        s_gen.update({k: s_genVar[k].x for k in [k for k in s_genVar.keys()\
                                                            if k[1] in periodsPerSubhorizon[b]]})
        s_load.update({k: s_loadVar[k].x for k in [k for k in s_loadVar.keys()\
                                                            if k[1] in periodsPerSubhorizon[b]]})
        s_Renewable.update({k: s_RenewableVar[k].x for k in [k for k in s_RenewableVar.keys()\
                                                            if k[1] in periodsPerSubhorizon[b]]})

    if postOptStatus in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):

        print('\n\nThe total present cost is ' +\
                                locale.currency(sum(presentCosts)/params.scalObjF, grouping = True))

        print('The future cost is ' +\
                                locale.currency(sum(futureCosts)/params.scalObjF, grouping = True))

        print('The total cost is ' +\
            locale.currency((sum(presentCosts)+sum(futureCosts))/params.scalObjF, grouping = True),\
                                                                                    flush = True)

        print('\nThe upper bound was ' +\
                    locale.currency(ub/params.scalObjF, grouping = True), flush = True)

        print('\nA difference of ' +\
                locale.currency((ub - (sum(presentCosts)+sum(futureCosts)))/params.scalObjF,\
                grouping = True) + f' ({100*(ub - (sum(presentCosts)+sum(futureCosts)))/ub:.4f}%)'\
                            , flush = True)

        print('\n\n', flush = True)

        f = open(params.outputFolder + '/final results - '+\
                    params.ps + ' - case ' + str(params.case) + '.csv', 'w', encoding = 'utf-8')
        f.write('Present cost ($);' + str(sum(presentCosts)/params.scalObjF) + '\n')
        f.write('Future cost ($);' + str(sum(futureCosts)/params.scalObjF) + '\n')
        f.write('Total cost ($);' + str((sum(presentCosts)+sum(futureCosts))/params.scalObjF)+'\n')
        f.write('Upper bound in the DDiP ($);' + str(ub/params.scalObjF)+'\n')
        f.write('Difference ($);' +\
                        str((ub - (sum(presentCosts)+sum(futureCosts)))/params.scalObjF)+'\n')
        f.write('Difference (%);' + str(100*(ub - (sum(presentCosts)+sum(futureCosts)))/ub))
        f.close()
        del f

        # Print all variables
        f = open(params.outputFolder + '/variables - Part 1 - '+\
                        params.ps + ' - case ' + str(params.case) + '.csv', 'w', encoding = 'utf-8')
        f.write('Var;x\n')
        for key, value in allVars.items():
            if 'theta' not in key and 'flowAC' not in key and 'flowDC' not in key:
                f.write(key + ';' + str(value) + '\n')
        f.close()
        del f

        f = open(params.outputFolder + '/variables - Part 2 - Angles and Flows - '+\
                        params.ps + ' - case ' + str(params.case) + '.csv', 'w', encoding = 'utf-8')
        f.write('Var;x;max;min\n')
        for key, value in allVars.items():
            if 'theta' in key or 'flowAC' in key or 'flowDC' in key:
                if 'theta' in key:
                    f.write(key + ';' + str(value) + ';2pi;-2pi\n')
                elif 'flowAC' in key:
                    f.write(key + ';' + str(value) + ';'\
                                        + str(network.AClineUBCap[int(key.split('_')[3])])+';'+\
                                        str(network.AClineLBCap[int(key.split('_')[3])]) + '\n')
                else:
                    f.write(key + ';' + str(value) + ';'\
                                        + str(network.DClinkCap[int(key.split('_')[3])]) +';'+\
                                        str(-1*network.DClinkCap[int(key.split('_')[3])]) + '\n')
        f.close()
        del f

        # Print only the slack variables
        f = open(params.outputFolder + '/network slack variables - '+\
                        params.ps + ' - case ' + str(params.case) + '.csv', 'w', encoding = 'utf-8')
        f.write('Var;x\n')
        for key, value in allVars.items():
            if 'slack' in key and not('ThermalGen' in key) and\
                                        not(('slackOutflowV' in key) or ('slackInflowV' in key)):
                f.write(key + ';' + str(value) + '\n')
        f.close()
        del f

        #print only the slack variables of maximum and minimum generation of groups of thermal units
        f = open(params.outputFolder + '/slack variables of max and min gen - '+\
                        params.ps + ' - case ' + str(params.case) + '.csv', 'w', encoding = 'utf-8')
        f.write('Var;x\n')
        for key, value in allVars.items():
            if 'ThermalGen' in key:
                f.write(key + ';' + str(value) + '\n')
        f.close()
        del f

        #print only the slack variables of reservoir volumes
        f = open(params.outputFolder + '/slack variables of reservoir volumes - '+\
                        params.ps + ' - case ' + str(params.case) + '.csv', 'w', encoding = 'utf-8')
        f.write('Var;x\n')
        for key, value in allVars.items():
            if ('slackOutflowV' in key) or ('slackInflowV' in key):
                f.write(key + ';' + str(value) + '\n')
        f.close()
        del f

        writeReadableSol(params, thermals, hydros, network, hgEachBus,\
                        tg, s_gen, s_load, s_Renewable, v, q, qBypass, qPump, s)

        writeReadableThermalSol(params, thermals,\
                {(g, t): stUpTG[g, t] for g in range(len(thermals.id)) for t in range(params.T)},\
                {(g, t): stDwTG[g, t] for g in range(len(thermals.id)) for t in range(params.T)},\
                {(g, t): dispStat[g, t] for g in range(len(thermals.id)) for t in range(params.T)},\
                {(g, t): tgDisp[g, t] for g in range(len(thermals.id)) for t in range(params.T)},\
                {(g, t): tg[g, t] for g in range(len(thermals.id)) for t in range(params.T)})

        if len(hydros.id) > 0:
            writeReadableHydroSol(params, hydros, v, q, s, qPump, qBypass,\
                                inflow, outflow, hgEachBus)
    else:
        print('The status of the post-optimization model is ' + str(postOptStatus), flush = True)

        m.write('postOptimization.lp')
        m.write('postOptimization.mps')
        m.write(params.outputFolder + 'postOptimization.lp')
        m.write(params.outputFolder + 'postOptimization.mps')

    return()
