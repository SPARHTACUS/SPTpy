# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""

import locale

from solver_interface.opt_model import Model, OptimizationStatus as OptS

from solver.write import write_generation, write_thermal_operation, write_hydro_operation
from addCompToModel.addHydro import addHydro
from addCompToModel.addNetwork import addNetwork
from addCompToModel.addThermal import thermalBin, thermalCont

def postOptimization(params, thermals, hydros, network, fixedVars, ub):
    """Optimize the model with the decision variables fixed and write the result to csv files"""

    locale.setlocale(locale.LC_ALL, '')

    for n_periods_per_subh in range(8, 0, -1):
        if (params.T % n_periods_per_subh) == 0:
            periods_per_subhorizon = {b: set(range(b*n_periods_per_subh,
                                                (b + 1)*n_periods_per_subh, 1))
                                            for b in range(int(params.T/n_periods_per_subh))}
            break

    present_costs, future_costs = ([0 for b in periods_per_subhorizon.keys()],
                                [0 for b in periods_per_subhorizon.keys()])

    hg, v, q, qBypass, qPump, s, inflow, outflow = {}, {}, {}, {}, {}, {}, {}, {}
    stUpTG, stDwTG, dispStat, tg, tgDisp = {}, {}, {}, {}, {}
    s_gen, s_load, s_Renewable  = {}, {}, {}
    allVars = {}

    params.N_SUBHORIZONS = len(periods_per_subhorizon.keys())

    for (b, periods) in periods_per_subhorizon.items():

        m = Model('m', solver_name = params.SOLVER, package = params.PACKAGE)

        m.verbose = 0

        (hg_var, vVar, qVar, qBypassVar, qPumpVar, sVar, inflowVar, outflowVar, alpha) =\
                                                                addHydro(m, params,
                                                                hydros, {}, {}, {}, {}, {},
                                                                fixedVars, periods)

        if b != max(list(periods_per_subhorizon.keys())):
            m.set_obj_coeff(alpha, 0)

        stUpTGVar, stDwTGVar, dispStatVar = thermalBin(m, params, thermals, network, hydros,
                                                    thermals.ID,
                                                    {}, {}, {}, fixedVars, periods,
                                                    varNature = 'B')

        tgVar, tgDispVar = thermalCont(m, params, thermals, network, thermals.ID,
                                        {}, fixedVars,
                                        periods, stUpTGVar, stDwTGVar, dispStatVar)

        _0, _1, _2, s_genVar, s_loadVar, s_RenewableVar = addNetwork(m,
                                        params, thermals, network, hydros, hg_var,
                                        tgVar, periods, {})

        for g in thermals.ID:
            for t in periods:
                m.add_constr(stUpTGVar[g, t] == fixedVars[params.MAP['stUpTG'][g,t]],
                                                                    name = f'constrStUpTG_{g}_{t}')
                m.add_constr(stDwTGVar[g, t] == fixedVars[params.MAP['stDwTG'][g,t]],
                                                                    name = f'constrStDwTG_{g}_{t}')
                m.add_constr(dispStatVar[g, t] == fixedVars[params.MAP['DpTG'][g,t]],
                                                                    name = f'constrDpTG_{g}_{t}')
                m.add_constr(tgDispVar[g,t] == fixedVars[params.MAP['DpGenTG'][g,t]],
                                                                    name = f'constrDpGenTG_{g}_{t}')

        for h in hydros.ID:
            for t in periods:
                m.add_constr(vVar[h, t] == fixedVars[params.MAP['v'][h, t]],
                                                                        name = f'constrV_{h}_{t}')
        for h in hydros.ID:
            for t in periods:
                if fixedVars[params.MAP['q'][h, t]] >= 1e-3:
                    m.add_constr(qVar[h, t] == fixedVars[params.MAP['q'][h, t]],
                                                                        name = f'constrQ_{h}_{t}')
                else:
                    m.add_constr(qVar[h, t] == 0, name = f'constrQ_{h}_{t}')

                m.add_constr(sVar[h, t] == fixedVars[params.MAP['s'][h, t]],
                                                                        name = f'constrS_{h}_{t}')
                if hydros.DOWN_RIVER_BY_PASS[h] is not None:
                    m.add_constr(qBypassVar[h, t] == fixedVars[params.MAP['QbyPass'][h,t]],
                                                                    name = f'constrQbyPass_{h}_{t}')
                if hydros.TURB_OR_PUMP[h] == 'Pump':
                    m.add_constr(qPumpVar[h, t] == fixedVars[params.MAP['pump'][h, t]],
                                                                    name = f'constrQpump_{h}_{t}')

        m.max_mip_gap = 1e-6
        postOptStatus = m.optimize()

        if postOptStatus not in (OptS.OPTIMAL, OptS.FEASIBLE):
            break

        print(f"Subhorizon {b}\t\t" +
                f"Gap (%): {100*((m.objective_value - m.objective_bound)/m.objective_value):.4f}")

        present_costs[b] = m.objective_value
        if b == max(list(periods_per_subhorizon.keys())):
            present_costs[b] = m.objective_value - m.get_var_x(alpha)
            future_costs[b] = m.get_var_x(alpha)

        allVars.update({m.get_name(var): m.get_var_x(var) for var in m.vars
                                    if int(m.get_name(var)[m.get_name(var).rfind('_')+1:])
                                                                >= min(periods)})

        hg.update({k: m.get_var_x(hg_var[k]) for k in [k for k in hg_var.keys()
                                                        if k[2] in periods]})
        v.update({k:m.get_var_x(vVar[k]) for k in [k for k in vVar.keys()
                                                            if k[1] in periods]})
        q.update({k:m.get_var_x(qVar[k]) for k in [k for k in qVar.keys()
                                                            if k[1] in periods]})
        qBypass.update({k: m.get_var_x(qBypassVar[k]) for k in [k for k in qBypassVar.keys()
                                                if not(isinstance(qBypassVar[k], (int, float))) and
                                                                k[1] in periods]})
        qPump.update({k: m.get_var_x(qPumpVar[k]) for k in [k for k in qPumpVar.keys()
                                                if not(isinstance(qPumpVar[k], (int, float))) and
                                                                k[1] in periods]})
        s.update({k:m.get_var_x(sVar[k]) for k in [k for k in sVar.keys()
                                                            if k[1] in periods]})
        inflow.update({k: m.get_var_x(inflowVar[k]) for k in [k for k in inflowVar.keys()
                                                if not(isinstance(inflowVar[k], (int, float))) and
                                                                k[1] in periods]})
        inflow.update({k: inflowVar[k] for k in [k for k in inflowVar.keys()
                                                if isinstance(inflowVar[k], (int, float)) and
                                                                k[1] in periods]})
        outflow.update({k: m.get_var_x(outflowVar[k]) for k in [k for k in outflowVar.keys()
                                                            if k[1] in periods]})
        stUpTG.update({k: m.get_var_x(stUpTGVar[k]) for k in [k for k in stUpTGVar.keys()
                                                            if k[1] in periods]})
        stDwTG.update({k: m.get_var_x(stDwTGVar[k]) for k in [k for k in stDwTGVar.keys()
                                                            if k[1] in periods]})
        dispStat.update({k: m.get_var_x(dispStatVar[k]) for k in [k for k in dispStatVar.keys()
                                                            if k[1] in periods]})
        tg.update({k: m.get_var_x(tgVar[k]) for k in [k for k in tgVar.keys()
                                                            if k[1] in periods]})
        tgDisp.update({k: m.get_var_x(tgDispVar[k]) for k in [k for k in tgDispVar.keys()
                                                            if k[1] in periods]})
        s_gen.update({k: m.get_var_x(s_genVar[k]) for k in [k for k in s_genVar.keys()
                                                            if k[1] in periods]})
        s_load.update({k: m.get_var_x(s_loadVar[k]) for k in [k for k in s_loadVar.keys()
                                                            if k[1] in periods]})
        s_Renewable.update({k: m.get_var_x(s_RenewableVar[k])
                                                    for k in [k for k in s_RenewableVar.keys()
                                                            if k[1] in periods]})

    if postOptStatus in (OptS.OPTIMAL, OptS.FEASIBLE):

        print('\n\nThe total present cost is ' +
                            locale.currency(sum(present_costs)/params.SCAL_OBJ_F, grouping=True))

        print('The future cost is ' +
                            locale.currency(sum(future_costs)/params.SCAL_OBJ_F, grouping=True))

        print('The total cost is ' +
            locale.currency((sum(present_costs)+sum(future_costs))/params.SCAL_OBJ_F,grouping=True),
                                                                                    flush = True)

        print('\nThe upper bound was ' +
                    locale.currency(ub/params.SCAL_OBJ_F, grouping = True), flush = True)

        print('\nA difference of ' +
                locale.currency((ub - (sum(present_costs)+sum(future_costs)))/params.SCAL_OBJ_F,
                grouping = True) + f' ({100*(ub - (sum(present_costs)+sum(future_costs)))/ub:.4f}%)'
                            , flush = True)

        print('\n\n', flush = True)

        f = open(params.OUT_DIR + '/final results - '+
                    params.PS + ' - case ' + str(params.CASE) + '.csv', 'w', encoding = 'utf-8')
        f.write('Present cost ($);' + str(sum(present_costs)/params.SCAL_OBJ_F) + '\n')
        f.write('Future cost ($);' + str(sum(future_costs)/params.SCAL_OBJ_F) + '\n')
        f.write('Total cost ($);' +str((sum(present_costs)+sum(future_costs))/params.SCAL_OBJ_F)
                                                                                            +'\n')
        f.write('Upper bound in the DDiP ($);' + str(ub/params.SCAL_OBJ_F)+'\n')
        f.write('Difference ($);' +\
                        str((ub - (sum(present_costs)+sum(future_costs)))/params.SCAL_OBJ_F)+'\n')
        f.write('Difference (%);' + str(100*(ub - (sum(present_costs)+sum(future_costs)))/ub))
        f.close()
        del f

        # Print all variables
        f = open(params.OUT_DIR + '/variables - Part 1 - '+
                        params.PS + ' - case ' + str(params.CASE) + '.csv', 'w', encoding = 'utf-8')
        f.write('Var;x\n')
        for key, value in allVars.items():
            if 'theta' not in key and 'flowAC' not in key and 'flowDC' not in key:
                f.write(key + ';' + str(value) + '\n')
        f.close()
        del f

        f = open(params.OUT_DIR + '/variables - Part 2 - Angles and Flows - '+
                        params.PS + ' - case ' + str(params.CASE) + '.csv', 'w', encoding = 'utf-8')
        f.write('Var;x;max;min\n')
        for key, value in allVars.items():
            if 'theta' in key or 'flowAC' in key or 'flowDC' in key:
                if 'theta' in key:
                    f.write(key + ';' + str(value) + ';2pi;-2pi\n')
                elif 'flowAC' in key:
                    f.write(key + ';' + str(value) + ';'
                                        + str(network.LINE_FLOW_UB[int(key.split('_')[3])])+';'+
                                        str(-1*network.LINE_FLOW_UB[int(key.split('_')[3])]) + '\n')
                else:
                    f.write(key + ';' + str(value) + ';'
                                        + str(network.LINK_MAX_P[int(key.split('_')[3])]) +';'+
                                        str(-1*network.LINK_MAX_P[int(key.split('_')[3])]) + '\n')
        f.close()
        del f

        # Print only the slack variables
        f = open(params.OUT_DIR + '/network slack variables - '+
                        params.PS + ' - case ' + str(params.CASE) + '.csv', 'w', encoding = 'utf-8')
        f.write('Var;x\n')
        for key, value in allVars.items():
            if ('slack' in key and not('ThermalGen' in key) and
                                        not(('slackOutflowV' in key) or ('slackInflowV' in key))):
                f.write(key + ';' + str(value) + '\n')
        f.close()
        del f

        #print only the slack variables of maximum and minimum generation of groups of thermal units
        f = open(params.OUT_DIR + '/slack variables of max and min gen - '+
                        params.PS + ' - case ' + str(params.CASE) + '.csv', 'w', encoding = 'utf-8')
        f.write('Var;x\n')
        for key, value in allVars.items():
            if 'ThermalGen' in key:
                f.write(key + ';' + str(value) + '\n')
        f.close()
        del f

        #print only the slack variables of reservoir volumes
        f = open(params.OUT_DIR + '/slack variables of reservoir volumes - '+
                        params.PS + ' - case ' + str(params.CASE) + '.csv', 'w', encoding = 'utf-8')
        f.write('Var;x\n')
        for (key, value) in [item for item in allVars.items()
                                if ('slack_outflow_' in item[0]) or ('slack_inflow_' in item[0])]:
            f.write(key + ';' + str(value) + '\n')
        f.close()
        del f

        write_generation(params, thermals, hydros, network, hg, tg, s_gen, s_load, s_Renewable)

        write_thermal_operation(params, thermals,
                            {(g, t): stUpTG[g, t] for g in thermals.ID for t in range(params.T)},
                            {(g, t): stDwTG[g, t] for g in thermals.ID for t in range(params.T)},
                            {(g, t): dispStat[g, t] for g in thermals.ID for t in range(params.T)},
                            {(g, t): tgDisp[g, t] for g in thermals.ID for t in range(params.T)},
                            {(g, t): tg[g, t] for g in thermals.ID for t in range(params.T)})

        if len(hydros.ID) > 0:
            write_hydro_operation(params, hydros, v, q, s, qPump, qBypass,
                                                                            inflow, outflow, hg)
    else:
        print('The status of the post-optimization model is ' + str(postOptStatus), flush = True)

        m.write('postOptimization.lp')
        m.write('postOptimization.mps')
        m.write(params.OUT_DIR + 'postOptimization.lp')
        m.write(params.OUT_DIR + 'postOptimization.mps')
