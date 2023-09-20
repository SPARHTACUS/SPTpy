# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""

import numpy as np

def write_event_tracker(params, event_tracker:list, W_RANK:int):
    """
        Write metrics for the general coordinator
    """

    name = 'generalCoordinator - ' if W_RANK == 0 else 'worker W_RANK ' + str(W_RANK) + ' - '

    f = open(params.OUT_DIR + name +
            params.PS + ' - case ' + str(params.CASE) + '.csv', 'w', encoding = 'ISO-8859-1')
    f.write('Event;Code line;Upper bound ($);Lower bound ($);')
    f.write('bRank origin;bRank destination;fRank origin;fRank destination;')
    f.write('Iteration of origin;Subhorizon of origin;')
    f.write('Iteration of destination;Subhorizon of destination;')
    f.write('Elapsed time (sec)')
    f.write('\n')
    if len(event_tracker) > 0:
        for row in event_tracker:
            for col in row[0:2]:
                f.write(str(col) + ';')

            if row[2] != ' ':
                f.write(str(row[2]/params.SCAL_OBJ_F) + ';')
            else:
                f.write(str(row[2]) + ';')

            if row[3] != ' ':
                f.write(str(row[3]/params.SCAL_OBJ_F) + ';')
            else:
                f.write(str(row[3]) + ';')

            for col in row[4:]:
                f.write(str(col) + ';')
            f.write('\n')
    f.close()
    del f

def write_generation(params, thermals, hydros, network,
                        hgEachBus, tg, s_gen, s_load, s_Renewable):
    """Write total generation per period of hydro and thermal plants to
        a csv file 'generation and load',
        along with net load, load curtailment, and generation surpluses
    """

    f = open(params.OUT_DIR + 'generation and load - ' + \
                params.PS + ' - case ' + str(params.CASE) + '.csv', 'w', encoding = 'ISO-8859-1')

    for h in hydros.ID:
        f.write(hydros.NAME[h] + ';')
        gen = [0 for t in range(params.T)]
        for t in range(params.T):
            for bus in [k[1] for k in hgEachBus.keys() if (k[0], k[2]) == (h, t)]:
                gen[t] += hgEachBus[h, bus, t]
        if hydros.TURB_OR_PUMP[h] == 'Pump':
            for t in range(params.T):
                f.write(str(round(-1*gen[t]*params.POWER_BASE, 4)) + ';')
        else:
            for t in range(params.T):
                f.write(str(round(gen[t]*params.POWER_BASE, 4)) + ';')
        f.write('\n')
    for g in thermals.ID:
        f.write(thermals.UNIT_NAME[g] + ';')
        for t in range(params.T):
            f.write(str(round(tg[g, t]*params.POWER_BASE, 4)) + ';')
        f.write('\n')

    f.write('Load;')
    for t in range(params.T):
        f.write(str(round(np.sum(network.NET_LOAD[:, t])*params.POWER_BASE, 4)) + ';')

    f.write('\n')

    f.write('Load Shedding;')
    for t in range(params.T):
        f.write(str(round(sum(s_gen[k] for k in s_gen.keys() if k[-1] == t)*params.POWER_BASE, 4))
                + ';')

    f.write('\n')
    f.write('Generation surplus;')
    for t in range(params.T):
        f.write(str(round(sum(s_load[k] for k in s_load.keys() if k[-1] == t)*params.POWER_BASE, 4))
                + ';')
    f.write('\n')

    f.write('Renewable generation curtailment;')
    for t in range(params.T):
        f.write(str(round(sum(s_Renewable[k] for k in s_Renewable.keys() if k[-1] == t)
                                                                    *params.POWER_BASE, 4)) + ';')

    f.close()

def write_thermal_operation(params, thermals, stUpTG, stDwTG, dispStat, tgDisp, tg):
    """Write the decisions for the thermal units"""

    f = open(params.OUT_DIR + 'thermal decisions - ' +
                params.PS + ' - case ' + str(params.CASE) + '.csv', 'w', encoding = 'ISO-8859-1')
    f.write('ID;Name;Period;')
    f.write('Start-up decision;Shut-down decision;Dispatch phase;')
    f.write('Generation in dispacth (MW);Total generation (MW)\n')
    for g in thermals.ID:
        for t in range(params.T):
            f.write(str(g) + ';')
            f.write(thermals.UNIT_NAME[g] + ';')
            f.write(str(t) + ';')
            f.write(str(stUpTG[g, t]) + ';')
            f.write(str(stDwTG[g, t]) + ';')
            f.write(str(dispStat[g, t]) + ';')
            f.write(str(round((tgDisp[g, t]+thermals.MIN_P[g]*dispStat[g, t])*params.POWER_BASE, 4))
                                                                                            + ';')
            f.write(str(round(tg[g, t]*params.POWER_BASE, 4))+';')
            f.write('\n')
    f.close()

def write_hydro_operation(params, hydros, v, q, s, qPump, qBypass, inflow, outflow, hg):
    """Write the decisions for the hydro plants to a csv file"""

    C_H = params.DISCRETIZATION*(3600*1e-6)

    f = open(params.OUT_DIR + 'hydro decisions - ' +
                params.PS + ' - case ' + str(params.CASE) + '.csv', 'w', encoding = 'ISO-8859-1')
    f.write("ID;Name;Period;Reservoir volume (hm3);")
    f.write("Turbine discharge (hm3);Turbine discharge (m3/s);")
    f.write("Spillage (hm3);Spillage (m3/s);")
    f.write("Pumping (hm3);Pumping (m3/s);Bypass (hm3);Bypass (m3/s);")
    f.write("Incremental inflow (hm3);Incremental inflow (m3/s);")
    f.write("Total inflow (hm3);Total inflow (m3/s);Total outflow (hm3);Total outflow (m3/s);")
    f.write("Total inflow minus total outflow (hm3);Total inflow minus total outflow (m3/s);")
    f.write("Total imbalance (hm3);Total imbalance (m3/s);")
    f.write('Power (MW);\n')
    for h in hydros.ID:
        for t in range(params.T):
            f.write(str(h) + ";")
            f.write(hydros.NAME[h] + ";")
            f.write(str(t) + ";")
            f.write(str(round(v[h, t], 4)) + ';')
            f.write(str(round(q[h, t]*C_H, 4)) + ';' + str(round(q[h, t], 4)) + ';')
            f.write(str(round(s[h, t]*C_H, 4)) + ';' + str(round(s[h, t], 4)) + ';')
            if hydros.TURB_OR_PUMP[h] == 'Pump':
                f.write(str(round(qPump[h, t]*C_H, 4)) + ';' + str(round(qPump[h, t], 4)) + ';')
            else:
                f.write('0;0;')
            if hydros.DOWN_RIVER_BY_PASS[h] is not None:
                f.write(str(round(qBypass[h, t]*C_H, 4)) + ';' + str(round(qBypass[h, t], 4)) + ';')
            else:
                f.write('0;0;')
            f.write(str(round(hydros.INFLOWS[h][t]*C_H, 4)) + ';')
            f.write(str(round(hydros.INFLOWS[h][t], 4)) + ';')
            f.write(str(round(inflow[h, t], 4)) + ';')
            f.write(str(round(inflow[h, t], 4)/C_H) + ';')
            f.write(str(round(outflow[h, t], 4)) + ';')
            f.write(str(round(outflow[h, t], 4)/C_H) + ';')
            # Total inflow minus total outflow (hm3)
            f.write(str(round(inflow[h,t] - outflow[h, t], 4)) + ';')
            f.write(str(round(inflow[h,t] - outflow[h, t], 4)/C_H) + ';')
            # Total imbalance (hm3)
            if t == 0:
                f.write(str(round((v[h,t] - hydros.V_0[h]) - (inflow[h,t] - outflow[h, t]), 4))+';')
            else:
                f.write(str(round((v[h,t] - v[h, t - 1]) - (inflow[h,t] - outflow[h, t]), 4))+';')
            if t == 0:
                f.write(str(round((v[h,t]-hydros.V_0[h])-(inflow[h,t]-outflow[h, t]),4)/C_H)+';')
            else:
                f.write(str(round((v[h,t] -v[h,t -1])-(inflow[h,t]-outflow[h, t]), 4)/C_H)+';')
            # Power (MW)
            if hydros.TURB_OR_PUMP[h] == 'Pump':
                f.write(str(round(-1*sum(hg[k] for k in hg.keys() if (k[0], k[2]) == (h, t))
                                                                *params.POWER_BASE, 4)) + ';')
            else:
                f.write(str(round(sum(hg[k] for k in hg.keys() if (k[0], k[2]) == (h, t))
                                                                *params.POWER_BASE, 4)) + ';')
            f.write('\n')
    f.close()

def writeDDiPdata(params, pLog, subhorizonInfo, backwardInfo, W_RANK):
    """Write data of the optimization process"""

    if W_RANK == 0:
        f = open(params.OUT_DIR + '/convergence - '+
                params.PS + ' - case ' + str(params.CASE) + '.csv', 'w', encoding = 'ISO-8859-1')
        f.write('Iteration;Lower bound;Upper bound;Gap (%);Forward time;')
        f.write('Backward time\n')
        for i in range(len(pLog['lb'])):
            f.write(str(i) + ';')
            f.write(str(pLog['lb'][i]/params.SCAL_OBJ_F) + ';')
            f.write(str(pLog['ub'][i]/params.SCAL_OBJ_F) + ';')
            f.write(str(pLog['gap'][i]*100) + ';')
            f.write(str(pLog['runTimeForward'][i]) + ';')
            f.write(str(pLog['runTimeBackward'][i]))
            f.write('\n')
        f.close()
        del f

    if params.I_AM_A_FORWARD_WORKER:
        f = open(params.OUT_DIR + '/forwardInfo - W_RANK ' + str(W_RANK) + ' - '+
                params.PS + ' - case ' + str(params.CASE) + '.csv', 'w', encoding = 'ISO-8859-1')
        f.write('Iteration;Subhorizon;Present costs ($);Future Costs ($)')
        f.write(';Time (sec);Iterations;Gap (%);Optimization Status')
        f.write(';Distance from previous solution (Euclidean distance)')
        f.write(';Distance from previous solution - binary variables')
        f.write(' (Hamming distance)')
        f.write(';Hamming distance to previous solution - thermal statuses')
        f.write(';Hamming distance to best solution - thermal statuses')
        f.write(';Communication (sec)')
        f.write(';Time to add cuts (sec);Time stamp (sec)')
        f.write('\n')
        for i in range(len(subhorizonInfo[0]['presentCots'])):
            for p in range(params.N_SUBHORIZONS):
                f.write(str(i) + ';')
                f.write(str(p) + ';')
                f.write(str(subhorizonInfo[p]['presentCots'][i]/params.SCAL_OBJ_F)+';')
                f.write(str(subhorizonInfo[p]['futureCosts'][i]/params.SCAL_OBJ_F)+';')
                f.write(str(subhorizonInfo[p]['time'][i]) + ';')
                f.write(str(subhorizonInfo[p]['iterations'][i]) + ';')
                f.write(str(subhorizonInfo[p]['gap'][i]*100) + ';')
                f.write(str(subhorizonInfo[p]['optStatus'][i]) + ';')
                f.write(str(subhorizonInfo[p]['distanceFromPreviousSol'][i]) + ';')
                f.write(str(subhorizonInfo[p]['distBinVars'][i]**2) + ';')
                f.write(str(subhorizonInfo[p]['distStatusBinVars'][i]**2) + ';')
                f.write(str(subhorizonInfo[p]['distStatusBinBestSol'][i]**2) + ';')
                f.write(str(subhorizonInfo[p]['communication'][i]) + ';')
                f.write(str(subhorizonInfo[p]['timeToAddCuts'][i]) + ';')
                f.write(str(subhorizonInfo[p]['timeStamp'][i]))
                f.write('\n')
        f.close()
        del f

    if params.I_AM_A_BACKWARD_WORKER:
        f = open(params.OUT_DIR + '/backwardInfo - W_RANK ' + str(W_RANK) + ' - ' +
                params.PS + ' - case ' + str(params.CASE) + '.csv', 'w', encoding = 'ISO-8859-1')
        f.write('Iteration;Subhorizon;LB ($);UB ($)')
        f.write(';Time (sec);Gap (%);Optimization Status')
        f.write(';Communication (sec)')
        f.write(';Time to add cuts (sec); Time stamp (sec)')
        f.write('\n')
        for i in range(len(backwardInfo[0]['lb'])):
            for p in range(params.N_SUBHORIZONS):
                f.write(str(i) + ';')
                f.write(str(p) + ';')
                f.write(str(backwardInfo[p]['lb'][i]/params.SCAL_OBJ_F) + ';')
                f.write(str(backwardInfo[p]['ub'][i]/params.SCAL_OBJ_F) + ';')
                f.write(str(backwardInfo[p]['time'][i]) + ';')
                f.write(str(backwardInfo[p]['gap'][i]*100) + ';')
                f.write(str(backwardInfo[p]['optStatus'][i]) + ';')
                f.write(str(backwardInfo[p]['communication'][i]) + ';')
                f.write(str(backwardInfo[p]['timeToAddCuts'][i]) + ';')
                f.write(str(backwardInfo[p]['timeStamp'][i]))
                f.write('\n')
        f.close()
        del f
