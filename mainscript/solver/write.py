# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""

import numpy as np

def writeEventTracker(params, eventTracker, wRank):
    '''Write metrics for the general coordinator'''

    name = 'generalCoordinator - ' if wRank == 0 else 'worker wRank ' + str(wRank) + ' - '

    f = open(params.outputFolder + name + \
            params.ps + ' - case ' + str(params.case) + '.csv', 'w', encoding = 'ISO-8859-1')
    f.write('Event;Code line; Upper bound ($); Lower bound ($);')
    f.write('bRank origin;bRank destination;fRank origin;fRank destination;')
    f.write('Iteration of origin;Subhorizon of origin;')
    f.write('Iteration of destination;Subhorizon of destination;')
    f.write('Elapsed time (sec)')
    f.write('\n')
    if len(eventTracker) > 0:
        for row in eventTracker:
            for col in row[0:2]:
                f.write(str(col) + ';')

            if row[2] != ' ':
                f.write(str(row[2]/params.scalObjF) + ';')
            else:
                f.write(str(row[2]) + ';')

            if row[3] != ' ':
                f.write(str(row[3]/params.scalObjF) + ';')
            else:
                f.write(str(row[3]) + ';')

            for col in row[4:]:
                f.write(str(col) + ';')
            f.write('\n')
    f.close()
    del f

    return()

def writeReadableSol(params, thermals, hydros, network,\
                    hgEachBus, tg, s_gen, s_load, s_Renewable, v,\
                    q, qBypass, qPump, s):
    '''Write the solution in a way humans can understand'''

    f = open(params.outputFolder + 'readableSolution - ' + \
                params.ps + ' - case ' + str(params.case) + '.csv', 'w', encoding = 'ISO-8859-1')

    for h in range(len(hydros.id)):
        f.write(hydros.name[h] + ';')
        gen = [0 for t in range(params.T)]
        for t in range(params.T):
            for bus in hydros.plantBuses[h]:
                gen[t] += hgEachBus[h, bus, t]
        if hydros.turbOrPump[h] == 'Pump':
            for t in range(params.T):
                f.write(str(round(-1*gen[t]*params.powerBase, 4)) + ';')
        else:
            for t in range(params.T):
                f.write(str(round(gen[t]*params.powerBase, 4)) + ';')
        f.write('\n')
    for g in range(len(thermals.id)):
        f.write(thermals.name[g] + ';')
        for t in range(params.T):
            f.write(str(round(tg[g, t]*params.powerBase, 4)) + ';')
        f.write('\n')
    f.write('Load;')
    for t in range(params.T):
        f.write(str(round(np.sum(network.load[t, :])*params.powerBase, 4)) + ';')

    f.write('\n')

    f.write('Load Shedding;')
    for t in range(params.T):
        f.write(str(round(sum([s_gen[bus, t] for bus in network.loadBuses])\
                                            *params.powerBase, 4)) + ';')

    f.write('\n')
    f.write('Generation surplus;')
    for t in range(params.T):
        f.write(str(round(sum([s_load[bus, t] for bus in\
                            (network.genBuses - network.renewableGenBuses)])\
                                            *params.powerBase, 4)) + ';')
    f.write('\n')

    f.write('Renewable generation curtailment;')
    for t in range(params.T):
        f.write(str(round(sum([s_Renewable[bus, t] for bus in network.renewableGenBuses])\
                                            *params.powerBase, 4)) + ';')

    f.close()

    if len(hydros.id) > 0:
        f = open(params.outputFolder + 'water transfer of hydro plants - '+\
                params.ps + ' - case ' + str(params.case) + '.csv', 'w', encoding = 'ISO-8859-1')
        f.write('<BEGIN>\n')
        f.write('water transfer of hydro plants in m3/s\n')
        for h in [h for h in range(len(hydros.id)) if len(hydros.downRiverTransferPlantID[h]) > 0]:
            f.write(hydros.name[h] + ';')
            for t in range(params.T):
                f.write(str(round(qBypass[h, t], 1)) + ';')
            f.write('\n')

        f.write('</END>')
        f.close()

        f = open(params.outputFolder + 'pumped water - '+\
                params.ps + ' - case ' + str(params.case) + '.csv', 'w', encoding = 'ISO-8859-1')
        f.write('<BEGIN>\n')
        f.write('Pumped water hydro plants in m3/s\n')
        for h in [h for h in range(len(hydros.id)) if hydros.turbOrPump[h] == 'Pump']:
            f.write(hydros.name[h] + ';')
            for t in range(params.T):
                f.write(str(round(qPump[h, t], 1)) + ';')
            f.write('\n')

        f.write('</END>')
        f.close()

        f = open(params.outputFolder +'total turbine discharge of hydro plants - '+\
                params.ps + ' - case ' + str(params.case) + '.csv', 'w', encoding = 'ISO-8859-1')
        f.write('<BEGIN>\n')
        f.write('Total turbine discharge of hydro plants in m3/s\n')
        for h in range(len(hydros.id)):
            f.write(hydros.name[h] + ';')
            for t in range(params.T):
                f.write(str(round(q[h, t], 1)) + ';')
            f.write('\n')

        f.write('</END>')
        f.close()

        f = open(params.outputFolder + 'spillage of hydro plants - ' + \
                params.ps + ' - case ' + str(params.case) + '.csv', 'w', encoding = 'ISO-8859-1')
        f.write('<BEGIN>\n')
        f.write('Spillage of hydropower plants in m3/s\n')
        for h in range(len(hydros.id)):
            f.write(hydros.name[h] + ';')
            for t in range(params.T):
                f.write(str(round(s[h, t], 1)) + ';')
            f.write('\n')
        f.write('</END>')
        f.close()

        f = open(params.outputFolder + 'reservoir volume of hydro plants - ' + \
                params.ps + ' - case ' + str(params.case) + '.csv', 'w', encoding = 'ISO-8859-1')
        f.write('<BEGIN>\n')
        f.write('Reservoir volume in hm3\n')
        for h in range(len(hydros.id)):
            f.write(hydros.name[h] + ';')
            for t in range(params.T):
                f.write(str(round(v[h, t], 1)) + ';')
            f.write('\n')
        f.write('</END>')
        f.close()

    return()

def writeReadableThermalSol(params, thermals, stUpTG, stDwTG, dispStat, tgDisp, tg):
    '''Write the decisions for the thermal units'''

    f = open(params.outputFolder + 'thermal decisions - ' + \
                params.ps + ' - case ' + str(params.case) + '.csv', 'w', encoding = 'ISO-8859-1')
    f.write('<BEGIN>\n')
    for g in range(len(thermals.id)):
        f.write('<Thermal>\n')
        f.write(thermals.name[g] + '\n')
        f.write('Start-up decision;Shut-down decision;Dispatch phase;')
        f.write('Generation in dispacth (MW);Total generation (MW)\n')
        for t in range(params.T):
            f.write(str(stUpTG[g, t]) + ';')
            f.write(str(stDwTG[g, t]) + ';')
            f.write(str(dispStat[g, t]) + ';')
            f.write(str(round((tgDisp[g, t]+thermals.minP[g]*dispStat[g, t])*params.powerBase, 4))\
                                                                                            + ';')
            f.write(str(round(tg[g, t]*params.powerBase, 4))+';')
            f.write('\n')
        f.write('</Thermal>\n')
    f.write('</END>')
    f.close()
    return()

def writeReadableHydroSol(params, hydros, v, q, s, qPump, qBypass, inflow, outflow, hgEachBus):
    '''Write the decisions for the hydro plants'''

    f = open(params.outputFolder + 'hydro decisions - ' + \
                params.ps + ' - case ' + str(params.case) + '.csv', 'w', encoding = 'ISO-8859-1')
    f.write('<BEGIN>\n')
    for h in range(len(hydros.id)):
        f.write('<Hydro>\n')
        f.write(hydros.name[h] + '\n')
        f.write('Reservoir volume (hm3);Turbine discharge (hm3);Spillage (hm3);')
        f.write('Pumping (hm3);Bypass (hm3);Incremental inflow (hm3);')
        f.write('Total inflow (hm3);Total outflow (hm3);')
        f.write('Total inflow minus total outflow (hm3);Total imbalance (hm3);')
        f.write('Power (MW);')
        f.write('\n')
        for t in range(params.T):
            f.write(str(round(v[h, t], 4)) + ';')
            f.write(str(round(q[h, t]*params.c_h, 4)) + ';')
            f.write(str(round(s[h, t]*params.c_h, 4)) + ';')
            if hydros.turbOrPump[h] == 'Pump':
                f.write(str(round(qPump[h, t]*params.c_h, 4)) + ';')
            else:
                f.write('0;')
            if len(hydros.downRiverTransferPlantID[h]) > 0:
                f.write(str(round(qBypass[h, t]*params.c_h, 4)) + ';')
            else:
                f.write('0;')
            f.write(str(round(hydros.inflows[h, t]*params.c_h, 4)) + ';')
            f.write(str(round(inflow[h, t], 4)) + ';')
            f.write(str(round(outflow[h, t], 4)) + ';')
            # Total inflow minus total outflow (hm3)
            f.write(str(round(inflow[h,t] - outflow[h, t], 4)) + ';')
            # Total imbalance (hm3)
            if t == 0:
                f.write(str(round((v[h,t] - hydros.V0[h]) - (inflow[h,t] - outflow[h, t]), 4))+';')
            else:
                f.write(str(round((v[h,t] - v[h, t - 1]) - (inflow[h,t] - outflow[h, t]), 4))+';')
            # Power (MW)
            if hydros.turbOrPump[h] == 'Pump':
                f.write(str(round(-1*sum([hgEachBus[h, bus, t] for bus in hydros.plantBuses[h]])\
                                                                *params.powerBase, 4)) + ';')
            else:
                f.write(str(round(sum([hgEachBus[h, bus, t] for bus in hydros.plantBuses[h]])\
                                                                *params.powerBase, 4)) + ';')

            f.write('\n')
        f.write('</Hydro>\n')
    f.write('</END>')
    f.close()
    return()

def writeDDiPdata(params, pLog, subhorizonInfo, backwardInfo, wRank):
    '''Write data of the optimization process'''

    if wRank == 0:
        f = open(params.outputFolder + '/convergence - '+\
                params.ps + ' - case ' + str(params.case) + '.csv', 'w', encoding = 'ISO-8859-1')
        f.write('Iteration;Lower bound;Upper bound;Gap (%);Forward time;')
        f.write('Backward time\n')
        for i in range(len(pLog['lb'])):
            f.write(str(i) + ';')
            f.write(str(pLog['lb'][i]/params.scalObjF) + ';')
            f.write(str(pLog['ub'][i]/params.scalObjF) + ';')
            f.write(str(pLog['gap'][i]*100) + ';')
            f.write(str(pLog['runTimeForward'][i]) + ';')
            f.write(str(pLog['runTimeBackward'][i]))
            f.write('\n')
        f.close()
        del f

    if params.I_am_a_forwardWorker:
        f = open(params.outputFolder + '/forwardInfo - wRank ' + str(wRank) + ' - '+\
                params.ps + ' - case ' + str(params.case) + '.csv', 'w', encoding = 'ISO-8859-1')
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
            for p in range(params.nSubhorizons):
                f.write(str(i) + ';')
                f.write(str(p) + ';')
                f.write(str(subhorizonInfo[p]['presentCots'][i]/params.scalObjF)+';')
                f.write(str(subhorizonInfo[p]['futureCosts'][i]/params.scalObjF)+';')
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

    if params.I_am_a_backwardWorker:
        f = open(params.outputFolder + '/backwardInfo - wRank ' + str(wRank) + ' - ' +\
                params.ps + ' - case ' + str(params.case) + '.csv', 'w', encoding = 'ISO-8859-1')
        f.write('Iteration;Subhorizon;LB ($);UB ($)')
        f.write(';Time (sec);Gap (%);Optimization Status')
        f.write(';Communication (sec)')
        f.write(';Time to add cuts (sec); Time stamp (sec)')
        f.write('\n')
        for i in range(len(backwardInfo[0]['lb'])):
            for p in range(params.nSubhorizons):
                f.write(str(i) + ';')
                f.write(str(p) + ';')
                f.write(str(backwardInfo[p]['lb'][i]/params.scalObjF) + ';')
                f.write(str(backwardInfo[p]['ub'][i]/params.scalObjF) + ';')
                f.write(str(backwardInfo[p]['time'][i]) + ';')
                f.write(str(backwardInfo[p]['gap'][i]*100) + ';')
                f.write(str(backwardInfo[p]['optStatus'][i]) + ';')
                f.write(str(backwardInfo[p]['communication'][i]) + ';')
                f.write(str(backwardInfo[p]['timeToAddCuts'][i]) + ';')
                f.write(str(backwardInfo[p]['timeStamp'][i]))
                f.write('\n')
        f.close()
        del f

    return()
