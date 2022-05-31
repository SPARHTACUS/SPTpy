# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""

import os
from time import sleep
import numpy as np
from mpi4py import MPI

from readAndLoadData.read_csv import readGenerators, readHydroGeneratingUnits,\
                        readNetwork, grossLoadAndRenewableGen, readIniStateThermal,\
                        readAggregHPF, readCostToGoFunction,\
                        readTrajectories,\
                        readBoundsOnGenOfThermals, readBoundsOnGenOfHydros,\
                        readInflows, resetGenCostsOfThermals, resetVolumeBounds,\
                            readPreviousStateOfHydroPlants
from network import Hydros, Thermals, Network
from optoptions import OptOptions
from preProcessing.removeEndOfLineBusesWithInj import removeEndOfLineBusesWithInj
from buildTheSystem import buildDCsystem
from createComm import createComm

def setParams(rootFolder, commWorld, rankWorld, sizeWorld, experiment, expName):
    '''Create an instance of OptOptions and set the initial values for its attributes'''

    params = OptOptions(rootFolder, rankWorld, sizeWorld, experiment['case'],\
                nSubhorizons = experiment['nSubhorizonsPerProcess'],\
                forwardWs = experiment['forwardWs'],\
                backwardWs = experiment['backwardWs'],\
                expName = expName,\
                solveOnlyFirstSubhorizonInput = experiment['solveOnlyFirstSubhorizonInput'])

    for k in [k for k in experiment.keys() if hasattr(params, k)]:
        oldValue = getattr(params, k)
        if isinstance(oldValue, list):
            setattr(params,k, [experiment[k] for i in range(len(oldValue))])
        else:
            setattr(params, k, experiment[k])

    params.outputFolder = rootFolder + '/output/' + params.ps + '/case '+params.case+'/'+expName+'/'

    if (rankWorld == 0) and not(os.path.isdir(params.outputFolder)):
        os.makedirs(params.outputFolder)

    commWorld.Barrier()

    if (rankWorld == 0) and (len(experiment) > 0):
        f = open(params.outputFolder + '/experiment - ' + params.ps +\
                                ' - case ' + str(params.case) + '.csv', 'w', encoding = 'utf-8')
        f.write('key;value\n')
        for k in experiment.keys():
            f.write(str(k) + ';' + str(experiment[k]) + '\n')
        f.close()
        del f

    return(params)

def shareBackwardAggrs(params, commWorld, rankWorld, sizeWorld):
    '''Make sure that all processes know how many subhorizons each of the backward processes has,
        and the periods in each of them. This will facilitate communication later on'''
    if params.I_am_a_backwardWorker and (sizeWorld > 1):
        bwPackageSend = np.array([  params.periodsPerSubhArray.shape[0],\
                                    params.periodsPerSubhArray.shape[1]], dtype = 'int')
        for r in [0]+[r for r in params.backwardWorkers if r != rankWorld]+params.forwardWorkers:
            # send to everyone except the process itself
            commWorld.Isend([bwPackageSend, MPI.INT], dest = r, tag = 35)
            commWorld.Isend([params.periodsPerSubhArray, MPI.INT], dest = r, tag = 36)

    if (sizeWorld > 1):

        corr = {r: -1e3 for r in params.backwardWorkers}    # correspondence between rankWorld, r,
                                                            # and bRank
        c = 1
        for r in params.backwardWorkers:
            corr[r] = c
            c += 1

        if rankWorld != 0 and not(params.I_am_a_forwardWorker):
            params.periodsOfBackwardWs[corr[rankWorld]] = params.periodsPerSubhArray

        bufferSize = {r: np.array([0, 0], dtype = 'int') for r in params.backwardWorkers}

        # receive from all backward processes. however, if the process is itself a backward
        # process, then it does not need to receive the info
        nMsgsToRecv = len(params.backwardWorkers) if not(params.I_am_a_backwardWorker) else\
                                                                len(params.backwardWorkers) - 1
        nMsgsRecv = 0

        status = MPI.Status()

        while nMsgsRecv < nMsgsToRecv:
            if (commWorld.Iprobe(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)):
                src = status.Get_source()
                tg = status.Get_tag()

                if tg == 35:
                    commWorld.Recv([bufferSize[src], MPI.INT], source = src, tag = 35)
                elif tg == 36:
                    c = corr[src]
                    params.periodsOfBackwardWs[c] = np.zeros((bufferSize[src][0],\
                                                            bufferSize[src][1]), dtype = 'int')
                    commWorld.Recv([params.periodsOfBackwardWs[c], MPI.INT], source = src, tag = 36)
                    nMsgsRecv += 1
                else:
                    raise Exception('Im wRank ' + str(rankWorld) + '. Ive got a message from '\
                                    + str(src) + ' with tag ' + str(tg) + ', and I dont know what'\
                                        + ' to do with it.')
            else:
                sleep(1)
    return()

def defineComplicatingVars(params, hydros, thermals):
    '''Creates arrays to store the values of the coupling variables
    Note that the primal decisions and dual variables are shared among processes through
    large one-dimensional arrays. To correctly recover demanded values, create dictionaries
    that store the index for each decision variable. For instance, if one is interested in the
    reservoir volume of hydro plant 1 in time period 6, then a map that takes volume, the index
    of the plant, and the index of the time period will return the index of the reservoir volume
    in the array.'''

    hydros.nUnits = {}
    hydros.nUnits.update({(h, t): 1 for t in range(params.T) for h in range(len(hydros.id))})

    # Populate the map params.mp with the time-coupling variables

    #### Thermal generating units
    # Start-up decision
    params.map['stUpTG'] = {(g, t): -1e6 for g in range(len(thermals.id)) for t in range(params.T)}
    # Shut-down decision
    params.map['stDwTG'] = {(g, t): -1e6 for g in range(len(thermals.id)) for t in range(params.T)}
    # Dispatch phase
    params.map['DpTG'] = {(g, t): -1e6 for g in range(len(thermals.id)) for t in range(params.T)}
    # Generation in the dispatch phase
    params.map['DpGenTG'] = {(g, t): -1e6 for g in range(len(thermals.id)) for t in range(params.T)}

    #### Hydro plants
    # Reservoir volume
    params.map['v'] = {(h, t): -1e6 for h in range(len(hydros.id)) for t in range(params.T)}
    # Turbine discharge
    params.map['q'] = {(h, t): -1e6 for h in range(len(hydros.id)) for t in range(params.T)}
    # Spillage
    params.map['s'] = {(h, t): -1e6 for h in range(len(hydros.id)) for t in range(params.T)}
    # Water transfer
    params.map['QbyPass'] = {(h, t): -1e6 for h in [h for h in range(len(hydros.id))\
                            if len(hydros.downRiverTransferPlantID[h]) >0] for t in range(params.T)}
    # Pumps
    params.map['pump'] = {(h, t): -1e6 for h in [h for h in range(len(hydros.id))\
                                        if hydros.turbOrPump[h] == 'Pump'] for t in range(params.T)}

    i = 0   # index of the time-coupling variable in the one-dimensional array that will be
            # created later

    for t in range(0, params.T, 1):
        for g in range(len(thermals.id)):
            params.map['stUpTG'][g, t] = i
            params.varsPerPeriod[t].append(i)
            params.binVarsPerPeriod[t].append(i)
            params.lbOnCouplVars.append(0)
            params.ubOnCouplVars.append(1)
            i += 1

        for g in range(len(thermals.id)):
            params.map['stDwTG'][g, t] = i
            params.varsPerPeriod[t].append(i)
            params.binVarsPerPeriod[t].append(i)
            params.lbOnCouplVars.append(0)
            params.ubOnCouplVars.append(1)
            i += 1
        for g in range(len(thermals.id)):
            params.map['DpTG'][g, t] = i
            params.varsPerPeriod[t].append(i)
            params.binVarsPerPeriod[t].append(i)
            params.binDispVarsPerPeriod[t].append(i)
            params.lbOnCouplVars.append(0)
            params.ubOnCouplVars.append(1)
            i += 1

    for t in range(0, params.T, 1):
        for g in range(len(thermals.id)):
            params.map['DpGenTG'][g, t] = i
            params.varsPerPeriod[t].append(i)
            params.conVarsPerPeriod[t].append(i)
            params.dispGenVarsPerPeriod[t].append(i)
            params.lbOnCouplVars.append(0)
            params.ubOnCouplVars.append(thermals.maxP[g] - thermals.minP[g])
            i += 1

        for h in range(len(hydros.id)):
            params.map['v'][h, t] = i
            params.varsPerPeriod[t].append(i)
            params.conVarsPerPeriod[t].append(i)
            params.lbOnCouplVars.append(hydros.minVol[h])
            params.ubOnCouplVars.append(hydros.maxVol[h])
            i += 1

        for h in range(len(hydros.id)):
            params.map['q'][h, t] = i
            params.varsPerPeriod[t].append(i)
            params.conVarsPerPeriod[t].append(i)
            params.lbOnCouplVars.append(0)

            if hydros.plantMaxPower[h] > 0:
                params.ubOnCouplVars.append(hydros.plantMaxTurbDisc[h])
            else:
                params.ubOnCouplVars.append(0)
            i += 1

        for h in range(len(hydros.id)):
            params.map['s'][h, t] = i
            params.varsPerPeriod[t].append(i)
            params.conVarsPerPeriod[t].append(i)
            params.lbOnCouplVars.append(0)
            params.ubOnCouplVars.append(hydros.maxSpil[h])
            i += 1

        for h in [h for h in range(len(hydros.id)) if len(hydros.downRiverTransferPlantID[h]) > 0]:
            params.map['QbyPass'][h, t] = i
            params.varsPerPeriod[t].append(i)
            params.conVarsPerPeriod[t].append(i)
            params.lbOnCouplVars.append(0)
            params.ubOnCouplVars.append(hydros.maxTransfer[h])
            i += 1

        for h in [h for h in range(len(hydros.id)) if hydros.turbOrPump[h] == 'Pump']:
            params.map['pump'][h, t] = i
            params.varsPerPeriod[t].append(i)
            params.conVarsPerPeriod[t].append(i)
            params.lbOnCouplVars.append(0)
            params.ubOnCouplVars.append(hydros.plantMaxTurbDisc[h])

            i += 1

    params.nComplVars = i   # total number of time-coupling variables

    # The following bounds will be used for rounding the time-coupling variables in
    # order to prevent numerical errors.
    params.lbOnCouplVars = np.array(params.lbOnCouplVars, dtype = 'd')
    params.ubOnCouplVars = np.array(params.ubOnCouplVars, dtype = 'd')

    for b in range(params.nSubhorizons):
        #### Binary variables
        for t in params.periodsPerSubhorizon[b]:
            params.varsPerSubh[b].extend(params.binVarsPerPeriod[t])
            params.binVarsPerSubh[b].extend(params.binVarsPerPeriod[t])
            params.binDispVarsPerSubh[b].extend(params.binDispVarsPerPeriod[t])

        #### Continuous variables
        for t in params.periodsPerSubhorizon[b]:
            params.varsPerSubh[b].extend(params.conVarsPerPeriod[t])
            params.conVarsPerSubh[b].extend(params.conVarsPerPeriod[t])

        #### Generation of thermal units in dispatch phase
        for t in params.periodsPerSubhorizon[b]:
            params.dispGenVarsPerSubh[b].extend(params.dispGenVarsPerPeriod[t])

        # Get all variables associated with time periods in the current subhorizon
        # and subhorizons that come before b
        for b2 in range(0, b + 1, 1):
            params.varsInPreviousAndCurrentSubh[b].extend(params.varsPerSubh[b2])
            params.contVarsInPreviousAndCurrentSubh[b].extend(params.conVarsPerSubh[b2])
            params.binVarsInPreviousAndCurrentSubh[b].extend(params.binVarsPerSubh[b2])

        # Get all variables associated with previous time periods
        for b2 in range(0, b, 1):
            params.varsInPreviousSubhs[b].extend(params.varsPerSubh[b2])

    for b in range(params.nSubhorizons):
        params.contVarsInPreviousAndCurrentSubh[b] = np.array(\
                                        params.contVarsInPreviousAndCurrentSubh[b], dtype = 'int')

        params.binVarsInPreviousAndCurrentSubh[b] = np.array(\
                                        params.binVarsInPreviousAndCurrentSubh[b], dtype = 'int')

    return()

def loadData(rootFolder, commWorld, sizeWorld, rankWorld, experiment, expName):
    '''Read csv files with system's data and operating conditions'''

    # create an instance of OptOptions (optoptions.py) with all parameters for the problem
    # and the solution process
    params = setParams(rootFolder, commWorld, rankWorld, sizeWorld, experiment, expName)

    shareBackwardAggrs(params, commWorld, rankWorld, sizeWorld)

    # create objects for the configurations of hydro plants, thermal plants, and the network model
    hydros, thermals, network = Hydros(), Thermals(),  Network()

    # read the parameters of the transmission network
    readNetwork(params.inputFolder + 'network - ' + params.ps +'.csv', params, network)

    # read data for the thermal and hydro generators
    readGenerators(params.inputFolder + 'powerPlants - ' + params.ps +'.csv',\
                                                                params, network, hydros, thermals)

    # read the data of hydro generating units
    if len(hydros.id) > 0:
        readHydroGeneratingUnits(params.inputFolder +'dataOfGeneratingUnits - ' +\
                                                                params.ps +'.csv', params, hydros)

    for b in range(len(network.busID)):
        network.loadHeader[network.busID[b]] = b

    network.genBuses = set(hydros.buses) | set(thermals.bus)

    network.dcBuses = set(  [network.DClinkFromTo[l][0] for l in network.DClinkFromTo]+\
                            [network.DClinkFromTo[l][1] for l in network.DClinkFromTo])

    # read the gross load and renewable generation
    grossLoadAndRenewableGen(params.inputFolder + 'case ' + str(params.case) +'/' +'gross load - '+\
                params.ps + ' - case ' + str(params.case) + '.csv',\
                params.inputFolder + 'case ' + str(params.case) +'/' + 'renewable generation - ' + \
                params.ps + ' - case ' + str(params.case) + '.csv' , params, network, rankWorld)

    # read the start-up and shut-down trajectories of thermal units
    readTrajectories(params.inputFolder + 'trajectories - ' + params.ps +'.csv', params, thermals)

    # read the incremental inflows to the reservoirs
    readInflows(params.inputFolder + 'case ' + str(params.case) +'/' + 'inflows - ' + \
                params.ps + ' - case ' + str(params.case) + '.csv', params, hydros)

    # bounds on the generation of groups of thermal units
    readBoundsOnGenOfThermals(params.inputFolder + 'case ' + str(params.case) +'/'+\
            'bounds on generation of groups of thermal units - ' + params.ps +\
                ' - case ' + str(params.case) + '.csv', params, thermals)

    # bounds on the generation of groups of hydro plants
    readBoundsOnGenOfHydros(params.inputFolder + 'case ' + str(params.case) +'/'+\
            'bounds on generation of groups of hydro plants - ' + params.ps +\
                ' - case ' + str(params.case) + '.csv', params, hydros)

    # read the initial state of the thermal units
    readIniStateThermal(params.inputFolder + 'case ' + str(params.case) + '/' +\
                                        'initialStateOfThermalPowerPlants - ' + params.ps +\
                                        ' - case ' + str(params.case) + '.csv', params, thermals)

    # reset generation costs
    resetGenCostsOfThermals(params.inputFolder + 'case ' + str(params.case) + '/' +\
                                        'reset generation costs of thermal units - ' + params.ps +\
                                        ' - case ' + str(params.case) + '.csv', params, thermals)

    # reset bounds on reservoir volumes
    resetVolumeBounds(params.inputFolder + 'case ' + str(params.case) + '/' +\
                                        'reset bounds on reservoir volumes - ' + params.ps +\
                                        ' - case ' + str(params.case) + '.csv', params, hydros)

    # read the previous states of hydro plants
    readPreviousStateOfHydroPlants(params.inputFolder + 'case ' + str(params.case) + '/' +\
                                        'initial reservoir volumes - ' + params.ps +\
                                        ' - case ' + str(params.case) + '.csv',\
                                        params.inputFolder + 'case ' + str(params.case) + '/' +\
                                        'previous water discharges of hydro plants - ' + params.ps+\
                                        ' - case ' + str(params.case) + '.csv', params, hydros)

    # the previous statuses of thermal units and their respective minimum up and down times, as well
    # as the ramping limits, might prevent the unit from being shut-down during a certain portion
    # of the planning horizon
    for g in range(len(thermals.id)):
        if (thermals.state0[g] == 1) and not(thermals.inShutDownTraj[g])\
            and not(thermals.tg0[g] <= thermals.minP[g]):
            thermals.sdDec[g] = params.T
            pDecrease = 0
            for t in range(params.T):
                pDecrease += thermals.rampDown[g]
                if (thermals.tg0[g] - pDecrease) <= thermals.minP[g]:
                    # The unit reaches the minimum at t, and can be turned off at t + 1
                    thermals.sdDec[g] = t + 1 #+ len(thermals.shutDownTraj[g])
                    # remember that the signal to shut down happens immediately after reaching the
                    # minimum, i.e., at t + 1
                    break
    ############################################################################

    readAggregHPF(params.inputFolder + 'case ' + str(params.case) +'/aggregated_3Dim - ' +\
                    params.ps +' - '+\
                    'case ' + str(params.case) + ' - HPF without binaries.csv', params, hydros)

    readCostToGoFunction(params.inputFolder + 'case ' + str(params.case) +'/' +\
                                    'cost-to-go function - ' + params.ps +\
                                    ' - case ' + str(params.case) + '.csv', params, hydros)

    buildDCsystem(params, hydros, thermals, network, rankWorld)

    defineComplicatingVars(params, hydros, thermals)

    if rankWorld == 0:
        f = open(params.outputFolder + 'Indices of complicating variables - ' + params.ps +\
                                ' - case ' + str(params.case) + '.csv', 'w', encoding = 'utf-8')
        f.write('varType;Plant;Time;Index;Lower bound on var;Upper bound on var\n')
        for key, _ in [(k, v) for k,v in params.map.items() if v is not {}]:
            for key2, value2 in params.map[key].items():
                f.write(key + ';' + str(key2[0]) + ';' + str(key2[1]))
                f.write(';' + str(value2))
                f.write(';' + str(params.lbOnCouplVars[value2]))
                f.write(';' + str(params.ubOnCouplVars[value2]))
                f.write('\n')
        f.close()
        del f

    # create communicators for the forward processes and backward processes. these will be used
    # for the workers to communicate with the general coordinator
    fComm, fRank, fSize, bComm, bRank, bSize = createComm(params, commWorld, rankWorld, sizeWorld)

    params.deficitCost = network.deficitCost

    if params.reduceSystem:
        removeEndOfLineBusesWithInj(params, hydros, thermals, network, rankWorld)

    return(params, hydros, thermals, network,\
            fComm, fRank, fSize, bComm, bRank, bSize)
