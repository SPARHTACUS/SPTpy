# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""

import numpy as np
from buildTheSystem import delMidPointBuses, updateLoadAndNetwork

def delEndOfLineBusesAndReassignInj(network, thermals, hydros,\
            candidateBuses, busesToDelete, indexOfBusesToDelete, linesDeleted):
    '''End-of-line buses are those connected to a single power line.
   Delete these buses and move their power injections to the neighbouring bus'''

    for bus in [bus for bus in candidateBuses if ((len(network.AClinesFromBus[bus]) \
                                                        + len(network.AClinesToBus[bus])) <= 1)]:
        busesToDelete.append(bus)
        indexOfBusesToDelete.append(network.busID.index(bus))

        island = []
        for i in network.busesInIsland.keys():
            if bus in network.busesInIsland[i]:
                island = i
                break
        subSystem = []
        for sub in network.subSysNames:
            if bus in network.busesInSubSys[sub]:
                subSystem = sub
                break
        # The elements connected to the bus to be deleted must be relocated to a new bus
        newBus = -1e12
        for l in (network.AClinesFromBus[bus] | network.AClinesToBus[bus]):
            if not(network.AClineFromTo[l][0] == bus):
                # Remove the line from the buses connected to 'bus'
                network.AClinesFromBus[network.AClineFromTo[l][0]].remove(l)
                newBus = network.AClineFromTo[l][0]
            elif not(network.AClineFromTo[l][1] == bus):
                # Remove the line from the buses connected to 'bus'
                network.AClinesToBus[network.AClineFromTo[l][1]].remove(l)
                newBus = network.AClineFromTo[l][1]

            del network.AClineFromTo[l]
            del network.AClineUBCap[l]
            del network.AClineLBCap[l]
            del network.AClineAdmt[l]
            network.lineIsland[island].remove(l)
            for sub in network.subSysNames:
                if l in network.linesInSubSys[sub]:
                    network.linesInSubSys[sub].remove(l)
            linesDeleted += 1

        del network.AClinesFromBus[bus]
        del network.AClinesToBus[bus]
        del network.DClinksFromBus[bus]
        del network.DClinksToBus[bus]
        network.busesInIsland[island].remove(bus)
        network.busesInSubSys[subSystem].remove(bus)

        #### Remove old bus and add elements to the new bus
        if (bus in network.loadBuses) or (bus in network.renewableGenBuses):
            network.load[:, network.loadHeader[newBus]] = np.add(\
                                                    network.load[:, network.loadHeader[newBus]],\
                                                    network.load[:, network.loadHeader[bus]])

        if bus in network.genBuses:
            network.genBuses.remove(bus)

            if not(newBus in network.genBuses):
                network.genBuses.add(newBus)

        if bus in network.renewableGenBuses:
            network.renewableGenBuses.remove(bus)

            if not(newBus in network.renewableGenBuses):
                network.renewableGenBuses.add(newBus)

        if bus in hydros.buses:
            hydros.buses.remove(bus)
            for h in [h for h in range(len(hydros.id)) if bus in hydros.plantBuses[h]]:
                if not(newBus in hydros.plantBuses[h]):
                    hydros.plantBuses[h][hydros.plantBuses[h].index(bus)] = newBus

                    for u in range(len(hydros.unitID[h])):
                        if hydros.unitBus[h][u] == bus:
                            hydros.unitBus[h][u] = newBus
                else:
                    # The hydro already injects power at bus 'newBus'
                    oldBusIndex = hydros.plantBuses[h].index(bus)

                    hydros.plantBusesCap[h][hydros.plantBuses[h].index(newBus)]\
                                                    += hydros.plantBusesCap[h][oldBusIndex]

                    del hydros.plantBuses[h][oldBusIndex]
                    del hydros.plantBusesCap[h][oldBusIndex]

                # Change buses of units
                for u in range(len(hydros.unitID[h])):
                    if hydros.unitBus[h][u] == bus:
                        hydros.unitBus[h][u] = newBus

                # Change buses of groups
                for group in range(len(hydros.groupsOfUnits[h])):
                    for b in range(len(hydros.busesOfEachGroup[h][group])):
                        if hydros.busesOfEachGroup[h][group][b] == bus:
                            hydros.busesOfEachGroup[h][group][b]= newBus

            if not(newBus in hydros.buses):
                hydros.buses.append(newBus)

        if bus in thermals.bus:
            for g in range(len(thermals.id)):
                if bus == thermals.bus[g]:
                    thermals.bus[g] = newBus

    return(linesDeleted)

def removeEndOfLineBusesWithInj(params, hydros, thermals, network, rank):
    '''End-of-line network buses with injections can be reallocated as long as its maximum
    injection is less than the line's capacity'''

    # Get the minimum and maximum loads of each bus during the planning horizon
    minload = np.min(network.load[0:params.T,:], axis = 0)
    maxload = np.max(network.load[0:params.T,:], axis = 0)

    #### Buses with fixed power injection
    l_ldBuses = list(network.loadBuses | network.renewableGenBuses)
    minLoadAtLoadBuses = {bus: 0 for bus in network.busID}
    maxLoadAtLoadBuses = {bus: 0 for bus in network.busID}

    for bus in l_ldBuses:
        minLoadAtLoadBuses[bus] = min(minload[network.loadHeader[bus]], 0)

    for bus in l_ldBuses:
        maxLoadAtLoadBuses[bus] = max(maxload[network.loadHeader[bus]], 0)

    for h in [h for h in range(len(hydros.id)) if hydros.turbOrPump[h] == 'Pump']:
        maxLoadAtLoadBuses[hydros.plantBuses[h][0]] +=hydros.plantMaxTurbDisc[h]*hydros.convMWm3s[h]

    maxGenOfBus = {bus: 0 for bus in network.busID}

    for g in range(len(thermals.id)):
        maxGenOfBus[thermals.bus[g]] += thermals.maxP[g]

    for h in range(len(hydros.id)):
        for b in range(len(hydros.plantBuses[h])):
            maxGenOfBus[hydros.plantBuses[h][b]] += hydros.plantBusesCap[h][b]

    done, it, busesDeleted, linesDeleted  = False, 0, 0, 0

    #### Set of candidate buses to be deleted
    candidateBuses = set()
    for bus in network.busID:
        if not(bus in network.dcBuses)\
            and ((len(network.AClinesFromBus[bus]) + len(network.AClinesToBus[bus])) <= 1):
            for l in (network.AClinesFromBus[bus] | network.AClinesToBus[bus]):
                if ((network.AClineLBCap[l] <= -(99999/params.powerBase)) and\
                    (network.AClineUBCap[l] >= 99999/params.powerBase))\
                    or\
                    (((-abs(-minLoadAtLoadBuses[bus] + maxGenOfBus[bus]) >= network.AClineLBCap[l])\
                    and (abs(-minLoadAtLoadBuses[bus]+maxGenOfBus[bus]) <= network.AClineUBCap[l]))\
                    and\
                    ((-abs(maxLoadAtLoadBuses[bus]) >= network.AClineLBCap[l])\
                    and (abs(maxLoadAtLoadBuses[bus]) <= network.AClineUBCap[l]))):

                    candidateBuses.add(bus)

    while not(done):
        it += 1
        busesToDelete, indexOfBusesToDelete = [], []

        #### Delete end-of-line buses
        linesDeleted = delEndOfLineBusesAndReassignInj(network, thermals, hydros, candidateBuses,\
                                                busesToDelete, indexOfBusesToDelete, linesDeleted)

        candidateBuses = candidateBuses - set(busesToDelete)

        busesDeleted = updateLoadAndNetwork(params, network,\
                                                indexOfBusesToDelete, busesToDelete, busesDeleted)
        ########################################################################

        #### Delete midpoint buses
        busesToDelete, indexOfBusesToDelete = [], []

        busesNoLoadNoGen = set(network.busID) - network.genBuses \
                            - network.loadBuses - set(network.refBusesOfIslands.values())\
                            - set(busesToDelete) - candidateBuses - network.dcBuses

        linesDeleted = delMidPointBuses(network, busesNoLoadNoGen, busesToDelete,\
                                        indexOfBusesToDelete, linesDeleted, [])

        busesDeleted = updateLoadAndNetwork(params, network, indexOfBusesToDelete,\
                                            busesToDelete, busesDeleted)
        ########################################################################

        # Get the minimum and maximum loads of each bus
        minload = np.min(network.load[0:params.T,:], axis = 0)
        maxload = np.max(network.load[0:params.T,:], axis = 0)

        #### Buses with fixed power injection
        l_ldBuses = list(network.loadBuses | network.renewableGenBuses)
        minLoadAtLoadBuses = {bus: 0 for bus in network.busID}
        maxLoadAtLoadBuses = {bus: 0 for bus in network.busID}

        for bus in l_ldBuses:
            minLoadAtLoadBuses[bus] =min(minload[network.loadHeader[bus]],0)

        for bus in l_ldBuses:
            maxLoadAtLoadBuses[bus] = max(maxload[network.loadHeader[bus]], 0)

        for h in range(len(hydros.id)):
            if hydros.turbOrPump[h] == 'Pump':
                maxLoadAtLoadBuses[hydros.plantBuses[h][0]] +=\
                                                    hydros.plantMaxTurbDisc[h]*hydros.convMWm3s[h]

        maxGenOfBus = {bus: 0 for bus in network.busID}

        for g in range(len(thermals.id)):
            maxGenOfBus[thermals.bus[g]] += thermals.maxP[g]

        for h in range(len(hydros.id)):
            for b in range(len(hydros.plantBuses[h])):
                maxGenOfBus[hydros.plantBuses[h][b]]+=hydros.plantBusesCap[h][b]

        for bus in network.busID:
            if not(bus in network.dcBuses)\
                    and ((len(network.AClinesFromBus[bus]) + len(network.AClinesToBus[bus])) <= 1):
                for l in (network.AClinesFromBus[bus] | network.AClinesToBus[bus]):
                    if (network.AClineLBCap[l] <= -(99999/params.powerBase)) and\
                            (network.AClineUBCap[l] >= (99999/params.powerBase))\
                        or\
                        (((-abs(-1*minLoadAtLoadBuses[bus] + maxGenOfBus[bus])\
                                                    >= network.AClineLBCap[l])\
                        and (abs(-1*minLoadAtLoadBuses[bus] + maxGenOfBus[bus])\
                                                    <= network.AClineUBCap[l]))\
                        and\
                        ((-abs(maxLoadAtLoadBuses[bus]) >= network.AClineLBCap[l])\
                        and (abs(maxLoadAtLoadBuses[bus]) <= network.AClineUBCap[l]))):

                        candidateBuses.add(bus)

        if len(candidateBuses) == 0:
            done = True

    if (rank == 0):
        print('\n\n\n')
        print(f'{it} iterations were performed')
        print(f'{busesDeleted} buses and {linesDeleted} lines were removed')
        print('\n\n\n', flush = True)
