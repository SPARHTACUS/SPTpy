# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""
from copy import deepcopy
import numpy as np
from readAndLoadData.read_csv import readNetworkExchanges

def updateLoadAndNetwork(params, network, indexOfBusesToDelete, busesToDelete, nBusesDeleted):
    '''Buses and/lines have been deleted. Update the the network object'''

    # The buses to be kept are
    indicesOfBusesToKeep=list(set([b for b in range(len(network.busID))])-set(indexOfBusesToDelete))

    indicesOfBusesToKeep.sort()
    # Update the load
    network.load = deepcopy(network.load[:, indicesOfBusesToKeep])

    for bus in busesToDelete:
        network.busName.remove(network.busName[network.busID.index(bus)])
        network.busID.remove(bus)
        del network.loadHeader[bus]

    # Now, get the buses with load
    network.loadBuses = set()
    for t in range(params.T):
        network.loadBuses = network.loadBuses | set([network.busID[b] for b in \
                                                                np.where(network.load[t] > 0)[0]])

    nBusesDeleted += len(busesToDelete)

    for b in range(len(network.busID)):
        network.loadHeader[network.busID[b]] = b

    return (nBusesDeleted)

def delEndOfLineBuses(network, busesNoLoadNoGen, busesToDelete, indexOfBusesToDelete,\
                        nLinesDeleted, linesToDelete):
    '''Delete no load and no generation buses connected to a single line'''

    for bus in busesNoLoadNoGen:
        if (len(network.AClinesFromBus[bus])+len(network.AClinesToBus[bus]))<=1:
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
            for l in (network.AClinesFromBus[bus] | network.AClinesToBus[bus]):
                if not(network.AClineFromTo[l][0] == bus):
                    # Remove the line from the buses connected to 'bus'
                    network.AClinesFromBus[network.AClineFromTo[l][0]].remove(l)

                elif not(network.AClineFromTo[l][1] == bus):
                    # Remove the line from the buses connected to 'bus'
                    network.AClinesToBus[network.AClineFromTo[l][1]].remove(l)

                linesToDelete.append(network.AClineFromTo[l])

                del network.AClineFromTo[l]
                del network.AClineUBCap[l]
                del network.AClineLBCap[l]
                del network.AClineAdmt[l]
                network.lineIsland[island].remove(l)
                for sub in network.subSysNames:
                    if l in network.linesInSubSys[sub]:
                        network.linesInSubSys[sub].remove(l)
                nLinesDeleted += 1

            del network.AClinesFromBus[bus]
            del network.AClinesToBus[bus]
            del network.DClinksFromBus[bus]
            del network.DClinksToBus[bus]
            network.busesInIsland[island].remove(bus)
            network.busesInSubSys[subSystem].remove(bus)

    return(nLinesDeleted)

def delMidPointBuses(network, busesNoLoadNoGen, busesToDelete, indexOfBusesToDelete,\
                    nLinesDeleted, linesToDelete):
    '''Delete buses with no generation and no load connected only to two lines'''

    for bus in busesNoLoadNoGen:
        if (len(network.AClinesFromBus[bus])+len(network.AClinesToBus[bus]))==2:
            busesToDelete.append(bus)
            indexOfBusesToDelete.append(network.busID.index(bus))
            # Add a new transmission line
            busesOfNewConnec = []
            cap, admt, reac = 1e12, 0, 0

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

            for l in (network.AClinesFromBus[bus] | network.AClinesToBus[bus]):
                if not(network.AClineFromTo[l][0] == bus) and \
                                                not(network.AClineFromTo[l][0] in busesOfNewConnec):
                    busesOfNewConnec.append(network.AClineFromTo[l][0])
                    # Remove the line from the buses connected to 'bus'
                    network.AClinesFromBus[network.AClineFromTo[l][0]].remove(l)

                elif not(network.AClineFromTo[l][1] == bus) and \
                                                not(network.AClineFromTo[l][1] in busesOfNewConnec):
                    busesOfNewConnec.append(network.AClineFromTo[l][1])
                    # Remove the line from the buses connected to 'bus'
                    network.AClinesToBus[network.AClineFromTo[l][1]].remove(l)

                cap = min(cap, network.AClineUBCap[l])
                reac += 1/network.AClineAdmt[l]

                linesToDelete.append(network.AClineFromTo[l])

                del network.AClineFromTo[l]
                del network.AClineUBCap[l]
                del network.AClineLBCap[l]
                del network.AClineAdmt[l]
                network.lineIsland[island].remove(l)
                for sub in network.subSysNames:
                    if l in network.linesInSubSys[sub]:
                        network.linesInSubSys[sub].remove(l)
                nLinesDeleted += 1
            admt = 1/reac

            # Add the new line. Note that this new line adopts the key l from the last deleted line
            busesOfNewConnec.sort()
            # Check if connection already exists
            found = False
            for l2 in network.AClinesFromBus[busesOfNewConnec[0]]:
                if (network.AClineFromTo[l2][1] == busesOfNewConnec[1]):
                    # Then the line already exists
                    found = True
                    dthetaMax = 1e12
                    if abs(cap/admt) < dthetaMax:
                        dthetaMax = cap/admt
                    if abs(network.AClineUBCap[l2]/network.AClineAdmt[l2]) < abs(dthetaMax):
                        dthetaMax = network.AClineUBCap[l2]/network.AClineAdmt[l2]

                    network.AClineUBCap[l2] = abs(dthetaMax)*abs((admt + network.AClineAdmt[l2]))
                    network.AClineLBCap[l2] = -1*network.AClineUBCap[l2]
                    network.AClineAdmt[l2] = network.AClineUBCap[l2]/dthetaMax
                    break
            if not(found):
                network.AClineFromTo[l] = (busesOfNewConnec[0], busesOfNewConnec[1])
                network.AClineUBCap[l] = cap
                network.AClineLBCap[l] = -1*cap
                network.AClineAdmt[l] = admt
                network.AClinesFromBus[busesOfNewConnec[0]].add(l)
                network.AClinesToBus[busesOfNewConnec[1]].add(l)
                network.lineIsland[island].append(l)
                network.linesInSubSys[subSystem].append(l)
                nLinesDeleted -= 1

            del network.AClinesFromBus[bus]
            del network.AClinesToBus[bus]
            del network.DClinksFromBus[bus]
            del network.DClinksToBus[bus]
            network.busesInIsland[island].remove(bus)
            network.busesInSubSys[subSystem].remove(bus)
    return(nLinesDeleted)

def buildDCsystem(params, hydros, thermals, network, rank):
    '''Build the DC network model'''

    network.subsystems = list(set(network.submarketID.values()))

    counter = 0
    for l in network.AClineFromTo.keys():
        f = network.submarketID[network.AClineFromTo[l][0]]
        t = network.submarketID[network.AClineFromTo[l][1]]
        if f != t:
            # The line connects different subsystems
            network.flowsBetweenSubsystems[f, t, counter] = network.AClineUBCap[l]
            counter += 1

    for l in network.DClinkFromTo.keys():
        f = network.submarketID[network.DClinkFromTo[l][0]]
        t = network.submarketID[network.DClinkFromTo[l][1]]
        if f != t:
            # The line connects different subsystems
            network.flowsBetweenSubsystems[f, t, counter] = network.DClinkCap[l]
            counter += 1

    for t in range(params.T):
        for subsys in network.subsystems:
            network.subsysLoad[t, subsys] = 0

        for bus in network.busID:
            network.subsysLoad[t, network.submarketID[bus]] +=\
                                                        network.load[t,network.loadHeader[bus]]

    # Get the thermal generating units that are in each subsystem
    network.listOfThermalsInSubsys = {subsys: [] for subsys in network.subsystems}
    for subsys in network.subsystems:
        for g in range(len(thermals.id)):
            if network.submarketID[thermals.bus[g]] == subsys:
                network.listOfThermalsInSubsys[subsys].append(g)

    # Get the hydro plants that are in each subsystem
    network.listOfHydrosInSubsys = {subsys: [] for subsys in network.subsystems}

    for h in range(len(hydros.id)):
        found = False
        for bus in hydros.plantBuses[h]:
            for subsys in network.subsystems:
                if network.submarketID[bus] == subsys:
                    network.listOfHydrosInSubsys[subsys].append(h)
                    found = True
                    break
            if found:
                break

    if params.reduceSystem:
        done, it, nBusesDeleted, nLinesDeleted = False, 0, 0, 0

        #### Set of candidate buses to be deleted
        busesNoLoadNoGen = set(network.busID) - network.genBuses \
                            - network.loadBuses - set(network.refBusesOfIslands.values())

        allBusesDeleted, allLinesDeleted = [], []

        while not(done):
            it += 1

            busesToDelete, indexOfBusesToDelete, linesToDelete = [], [], []

            nLinesDeleted = delEndOfLineBuses(network, busesNoLoadNoGen,\
                                busesToDelete, indexOfBusesToDelete, nLinesDeleted, linesToDelete)

            busesNoLoadNoGen = busesNoLoadNoGen - set(busesToDelete)

            nLinesDeleted = delMidPointBuses(network, busesNoLoadNoGen,\
                                busesToDelete, indexOfBusesToDelete, nLinesDeleted, linesToDelete)

            busesNoLoadNoGen = busesNoLoadNoGen - set(busesToDelete)

            nBusesDeleted = updateLoadAndNetwork(params, network, indexOfBusesToDelete,\
                                                busesToDelete, nBusesDeleted)

            allBusesDeleted = allBusesDeleted + busesToDelete
            allLinesDeleted = allLinesDeleted + linesToDelete

            if len(busesToDelete) == 0:
                done = True

        if (rank == 0):
            print('\n\n\n' + f"{it} iterations were performed")
            print(f"{nBusesDeleted} buses and {nLinesDeleted} lines were removed" + "\n\n\n")

    return()
