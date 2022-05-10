# -*- coding: ISO-8859-1 -*-
"""
@author: Colonetti
"""

from datetime import date
import csv
from stat import S_ENFMT
import numpy as np

def grossLoadAndRenewableGen(filenameGrossLoad, filenameRenewableGen, params, network, rankWorld):
    '''Read gross load and renewable generation. compute the net load and print it'''

    network.load = np.zeros((params.T, len(network.busID)), dtype = 'd')
    renewableGen = np.zeros((params.T, len(network.busID)), dtype = 'd')

    f = open(filenameGrossLoad, 'r', encoding = 'utf-8')
    foundBus = [False for b in range(len(network.busID))]
    reader = csv.reader(f, delimiter = ';')
    row = next(reader) # <BEGIN>
    row = next(reader) # Header
    row = next(reader) # either the first bus or end
    while row[0].strip() != '</END>':
        try:
            b = network.busName.index(row[0].strip())
        except ValueError as err:
            raise ValueError(f'Bus {row[0].strip()} is not in the system') from err

        for t in range(params.T):
            network.load[t, b] = float(row[1 + t].strip())

        foundBus[b] = True
        row = next(reader) # next bus or end

    for b in range(len(network.busID)):
        if not(foundBus[b]):
            raise Exception('No load has been found for bus ' + network.busName[b])

    f = open(filenameRenewableGen, 'r', encoding = 'utf-8')
    foundBus = [False for b in range(len(network.busID))]
    reader = csv.reader(f, delimiter = ';')
    row = next(reader) # <BEGIN>
    row = next(reader) # Header
    row = next(reader) # either the first bus or end
    while row[0].strip() != '</END>':
        try:
            b = network.busName.index(row[0].strip())
        except ValueError as err:
            raise ValueError(f'Thermal unit {row[1].strip()} is not in the system') from err

        for t in range(params.T):
            renewableGen[t, b] = float(row[1 + t].strip())

        foundBus[b] = True
        row = next(reader) # next bus or end

    for b in range(len(network.busID)):
        if not(foundBus[b]):
            raise Exception('No renewable generation has been found for bus ' + network.busName[b])

    network.load = np.multiply(np.subtract(network.load, renewableGen), 1/params.powerBase)

    if rankWorld == 0:
        f = open(params.outputFolder + 'net load - ' + params.ps\
                                                    + ' - case ' + str(params.case) + '.csv', 'w',\
                                                        encoding = 'utf-8')
        f.write('<BEGIN>\nBus/Hour;')

        for t in range(params.T):
            f.write(str(t) + ';')
        f.write('\n')
        b = 0
        for _ in network.busID:
            f.write(network.busName[b] + ';')
            for t in range(params.T):
                f.write(str(network.load[t][b]*params.powerBase) + ';')
            b += 1
            f.write('\n')

        f.write('</END>')
        f.close()
        del f

    for t in range(params.T):
        network.loadBuses = network.loadBuses | set([network.busID[b] for b in \
                                                                np.where(network.load[t] > 0)[0]])

        network.genBuses = network.genBuses | set([network.busID[b] for b in \
                                                                np.where(network.load[t] < 0)[0]])
        network.renewableGenBuses = network.renewableGenBuses | set([network.busID[b] for b in \
                                                                np.where(network.load[t] < 0)[0]])

    return()

def resetGenCostsOfThermals(filename, params, thermals):
    '''Reset the unitary generation costs for the thermal units'''

    f = open(filename, 'r', encoding = 'ISO-8859-1')
    reader = csv.reader(f, delimiter = ';')
    row = next(reader) # <BEGIN>
    row = next(reader) # Header

    row = next(reader) # Either the first hydro or end

    while row[0].strip() != '</END>':
        try:
            g = thermals.name.index(row[1].strip())
        except ValueError as err:
            raise ValueError(f'Thermal unit {row[1].strip()} is not in the system') from err

        thermals.genCost[g] = params.discretization*\
                                            params.powerBase*float(row[2].strip())*params.scalObjF

        row = next(reader) # next thermal unit or end

    f.close()
    del f

    return()

def resetVolumeBounds(filename, params, hydros):
    '''Reset bounds on reservoir volumes'''

    f = open(filename, 'r', encoding = 'ISO-8859-1')
    reader = csv.reader(f, delimiter = ';')
    row = next(reader) # <BEGIN>
    row = next(reader) # Header

    row = next(reader) # Either the first hydro or end

    while row[0].strip() != '</END>':
        try:
            h = hydros.name.index(row[1].strip())
        except ValueError as err:
            raise ValueError(f'Reservoir {row[1].strip()} is not in the system') from err

        assert float(row[2].strip()) >= 0 and float(row[3].strip()) >= 0,\
                                                    'Negative reservoir volumes are not allowed'

        hydros.maxVol[h] = float(row[2].strip())
        hydros.minVol[h] = float(row[3].strip())

        row = next(reader) # next hydro or end

    f.close()
    del f

    return()

def readPreviousStateOfHydroPlants(filenameIniVol, filenamePrevDisch, params, hydros):
    '''Read initial reservoir volume and previous discharges'''

    f = open(filenameIniVol, 'r', encoding = 'ISO-8859-1')
    reader = csv.reader(f, delimiter = ';')
    row = next(reader) # <BEGIN>
    row = next(reader) # Header

    foundHydro = [False for h in range(len(hydros.id))]

    row = next(reader) # Either the first hydro or end

    while row[0].strip() != '</END>':
        try:
            h = hydros.name.index(row[1].strip())
        except ValueError as err:
            raise ValueError(f'Reservoir {row[1].strip()} is not in the system') from err

        assert float(row[2].strip()) >= 0, 'Negative reservoir volumes are not allowed'

        hydros.V0[h] = float(row[2].strip())
        foundHydro[h] = True

        row = next(reader) # next hydro or end

    f.close()
    del f

    for h in range(len(hydros.id)):
        if not(foundHydro[h]):
            s = 'No initial reservoir volume has been found for hydro plant ' + hydros.name[h]
            raise Exception(s)

    # now read the previous discharges
    hydros.spil0 = {(h, t): 0 for h in range(len(hydros.id)) for t in range(-1440, 0, 1)}

    f = open(filenamePrevDisch, 'r', encoding = 'ISO-8859-1')
    reader = csv.reader(f, delimiter = ';')
    row = next(reader) # <BEGIN>
    row = next(reader) # Header

    foundHydro = [False for h in range(len(hydros.id))]

    row = next(reader) # Either the first hydro or end

    while row[0].strip() != '</END>':
        try:
            h = hydros.name.index(row[1].strip())
        except ValueError as err:
            raise ValueError(f'Reservoir {row[1].strip()} is not in the system') from err

        i = 0
        for t in range(-1440, 0, 1):
            assert float(row[2 + i].strip()) >= 0, 'Negative discharges are not allowed'
            hydros.spil0[h, t] = float(row[2 + i].strip())
            i += 1
        foundHydro[h] = True

        row = next(reader) # next hydro or end

    f.close()
    del f

    for h in range(len(hydros.id)):
        if not(foundHydro[h]):
            s = 'No previous discharge has been found for hydro plant ' + hydros.name[h]
            raise Exception(s)
    return()

def readInflows(filename, params, hydros):
    '''Read inflows in m3/s'''

    hydros.inflows = np.zeros((len(hydros.id), params.T), dtype = 'd')

    foundHydro = [False for h in range(len(hydros.id))]

    f = open(filename, 'r', encoding = 'ISO-8859-1')
    reader = csv.reader(f, delimiter = ';')

    while True:
        try:
            row = next(reader)
        except StopIteration:
            break

        try:
            h = hydros.name.index(row[0].strip())
        except ValueError as err:
            raise ValueError(f'Reservoir {row[0].strip()} is not in the system') from err

        foundHydro[h] = True

        for t in range(params.T):
            assert float(row[1 + t]) >= 0, f'Negative inflow for plant {hydros.name[h]} at time {t}'
            hydros.inflows[h, t] = float(row[1 + t])

    for h in range(len(hydros.id)):
        if not(foundHydro[h]):
            s = 'No inflow has been found for hydro plant ' + hydros.name[h]
            raise Exception(s)

    return()

def readNetworkExchanges(filename, params):
    '''Get the subsystem to which each bus belongs'''
    def readBusCorresponde(row, reader, params):
        row = next(reader)  # header
        header = {}
        header['ID'] = row.index('ID')
        header['System ID'] = row.index('System ID')

        busCorrespondence = {}

        row = next(reader)#either the first bus or </CorrespondenceBusSubsystem>
        while not(row[0] == '</CorrespondenceBusSubsystem>'):
            busCorrespondence[int(row[header['ID']].strip())] =int(row[header['System ID']].strip())
            row = next(reader) # next bus or </CorrespondenceBusSubsystem>
        return(busCorrespondence)

    f = open(filename, 'r', encoding = 'ISO-8859-1')
    reader = csv.reader(f, delimiter = ';')

    row = next(reader)  # '<BEGIN>'
    row = next(reader)  # either </END>, or <Hydro plants>, or <Thermal plants>

    while not(row[0] == '</END>'):
        if (row[0] == '<CorrespondenceBusSubsystem>'):
            busCorrespondence = readBusCorresponde(row, reader, params)
        row = next(reader)
    f.close()
    return(busCorrespondence)

def readTrajectories(filename, params, thermals):
    '''Read shut-down and start-up tracjetories'''

    f = open(filename, 'r', encoding = 'ISO-8859-1')
    reader = csv.reader(f, delimiter = ';')

    row = next(reader)  # <BEGIN>
    row = next(reader)  # Header

    header = {}
    header['ID'] = row.index('ID')
    header['Name'] = row.index('Name')
    header['Traj'] = row.index('Trajectory')
    header['Step0'] = row.index('Step 0 (MW)')

    row = next(reader)  # Either </END> or first generator

    while row[0].strip() != '</END>':
        try:
            g = thermals.name.index(row[header['Name']])
            found = True
        except ValueError:
            found = False

        if found:
            if row[header['Traj']] == 'Start-up':
                for step in row[header['Step0']:]:
                    if len(step) > 0 and float(step) != 0:
                        thermals.startUpTraj[g].append(float(step)/params.powerBase)
            elif row[header['Traj']] == 'Shut-down':
                for step in row[header['Step0']:]:
                    if len(step) > 0 and float(step) != 0:
                        thermals.shutDownTraj[g].append(float(step)/params.powerBase)
            else:
                print('I dont know what ' + row[header['Traj']] + ' is',flush=True)

        row = next(reader)

    f.close()
    del f
    return()

def readBoundsOnGenOfHydros(filename, params, hydros):
    '''Read bounds on generation of groups of hydro units'''

    def appendToListOfConstraints(params, hydros, listOfConstraints, row, reader):
        '''Append a new constraint to a list of thermal generation constraints'''

        listOfConstraints.append([[],[],[], 0]) # first position is the list of plants
                                                # second is the list of groups for each plant in the
                                                # constraint
                                                # third is the list of periods
                                                # fourth is the RHS of the constraint

        row = next(reader) # hydro plants in constraint
        # check for leading zeros
        for h in row[1].strip().split(','):
            if len(h) > 1 and h[0] == '0':
                raise Exception('One or more plants is identified with a leading zero in '+\
                                'the input file of bounds on generation of groups of plants')

        plants = [hName.strip() for hName in row[1].strip().split(',') if hName.strip() != '']
        assert len(plants) ==len(set(plants)),'A hydro plant appears more than once in a constraint'
        assert (set(plants) & set(hydros.name)) == set(plants),\
                                                'Not all plants in the constraint are in the system'
        listOfConstraints[-1][0] = [hydros.name.index(hName) for hName in plants]

        row = next(reader) # groups of units
        groupsForEachplant = [grp for grp in row[1].strip().split(',') if grp != '']

        assert len(groupsForEachplant) == len(plants),\
                        'Groups of units are missing for one of more hydro plants in a constraint'

        for plant in range(len(plants)):
            groupsForEachplant[plant] = groupsForEachplant[plant].replace('[', '')
            groupsForEachplant[plant] = groupsForEachplant[plant].replace(']', '')
            groupsForEachplant[plant] = groupsForEachplant[plant].strip().split(',')
            groupsForEachplant[plant] = [groupName.strip()\
                                                        for groupName in groupsForEachplant[plant]]

        listOfConstraints[-1][1] = groupsForEachplant

        row = next(reader) # times in constraint
        times = list(range(int(row[1].strip()), int(row[2].strip()) + 1, 1))
        assert (set(times) & set(range(params.T))) == set(times),\
                                'All time periods in constraint must be within the planning horizon'
        listOfConstraints[-1][2] = times

        row = next(reader) # RHS
        listOfConstraints[-1][3] = float(row[1].strip())/params.powerBase
        return()

    f = open(filename, 'r', encoding = 'ISO-8859-1')
    reader = csv.reader(f, delimiter = ';')

    row = next(reader)  # <BEGIN>
    row = next(reader)  # either <constr> to indicate the first constraint or </END> to finish

    while row[0].strip() != '</END>':
        # each constraint should have exactly 5 fields
        row = next(reader) # Comment. Skip
        row = next(reader) # Constraint type
        assert row[1].strip() in ('UB', 'LB', 'EQ'), 'Invalid constraint type'

        if row[1].strip() == 'UB':
            appendToListOfConstraints(params, hydros, hydros.maxGen, row, reader)

        elif row[1].strip() == 'LB':
            appendToListOfConstraints(params, hydros, hydros.minGen, row, reader)

        else:
            appendToListOfConstraints(params, hydros, hydros.equalityConstrs, row, reader)

        row = next(reader)
        if row[0].strip() != '</constr>':
            raise Exception('Error reading constraint. There seems to be less fields than required')

        row = next(reader)  # either a new constraint or </END>

    f.close()
    del f
    return()

def readBoundsOnGenOfThermals(filename, params, thermals):
    '''Read bounds on generation of groups of thermal units'''

    def appendToListOfConstraints(params, thermals, listOfConstraints, row, reader):
        '''Append a new constraint to a list of thermal generation constraints'''

        listOfConstraints.append([[], [], 0])   # first position is the list of units
                                                # second is the list of periods
                                                # third is the RHS of the constraint

        row = next(reader) # thermal units in constraint
        # check for leading zeros
        for u in row[1].strip().split(','):
            if len(u) > 1 and u[0] == '0':
                raise Exception('One or more units is identified with a leading zero in '+\
                                'the input file of bounds on generation of groups of units')

        units = [int(u) for u in row[1].strip().split(',') if u != '']
        assert len(units) == len(set(units)),'A thermal unit appears more than once in a constraint'
        assert (set(units) & set(thermals.id)) == set(units),\
                                                'Not all units in the constraint are in the system'
        listOfConstraints[-1][0] = units

        row = next(reader) # times in constraint
        times = list(range(int(row[1].strip()), int(row[2].strip()) + 1, 1))
        assert (set(times) & set(range(params.T))) == set(times),\
                                'All time periods in constraint must be within the planning horizon'
        listOfConstraints[-1][1] = times

        row = next(reader) # RHS
        listOfConstraints[-1][2] = float(row[1].strip())/params.powerBase
        return()

    f = open(filename, 'r', encoding = 'ISO-8859-1')
    reader = csv.reader(f, delimiter = ';')

    row = next(reader)  # <BEGIN>
    row = next(reader)  # either <constr> to indicate the first constraint or </END> to finish

    while row[0].strip() != '</END>':
        # each constraint should have exactly 5 fields
        row = next(reader) # Comment. Skip
        row = next(reader) # Constraint type
        assert row[1].strip() in ('UB', 'LB', 'EQ'), 'Invalid constraint type'

        if row[1].strip() == 'UB':
            appendToListOfConstraints(params, thermals, thermals.maxGen, row, reader)

        elif row[1].strip() == 'LB':
            appendToListOfConstraints(params, thermals, thermals.minGen, row, reader)

        else:
            appendToListOfConstraints(params, thermals, thermals.equalityConstrs, row, reader)

        row = next(reader)
        if row[0].strip() != '</constr>':
            raise Exception('Error reading constraint. There seems to be less fields than required')

        row = next(reader)  # either a new constraint or </END>

    f.close()
    del f
    return ()

def readCostToGoFunction(filename, params, hydros):
    '''Read the cost-to-go function'''

    f = open(filename, 'r', encoding = 'ISO-8859-1')
    reader = csv.reader(f, delimiter = ';')

    row = next(reader)  # <BEGIN>
    row = next(reader)  # Header
    header = {}
    for h in range(len(hydros.id)):
        try:
            header[hydros.nameDECOMP[h]] = row.index(hydros.nameDECOMP[h])
        except ValueError as err:
            raise ValueError(f'Hydro plant {hydros.name[h]} has not ' +\
                                'been found in the cost-to-go function file') from err

    header['rhs'] = row.index('RHS ($)')
    row = next(reader)  # First hydro or </END>
    while not(row[0] == '</END>'):
        hydros.CTFrhs.append(float(row[header['rhs']])*params.scalObjF)
        hydros.CTF.append([])
        for h in range(len(hydros.id)):
            hydros.CTF[-1].append((float(row[header[hydros.nameDECOMP[h]]]))*params.scalObjF)
        row = next(reader)
    f.close()
    del f
    # Check if there is any cut that appears more than once
    uniqueCuts = []
    for c in range(len(hydros.CTFrhs)):
        if [hydros.CTF[c] + [hydros.CTFrhs[c]]] not in uniqueCuts:
            uniqueCuts.append([hydros.CTF[c] + [hydros.CTFrhs[c]]])
    hydros.CTF = [[uniqueCuts[c][0][h] for h in range(len(hydros.id))]\
                                                                for c in range(len(uniqueCuts))]
    hydros.CTFrhs = [uniqueCuts[c][0][-1] for c in range(len(uniqueCuts))]

    return ()

def readIniStateThermal(filename, params, thermals):
    '''Read the initial state of the thermal units'''

    f = open(filename, 'r', encoding = 'ISO-8859-1')
    reader = csv.reader(f, delimiter = ';')

    row = next(reader)  # '<BEGIN>'
    row = next(reader)  # <Thermal plants>
    row = next(reader)  # header

    header = {}
    header['ID'] = row.index('ID')
    header['Name'] = row.index('Name')
    header['iniGen'] = row.index('Generation in time t = -1 in MW')
    s = 'State in t = -1. Either 1, if up, or 0, if down'
    header['iniState'] = row.index(s)
    s = 'Number of hours (> 0) in the state of t = -1'
    header['nHoursInIniState'] = row.index(s)
    s = 'Start-up trajectory (TRUE or FALSE)'
    header['inStartupTraj'] = row.index(s)
    s = 'Shut-down trajectory (TRUE or FALSE)'
    header['inShutDownTraj'] = row.index(s)

    row = next(reader)  # either first thermal or </Thermal plants>
    while not(row[0] == '</Thermal plants>'):
        plantID = int(row[header['ID']])
        plantName = row[header['Name']].strip()
        iniGen = float(row[header['iniGen']])
        iniState = int(row[header['iniState']])
        nHoursInIniState = int(row[header['nHoursInIniState']])

        inStartUp = True if row[header['inStartupTraj']].strip() in ('TRUE', 'True') else False
        inShutDown = True if row[header['inShutDownTraj']].strip() in ('TRUE', 'True') else False

        try:
            g = thermals.name.index(plantName)
        except ValueError as err:
            raise ValueError(f'Thermal unit {plantName} is not in the system') from err

        if not(inStartUp) and not(inShutDown) and\
                                (iniState == 1 and (iniGen < thermals.minP[g]*params.powerBase)):
            raise Exception(f'Thermal plant {plantName} is not in a start-up or shut-down '+\
                            'trajectory and yet its initial generation is less than its minimum')

        thermals.tg0[g] = iniGen/params.powerBase
        thermals.state0[g] = iniState
        thermals.nHoursInPreState[g] = nHoursInIniState
        thermals.inStartUpTraj[g] = inStartUp
        thermals.inShutDownTraj[g] = inShutDown
        row = next(reader)
    f.close()
    return()

def readGenerators(filename, params, network, hydros, thermals):
    '''Read data of hydro, thermal and wind generators'''

    def readHydros(row, reader, params, hydros):
        '''Read data of hydropower plants'''
        row = next(reader)  # header
        header = {}
        header['ID'] = row.index('ID')
        header['Name'] = row.index('Name')
        header['MinVol'] = row.index('Minimum reservoir volume (hm3)')
        header['MaxVol'] = row.index('Maximum reservoir volume (hm3)')
        header['Downriver reservoir'] = row.index('Name of downriver reservoir')
        header['WaterTravT'] = row.index('Water travelling time (h)')
        header['Run-of-river'] = row.index('Run-of-river plant? TRUE or FALSE')
        header['MinForebay'] = row.index('Minimum forebay level (m)')
        header['MaxForebay'] = row.index('Maximum forebay level (m)')
        header['MaxSpil'] = row.index('Maximum spillage (m3/s)')
        header['Basin'] = row.index('Basin')
        header['InflOfSpill'] =\
                        row.index('Influence of spillage on the HPF? Yes or No')
        header['MaxSpilHPF'] = row.index('Maximum spillage - HPF')
        header['DRTransfer'] =row.index('Downriver plant of transfer discharge')
        header['MaxTransfer'] = row.index('Maximum transfer discharge (m3/s)')
        header['TransferTravelTime'] =\
                row.index('Water travel time in the transfer process (h)')

        header['DRPump'] = row.index('Downriver reservoir of pump units')
        header['UPRPump'] = row.index('Upriver reservoir of pump units')
        header['PumpTravelTime'] = row.index('Water travel time in pumping (h)')

        header['idDECOMP'] = row.index('ID DECOMP')
        header['nameDECOMP'] = row.index('Name DECOMP')

        header['Subbasin'] = row.index('Subbasin')

        row = next(reader)  # either the first hydro plant or </Hydro plants>
        while not(row[0] == '</Hydro plants>'):
            hydros.addNewHydro(params, row, header)
            row = next(reader) # next hydro plant or </Hydro plants>

        for h in range(len(hydros.id)):
            if not(hydros.downRiverPlantName[h] == '0'):
                hydros.downRiverPlantID.append([hydros.name.index(hydros.downRiverPlantName[h])])
            else:
                hydros.downRiverPlantID.append([])

        for h in range(len(hydros.id)):
            try:
                hydros.upRiverPlantNames[hydros.downRiverPlantID[h][0]].append(hydros.name[h])
                hydros.upRiverPlantIDs[hydros.downRiverPlantID[h][0]].append(h)
            except:
                pass

        for h in range(len(hydros.id)):
            if hydros.turbOrPump[h] == 'Pump':
                # water will go from hydros.dnrOfPumpsID[h] (downriver) to hydros.uprOfPumpsID[h]
                # (upriver)
                hydros.downRiverPumps[hydros.uprOfPumpsID[h]].append(hydros.dnrOfPumpsID[h])

        for h in range(len(hydros.id)):
            if not(hydros.downRiverPlantTransferName[h] == '0'):
                hydros.downRiverTransferPlantID.append([hydros.name.index(\
                                                            hydros.downRiverPlantTransferName[h])])
            else:
                hydros.downRiverTransferPlantID.append([])

        for h in range(len(hydros.id)):
            try:
                hydros.upRiverTransferPlantID[hydros.downRiverTransferPlantID[h][0]].append(h)
            except:
                pass

        for h in range(len(hydros.id)):
            if not(hydros.dnrOfPumps[h] == '0'):
                hydros.dnrOfPumpsID.append(hydros.name.index(hydros.dnrOfPumps[h]))
            else:
                hydros.dnrOfPumpsID.append([])

            if not(hydros.uprOfPumps[h] == '0'):
                hydros.uprOfPumpsID.append(hydros.name.index(hydros.uprOfPumps[h]))
            else:
                hydros.uprOfPumpsID.append([])

        return()

    def readThermals(row, reader, params, thermals):
        '''Read data of thermal generators'''
        row = next(reader)  # header
        header = {}
        header['ID'] = row.index('ID')
        header['Name'] = row.index('Name')
        header['minP'] = row.index('Minimum power output (MW)')
        header['maxP'] = row.index('Maximum power output (MW)')
        header['genCost'] = row.index('Unitary linear cost ($/MW)')
        header['rampUp'] = row.index('Ramp-up limit (MW/h)')
        header['rampDown'] = row.index('Ramp-down limit (MW/h)')
        header['minUp'] = row.index('Minimum up-time (h)')
        header['minDown'] = row.index('Minimum down-time (h)')
        header['bus'] = row.index('Bus id')
        header['constCost'] = row.index('Constant cost ($)')
        header['stUpCost'] = row.index('Start-up cost ($)')
        header['stDwCost'] = row.index('Shut-down cost ($)')
        header['plantsIDDESSEM'] = row.index('Plants ID in DESSEM')
        header['plantsNameDESSEM'] = row.index('Plants name in DESSEM')
        header['unitsIDDESSEM'] = row.index('Units ID in DESSEM')

        row = next(reader) # either the first thermal plant or </Thermal plants>
        while not(row[0] == '</Thermal plants>'):
            thermals.addNewThermal(params, row, header)
            row = next(reader) # next thermal plant or </Thermal plants>

        return()

    f = open(filename, 'r', encoding = 'ISO-8859-1')
    reader = csv.reader(f, delimiter = ';')

    row = next(reader)  # '<BEGIN>'
    row = next(reader)  # either </END>, or <Hydro plants>, or <Thermal plants>

    while not(row[0] == '</END>'):
        if (row[0] == '<Hydro plants>'):
            readHydros(row, reader, params, hydros)

        elif (row[0] == '<Thermal plants>'):
            readThermals(row, reader, params, thermals)

        elif (row[0] == '<Deficit cost>'):
            row = next(reader) # Header of the deficit cost
            row = next(reader) # Deficit cost
            network.deficitCost = float(row[0].strip())*params.powerBase*params.scalObjF*\
                                                        params.baseTimeStep
            row = next(reader) # </Deficit cost>

        row = next(reader)
    f.close()
    return ()

def readHydroGeneratingUnits(filename, params, hydros):
    'Read the hydro generating units data'
    f = open(filename, 'r', encoding = 'ISO-8859-1')
    reader = csv.reader(f, delimiter = ';')

    row = next(reader)  # '<BEGIN>'
    row = next(reader)  # '<Hydro generating units>'
    row = next(reader)  # header

    header = {}
    header['ID'] = row.index('ID')
    header['Name'] = row.index('Name')
    header['plantID'] = row.index('Hydro plant ID')
    header['plantName'] = row.index('Hydro plant name')
    header['groupID'] = row.index('Group ID')
    header['groupName'] = row.index('Group name')
    header['bus'] = row.index('Bus ID')
    header['minGen'] = row.index('Minimum generation (MW)')
    header['maxGen'] = row.index('Maximum generation (MW)')
    header['minTurbDisc'] = row.index('Minimum turbine discharge (m3/s)')
    header['maxTurbDisc'] = row.index('Maximum turbine discharge (m3/s)')
    header['Turbinetype'] = row.index('Turbine type')
    header['turbineOrPUMP'] = row.index('Turbine or Pump?')
    header['conversionMWm3s'] =row.index('Conversion rate MW/(m3/s/) for pumps')

    row = next(reader)  # either first unit or </END>

    while not(row[0] == '</Hydro generating units>'):
        genUnitsID = int(row[header['ID']])
        genUnitsName = row[header['Name']].strip()
        plantID = int(row[header['plantID']])
        group = row[header['groupName']]
        bus = int(row[header['bus']])
        turbType = row[header['Turbinetype']]
        minPower = float(row[header['minGen']])/params.powerBase
        maxPower = float(row[header['maxGen']])/params.powerBase
        minTurbDisc = float(row[header['minTurbDisc']])
        maxTurbDisc = float(row[header['maxTurbDisc']])
        turbOrPump = row[header['turbineOrPUMP']].strip()
        conversionMWm3s = float(row[header['conversionMWm3s']])/(params.powerBase)

        h = hydros.id.index(plantID)
        # The following are attributes of individual generating units
        hydros.unitID[h].append(genUnitsID)
        hydros.unitName[h].append(genUnitsName)
        hydros.unitGroup[h].append(group)
        hydros.unitBus[h].append(bus)
        hydros.turbType[h].append(turbType)

        if not(bus in hydros.plantBuses[h]):
            hydros.plantBuses[h].append(bus)
            hydros.plantBusesCap[h].append(maxPower)
        else:
            hydros.plantBusesCap[h][hydros.plantBuses[h].index(bus)] += maxPower

        hydros.groupOfGivenUnit[h].append(-100000)  # The group to which this
                                                    # unit belongs
        if not(group in hydros.groupsOfUnits[h]):
            hydros.groupsOfUnits[h].append(group)
            hydros.unitsInGroups[h].append([])
            hydros.unitsInGroups[h][-1].append(len(hydros.unitID[h]) - 1)
            hydros.groupOfGivenUnit[h][-1] = len(hydros.groupsOfUnits[h]) - 1
            # The bus to each the units in this group are connected to
            hydros.busesOfEachGroup[h].append([bus])
        else:
            hydros.unitsInGroups[h][hydros.groupsOfUnits[h].index(group)]\
                                                                .append(len(hydros.unitID[h]) - 1)
            hydros.groupOfGivenUnit[h][-1] = hydros.groupsOfUnits[h].index(group)
            if not(bus in hydros.busesOfEachGroup[h][hydros.groupsOfUnits[h].index(group)]):
                hydros.busesOfEachGroup[h][hydros.groupsOfUnits[h].index(group)].append(bus)

        if turbOrPump == 'Pump':
            #### Assuming there is a maximum of one pump at each plant
            hydros.turbOrPump[h] = 'Pump'
            hydros.convMWm3s[h] = conversionMWm3s

        hydros.unitMinPower[h].append(minPower)
        hydros.unitMaxPower[h].append(maxPower)
        hydros.unitMinTurbDisc[h].append(minTurbDisc)
        hydros.unitMaxTurbDisc[h].append(maxTurbDisc)
        # The following attributes are related to the plant as a whole, and
        # they are computed by summing the attributes of the generating units
        hydros.plantMinPower[h] = min(minPower, hydros.plantMinPower[h])
        hydros.plantMaxPower[h] += maxPower
        hydros.plantMinTurbDisc[h] = min(minTurbDisc,hydros.plantMinTurbDisc[h])
        hydros.plantMaxTurbDisc[h] += maxTurbDisc

        if turbOrPump == 'Pump':
            hydros.plantMinPower[h] = -1*hydros.convMWm3s[h]*hydros.plantMaxTurbDisc[h]
            hydros.unitMinPower[h][-1] = -3e2
            hydros.unitMaxPower[h][-1] = 3e2

        row = next(reader)

    f.close()

    #### Some reservoirs have no turbines. Thus, change their minimum discharge
    #### and minimum power output to 0
    for h in range(len(hydros.id)):
        if hydros.plantMinTurbDisc[h] == 1e12:
            hydros.plantMinTurbDisc[h] = 0
        if hydros.plantMinPower[h] == 1e12:
            hydros.plantMinPower[h] = 0

    # First, get the buses with generating units. Get the bus of the first generating unit
    hydros.buses = [bus for h in range(len(hydros.id)) for bus in hydros.plantBuses[h]]

    return ()

def readAggregHPF(filename, params, hydros):
    '''Read the three-dimensional HPF'''

    f = open(filename, 'r', encoding = 'ISO-8859-1')
    reader = csv.reader(f, delimiter = ';')

    row = next(reader)  # <BEGIN>
    row = next(reader)  # First hydro or </END>

    while not(row[0] == '</END>'):
        row = next(reader)  # ID
        row = next(reader)  # Hydro plant name

        try:
            h = hydros.name.index(row[0].strip())
        except ValueError:
            h = -1000
        row = next(reader)  # <HPF>
        row = next(reader)  # Header
        row = next(reader)  # Either first cut or </HPF>

        if h == -1000:
            while not(row[0] == '</Hydro>'):
                while not(row[0] == '</HPF>'):

                    row = next(reader)  # Either next cut or </HPF>

                row = next(reader)  # <Underestimator>
                row = next(reader)  # Header
                row = next(reader)  # Either </Underestimator> or underestimator
                while not(row[0] == '</Underestimator>'):
                    row = next(reader)
                row = next(reader)
        else:
            while not(row[0] == '</Hydro>'):
                while not(row[0] == '</HPF>'):
                    hydros.A0[h].append((float(row[0]))/params.powerBase)
                    hydros.A1[h].append((float(row[1]))/params.powerBase)
                    hydros.A2[h].append((float(row[2]))/params.powerBase)
                    hydros.A3[h].append(float(row[3])/params.powerBase)
                    row = next(reader)  # Either next cut or </HPF>

                row = next(reader)

        row = next(reader)
    f.close()
    del f

    for h in range(len(hydros.id)):
        if (hydros.plantMaxPower[h] > 0):
            if len(hydros.A0[h]) == 0:
                s = 'The hydropower function of ' + hydros.name[h] + ' has not been found'
                raise Exception(s)
    return()

def readNetwork(filename, params, network):
    '''Read all network components of the system: buses and lines'''

    def readSubSystems(row, reader, network):
        '''Subsystems'''
        row = next(reader)  # header
        header = {'ID': row.index('ID'), 'Name': row.index('Name'), 'Islands': row.index('Islands')}

        row = next(reader)  # either the first bus or </Subsystems>
        while not(row[0] == '</Subsystems>'):
            network.addNewSubSystem(row, header)
            row = next(reader) # next bus or </Subsystems>
        return()

    def readBuses(row, reader, network):
        '''Buses'''
        row = next(reader)  # header
        header = {}
        header['ID'] = row.index('ID')
        header['Name'] = row.index('Name')
        header['Reference bus'] = row.index('Reference bus')
        header['ElecSub'] = row.index('Electrical subsystem')
        header['Island'] = row.index('Island')
        header['baseVoltage'] = row.index('Base voltage (kV)')
        header['area'] = row.index('Area')
        header['submName'] = row.index('Subsystem market - Name')
        header['submID'] = row.index('Subsystem market - ID')

        row = next(reader)  # either the first bus or </Buses>
        while not(row[0] == '</Buses>'):
            network.addNewBus(row, header)
            row = next(reader) # next bus or </Buses>
        return()

    def readAClines(row, reader, params, network):
        '''AC lines'''

        func = network.addNewAClineReducedSystem if params.removeParallelTransmLines else\
                                                                            network.addNewACline

        row = next(reader)  # header
        header = {}
        header['From (ID)'] = row.index('From (ID)')
        header['From (Name)'] = row.index('From (Name)')
        header['To (ID)'] = row.index('To (ID)')
        header['To (Name)'] = row.index('To (Name)')
        header['Cap'] = row.index('Line rating (MW)')
        header['Reac'] = row.index('Reactance (%) - 100-MVA base')
        header['Island'] = row.index('Island')
        header['ElecSub'] = row.index('Electrical subsystem')
        header['Tie line - sys from'] = row.index('Tie line - system from')
        header['Tie line - sys to'] = row.index('Tie line - system to')

        row = next(reader) # either the line or </AC Transmission lines>
        while not(row[0] == '</AC Transmission lines>'):
            func(params, row, header)
            row = next(reader) # next line or </AC Transmission lines>
        return()

    def readDClinks(row, reader, params, network):
        '''DC Links'''

        row = next(reader)  # header
        header = {}
        header['From (ID)'] = row.index('From (ID)')
        header['To (ID)'] = row.index('To (ID)')
        header['Cap'] = row.index('Rating (MW)')
        header['ElecSub'] = row.index('Electrical subsystem')
        header['Tie line - sys from'] = row.index('Tie line - system from')
        header['Tie line - sys to'] = row.index('Tie line - system to')
        row = next(reader) # either the first DC link or </DC Links>
        while not(row[0] == '</DC Links>'):
            network.addNewDClink(params, row, header)
            row = next(reader) # next DC link or </DC Links>
        return()

    f = open(filename, 'r', encoding = 'ISO-8859-1')
    reader = csv.reader(f, delimiter = ';')

    row = next(reader)  # '<BEGIN>'
    row = next(reader)  # either </END>, or <Hydro plants>, or <Thermal plants>

    while not(row[0] == '</END>'):
        if (row[0] == '<Subsystems>'):
            readSubSystems(row, reader, network)
        elif (row[0] == '<Buses>'):
            readBuses(row, reader, network)
        elif (row[0] == '<AC Transmission lines>'):
            readAClines(row, reader, params, network)
        elif (row[0] == '<DC Links>'):
            readDClinks(row, reader, params, network)
        row = next(reader)
    f.close()

    return()
