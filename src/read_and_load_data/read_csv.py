# -*- coding: ISO-8859-1 -*-
"""
@author: Colonetti
"""

import os
from datetime import date
import csv
from stat import S_ENFMT
import numpy as np

def gross_load_and_renewableGen(filename_gross_load,
                                    filename_renewable_gen,
                                        params, network):
    """Read gross load and renewable generation. compute the net load and print it"""

    network.NET_LOAD = np.zeros((len(network.BUS_ID), params.T), dtype = 'd')
    renewable_gen = np.zeros((len(network.BUS_ID), params.T), dtype = 'd')

    f = open(filename_gross_load, 'r', encoding = 'utf-8')
    found_bus = {bus: False for bus in network.BUS_ID}
    reader = csv.reader(f, delimiter = ';')
    row = next(reader) # <BEGIN>
    row = next(reader) # Header
    row = next(reader) # either the first bus or end
    while row[0].strip() != '</END>':
        try:
            bus = [bus for bus in network.BUS_ID if network.BUS_NAME[bus] == row[0].strip()][0]
        except IndexError:
            raise ValueError(f'Bus {row[0].strip()} is not in the system')

        b = network.BUS_ID.index(bus)

        for t in range(params.T):
            network.NET_LOAD[b, t] = float(row[1 + t].strip())

        found_bus[bus] = True
        row = next(reader) # next bus or end

    if not(all(found_bus.values())):
        raise ValueError('No load has been found for buses ' +
                                            [bus for bus in network.BUS_ID if not(found_bus[bus])])

    if os.path.isfile(filename_renewable_gen):
        f = open(filename_renewable_gen, 'r', encoding = 'utf-8')
        found_bus = {bus: False for bus in network.BUS_ID}
        reader = csv.reader(f, delimiter = ';')
        row = next(reader) # <BEGIN>
        row = next(reader) # Header
        row = next(reader) # either the first bus or end
        while row[0].strip() != '</END>':
            try:
                bus = [bus for bus in network.BUS_ID if network.BUS_NAME[bus] == row[0].strip()][0]
            except IndexError:
                raise ValueError(f'Bus {row[0].strip()} is not in the system')

            b = network.BUS_ID.index(bus)

            for t in range(params.T):
                renewable_gen[b ,t] = float(row[1 + t].strip())

            found_bus[network.BUS_ID[b]] = True
            row = next(reader) # next bus or end

        if not(all(found_bus.values())):
            raise ValueError('No load has been found for buses ' +
                                            [bus for bus in network.BUS_ID if not(found_bus[bus])])
    else:
        print("No file of renewable generation found. Assuming no renewable generation",flush=True)

    network.NET_LOAD = np.multiply(np.subtract(network.NET_LOAD, renewable_gen),1/params.POWER_BASE)

    f = open(params.OUT_DIR + 'net load - ' + params.PS
                                                + ' - case ' + str(params.CASE) + '.csv', 'w',
                                                    encoding = 'utf-8')
    f.write('<BEGIN>\nBus/Hour;')

    for t in range(params.T):
        f.write(str(t) + ';')
    f.write('\n')
    for b, bus in enumerate(network.BUS_ID):
        f.write(network.BUS_NAME[bus] + ';')
        for t in range(params.T):
            f.write(str(network.NET_LOAD[b][t]*params.POWER_BASE) + ';')
        f.write('\n')

    f.write('</END>')
    f.close()
    del f

def reset_gen_costs_of_thermals(filename, params, thermals):
    """Reset the unitary generation costs for the thermal units"""

    f = open(filename, 'r', encoding = 'ISO-8859-1')
    reader = csv.reader(f, delimiter = ';')
    row = next(reader) # <BEGIN>
    row = next(reader) # Header
    row = next(reader) # Either the first hydro or end

    while row[0].strip() != '</END>':
        try:
            g = [g for g in thermals.ID if thermals.UNIT_NAME[g] == row[1].strip()][0]
        except IndexError:
            raise ValueError(f'Thermal unit {row[1].strip()} is not in the system')

        thermals.GEN_COST[g] = (params.DISCRETIZATION*
                                        params.POWER_BASE*float(row[2].strip())*params.SCAL_OBJ_F)

        row = next(reader) # next thermal unit or end

    f.close()
    del f

def reset_volume_bounds(filename, params, hydros):
    """Reset bounds on reservoir volumes"""

    f = open(filename, 'r', encoding = 'ISO-8859-1')
    reader = csv.reader(f, delimiter = ';')
    row = next(reader) # <BEGIN>
    row = next(reader) # Header
    row = next(reader) # Either the first hydro or end

    while row[0].strip() != '</END>':
        try:
            h = hydros.ID[hydros.ID.index(int(row[0].strip()))]
        except IndexError:
            raise ValueError(f'Reservoir {row[1].strip()} ({row[0].strip()}) is not in the system')

        assert float(row[2].strip()) >= 0 and float(row[3].strip()) >= 0,\
                                                    'Negative reservoir volumes are not allowed'

        hydros.MAX_VOL[h], hydros.MIN_VOL[h] = float(row[2].strip()), float(row[3].strip())

        row = next(reader) # next hydro or end

    f.close()
    del f

def read_previous_state_of_hydro_plants(filenameIniVol, filenamePrevDisch, params, hydros):
    """Read initial reservoir volume and previous discharges"""

    f = open(filenameIniVol, 'r', encoding = 'ISO-8859-1')
    reader = csv.reader(f, delimiter = ';')
    row = next(reader) # <BEGIN>
    row = next(reader) # Header

    found_hydro = {h: False for h in hydros.ID}

    row = next(reader) # Either the first hydro or end

    while row[0].strip() != '</END>':
        try:
            h = hydros.ID[hydros.ID.index(int(row[0].strip()))]
        except IndexError:
            raise ValueError(f'Reservoir {row[1].strip()} is not in the system')

        assert float(row[2].strip()) >= 0, 'Negative reservoir volumes are not allowed'

        hydros.V_0[h] = float(row[2].strip())
        found_hydro[h] = True

        row = next(reader) # next hydro or end

    f.close()
    del f

    for h in hydros.ID:
        if not(found_hydro[h]):
            s = 'No initial reservoir volume has been found for hydro plant ' + hydros.NAME[h]
            raise Exception(s)

    # now read the previous discharges
    hydros.SPIL_0 = {(h, t): 0 for h in hydros.ID for t in range(-1440, 0, 1)}

    f = open(filenamePrevDisch, 'r', encoding = 'ISO-8859-1')
    reader = csv.reader(f, delimiter = ';')
    row = next(reader) # <BEGIN>
    row = next(reader) # Header

    found_hydro = {h: False for h in hydros.ID}

    row = next(reader) # Either the first hydro or end

    while row[0].strip() != '</END>':
        try:
            h = [h2 for h2 in hydros.ID if hydros.NAME[h2] == row[1].strip()][0]
        except IndexError:
            raise ValueError(f'Reservoir {row[1].strip()} is not in the system')

        i = 0
        for t in range(-1440, 0, 1):
            assert float(row[2 + i].strip()) >= 0, 'Negative discharges are not allowed'
            hydros.SPIL_0[h, t] = float(row[2 + i].strip())
            i += 1
        found_hydro[h] = True

        row = next(reader) # next hydro or end

    f.close()
    del f

    for h in hydros.ID:
        if not(found_hydro[h]):
            s = 'No previous discharge has been found for hydro plant ' + hydros.NAME[h]
            raise Exception(s)


def read_inflows(filename, params, hydros):
    """Read inflows in m3/s"""

    found_hydro = {h: False for h in hydros.ID}

    f = open(filename, 'r', encoding = 'ISO-8859-1')
    reader = csv.reader(f, delimiter = ';')

    while True:
        try:
            row = next(reader)
        except StopIteration:
            break

        try:
            h = [h2 for h2 in hydros.ID if hydros.NAME[h2] == row[0].strip()][0]
        except IndexError:
            raise ValueError(f'Reservoir {row[0].strip()} is not in the system')

        found_hydro[h] = True

        for t in range(params.T):
            assert float(row[1 + t]) >= 0, f'Negative inflow for plant {hydros.NAME[h]} at time {t}'
            hydros.INFLOWS[h][t] = float(row[1 + t])

    for h in hydros.ID:
        if not(found_hydro[h]):
            raise ValueError('No inflow has been found for hydro plant ' + hydros.NAME[h])

def read_trajectories(filename, params, thermals):
    """
        Read shut-down and start-up tracjetories
    """

    f = open(filename, 'r', encoding = 'ISO-8859-1')
    reader = csv.reader(f, delimiter = ';')

    row = next(reader)  # <BEGIN>
    row = next(reader)  # Header

    header = {"ID": row.index('ID'), "Name": row.index('Name'),
                "Traj": row.index('Trajectory'), "Step0": row.index('Step 0 (MW)')}

    row = next(reader)  # Either </END> or first generator

    while row[0].strip() != '</END>':
        try:
            g = [g for g in thermals.ID if thermals.UNIT_NAME[g] == row[header['Name']]][0]
        except IndexError:
            raise ValueError(f"Thermal unit {row[header['Name']]} is not in the system")

        if row[header['Traj']] == 'Start-up':
            for step in row[header['Step0']:]:
                if len(step) > 0 and float(step) != 0:
                    thermals.STUP_TRAJ[g].append(float(step)/params.POWER_BASE)
        elif row[header['Traj']] == 'Shut-down':
            for step in row[header['Step0']:]:
                if len(step) > 0 and float(step) != 0:
                    thermals.STDW_TRAJ[g].append(float(step)/params.POWER_BASE)
        else:
            raise ValueError(f"I dont know what {row[header['Traj']]} is in the trajectories file")

        row = next(reader)

    f.close()
    del f

def read_bounds_on_gen_of_hydros(filename, params, hydros):
    """Read bounds on generation of groups of hydro units"""

    def appendToListOfConstraints(params, hydros, listOfConstraints, row, reader):
        """Append a new constraint to a list of thermal generation constraints"""

        listOfConstraints.append([[],[],[], 0]) # first position is the list of plants
                                                # second is the list of groups for each plant in the
                                                # constraint
                                                # third is the list of periods
                                                # fourth is the RHS of the constraint

        row = next(reader) # hydro plants in constraint
        # check for leading zeros
        for h in row[1].strip().split(','):
            if len(h) > 1 and h[0] == '0':
                raise ValueError('One or more plants is identified with a leading zero in '+
                                'the input file of bounds on generation of groups of plants')

        plants = [hName.strip() for hName in row[1].strip().split(',') if hName.strip() != '']
        assert len(plants) ==len(set(plants)),'A hydro plant appears more than once in a constraint'
        assert (set(plants) & set(hydros.NAME.values())) == set(plants),\
                                                'Not all plants in the constraint are in the system'
        listOfConstraints[-1][0] = [h for h in hydros.ID if hydros.NAME[h] in plants]

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
        listOfConstraints[-1][3] = float(row[1].strip())/params.POWER_BASE

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

def read_bounds_on_gen_of_thermals(filename, params, thermals):
    """Read bounds on generation of groups of thermal units"""

    def append_new_constr(params, thermals, listOfConstraints, row, reader):
        """Append a new constraint to a list of thermal generation constraints"""

        listOfConstraints.append([[], [], 0])   # first position is the list of units
                                                # second is the list of periods
                                                # third is the RHS of the constraint

        row = next(reader) # thermal units in constraint
        # check for leading zeros
        for u in row[1].strip().split(','):
            if len(u) > 1 and u[0] == '0':
                raise ValueError('One or more units is identified with a leading zero in '+\
                                'the input file of bounds on generation of groups of units')

        units = [int(u) for u in row[1].strip().split(',') if u != '']
        assert len(units) == len(set(units)),'A thermal unit appears more than once in a constraint'
        assert (set(units) & set(thermals.ID)) == set(units),\
                                                'Not all units in the constraint are in the system'
        listOfConstraints[-1][0] = units

        row = next(reader) # times in constraint
        times = list(range(int(row[1].strip()), int(row[2].strip()) + 1, 1))
        assert (set(times) & set(range(params.T))) == set(times),\
                                'All time periods in constraint must be within the planning horizon'
        listOfConstraints[-1][1] = times

        row = next(reader) # RHS
        listOfConstraints[-1][2] = float(row[1].strip())/params.POWER_BASE

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
            append_new_constr(params, thermals, thermals.ADDTNL_MAX_P, row, reader)

        elif row[1].strip() == 'LB':
            append_new_constr(params, thermals, thermals.ADDTNL_MIN_P, row, reader)

        else:
            append_new_constr(params, thermals, thermals.EQ_GEN_CONSTR, row, reader)

        row = next(reader)
        if row[0].strip() != '</constr>':
            raise Exception('Error reading constraint. There seems to be less fields than required')

        row = next(reader)  # either a new constraint or </END>

    f.close()
    del f

def read_cost_to_go_function(filename, params, hydros):
    """Read the cost-to-go function"""

    f = open(filename, 'r', encoding = 'ISO-8859-1')
    reader = csv.reader(f, delimiter = ';')

    row = next(reader)  # <BEGIN>
    row = next(reader)  # Header
    header = {}
    for h in hydros.ID:
        try:
            header[hydros.NAME[h]] = row.index(hydros.NAME[h])
        except ValueError as err:
            raise ValueError(f'Hydro plant {hydros.NAME[h]} has not ' +
                                'been found in the cost-to-go function file') from err

    header['rhs'] = row.index('RHS ($)')
    row = next(reader)  # First hydro or </END>
    c = 0
    while not(row[0] == '</END>'):
        hydros.CTFrhs[c] = float(row[header['rhs']])*params.SCAL_OBJ_F
        for h in hydros.ID:
            hydros.CTF[h][c] = float(row[header[hydros.NAME[h]]])*params.SCAL_OBJ_F
        row = next(reader)
        c += 1
    f.close()
    del f

    # Check if there is any cut that appears more than once
    unique_cuts = []
    unique_cuts_idxs = []
    for c in hydros.CTFrhs.keys():
        if [[hydros.CTFrhs[c]] + [hydros.CTF[h][c] for h in hydros.ID]] not in unique_cuts:
            unique_cuts.append([[hydros.CTFrhs[c]] + [hydros.CTF[h][c] for h in hydros.ID]])
            unique_cuts_idxs.append(c)

    hydros.CTFrhs = {c: hydros.CTFrhs[c] for c in unique_cuts_idxs}
    hydros.CTF = {h: {c: hydros.CTF[h][c] for c in unique_cuts_idxs}for h in hydros.ID}

def read_ini_state_thermal(filename, params, thermals):
    """Read the initial state of the thermal units"""

    f = open(filename, 'r', encoding = 'ISO-8859-1')
    reader = csv.reader(f, delimiter = ';')

    row = next(reader)  # '<BEGIN>'
    row = next(reader)  # <Thermal plants>
    row = next(reader)  # header

    header = {}
    header['ID'] = row.index('ID')
    header['Name'] = row.index('Name')
    header['iniGen'] = row.index('Generation in time t = -1 in MW')
    header['iniState'] = row.index('State in t = -1. Either 1, if up, or 0, if down')
    header['nHoursInIniState'] = row.index('Number of hours (> 0) in the state of t = -1')
    header['inStartupTraj'] = row.index('Start-up trajectory (TRUE or FALSE)')
    header['inShutDownTraj'] = row.index('Shut-down trajectory (TRUE or FALSE)')

    row = next(reader)  # either first thermal or </Thermal plants>
    while not(row[0] == '</Thermal plants>'):
        plant_name = row[header['Name']].strip()
        ini_gen = float(row[header['iniGen']])
        ini_state = int(row[header['iniState']])
        n_hours_in_ini_state = int(row[header['nHoursInIniState']])

        inStartUp = True if row[header['inStartupTraj']].strip() in ('TRUE', 'True') else False
        inShutDown = True if row[header['inShutDownTraj']].strip() in ('TRUE', 'True') else False

        try:
            g = [g for g in thermals.ID if thermals.UNIT_NAME[g] == row[header['Name']]][0]
        except IndexError:
            raise ValueError(f'Thermal unit {plant_name} is not in the system')

        if (not(inStartUp) and not(inShutDown) and
                    (ini_state == 1 and (ini_gen < (thermals.MIN_P[g]*params.POWER_BASE - 1e-4)))):
            raise ValueError(f'Thermal plant {plant_name} is not in a start-up or shut-down '+
                            'trajectory and yet its initial generation is less than its minimum')

        thermals.STATE_0[g] = ini_state
        thermals.N_HOURS_IN_PREVIOUS_STATE[g] = n_hours_in_ini_state

        if ini_state == 1:
            if ((thermals.MIN_P[g] + 1e-4) <= (ini_gen/params.POWER_BASE)
                                                                    <= (thermals.MAX_P[g] - 1e-4)):
                thermals.T_G_0[g] = ini_gen/params.POWER_BASE
            elif ini_gen/params.POWER_BASE > (thermals.MAX_P[g] - 1e-4):
                thermals.T_G_0[g] = thermals.MAX_P[g]
            else:
                thermals.T_G_0[g] = thermals.MIN_P[g]
        else:
            thermals.T_G_0[g] = 0

        row = next(reader)
    f.close()

def read_generators(filename, params, network, hydros, thermals):
    """Read data of hydro, thermal and wind generators"""

    def readHydros(row, reader, params, hydros):
        """Read data of hydropower plants"""
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
        header['InflOfSpill'] = row.index('Influence of spillage on the HPF? Yes or No')
        header['MaxSpilHPF'] = row.index('Maximum spillage - HPF')
        header['DRBypass'] =row.index('Downriver plant of bypass discharge')
        header['MaxBypass'] = row.index('Maximum bypass discharge (m3/s)')
        header['BypassTravelTime'] = row.index('Water travel time in the bypass process (h)')
        header['DRPump'] = row.index('Downriver reservoir of pump units')
        header['UPRPump'] = row.index('Upriver reservoir of pump units')
        header['PumpTravelTime'] = row.index('Water travel time in pumping (h)')

        row = next(reader)  # either the first hydro plant or </Hydro plants>
        while not(row[0] == '</Hydro plants>'):
            hydros.addNewHydro(params, row, header)
            row = next(reader) # next hydro plant or </Hydro plants>

    def readThermals(row, reader, params, thermals):
        """
            Read data of thermal generators
        """

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

        row = next(reader) # either the first thermal plant or </Thermal plants>
        while not(row[0] == '</Thermal plants>'):
            thermals.add_new_thermal(params, row, header)
            row = next(reader) # next thermal plant or </Thermal plants>

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
            network.DEFICIT_COST = (float(row[0].strip())*params.POWER_BASE*params.SCAL_OBJ_F*
                                                        params.BASE_TIME_STEP)
            row = next(reader) # </Deficit cost>

        row = next(reader)
    f.close()

def read_hydro_generating_units(filename, params, hydros):
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
    header['turbineOrPUMP'] = row.index('Turbine or Pump?')
    header['conversionMWm3s'] = row.index('Conversion rate MW/(m3/s/) for pumps')

    row = next(reader)  # either first unit or </END>

    while not(row[0] == '</Hydro generating units>'):
        unit_id = int(row[header['ID']])
        unit_name = row[header['Name']].strip()
        plant_id = int(row[header['plantID']])
        group = row[header['groupName']]
        bus = int(row[header['bus']])
        min_power = float(row[header['minGen']])/params.POWER_BASE
        max_power = float(row[header['maxGen']])/params.POWER_BASE
        min_turb_disch = float(row[header['minTurbDisc']])
        max_turb_disch = float(row[header['maxTurbDisc']])
        turbine_or_pump = row[header['turbineOrPUMP']].strip()
        conversion_MW_m3s = float(row[header['conversionMWm3s']])/(params.POWER_BASE)

        assert plant_id in hydros.ID, f"Hydro plant {plant_id} is not in the system"

        # The following are attributes of individual generating units
        hydros.UNIT_ID[plant_id].append(unit_id)
        hydros.UNIT_NAME[plant_id][unit_id] = unit_name
        hydros.UNIT_GROUP[plant_id][unit_id] = group
        hydros.UNIT_BUS[plant_id][unit_id] = [bus]
        hydros.UNIT_BUS_COEFF[plant_id][unit_id] = {bus: 1.000}

        if turbine_or_pump == 'Pump':
            #### Assuming there is a maximum of one pump at each plant
            hydros.TURB_OR_PUMP[plant_id] = 'Pump'
            hydros.PUMP_CONVERSION_FACTOR[plant_id] = conversion_MW_m3s

        hydros.UNIT_MIN_P[plant_id][unit_id] = min_power
        hydros.UNIT_MAX_P[plant_id][unit_id] = max_power
        hydros.UNIT_MIN_TURB_DISCH[plant_id][unit_id] = min_turb_disch
        hydros.UNIT_MAX_TURB_DISCH[plant_id][unit_id] = max_turb_disch

        row = next(reader)

    f.close()

def read_aggreg_HPF(filename, params, hydros):
    """Read the three-dimensional HPF"""

    f = open(filename, 'r', encoding = 'ISO-8859-1')
    reader = csv.reader(f, delimiter = ';')

    row = next(reader)  # <BEGIN>
    row = next(reader)  # First hydro or </END>

    while not(row[0] == '</END>'):
        row = next(reader)  # ID
        row = next(reader)  # Hydro plant name

        try:
            h = [h2 for h2 in hydros.ID if hydros.NAME[h2] == row[0].strip()][0]
        except IndexError:
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
                    hydros.A0[h].append(float(row[0])/params.POWER_BASE)
                    hydros.A1[h].append(float(row[1])/params.POWER_BASE)
                    hydros.A2[h].append(float(row[2])/params.POWER_BASE)
                    hydros.A3[h].append(float(row[3])/params.POWER_BASE)
                    row = next(reader)  # Either next cut or </HPF>

                row = next(reader)

        row = next(reader)
    f.close()
    del f

    for h in hydros.ID:
        if sum(hydros.UNIT_MAX_P[h].values()) > 0:
            if len(hydros.A0[h]) == 0:
                raise ValueError(f'The hydropower function of {hydros.NAME[h]} has not been found')

def read_network(filename, params, network):
    """Read all network components of the system: buses and lines"""

    def read_buses(row, reader, network):
        """read bus data"""
        row = next(reader)  # header
        header = {}
        header['ID'] = row.index('ID')
        header['Name'] = row.index('Name')
        header['Reference bus'] = row.index('Reference bus')
        header['baseVoltage'] = row.index('Base voltage (kV)')
        header['area'] = row.index('Area')
        header['submName'] = row.index('Subsystem market - Name')
        header['submID'] = row.index('Subsystem market - ID')

        row = next(reader)  # either the first bus or </Buses>
        while not(row[0] == '</Buses>'):
            network.add_new_bus(row, header)
            row = next(reader) # next bus or </Buses>

    def read_lines(row, reader, params, network):
        """read transmission line data"""

        row = next(reader)  # header
        header = {}
        header['From (ID)'] = row.index('From (ID)')
        header['From (Name)'] = row.index('From (Name)')
        header['To (ID)'] = row.index('To (ID)')
        header['To (Name)'] = row.index('To (Name)')
        header['Cap'] = row.index('Line rating (MW)')
        header['Reac'] = row.index('Reactance (p.u.) - 100-MVA base')

        row = next(reader) # either the line or </AC Transmission lines>
        while not(row[0] == '</AC Transmission lines>'):
            network.add_new_AC_line(params, row, header)
            row = next(reader) # next line or </AC Transmission lines>

    def read_DC_links(row, reader, params, network):
        """read DC Link data"""

        row = next(reader)  # header
        header = {}
        header['From (ID)'] = row.index('From (ID)')
        header['To (ID)'] = row.index('To (ID)')
        header['Cap'] = row.index('Rating (MW)')
        row = next(reader) # either the first DC link or </DC Links>
        while not(row[0] == '</DC Links>'):
            network.add_new_DC_link(params, row, header)
            row = next(reader) # next DC link or </DC Links>

    f = open(filename, 'r', encoding = 'ISO-8859-1')
    reader = csv.reader(f, delimiter = ';')

    row = next(reader)  # '<BEGIN>'
    row = next(reader)  # either </END>, or <Hydro plants>, or <Thermal plants>

    while not(row[0] == '</END>'):
        if (row[0] == '<Buses>'):
            read_buses(row, reader, network)
        elif (row[0] == '<AC Transmission lines>'):
            read_lines(row, reader, params, network)
        elif (row[0] == '<DC Links>'):
            read_DC_links(row, reader, params, network)
        row = next(reader)
    f.close()
