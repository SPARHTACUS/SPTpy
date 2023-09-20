# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""

import numpy as np
from math import inf, pi

class Hydros:
    """
        Class of hydro plants
    """
    def __init__(self):
        self.setup()

    def setup(self):
        """Initialize the attributes"""

        self.ID = []
        self.NAME = {}

        self.TURB_OR_PUMP, self.PUMP_CONVERSION_FACTOR = {}, {}

        self.MIN_VOL, self.MAX_VOL = {}, {}

        self.DOWN_RIVER_RESERVOIR = {}

        self.WATER_TRAVEL_TIME = {}

        self.MAX_SPIL = {}

        self.DOWN_RIVER_BY_PASS = {}            # Plant that receives
        self.upRiverTransferPlantID = {}        # Plant that transfers
        self.MAX_BY_PASS = {}
        self.BY_PASS_TRAVEL_TIME = {}

        self.DOWN_RIVER_PUMP, self.UP_RIVER_PUMP = {}, {}

        self.PUMP_TRAVEL_TIME = {}

        # Initial state
        self.V_0, self.Q_0, self.SPIL_0 = {}, {}, {}

        # The following are attributes of individual generating units
        (self.UNIT_ID, self.UNIT_NAME, self.UNIT_GROUP,
                                        self.UNIT_BUS, self.UNIT_BUS_COEFF) = ({}, {}, {}, {}, {})

        (self.UNIT_MIN_P, self.UNIT_MAX_P,
                        self.UNIT_MIN_TURB_DISCH, self.UNIT_MAX_TURB_DISCH) = ({}, {}, {}, {})

        # The coefficients of the aggregated hydropower function are as follows
        self.A0 = {}    # Coefficient of the turbine discharge
        self.A1 = {}    # Coefficient of the reservoir volume
        self.A2 = {}    # Coefficient of the spillage
        self.A3 = {}    # Constant term

        # Coefficients of the cost-to-go function
        self.CTF = {}
        # RHS of the cost-to-go function
        self.CTFrhs = {}

        # inflows to reservoirs in m3/s
        self.INFLOWS = {}

        # bounds on generation of groups of plants. Analogous to that of the thermal units
        self.maxGen, self.minGen, self.equalityConstrs = [], [], []

    def addNewHydro(self, params, row, header):
        """
            Add a new hydro plant
        """

        self.ID.append(int(row[header['ID']]))
        self.NAME[self.ID[-1]] = row[header['Name']].strip()

        self.MIN_VOL[self.ID[-1]] = round(float(row[header['MinVol']]), 6)
        self.MAX_VOL[self.ID[-1]] = round(float(row[header['MaxVol']]), 6)

        self.DOWN_RIVER_RESERVOIR[self.ID[-1]] = (row[header['Downriver reservoir']].strip() if
                                                    row[header['Downriver reservoir']].strip() !='0'
                                                        else None
                                                )

        for t in range(params.T):
            if params.DISCRETIZATION <= params.BASE_TIME_STEP:
                self.WATER_TRAVEL_TIME[(row[header['Name']].strip(),
                                row[header['Downriver reservoir']].strip(), t)] =\
                                int(int(float(row[header['WaterTravT']]))*(1/params.BASE_TIME_STEP))
            else:
                self.WATER_TRAVEL_TIME[(row[header['Name']].strip(),
                                row[header['Downriver reservoir']].strip(), t)] = 0

        self.MAX_SPIL[self.ID[-1]] = float(row[header['MaxSpil']])

        self.DOWN_RIVER_BY_PASS[self.ID[-1]] = (row[header['DRBypass']].strip()
                                                    if row[header['DRBypass']].strip() != '0'
                                                        else None)
        self.MAX_BY_PASS[self.ID[-1]] = float(row[header['MaxBypass']])
        self.BY_PASS_TRAVEL_TIME.update({(len(self.ID), t): 0 for t in range(params.T)})
        for t in range(params.T):
            if params.DISCRETIZATION <= params.BASE_TIME_STEP:
                self.BY_PASS_TRAVEL_TIME[len(self.ID), t] = int(row[header['BypassTravelTime']])

        self.DOWN_RIVER_PUMP[self.ID[-1]] = (row[header['DRPump']].strip()
                                            if row[header['DRPump']].strip() != '0'
                                                else None)
        self.UP_RIVER_PUMP[self.ID[-1]] = (row[header['UPRPump']].strip()
                                            if row[header['UPRPump']].strip() != '0'
                                                else None)

        self.PUMP_TRAVEL_TIME.update({(self.ID[-1], t): 0 for t in range(params.T)})
        for t in range(params.T):
            if params.DISCRETIZATION <= params.BASE_TIME_STEP:
                self.PUMP_TRAVEL_TIME[self.ID[-1], t] = int(row[header['PumpTravelTime']])

        self.V_0[self.ID[-1]] = 0
        self.Q_0[self.ID[-1]] = 0
        self.SPIL_0[self.ID[-1]] = 0

        self.UNIT_ID[self.ID[-1]] = []
        self.UNIT_NAME[self.ID[-1]] = {}
        self.UNIT_GROUP[self.ID[-1]] = {}
        self.UNIT_BUS[self.ID[-1]] = {}
        self.UNIT_BUS_COEFF[self.ID[-1]] = {}

        self.UNIT_MIN_P[self.ID[-1]] = {}
        self.UNIT_MAX_P[self.ID[-1]] = {}
        self.UNIT_MIN_TURB_DISCH[self.ID[-1]] = {}
        self.UNIT_MAX_TURB_DISCH[self.ID[-1]] = {}

        self.TURB_OR_PUMP[self.ID[-1]] = ''
        self.PUMP_CONVERSION_FACTOR[self.ID[-1]] = 0

        self.A0[self.ID[-1]] = []
        self.A1[self.ID[-1]] = []
        self.A2[self.ID[-1]] = []
        self.A3[self.ID[-1]] = []

        self.INFLOWS[self.ID[-1]] = {t: 0 for t in range(params.T)}

        self.CTF[self.ID[-1]] = {}

class Thermals:
    """Class of thermal units"""
    def __init__(self):
        """
            initialize the attributes
        """
        self.ID = []
        self.UNIT_NAME = {}

        self.MIN_P, self.MAX_P = {}, {}      # power limits

        self.GEN_COST = {}                   # unitary cost in $/(p.u.) for a 30-min period

        self.RAMP_UP, self.RAMP_DOWN = {}, {} # ramps

        self.MIN_UP, self.MIN_DOWN = {}, {}   # minimum times

        self.BUS = {}                       # bus to which the unit is connected
        self.BUS_COEFF = {}                 # coefficient of the generation in each bus. default=1

        self.CONST_COST, self.ST_UP_COST, self.ST_DW_COST = {}, {}, {}

        # Previous state
        self.STATE_0 = {}
        self.T_G_0 = {}
        self.N_HOURS_IN_PREVIOUS_STATE = {}

        # minimum and maximum generation of groups of units
        self.ADDTNL_MIN_P, self.ADDTNL_MAX_P = [], []
        # groups of thermal units may have their combined power output set to a fixed level.
        self.EQ_GEN_CONSTR = []

        self.STUP_TRAJ = {}         # power steps in the start-up trajectory
        self.STDW_TRAJ = {}        # power steps in the shut-down trajectory

    def add_new_thermal(self, params, row, header):
        """
            Add a new thermal unit
        """

        self.ID.append(int(row[header['ID']]))
        self.UNIT_NAME[self.ID[-1]] = row[header['Name']]
        self.MIN_P[self.ID[-1]] = float(row[header['minP']])/params.POWER_BASE
        self.MAX_P[self.ID[-1]] = float(row[header['maxP']])/params.POWER_BASE
        self.GEN_COST[self.ID[-1]] = (params.DISCRETIZATION*
                                        params.POWER_BASE*float(row[header['genCost']])*
                                            params.SCAL_OBJ_F
                                        )

        self.RAMP_UP[self.ID[-1]] = (params.DISCRETIZATION*
                                        float(row[header['rampUp']])/params.POWER_BASE
                                        )
        self.RAMP_DOWN[self.ID[-1]] = (params.DISCRETIZATION*
                                        float(row[header['rampDown']])/params.POWER_BASE
                                    )

        self.MIN_UP[self.ID[-1]] = int(row[header['minUp']])
        self.MIN_DOWN[self.ID[-1]] = int(row[header['minDown']])

        self.BUS[self.ID[-1]] = [(int(row[header['bus']]))]
        self.BUS_COEFF[self.ID[-1]] = {(int(row[header['bus']])): 1.00}

        self.CONST_COST[self.ID[-1]] = (float(row[header['constCost']])*params.SCAL_OBJ_F)
        self.ST_UP_COST[self.ID[-1]] = (float(row[header['stUpCost']])*params.SCAL_OBJ_F)
        self.ST_DW_COST[self.ID[-1]] = (float(row[header['stDwCost']])*params.SCAL_OBJ_F)

        self.STATE_0[self.ID[-1]] = (0)
        self.T_G_0[self.ID[-1]] = (0)
        self.N_HOURS_IN_PREVIOUS_STATE[self.ID[-1]] = (0)

        self.STUP_TRAJ[self.ID[-1]] = []
        self.STDW_TRAJ[self.ID[-1]] = []

def get_buses_bounds_on_injections(params, network, thermals, hydros):
    """
        for each bus in the system, get the most negative (largest load) and most positive
        (largest generation) injections based on the elements connected to the bus

        the bounds are given as minimum and maximum over the entire planning horizon
        and also as per-period bounds

        in computing these bounds, transmission elements are not considered
    """

    # get first the minimum and maximum injections for each period of the planning horizon
    min_inj_per_period = {bus: {t: 0 for t in range(params.T)} for bus in network.BUS_ID}
    max_inj_per_period = {bus: {t: 0 for t in range(params.T)} for bus in network.BUS_ID}

    for g in thermals.UNIT_NAME.keys():
        for bus in thermals.BUS[g]:
            for t in range(params.T):
                max_inj_per_period[bus][t] += thermals.BUS_COEFF[g][bus]*thermals.MAX_P[g]

    for h in [h for h in hydros.ID if hydros.TURB_OR_PUMP[h] != 'Pump']:
        for u in hydros.UNIT_ID[h]:
            for bus in hydros.UNIT_BUS[h][u]:
                for t in range(params.T):
                    max_inj_per_period[bus][t] += (hydros.UNIT_BUS_COEFF[h][u][bus]
                                                                        *hydros.UNIT_MAX_P[h][u])

    for h in [h for h in hydros.ID if hydros.TURB_OR_PUMP[h] == 'Pump']:
        for u in hydros.UNIT_ID[h]:
            for bus in hydros.UNIT_BUS[h][u]:
                for t in range(params.T):
                    min_inj_per_period[bus][t] -= hydros.UNIT_BUS_COEFF[h][u][bus]*(
                                                            hydros.UNIT_MAX_TURB_DISCH[h][u]
                                                                *hydros.PUMP_CONVERSION_FACTOR[u])

    non_zero_inj_buses = [bus for bus in network.BUS_ID
                            if np.max(np.abs(network.NET_LOAD[network.BUS_HEADER[bus],:]),axis=0)>0]
    for t in range(params.T):
        for bus in non_zero_inj_buses:
            min_inj_per_period[bus][t] = (min_inj_per_period[bus][t]
                                                    - network.NET_LOAD[network.BUS_HEADER[bus], t])
            max_inj_per_period[bus][t] = (max_inj_per_period[bus][t]
                                                    - network.NET_LOAD[network.BUS_HEADER[bus], t])

    # now compute the bounds over the entire planning horizon by basically taking the
    # minimum of the minimums and the maximum of the maximums
    min_inj = {bus: min(min_inj_per_period[bus][t] for t in range(params.T))
                                                                        for bus in network.BUS_ID}
    max_inj = {bus: max(max_inj_per_period[bus][t] for t in range(params.T))
                                                                        for bus in network.BUS_ID}

    return (min_inj, max_inj, min_inj_per_period, max_inj_per_period)

def add_new_parallel_line(
                        resistance_line_1, reactance_line_1,
                        shunt_conductance_line_1, shunt_susceptance_line_1,
                        normal_UB_line_1, normal_LB_line_1,
                        emerg_UB_line_1, emerg_LB_line_1,
                        resistance_line_2, reactance_line_2,
                        shunt_conductance_line_2, shunt_susceptance_line_2,
                        normal_UB_line_2, normal_LB_line_2,
                        emerg_UB_line_2, emerg_LB_line_2):
    """Add a new parallel line to the system. all parameters must be in p.u."""

    # squared norm 2 of the impedance of line 1
    sq_norm_2_imp_line_1 = resistance_line_1**2 + reactance_line_1**2
    cond_line_1 = resistance_line_1/sq_norm_2_imp_line_1
    suscep_line_1 = - reactance_line_1/sq_norm_2_imp_line_1
    # magnitude of line 1's admittance
    mag_admt_line_1 = (cond_line_1**2 + suscep_line_1**2)**(1/2)

    # squared norm 2 of the impedance of line 2
    sq_norm_2_imp_line_2 = resistance_line_2**2 + reactance_line_2**2
    cond_line_2 = resistance_line_2/sq_norm_2_imp_line_2
    suscep_line_2 = - reactance_line_2/sq_norm_2_imp_line_2
    # magnitude of line 2's admittance
    mag_admt_line_2 = (cond_line_2**2 + suscep_line_2**2)**(1/2)

    #### parameters of the equivalent line
    # squared norm 2 of the admittance of the equivalent line
    sq_norm_2_admt_eq_line = (cond_line_1 + cond_line_2)**2 + (suscep_line_1 + suscep_line_2)**2
    conductance_eq_line = (cond_line_1 + cond_line_2)
    susceptance_eq_line = (suscep_line_1 + suscep_line_2)
    resistance_eq_line = conductance_eq_line/sq_norm_2_admt_eq_line
    reactance_eq_line = - susceptance_eq_line/sq_norm_2_admt_eq_line
    shunt_conductance_eq_line = shunt_conductance_line_1 + shunt_conductance_line_2
    shunt_susceptance_eq_line = shunt_susceptance_line_1 + shunt_susceptance_line_2
    # magnitude of the equivalent line's admittance
    mag_admt_eq_line = sq_norm_2_admt_eq_line**(1/2)

    # the magnitude of the apparent power flowing in a line i,j is given by
    # |s_i_j| = |V_i| * |conj(y_i_j)| * |conj(V_i - V_j)|
    # given that two parallel lines have the same end-point voltages and that their equivalent
    # admittance is y_eq_i_j, for the total flow between i,j and there are at least two
    # parallel lines, |s_i_j| is
    # |s_i_j| = |V_i| * |conj(y_eq_i_j)| * |conj(V_i - V_j)|
    # and yet, the acceptable values for V_i and V_j will depend on the individual limits of the
    # lines. thus, given that |V_i| * |conj(V_i - V_j)| is the same for all parallel lines
    # connecting i,j, then a maximum value for this product can be found by taking the minimum of
    # max_mag_prod = min{S_max_l/|conj(y_i_j_l)| for all l in L_i_j},
    # where L_i_j is the set of parallel lines between i,j

    normal_UB_eq_line = np.array(len(normal_UB_line_1)*[0.000])
    normal_LB_eq_line = np.array(len(normal_UB_line_1)*[0.000])
    emerg_UB_eq_line = np.array(len(emerg_UB_line_1)*[0.000])
    emerg_LB_eq_line = np.array(len(emerg_UB_line_1)*[0.000])
    for t in range(len(emerg_UB_line_1)):
        max_mag_prod_normal_UB = min(normal_UB_line_1[t]/mag_admt_line_1,
                                                            normal_UB_line_2[t]/mag_admt_line_2)
        max_mag_prod_emerg_UB = min(emerg_UB_line_1[t]/mag_admt_line_1,
                                                            emerg_UB_line_2[t]/mag_admt_line_2)

        normal_UB_eq_line[t] = max_mag_prod_normal_UB*mag_admt_eq_line
        emerg_UB_eq_line[t] = max_mag_prod_emerg_UB*mag_admt_eq_line

        max_mag_prod_normal_LB = max(normal_LB_line_1[t]/mag_admt_line_1,
                                                            normal_LB_line_2[t]/mag_admt_line_2)
        max_mag_prod_emerg_LB = max(emerg_LB_line_1[t]/mag_admt_line_1,
                                                            emerg_LB_line_2[t]/mag_admt_line_2)

        normal_LB_eq_line[t] = max_mag_prod_normal_LB*mag_admt_eq_line
        emerg_LB_eq_line[t] = max_mag_prod_emerg_LB*mag_admt_eq_line

    return(resistance_eq_line, reactance_eq_line,
            conductance_eq_line, susceptance_eq_line,
            shunt_conductance_eq_line, shunt_susceptance_eq_line,
            normal_UB_eq_line, normal_LB_eq_line,
            emerg_UB_eq_line, emerg_LB_eq_line)

class Network:
    'Class of transmission network with DC model'
    def __init__(self):
        self.setup()

    def setup(self):
        'Initialize the attributes'

        self.BUS_ID = []                # list of ints with the buses` ids
        self.BUS_NAME = {}              # list of strs with the buses` names

        self.REF_BUS_ID = []

        self.LINES_FROM_BUS = {}
        self.LINES_TO_BUS = {}

        self.LINKS_FROM_BUS = {}
        self.LINKS_TO_BUS = {}

        # these will be defined for each AC transmission line (as opposed to a DC link) of the
        # system. Despite the name, in the model, flows in the AC transmission lines are linear
        # functions of bus angles
        self.LINE_ID = []
        self.LINE_F_T = {}
        self.LINE_FLOW_UB = {}
        self.LINE_FLOW_LB = {}
        self.LINE_X = {}

        # the following two dicts are defined for each DC link of the system. Different from the
        # AC lines, the flows in these links are not functions of bus angles
        self.LINK_ID = []
        self.LINK_F_T = {}
        self.LINK_MAX_P = {}

        self.BUS_HEADER = {}                # Gets bus ID and returns bus index in the load
        self.NET_LOAD = []                  # will be a numpy array containing the load at each
                                            # bus and time period

        self.DEFICIT_COST = -1e12

        self.THETA_BOUND = 50*24*pi     # bound on the buses' voltage angles

        self.DISJOINT_AC_SUBS = None

        self.PTDF = None

        self.ACTIVE_BOUNDS = {}
        self.ACTIVE_UB_PER_PERIOD = {}
        self.ACTIVE_LB_PER_PERIOD = {}
        self.ACTIVE_UB = {}
        self.ACTIVE_LB = {}

        self.SEC_CONSTRS = {}           # first keys are time periods, second keys are ids
                                        # self.SEC_CONSTRS[id] = {'name': '',
                                        #                           'net load': [],
                                        #                           'factors': {comp: 0}
                                        #                           'LB': -inf,
                                        #                           'UB': inf}
                                        # in self.SEC_CONSTRS[id], key 'net load'
                                        # returns the net load in the constraint.
                                        # key 'factors' returns the coefficients
                                        # of the participating components in the constraint.
                                        # the bounds are givein in 'LB' and 'UB' in p.u.


    def add_new_bus(self, row, header):
        'Add a new bus'

        bus = int(row[header['ID']].strip())

        self.BUS_ID.append(bus)
        self.BUS_NAME[self.BUS_ID[-1]] = row[header['Name']].strip()

        self.LINES_FROM_BUS[bus] = []
        self.LINES_TO_BUS[bus] = []

        self.LINKS_FROM_BUS[bus] = []
        self.LINKS_TO_BUS[bus] = []

        if (row[header['Reference bus']].strip() == 'Ref'):
            self.REF_BUS_ID.append(bus)

    def add_new_line(self, params, line_id:int,
                    from_id:int, to_id:int,
                        reactance:float, resistance:float, shunt_conduc:float, shunt_suscep:float,
                            line_rating:float, emergency_rating:float,
                                tap_setting:float, min_tap:float, max_tap:float,
                                    phase_shift:float, controlled_bus:int):
        """Add a new transmission line to the network"""

        if from_id < to_id:
            f, t = from_id, to_id
        else:
            f, t = to_id, from_id

        cap = line_rating

        if (f, t) in self.LINE_F_T.values():
            l = [l for l in self.LINE_ID if self.LINE_F_T[l] == (f, t)][0]

            (_, self.LINE_X[l], _1, _2, _3, _4, self.LINE_FLOW_UB[l], self.LINE_FLOW_LB[l],
                                    _5, _6) = add_new_parallel_line(
                                                    0, reactance, 0, 0,
                                                    np.array(params.T*[cap/params.POWER_BASE]),
                                                    -1*np.array(params.T*[cap/params.POWER_BASE]),
                                                    np.array(params.T*[cap/params.POWER_BASE]),
                                                    -1*np.array(params.T*[cap/params.POWER_BASE]),
                                                    0,
                                                    self.LINE_X[l], 0, 0,
                                                    self.LINE_FLOW_UB[l], self.LINE_FLOW_LB[l],
                                                    self.LINE_FLOW_UB[l], self.LINE_FLOW_LB[l])

            self.ACTIVE_BOUNDS[l] = self.ACTIVE_BOUNDS[l] or cap < 99999
            self.ACTIVE_UB[l] = self.ACTIVE_BOUNDS[l]
            self.ACTIVE_LB[l] = self.ACTIVE_BOUNDS[l]
            self.ACTIVE_UB_PER_PERIOD[l] = {t: self.ACTIVE_BOUNDS[l] for t in range(params.T)}
            self.ACTIVE_LB_PER_PERIOD[l] = {t: self.ACTIVE_BOUNDS[l] for t in range(params.T)}

        else:
            l = line_id
            self.LINE_ID.append(l)
            self.LINE_F_T[l] = (f, t)
            self.LINE_FLOW_UB[l] = np.array(params.T*[cap/params.POWER_BASE])
            self.LINE_FLOW_LB[l] = -1*np.array(params.T*[cap/params.POWER_BASE])
            self.LINE_X[l] = reactance

            self.ACTIVE_BOUNDS[l] = cap*params.POWER_BASE < 99999
            self.ACTIVE_UB[l] = self.ACTIVE_BOUNDS[l]
            self.ACTIVE_LB[l] = self.ACTIVE_BOUNDS[l]
            self.ACTIVE_UB_PER_PERIOD[l] = {t: self.ACTIVE_BOUNDS[l] for t in range(params.T)}
            self.ACTIVE_LB_PER_PERIOD[l] = {t: self.ACTIVE_BOUNDS[l] for t in range(params.T)}

    def add_new_AC_line(self, params, row, header):
        'add a new AC line: if it is parallel to a existing line, then combine them'

        if int(row[header['From (ID)']].strip()) < int(row[header['To (ID)']].strip()):
            f, t = int(row[header['From (ID)']].strip()), int(row[header['To (ID)']].strip())
        else:
            f, t = int(row[header['To (ID)']].strip()), int(row[header['From (ID)']].strip())

        cap = float(row[header['Cap']].strip())/params.POWER_BASE
        x = float(row[header['Reac']].strip())*(100/params.POWER_BASE)

        if (f, t) in self.LINE_F_T.values():
            l = [l for l in self.LINE_ID if self.LINE_F_T[l] == (f, t)][0]

            (_, self.LINE_X[l], _1, _2, _3, _4, self.LINE_FLOW_UB[l], self.LINE_FLOW_LB[l],
                                    _5, _6) = add_new_parallel_line(
                                                    0, x, 0, 0,
                                                    np.array(params.T*[cap]),
                                                    -1*np.array(params.T*[cap]),
                                                    np.array(params.T*[cap]),
                                                    -1*np.array(params.T*[cap]),
                                                    0,
                                                    self.LINE_X[l], 0, 0,
                                                    self.LINE_FLOW_UB[l], self.LINE_FLOW_LB[l],
                                                    self.LINE_FLOW_UB[l], self.LINE_FLOW_LB[l])

            self.ACTIVE_BOUNDS[l] = self.ACTIVE_BOUNDS[l] or cap*params.POWER_BASE < 99999
            self.ACTIVE_UB[l] = self.ACTIVE_BOUNDS[l]
            self.ACTIVE_LB[l] = self.ACTIVE_BOUNDS[l]
            self.ACTIVE_UB_PER_PERIOD[l] = {t: self.ACTIVE_BOUNDS[l] for t in range(params.T)}
            self.ACTIVE_LB_PER_PERIOD[l] = {t: self.ACTIVE_BOUNDS[l] for t in range(params.T)}

        else:
            l = max(self.LINE_ID,default=0) + 1
            self.LINE_ID.append(l)
            self.LINE_F_T[l] = (f, t)
            self.LINE_FLOW_UB[l] = np.array(params.T*[cap])
            self.LINE_FLOW_LB[l] = -1*np.array(params.T*[cap])
            self.LINE_X[l] = x

            self.LINES_FROM_BUS[f].append(l)
            self.LINES_TO_BUS[t].append(l)

            self.ACTIVE_BOUNDS[l] = cap*params.POWER_BASE < 99999
            self.ACTIVE_UB[l] = self.ACTIVE_BOUNDS[l]
            self.ACTIVE_LB[l] = self.ACTIVE_BOUNDS[l]
            self.ACTIVE_UB_PER_PERIOD[l] = {t: self.ACTIVE_BOUNDS[l] for t in range(params.T)}
            self.ACTIVE_LB_PER_PERIOD[l] = {t: self.ACTIVE_BOUNDS[l] for t in range(params.T)}

    def add_new_DC_link(self, params, row, header):
        """
            Add a new DC link
        """

        f, t = int(row[header['From (ID)']].strip()), int(row[header['To (ID)']].strip())

        l = len(self.LINK_ID)

        self.LINK_ID.append(l)

        self.LINK_F_T[l] = (f, t)

        self.LINK_MAX_P[l] = float(row[header['Cap']].strip())/params.POWER_BASE

        self.LINKS_FROM_BUS[f].append(l)
        self.LINKS_TO_BUS[t].append(l)

    def get_gen_buses(self, thermals, hydros):
        """
            get the buses to which controllable generating elements are connected to
        """

        ALL_HYDRO_BUSES = {bus for h in hydros.ID for u in hydros.UNIT_ID[h]
                                                                for bus in hydros.UNIT_BUS[h][u]}
        ALL_THERMAL_BUSES = {bus for g in thermals.UNIT_NAME.keys() for bus in thermals.BUS[g]}

        return (ALL_THERMAL_BUSES | ALL_HYDRO_BUSES)

    def get_load_buses(self, thermals, hydros):
        """
            get the buses for which in at least one of the periods there is a nonzero load
        """
        ALL_PUMP_BUSES = {bus for h in hydros.ID for u in hydros.UNIT_ID[h]
                            for bus in hydros.UNIT_BUS[h][u] if hydros.TURB_OR_PUMP[h] == 'Pump'}
        return ({bus for bus in self.BUS_ID
                    if np.max(self.NET_LOAD[self.BUS_HEADER[bus]][:]) > 0 or bus in ALL_PUMP_BUSES})

    def get_renewable_gen_buses(self, thermals, hydros):
        """
            get the buses for which in at least one of the periods
                there is a nonzero fixed generation
        """
        return {bus for bus in self.BUS_ID if min(self.NET_LOAD[self.BUS_HEADER[bus]][:]) < 0}
