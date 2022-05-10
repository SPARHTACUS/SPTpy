# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""

from math import inf

class Hydros:
    'Class of hydro plants'
    def __init__(self):
        self.setup()

    def setup(self):
        'Initialize the attributes'
        self.id = []
        self.name = []

        self.turbType , self.turbOrPump, self.convMWm3s = [], [], []

        self.minVol, self.maxVol = [], []

        self.downRiverPlantName, self.downRiverPlantID = [], []

        self.upRiverPlantNames, self.upRiverPlantIDs = [], []

        self.travelTime = {}

        self.minForebayL, self.maxForebayL = [], []

        self.maxSpil, self.maxSpilHPF = [], []

        self.minOutflowHPF,self.maxOutflowHPF,self.minTurbDischHPF,self.maxTurbDischHPF=[],[],[],[]

        self.hpfModel = []
        self.inflOfSpil = []

        self.nQPoints, self.nVPoints, self.nSPoints, self.vRange = [], [], [], []

        # Parameters of the 'original' hydropower function
        self.avrgProd, self.headLossType, self.headLoss, self.hpfFB, self.hpfTR, self.nTailracePol=\
                                                                            [], [], [], [], [], []

        self.downRiverPlantTransferName = []
        self.downRiverTransferPlantID = []      # Plant that receives
        self.upRiverTransferPlantID = []        # Plant that transfers
        self.maxTransfer = []
        self.transferTravelTime = {}

        self.downRiverPumps = []

        self.dnrOfPumps, self.uprOfPumps, self.dnrOfPumpsID, self.uprOfPumpsID  = [], [], [], []

        self.plantsMaxPump = []
        self.pumpageWaterTTime = {}

        # Plants ids and names in the short-term model DECOMP
        self.idDECOMP, self.nameDECOMP = [], []

        self.basin = []             # plants' basin
        self.basinOfSubbasin = {}   # The basin to which a subbasin belongs to
        self.buses = []
        self.groupsOfUnits = []     # number of groups of units
        self.unitsInGroups = []     # units in each of the plants' groups
        self.groupOfGivenUnit = []  # returns the group of a specific unit
        self.busesOfEachGroup = []  #

        # Initial state
        self.V0, self.q0, self.spil0 = [], [], []

        # The following are attributes of individual generating units
        self.unitID, self.unitName, self.unitGroup, self.unitBus = [], [], [], []

        self.unitMinPower,self.unitMaxPower,self.unitMinTurbDisc,self.unitMaxTurbDisc = [],[],[],[]

        # The following attributes are related to the plant as a whole, and
        # they are computed by summing the attributes of the generating units
        self.plantMinPower, self.plantMaxPower = [], []

        self.plantMinTurbDisc, self.plantMaxTurbDisc = [], []

        self.plantBuses = []        # Buses to which generating units are connected
        self.plantBusesCap = []     # Installed capacity connected to the buses in self.plantBuses

        # The coefficients of the aggregated hydropower function are as follows
        self.A0 = []    # Coefficient of the turbine discharge
        self.A1 = []    # Coefficient of the reservoir volume
        self.A2 = []    # Coefficient of the spillage
        self.A3 = []    # Constant term

        # Coefficients of the cost-to-go function
        self.CTF = []
        # RHS of the cost-to-go function
        self.CTFrhs = []

        # Include the zero turbine discharge points when approximating the HPF
        self.includeZeroQ = []

        # inflows to reservoirs in m3/s
        self.inflows = []

        # bounds on generation of groups of plants. Analogous to that of the thermal units
        self.maxGen, self.minGen, self.equalityConstrs = [], [], []

    def addNewHydro(self, params, row, header):
        'Add a new hydro plant'

        self.id.append(int(row[header['ID']]))
        self.name.append(row[header['Name']].strip())
        self.minVol.append(round(float(row[header['MinVol']]), 6))
        self.maxVol.append(round(float(row[header['MaxVol']]), 6))
        self.downRiverPlantName.append(row[header['Downriver reservoir']].strip())

        for t in range(params.T):
            if params.discretization <= params.baseTimeStep:
                self.travelTime[(row[header['Name']].strip(),\
                                row[header['Downriver reservoir']].strip(), t)] =\
                                int(int(float(row[header['WaterTravT']]))*(1/params.baseTimeStep))
            else:
                self.travelTime[(row[header['Name']].strip(),\
                                row[header['Downriver reservoir']].strip(), t)] = 0

        self.minForebayL.append(float(row[header['MinForebay']]))
        self.maxForebayL.append(float(row[header['MaxForebay']]))
        self.maxSpil.append(float(row[header['MaxSpil']]))

        if (row[header['InflOfSpill']] == 'No'):
            self.inflOfSpil.append(False)
        elif (row[header['InflOfSpill']] == 'Yes'):
            self.inflOfSpil.append(True)
        else:
            raise Exception(row[header['InflOfSpill']] + ' is not a valid '\
              'value for the influence of spillage on the HPF of hydro plant '+ row[header['Name']])

        self.maxSpilHPF.append(float(row[header['MaxSpilHPF']]))
        self.minOutflowHPF.append(0)
        self.maxOutflowHPF.append(0)
        self.minTurbDischHPF.append(0)
        self.maxTurbDischHPF.append(0)

        self.hpfModel.append('')

        self.downRiverPumps.append([])

        self.downRiverPlantTransferName.append(row[header['DRTransfer']].strip())
        self.upRiverTransferPlantID.append([])
        self.maxTransfer.append(float(row[header['MaxTransfer']]))

        self.transferTravelTime.update({(len(self.id), t): 0\
                                                    for t in range(params.T)})
        for t in range(params.T):
            if params.discretization <= params.baseTimeStep:
                self.transferTravelTime[len(self.id), t] = int(row[header['TransferTravelTime']])

        self.dnrOfPumps.append(row[header['DRPump']].strip())
        self.uprOfPumps.append(row[header['UPRPump']].strip())
        self.plantsMaxPump.append(0)

        self.pumpageWaterTTime.update({(len(self.id), t): 0\
                                                    for t in range(params.T)})
        for t in range(params.T):
            if params.discretization <= params.baseTimeStep:
                self.pumpageWaterTTime[len(self.id), t] = int(row[header['PumpTravelTime']])

        self.idDECOMP.append(int(row[header['idDECOMP']]))
        self.nameDECOMP.append(row[header['nameDECOMP']].strip())

        self.nQPoints.append(-1e3)
        self.nVPoints.append(-1e3)
        self.nSPoints.append(-1e3)
        self.vRange.append(-1e3)

        self.basin.append(row[header['Basin']].strip())
        self.groupsOfUnits.append([])
        self.unitsInGroups.append([])
        self.groupOfGivenUnit.append([])
        self.busesOfEachGroup.append([])

        self.upRiverPlantNames.append([])
        self.upRiverPlantIDs.append([])

        self.hpfFB.append({'F0': 0, 'F1': 0, 'F2': 0, 'F3': 0, 'F4': 0})
        self.hpfTR.append({'T0': [], 'T1': [], 'T2': [], 'T3': [], 'T4': []})

        self.avrgProd.append(0)
        self.headLossType.append(0)
        self.headLoss.append(0)
        self.nTailracePol.append(0)

        self.V0.append(0)
        self.q0.append(0)
        self.spil0.append(0)

        self.unitID.append([])
        self.unitName.append([])
        self.unitGroup.append([])
        self.unitBus.append([])
        self.unitMinPower.append([])
        self.unitMaxPower.append([])
        self.unitMinTurbDisc.append([])
        self.unitMaxTurbDisc.append([])
        self.turbType.append([])

        self.turbOrPump.append('')
        self.convMWm3s.append(0)

        self.plantMinPower.append(1e12)
        self.plantMaxPower.append(0)
        self.plantMinTurbDisc.append(1e12)
        self.plantMaxTurbDisc.append(0)
        self.plantBuses.append([])
        self.plantBusesCap.append([])

        self.A0.append([])
        self.A1.append([])
        self.A2.append([])
        self.A3.append([])

        self.includeZeroQ.append(True)

        return()

class Thermals:
    'Class of thermal units'
    def __init__(self):
        self.setup()

    def setup(self):
        'Initialize the attributes'
        self.id = []
        self.name = []

        self.minP, self.maxP  = [], []      # power limits

        self.genCost = []                   # unitary cost in $/(p.u.) for a 30-min period

        self.rampUp, self.rampDown = [], [] # ramps

        self.minUp, self.minDown = [], []   # minimum times

        self.bus = []                       # bus to which the unit is connected
        self.constCost, self.stUpCost, self.stDwCost = [], [], []

        # Previous state
        self.state0 = []
        self.tg0 = []
        self.nHoursInPreState = []

        # minimum and maximum generation of groups of units
        self.minGen, self.maxGen = [], []

        # plants and units IDs and names in the DESSEM model. only used for the Brazilian system
        self.plantsIDDESSEM = []
        self.plantsNameDESSEM = []
        self.unitsIDDESSEM = []

        self.startUpTraj = []       # power steps in the start-up trajectory
        self.shutDownTraj = []      # power steps in the shut-down trajectory

        self.inStartUpTraj = []     # indicates whether that units starts the optimization horizon
                                    # in the start-up trajectory, i.e., it was in this trajectory
                                    # before
        self.inShutDownTraj = []

        self.sdDec = []             # gives the minimum time period at which the unit can be turned
                                    # off. it is defined in readAndLoadData.loadDatap.py and it
                                    # depends on the unit's previous status as well as its limits

        self.equalityConstrs = []   # groups of thermal units may have their combined power output
                                    # set to a fixed level.

        self.maxPeriodsInDisp = []
        self.minPeriodsInDisp = []

        return()

    def addNewThermal(self, params, row, header):
        'Add a new thermal unit'

        self.id.append(int(row[header['ID']]))
        self.name.append(row[header['Name']])
        self.minP.append(float(row[header['minP']])/params.powerBase)
        self.maxP.append(float(row[header['maxP']])/params.powerBase)
        self.genCost.append(params.discretization*\
                                    params.powerBase*float(row[header['genCost']])*params.scalObjF)

        self.rampUp.append(params.discretization*float(row[header['rampUp']])/params.powerBase)
        self.rampDown.append(params.discretization*float(row[header['rampDown']])/params.powerBase)

        self.minUp.append(int(row[header['minUp']]))
        self.minDown.append(int(row[header['minDown']]))
        self.bus.append(int(row[header['bus']]))
        self.constCost.append(float(row[header['constCost']])*params.scalObjF)
        self.stUpCost.append(float(row[header['stUpCost']])*params.scalObjF)
        self.stDwCost.append(float(row[header['stDwCost']])*params.scalObjF)
        self.state0.append(0)
        self.tg0.append(0)
        self.nHoursInPreState.append(0)

        self.plantsIDDESSEM.append(int(row[header['plantsIDDESSEM']].strip()))
        self.plantsNameDESSEM.append(row[header['plantsNameDESSEM']].strip())
        self.unitsIDDESSEM.append(int(row[header['unitsIDDESSEM']].strip()))

        self.inStartUpTraj.append(False)
        self.inShutDownTraj.append(False)

        self.sdDec.append(0)

        self.startUpTraj.append([])
        self.shutDownTraj.append([])

        self.maxPeriodsInDisp.append(params.T)
        self.minPeriodsInDisp.append(0)

        return()

class Network:
    'Class of transmission network with DC model'
    def __init__(self):
        self.setup()

    def setup(self):
        'Initialize the attributes'

        self.subSysIDs = []
        self.subSysNames = []
        self.busesInSubSys = {}
        self.linesInSubSys = {}
        self.subSysIslands = {}
        self.dcsInSubSys = {}

        self.busID = []
        self.busName = []
        self.busSubSyst = {}
        self.baseVoltage = {}
        self.area = {}
        self.submarketName = {}
        self.submarketID = {}
        self.busesInIsland = {}
        self.refBusesOfIslands = {}

        self.genBuses = set()           # buses with generation
        self.renewableGenBuses = set()  # is a subset of self.genBuses
        self.loadBuses = set()          # buses with loads
        self.dcBuses = set()            # buses to which a dc link is connected to

        self.tieLines = {}
        self.tieDClinks = {}
        self.lineIsland = {}
        self.islandOfLine = {}

        # the following dictionary will be populated for each bus of the system, i.e., each bus will
        # be a dict key.
        self.AClinesFromBus = {}
        self.AClinesToBus = {}
        self.DClinksFromBus = {}
        self.DClinksToBus = {}

        # these will be defined for each AC transmission line (as opposed to a DC link) of the
        # system. Despite the name, in the model, flows in the AC transmission lines are linear
        # functions of bus angles
        self.AClineFromTo = {}
        self.AClineUBCap = {}
        self.AClineLBCap = {}
        self.AClineAdmt = {}

        # the following two dicts are defined for each DC link of the system. Different from the
        # AC lines, the flows in these links are not functions of bus angles
        self.DClinkFromTo = {}
        self.DClinkCap = {}

        self.loadHeader = {}                # Gets bus ID and returns bus index in the load
        self.load = []                      # will be a numpy array containing the load at each
                                            # bus and time period
        self.subsysLoad = {}                # total load in p.u. for each subsystem at each
                                            # time period

        self.deficitCost = -1e12

        self.flowsBetweenSubsystems = {}    # dictionary of maximum power exchanges between
                                            # subsystems

        self.subsystems = []                # subsystems within the system

        self.listOfThermalsInSubsys = {}    # list of thermal units in a subsystem
        self.listOfHydrosInSubsys = {}      # list of hydro plants in a subsystem

        # the following is only used for the Brazilian system
        self.ANDEload =None     # is Paraguay's load at each time step (it is on the 50 Hz)

        self.thetaBound = inf   # bound on the buses' voltage angles

        return()

    def addNewSubSystem(self, row, header):
        'Add a new subsystem'

        self.subSysIDs.append(int(row[header['ID']].strip()))
        self.subSysNames.append(row[header['Name']].strip())
        self.busesInSubSys[self.subSysNames[-1]] = []
        self.linesInSubSys[self.subSysNames[-1]] = []
        self.dcsInSubSys[self.subSysNames[-1]] = []
        self.tieLines[self.subSysNames[-1]] = []
        self.subSysIslands[self.subSysNames[-1]] = [sub.strip() \
                                            for sub in row[header['Islands']].strip().split(',')]
        return()

    def addNewBus(self, row, header):
        'Add a new bus'

        bus = int(row[header['ID']].strip())
        self.busID.append(bus)
        self.busName.append(row[header['Name']].strip())
        self.busSubSyst[bus] = row[header['ElecSub']].strip()
        self.baseVoltage[bus] = float(row[header['baseVoltage']].strip())
        self.area[bus] = int(row[header['area']].strip())
        self.submarketName[bus] = row[header['submName']].strip()
        self.submarketID[bus] = int(row[header['submID']].strip())

        self.busesInSubSys[row[header['ElecSub']].strip()].append(bus)

        if not(row[header['Island']].strip() in self.busesInIsland.keys()):
            self.busesInIsland[row[header['Island']].strip()] = [bus]
        else:
            self.busesInIsland[row[header['Island']].strip()].append(bus)

        if (row[header['Reference bus']].strip() == 'Ref'):
            self.refBusesOfIslands[row[header['Island']].strip()] = bus

        self.AClinesFromBus[bus] = set()
        self.AClinesToBus[bus] = set()
        self.DClinksFromBus[bus] = set()
        self.DClinksToBus[bus] = set()
        return()

    def addNewACline(self, params, row, header):
        'Add a new AC line'

        if int(row[header['From (ID)']].strip()) < \
                                            int(row[header['To (ID)']].strip()):
            f = int(row[header['From (ID)']].strip())
            t = int(row[header['To (ID)']].strip())
            fromSys = row[header['Tie line - sys from']].strip()
            toSys = row[header['Tie line - sys to']].strip()
        else:
            f = int(row[header['To (ID)']].strip())
            t = int(row[header['From (ID)']].strip())
            fromSys = row[header['Tie line - sys to']].strip()
            toSys = row[header['Tie line - sys from']].strip()

        l = len(self.AClineFromTo)
        self.AClineFromTo[l] = (f, t)
        self.AClineUBCap[l] = float(row[header['Cap']].strip())\
                                                    /params.powerBase
        self.AClineLBCap[l] = -1*self.AClineUBCap[l]
        self.AClineAdmt[l] = (1/(float(row[header['Reac']].strip())/100))

        if not(row[header['Island']].strip() in self.lineIsland.keys()):
            self.lineIsland[row[header['Island']].strip()] = [l]
        else:
            self.lineIsland[row[header['Island']].strip()].append(l)

        self.islandOfLine[l] = row[header['Island']].strip()

        if (row[header['ElecSub']].strip() == 'Tie line'):
            self.tieLines[fromSys].append(l)
            self.tieLines[toSys].append(l)

            self.linesInSubSys[fromSys].append(l)
            self.linesInSubSys[toSys].append(l)
        else:
            self.linesInSubSys[row[header['ElecSub']].strip()].append(l)

        self.AClinesFromBus[f].add(l)
        self.AClinesToBus[t].add(l)

        return()

    def addNewAClineReducedSystem(self, params, row, header):
        'add a new AC line: if it is parallel to a existing line, then combine them'

        if int(row[header['From (ID)']].strip()) < int(row[header['To (ID)']].strip()):
            f = int(row[header['From (ID)']].strip())
            t = int(row[header['To (ID)']].strip())
            fromSys = row[header['Tie line - sys from']].strip()
            toSys = row[header['Tie line - sys to']].strip()
        else:
            f = int(row[header['To (ID)']].strip())
            t = int(row[header['From (ID)']].strip())
            fromSys = row[header['Tie line - sys to']].strip()
            toSys = row[header['Tie line - sys from']].strip()

        cap = float(row[header['Cap']].strip())/params.powerBase
        admt = (1/(float(row[header['Reac']].strip())/100))

        if ((f, t) in list(self.AClineFromTo.values())):
            l = list(self.AClineFromTo.values()).index((f, t))
            dthetaMax = 1e12
            if abs(cap/admt) < dthetaMax:
                dthetaMax = cap/admt
            if abs(self.AClineUBCap[l]/self.AClineAdmt[l]) < abs(dthetaMax):
                dthetaMax = self.AClineUBCap[l]/self.AClineAdmt[l]
            #dthetaMax = min(cap/admt, self.AClineUBCap[l]/self.AClineAdmt[l])
            self.AClineUBCap[l] = abs(dthetaMax)*abs(admt + self.AClineAdmt[l])
            self.AClineAdmt[l] = self.AClineUBCap[l]/dthetaMax
            self.AClineLBCap[l] = -1*self.AClineUBCap[l]
        else:
            l = len(self.AClineFromTo)
            self.AClineFromTo[l] = (f, t)
            self.AClineUBCap[l] = cap
            self.AClineLBCap[l] = -1*self.AClineUBCap[l]
            self.AClineAdmt[l] = admt

            if not(row[header['Island']].strip() in self.lineIsland.keys()):
                self.lineIsland[row[header['Island']].strip()] = [l]
            else:
                self.lineIsland[row[header['Island']].strip()].append(l)

            self.islandOfLine[l] = row[header['Island']].strip()

            if (row[header['ElecSub']].strip() == 'Tie line'):
                self.tieLines[fromSys].append(l)
                self.tieLines[toSys].append(l)

                self.linesInSubSys[fromSys].append(l)
                self.linesInSubSys[toSys].append(l)
            else:
                self.linesInSubSys[row[header['ElecSub']].strip()].append(l)

            self.AClinesFromBus[f].add(l)
            self.AClinesToBus[t].add(l)

        return()

    def addNewDClink(self, params, row, header):
        'Add a new DC link'

        f = int(row[header['From (ID)']].strip())
        t = int(row[header['To (ID)']].strip())
        fromSys = row[header['Tie line - sys from']].strip()
        toSys = row[header['Tie line - sys to']].strip()

        l = len(self.DClinkFromTo)

        if (row[header['ElecSub']].strip() == 'Tie line'):
            self.tieDClinks[l] = {'FromSys': fromSys, 'ToSys': toSys}

            self.dcsInSubSys[fromSys].append(l)
            self.dcsInSubSys[toSys].append(l)
        else:
            self.dcsInSubSys[row[header['ElecSub']].strip()].append(l)

        self.DClinkFromTo[l] = (f, t)
        self.DClinkCap[l] = float(row[header['Cap']].strip())/params.powerBase

        self.DClinksFromBus[f].add(l)
        self.DClinksToBus[t].add(l)

        return()
