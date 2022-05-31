# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""

from mip import CBC, GUROBI
from numpy import array, zeros

class OptOptions:
    'Class of optimization and problem parameters'
    def __init__(self, rootFolder: str, rankWorld: int, sizeWorld: int, case: str,\
                    nSubhorizons: list, forwardWs: list, backwardWs: list, expName: str,\
                    solveOnlyFirstSubhorizonInput = None):

        self.setup(rootFolder, rankWorld, sizeWorld, case, nSubhorizons, forwardWs, backwardWs,\
                    expName, solveOnlyFirstSubhorizonInput)

    def setup(self, rootFolder: str, rankWorld: int, sizeWorld: int, case: str,\
                    nSubhorizons: list, forwardWs: list, backwardWs: list, expName:str,\
                    solveOnlyFirstSubhorizonInput):
        'Initialize the attributes'

        self.expName = expName

        self.I_am_a_forwardWorker = (rankWorld in forwardWs) or (sizeWorld == 1)
        self.I_am_a_backwardWorker = (rankWorld in backwardWs) or (sizeWorld == 1)

        self.nSubhorizons = nSubhorizons[rankWorld] # Number of subhorizons

        self.asynchronous = False if sizeWorld == 1 else False

        self.T = 48                     # number of periods in the planning horizon
        self.relGapDDiP = 1e-3          # relative gap tolerance for the DDiP
        self.maxItDDiP =200 if self.asynchronous else 200# maximum number of iterations for the DDiP
        self.timeLimit = 24*3600        # time limit in seconds
        self.case = case#'1'#           # id of the instance
        self.caseDate = None            # only used for the Brazilian system
        self.start = 0                  # used to control the time limit. It is the time stamp
                                        # at which the optimization process begins
        self.lastTime = 0               # used to control the time limit. It is the final time.
                                        # self.lastTime = self.start + self.timeLimit
        self.ps = 'SIN'                 # power system ('SIN')

        self.inputFolder = rootFolder + '/input/' + self.ps +'/'
        self.outputFolder = ''

        self.solver = CBC # GUROBI or CBC

        if not(self.asynchronous) and rankWorld == 0 and sizeWorld > 1:
            if len(forwardWs) > 1:
                subhsOfBackwards = set(nSubhorizons[bw] for bw in backwardWs)
                for ns in subhsOfBackwards:
                    assert len([bw for bw in backwardWs if nSubhorizons[bw]==ns]) >=len(forwardWs),\
                            'For each forward worker, there should be at least one backward worker'\
                                ' for each backward worker partition to avoid ambiguity.' +\
                                    ' If you have two forward workers and you want a backward pass'\
                                    + ' with 24 subhorizons, then you need two backward workers '\
                                    + 'with 24 subhorizons. For three forward workers, you would '\
                                    + 'need three backward workers, and so on.'

        # if synchronous optimization is used, then, at the synchronization points, there is a
        # minimum of information that must be received by forward and backward workers before
        # they can proceed
        self.nCutsTobeReceived = len([r for r in range(1, sizeWorld, 1)\
                                    if nSubhorizons[r] <= self.nSubhorizons and\
                                    r in backwardWs and not(solveOnlyFirstSubhorizonInput[r])])\
                                if self.I_am_a_forwardWorker and not(self.asynchronous) else None

        self.threads = 0 if sizeWorld == 1 else (2 if self.asynchronous else 1)

        if self.solver == GUROBI:
            # for GUROBI, it is -1 (default), 0 (off), # 1 (conservative), or 2 (aggressive)
            # changes GUROBI's presolve parameter
            self.preprocess = 2
        else:
            # for CBC, it is -1 (default), 0 (off), 1 (on), or 2 (equal), 3 (aggregate), 4 (sos)
            self.preprocess = 0

        if self.solver == GUROBI:
            self.heuristics = 0                                 # float 0 <= self.heuristics <= 1
        else:
            self.heuristics = "off"                             # for CBC, either "on" or "off"

        self.verbose = 0 if sizeWorld > 1 or self.nSubhorizons > 1 else 1   # print log to console

        self.baseTimeStep = 0.5                                 # In hours

        self.discretization = 0.5                               # In hours

        # BDSubhorizonProb applies Benders decomposition to subhorizon problems by separating the
        # binary variables
        self.BDSubhorizonProb = False if self.I_am_a_forwardWorker and self.nSubhorizons <= 16 else False
        # decompose the problem into a generation problem and a network problem
        self.BDnetworkSubhorizon = False if self.I_am_a_forwardWorker else False
        # BDBackwardProb is used to solve the backward subhorizon's problems with BD
        # by separating the network
        self.BDBackwardProb = False if self.I_am_a_backwardWorker and self.nSubhorizons <= 4 else False

        if ((self.T % self.nSubhorizons)) > 0:
            raise Exception('Im wRank ' + str(rankWorld) + '. My temporal decomposition does not' +\
                                            ' match the planning horizon and the discretization.')

        if not(self.I_am_a_forwardWorker) and self.BDSubhorizonProb:
            raise Exception('Backward workers cannot use the classical BD')

        if not(self.I_am_a_backwardWorker) and self.BDBackwardProb:
            raise Exception('Flag BDBackwardProb is not valid for a forward worker. Use'+\
                        ' BDnetworkSubhorizon instead if you want to separate the network problem.')

        if rankWorld == 0:
            self.nSubhsOfEachProcess = [nSubhorizons[r] for r in range(sizeWorld)]
            nPeriodsPerSubhOfEachProcess = [nSubhorizons[r]*\
                        [int(self.T/nSubhorizons[r])] for r in range(sizeWorld)]
            self.periodsPerSubhOfEachProcess=[[set(range(sum(nPeriodsPerSubhOfEachProcess[r][0:b]),\
                                                sum(nPeriodsPerSubhOfEachProcess[r][0:b])\
                                                    + nPeriodsPerSubhOfEachProcess[r][b], 1))\
                                                        for b in range(nSubhorizons[r])]\
                                                            for r in range(sizeWorld)]

            # Get aggregation of forward and backward workers separately
            self.aggForwardWorkers = [0]    # the first element accounts for the coordinator
            self.aggBackwardWorkers = [0]   # the first element accounts for the coordinator

            for r in range(sizeWorld):
                if r in forwardWs:
                    self.aggForwardWorkers.append(nSubhorizons[r])
                if r in backwardWs:
                    self.aggBackwardWorkers.append(nSubhorizons[r])
        else:
            # Get aggregation of forward and backward workers separately
            aggBackwardWorkers = [0]   # the first element accounts for the coordinator

            for r in range(sizeWorld):
                if r in backwardWs:
                    aggBackwardWorkers.append(nSubhorizons[r])

        nPeriodsPerSubhorizon = self.nSubhorizons*[int(self.T/self.nSubhorizons)]

        self.periodsPerSubhorizon = [set(range(sum(nPeriodsPerSubhorizon[0:b]),\
                                    sum(nPeriodsPerSubhorizon[0:b]) + nPeriodsPerSubhorizon[b], 1))\
                                                    for b in range(self.nSubhorizons)]

        # Create a numpy-array version of self.periodsPerSubhorizon
        periodsPerSubhorizonList = [list(self.periodsPerSubhorizon[b])\
                                                                for b in range(self.nSubhorizons)]

        # Note that, for an array, all rows must have same number of columns
        # Thus, for rows with less than the maximum number of columns, i.e.,
        # the maximum number of periods in a single subhorizon, add periods 1e6
        maxNumberOfPeriods=max([len(periodsPerSubhorizonList[b]) for b in range(self.nSubhorizons)])

        for b in range(self.nSubhorizons):
            periodsPerSubhorizonList[b] = periodsPerSubhorizonList[b] +\
                                    int(maxNumberOfPeriods - len(periodsPerSubhorizonList[b]))*[1e6]

        for b in range(self.nSubhorizons):
            periodsPerSubhorizonList[b].sort()

        self.periodsPerSubhArray = array(periodsPerSubhorizonList, dtype = 'int')

        # The same, but now for the general coordinator
        self.periodsOfBackwardWs = [[] for r in backwardWs] + [[]]
        self.periodsOfBackwardWs[0] = zeros((1, 1), dtype = 'int') #GenCoord

        # the following attribute is used only in the parallel strategy and, in that case, only
        # by backward workers. As the name suggests, it indicates that only the first
        # subhorizon problem is to be solved
        if rankWorld in forwardWs:
            self.solveOnlyFirstSubhorizon = False
        else:
            self.solveOnlyFirstSubhorizon = solveOnlyFirstSubhorizonInput[rankWorld]

        if rankWorld == 0:
            self.firstSubhorizonFlags = []
            for r in range(sizeWorld):
                self.firstSubhorizonFlags.append(solveOnlyFirstSubhorizonInput[r])

        self.forwardSendPoints = []     # subhorizons in the forward process that, after
                                        # solved, guarantee that the current, possibly
                                        # partial, solution can be evaluated in at least
                                        # one of the backward workers
        if (sizeWorld > 1) and rankWorld in forwardWs:
            for r in backwardWs:
                backwardSubhorizon = nSubhorizons[r] - 1#last subhorizon in this backward worker
                PeriodsPerSubhorizonBackward = nSubhorizons[r]*[int(self.T/nSubhorizons[r])]
                periodsInSubhorizons = [set(range(sum(PeriodsPerSubhorizonBackward[0:b]),\
                                        sum(PeriodsPerSubhorizonBackward[0:b])\
                                        + PeriodsPerSubhorizonBackward[b], 1))\
                                                                    for b in range(nSubhorizons[r])]
                firstPeriodInLastSubhorizon = min(periodsInSubhorizons[backwardSubhorizon])
                for fb in range(self.nSubhorizons):
                    if (firstPeriodInLastSubhorizon - max(self.periodsPerSubhorizon[fb])) == 1:
                        self.forwardSendPoints.append(fb)
                        break

        self.backwardSendPoints = []

        if (sizeWorld > 1) and rankWorld in backwardWs:
            for r in [r for r in backwardWs if r!=rankWorld and self.nSubhorizons<=nSubhorizons[r]]:
                # self.backwardSendPoints defines what partial dual solutions are to be sent to
                # other backward workers. for partial solutions, a backward worker only sends them
                # to backward workers whose number of subhorizons is strictly more than its own
                self.backwardSendPoints.extend(list(range(2, self.nSubhorizons, 1)))
                # Remember that the last subhorizon solved in the backward is always
                # sent as a complete solution to the general coordinator

        self.backwardSendPoints = set(self.backwardSendPoints)

        self.synchronizationPoints = set()
                                    # only used if the algorithm is synchronous. these are points
                                    # at which processes have to wait until a certain message is
                                    # received. It needs to match self.backwardSendPoints and
                                    # backwardWorkersMatches of the general coordinator

        self.cutsToBeReceivedAtSynchPoints = {-1: 0}

        if self.I_am_a_backwardWorker and not(self.asynchronous) and sizeWorld > 1:
            for r in [r for r in backwardWs if r != rankWorld and\
                                                        not(solveOnlyFirstSubhorizonInput[r]) and\
                                                        self.nSubhorizons >= nSubhorizons[r]]:
                # the synchronization points come from backward processes whose number of
                # subhorizons is less than its own
                if self.nSubhorizons >= nSubhorizons[r] or self.solveOnlyFirstSubhorizon:
                    # first time period of the first subhorizon of backward worker r
                    firstTinSubh2 = 0
                    for subhOfB2 in range(1, nSubhorizons[r], 1):
                        firstTinSubh2 += int(self.T/nSubhorizons[r])

                        firstSubh = 0 if self.solveOnlyFirstSubhorizon else 1
                        lastTinSubh1 = int(self.T/self.nSubhorizons) -1\
                                                        if self.solveOnlyFirstSubhorizon else\
                                                                int(2*(self.T/self.nSubhorizons)) -1
                                                        # minus 1 because the time starts from zero
                        for subhOfB1 in range(firstSubh, self.nSubhorizons - 1, 1):
                            if (lastTinSubh1 + 1) == firstTinSubh2:
                                # then, the last period in subhorizon subhOfB1 of backward worker
                                # rankWorld commes immediatelly before the first time period of
                                # subhorizon subhOfB2 of backward worker r. thus, a subgradient from
                                # subhorizon subhOfB2 of backward worker r can be used
                                # for approximating the future in subhorizon subhOfB1 of
                                # backward worker rankWorld
                                if self.solveOnlyFirstSubhorizon:
                                    self.synchronizationPoints.add(subhOfB1)

                                    if subhOfB1 not in self.cutsToBeReceivedAtSynchPoints:
                                        self.cutsToBeReceivedAtSynchPoints[subhOfB1] = 1
                                    else:
                                        self.cutsToBeReceivedAtSynchPoints[subhOfB1] += 1
                                else:
                                    # to avoid deadlocks, for processes with the same number of
                                    # subhorizons, add all cuts at the end of the iteration, when
                                    # all backward subhorizon problems have been solved.
                                    self.synchronizationPoints.add(-1)
                                    self.cutsToBeReceivedAtSynchPoints[-1] += 1
                                                            # the '-1' key indicates that it is not
                                                            # to be added during the backward pass
                                                            # because cuts from the first and second
                                                            # subhorizons are not shared among
                                                            # these backward workers
                            lastTinSubh1 += int(self.T/self.nSubhorizons)

        # for each synchronization point in elf.synchronizationPoints, there might be more
        # than one cut to be received. Moreover, the cuts also might be received at varying times.
        # Therefore it is important to keep track of the cuts already received
        self.countCutsReceived = {subh: {it: 0 for it in range(self.maxItDDiP)}\
                                                    for subh in self.cutsToBeReceivedAtSynchPoints}

        self.forwardWorkers = [0] if sizeWorld == 1 else forwardWs
        self.backwardWorkers = [0] if sizeWorld == 1 else backwardWs

        if not(self.BDSubhorizonProb) and not(self.BDnetworkSubhorizon) and\
                                            not(self.BDBackwardProb) and (self.nSubhorizons == 1):
            # In case there is no decomposition
            self.gapMILPSubh = self.relGapDDiP
        else:
            self.gapMILPSubh = 1e-4     # relative gap tolerance for the subhorizon MILPs
                                        # solved directly by the optimization software

        self.gapInnerBD = 1e-4          # relative gap tolerance for the BD
        self.gapInnerBDMP = 1e-5        # relative gap tolerance for the master problem in the BD
        self.gapInnerBDnetwork = 1e-4   # relative gap tolerance for the network BD
        self.maxITinnerBD =  5          # maximum number of iterations for the BD

        if self.I_am_a_backwardWorker:
            if self.solveOnlyFirstSubhorizon:
                self.maxITinnerBDnetwork = 2    # maximum number of iterations for the inner BD
                                                # i.e., the BD with the network decomposition
            else:
                self.maxITinnerBDnetwork = 2
        else:
            self.maxITinnerBDnetwork = 2

        self.deficitCost = -1e12                # it will be set latter

        self.status = None

        self.removeParallelTransmLines, self.reduceSystem,\
                    self.redundTransmConstr, self.redundCostToGoConstr = True, True, True, True

        self.powerBase = 100        # in MW
        # scaling for the objective function
        self.scalObjF = 1e-6
        self.massBal = 1e2          # scaling constraints the water-balance constraints

        self.costOfVolViolation = 1e8*self.scalObjF # per hm3

        self.c_h = self.discretization*(3600*1e-6)  # from flow in m3/s to volume in hm3

        self.nComplVars = -1000 # number of time-coupling variables. will be set later
        # lower and upper bounds on the time-coupling variables. will be set later
        self.lbOnCouplVars, self.ubOnCouplVars  = [], []

        self.map ={}# will be used for retrieving variable values from a one-dimensional numpy array

        # indices of all vars in each period, continuous var in t, binary vars in t
        # indices of thermal generation in dispatch phase variables in each period,
        # indices of binary dispatch-phase status in each period
        self.varsPerPeriod, self.conVarsPerPeriod, self.binVarsPerPeriod, \
                                    self.dispGenVarsPerPeriod, self.binDispVarsPerPeriod =\
                                                    {t: [] for t in range(self.T)},\
                                                        {t: [] for t in range(self.T)},\
                                                            {t: [] for t in range(self.T)},\
                                                                {t: [] for t in range(self.T)},\
                                                                    {t: [] for t in range(self.T)}

        self.varsInPreviousSubhs, self.varsInPreviousAndCurrentSubh, self.binVarsPerSubh,\
                    self.varsPerSubh, self.binDispVarsPerSubh,\
                                self.conVarsPerSubh, self.dispGenVarsPerSubh =\
                            {b: [] for b in range(self.nSubhorizons)},\
                                    {b: [] for b in range(self.nSubhorizons)},\
                                        {b: [] for b in range(self.nSubhorizons)},\
                                            {b: [] for b in range(self.nSubhorizons)},\
                                                {b: [] for b in range(self.nSubhorizons)},\
                                                    {b: [] for b in range(self.nSubhorizons)},\
                                                        {b: [] for b in range(self.nSubhorizons)}


        self.contVarsInPreviousAndCurrentSubh, self.binVarsInPreviousAndCurrentSubh =\
                                                {b: [] for b in range(self.nSubhorizons)},\
                                                        {b: [] for b in range(self.nSubhorizons)}

        self.genConstrsPerPeriod = {b: {t: [] for t in range(self.T)}\
                                                    for b in range(self.nSubhorizons)}

        if not(self.asynchronous) and rankWorld in backwardWs:
            self.delayedCuts = {it: {subh: {bRank: {subhOfSending: None\
                                        for subhOfSending in range(aggBackwardWorkers[bRank])}\
                                            for bRank in range(len(backwardWs) + 1)}\
                                                for subh in range(self.nSubhorizons)}\
                                                    for it in range(self.maxItDDiP)}

        self.winBestSol, self.winRedFlag, self.winUB, self.winLB = None, None, None, None
