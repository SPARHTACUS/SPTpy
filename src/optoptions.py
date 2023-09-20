# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""

from os import path
from csv import reader
from numpy import zeros
from enum import Enum

class NetworkModel(Enum):
    """
        Network models
    """
    SINGLE_BUS = 1
    FLUXES = 2
    B_THETA = 3
    PTDF = 4
    @classmethod
    def _missing_(cls, value):
        raise ValueError(f"Valid types: {(', '.join([repr(member.name) for member in cls]),)}")

class Package(Enum):
    """
        Package options
    """
    MIP = 1
    PYOMO_KERNEL = 2
    PYOMO_CONCRETE = 3
    @classmethod
    def _missing_(cls, value):
        raise ValueError(f"Valid types: {(', '.join([repr(member.name) for member in cls]),)}")

class Solver(Enum):
    """
        Solver options
    """
    CBC = 'CBC'
    GRB = 'GRB'
    SCIP = 'SCIP'
    HiGHS = 'HiGHS'
    gurobi_persistent = 'gurobi_persistent'
    mosek_persistent = 'mosek_persistent'
    cplex_persistent = 'cplex_persistent'
    @classmethod
    def _missing_(cls, value):
        raise ValueError(f"Valid types: {(', '.join([repr(member.name) for member in cls]),)}")

class Coupling(Enum):
    """
        Solver options
    """
    VARS = 'vars'
    CONSTRS = 'constrs'
    @classmethod
    def _missing_(cls, value):
        raise ValueError(f"Valid types: {(', '.join([repr(member.name) for member in cls]),)}")

def _set_params_from_file(params, file_name):
    """
        check if there is a params.txt file at the system folder
        in case there is such a file, read its contents and
        overwrite the parameters according to them
    """

    if path.isfile(file_name):
        print('\n\n')
        print(f"Attributes found in file {params.IN_DIR + params.CASE + '/params.txt'}" +
                                            " will overwrite default values of parameters")
        with open(file_name, encoding="ISO-8859-1") as csv_file:
            csv_reader = reader(csv_file, delimiter = '\t')
            for row in [r for r in csv_reader if len(r) > 0 and len(r[0]) > 0 and r[0][0]!="#"]:
                if hasattr(params, row[0].strip()):
                    if not(isinstance(getattr(params, row[0].strip()), dict)):
                        old_value = getattr(params, row[0].strip())
                        new_value = type(getattr(params, row[0].strip()))(row[1].strip())
                        setattr(params, row[0].strip(), new_value)
                        print(f"Attribute {row[0].strip()} changed from {old_value} to {new_value}")
                    else:
                        # if it is a dict
                        old_value = dict(getattr(params, row[0].strip()).items())
                        dict_keys = list(old_value.keys())
                        if isinstance(getattr(params, row[0].strip())[dict_keys[0]], bool):
                            assert row[1].strip() in ('True', 'False', '1', '0')
                            new_value = row[1].strip() in ('True', '1')
                        else:
                            new_value =\
                                type(getattr(params,row[0].strip())[dict_keys[0]])(row[1].strip())
                        for k in dict_keys:
                            getattr(params, row[0].strip())[k] = new_value

                        print(f"Attribute {row[0].strip()} changed from {old_value} to {new_value}")
                else:
                    raise AttributeError("\n\n\n" +
                                    f"Params has no attribute {row[0]}.\n" +
                                    "The .txt file used to overwrite attribute values must "+
                                    "have no header, one pair attribute\tvalue " +
                                    "should be given in each line\n" +
                                    "Attributes and values are separated by tabs and "+
                                    "no special characters are allowed\n" +
                                    "Lines starting with # are ignored")

class OptOptions:
    """
        Optimization and problem parameters
    """

    def __init__(self, root_folder:str, w_rank:int, w_size:int, case:str,
                    n_subhorizons:list, forward_workers:list, backward_workers:list, exp_name:str,
                    solve_only_first_subhorizon = None):
        """
            Initialize the attributes
        """

        self.EXP_NAME = exp_name

        self.I_AM_A_FORWARD_WORKER = (w_rank in forward_workers) or (w_size == 1)
        self.I_AM_A_BACKWARD_WORKER = (w_rank in backward_workers) or (w_size == 1)

        self.N_SUBHORIZONS = n_subhorizons[w_rank] # Number of subhorizons

        self.ASYNCHRONOUS = False if w_size == 1 else True

        self.T = 36                         # number of periods in the planning horizon
        self.REL_GAP_DDiP = 1e-3            # relative gap tolerance for the DDiP
        # maximum number of iterations for the DDiP
        self.MAX_IT_DDiP = 200 if self.ASYNCHRONOUS else 200
        self.TIME_LIMIT = 3*3600            # time limit in seconds
        self.CASE = case#'1'#               # id of the instance
        self.CASE_DATE = None               # only used for the Brazilian system
        self.START = 0                      # used to control the time limit. It is the time stamp
                                            # at which the optimization process begins
        self.LAST_TIME = 0                  # used to control the time limit. It is the final time.
                                            # self.LAST_TIME = self.START + self.TIME_LIMIT
        self.PS = 'ieee118'              # power system ('SIN')

        self.IN_DIR = root_folder + '/input/' + self.PS +'/'
        self.OUT_DIR = ''

        self.PACKAGE = Package.PYOMO_CONCRETE

        self.SOLVER = Solver.HiGHS

        self.COUPLING = Coupling.VARS

        if not(self.ASYNCHRONOUS) and w_rank == 0 and w_size > 1:
            if len(forward_workers) > 1:
                subhsOfBackwards = set(n_subhorizons[bw] for bw in backward_workers)
                if any((len([bw for bw in backward_workers
                                    if n_subhorizons[bw] == ns]) < len(forward_workers)
                                        for ns in subhsOfBackwards)):
                    raise ValueError(
                            'For each forward worker, there should be at least one backward worker'
                                ' for each backward worker partition to avoid ambiguity.' +
                                    ' If you have two forward workers and you want a backward pass'
                                    + ' with 24 subhorizons, then you need two backward workers '
                                    + 'with 24 subhorizons. For three forward workers, you would '
                                    + 'need three backward workers, and so on.')

        # if synchronous optimization is used, then, at the synchronization points, there is a
        # minimum of information that must be received by forward and backward workers before
        # they can proceed
        self.N_CUTS_TO_BE_RECVD = (
                                    len([r for r in range(1, w_size, 1)
                                    if n_subhorizons[r] <= self.N_SUBHORIZONS and
                                    r in backward_workers and not(solve_only_first_subhorizon[r])])
                                        if self.I_AM_A_FORWARD_WORKER and not(self.ASYNCHRONOUS)
                                            else None
                                )

        self.THREADS = 0 if w_size == 1 else (2 if self.ASYNCHRONOUS else 1)

        self.VERBOSE = 0 if w_size > 1 or self.N_SUBHORIZONS > 1 else 1   # print log to console

        self.BASE_TIME_STEP = 1.0                               # In hours

        self.DISCRETIZATION = 1.0                               # In hours

        if w_rank in forward_workers:
            self.SOLVE_ONLY_1st_SUBH = False
        else:
            self.SOLVE_ONLY_1st_SUBH = solve_only_first_subhorizon[w_rank]

        if w_rank == 0:
            self.FIRST_SUBH_FLAGS = [solve_only_first_subhorizon[r] for r in range(w_size)]

        # BDSubhorizonProb applies Benders decomposition to subhorizon problems by separating the
        # binary variables
        self.BD_SUBHORIZON_PROB = (
                                    False if self.I_AM_A_FORWARD_WORKER and self.N_SUBHORIZONS <= 16
                                        else False
                                )
        # decompose the problem into a generation problem and a network problem
        self.BD_NETWORK_SUBHORIZON = (
                                    False if self.I_AM_A_FORWARD_WORKER and self.N_SUBHORIZONS <= 16
                                        else False
                                )
        # BDBackwardProb is used to solve the backward subhorizon's problems with BD
        # by separating the network
        self.BD_BACKWARD_PROB = (
                                False if self.I_AM_A_BACKWARD_WORKER and self.N_SUBHORIZONS <= 4
                                    else False
                                )

        if (not(self.BD_SUBHORIZON_PROB) and not(self.BD_NETWORK_SUBHORIZON) and
                                        not(self.BD_BACKWARD_PROB) and (self.N_SUBHORIZONS == 1)):
            # In case there is no decomposition
            self.MILP_GAP_SUBH = self.REL_GAP_DDiP
        else:
            self.MILP_GAP_SUBH = 1e-4   # relative gap tolerance for the subhorizon MILPs
                                        # solved directly by the optimization software

        # relative gap tolerance for the BD
        self.INNER_BD_GAP = 1e-4
        # relative gap tolerance for the master problem in the BD
        self.INNER_BD_MP_GAP = 1e-5
        # relative gap tolerance for the network BD
        self.INNER_BD_NETWORK_GAP = 1e-4
        # maximum number of iterations for the BD
        self.INNER_BD_MAX_IT = 5

        if self.I_AM_A_BACKWARD_WORKER:
            if self.SOLVE_ONLY_1st_SUBH:
                self.INNER_BD_NETWORK_MAX_IT = 2    # maximum number of iterations for the inner BD
                                                    # i.e., the BD with the network decomposition
            else:
                self.INNER_BD_NETWORK_MAX_IT = 2
        else:
            self.INNER_BD_NETWORK_MAX_IT = 2

        self.DEFICIT_COST = -1e12                # it will be set latter

        self.status = None

        (self.REDUCE_SYSTEM,
            self.REDUNDANT_TRANSM_CONSTRS, self.REDUNDANT_CTF_CONSTRS) = (False, True, True)

        self.POWER_BASE = 100        # in MW
        # scaling for the objective function
        self.SCAL_OBJ_F = 1e-3

        self.winBestSol, self.win_red_flag, self.winUB, self.winLB = (None, None, None, None)

        self.MIN_GEN_CUT_MW:float = 1.00

        self.PTDF_COEFF_TOL:float = 1e-4

        self.MAX_PROCES:int = 1
        self.MAX_NUMBER_OF_CONNECTIONS:int = 20

        self.NETWORK_MODEL = NetworkModel.B_THETA

        _set_params_from_file(self, self.IN_DIR + "/params.txt")
        _set_params_from_file(self, self.IN_DIR + "case " + self.CASE + "/params.txt")

        self.COST_OF_VOL_VIOL = 1e8*self.SCAL_OBJ_F # per hm3

        if abs((self.T % self.N_SUBHORIZONS)) > 0:
            raise ValueError(f'Im W_RANK {w_rank}. My temporal decomposition does not' +
                                            ' match the planning horizon and the discretization.' +
                        f" T is {self.T} while the number of subhorizons is {self.N_SUBHORIZONS}")

        if not(self.I_AM_A_FORWARD_WORKER) and self.BD_SUBHORIZON_PROB:
            raise ValueError('Backward workers cannot use the classical BD')

        if not(self.I_AM_A_BACKWARD_WORKER) and self.BD_BACKWARD_PROB:
            raise ValueError('Flag BD_BACKWARD_PROB is not valid for a forward worker. Use'+
                        ' BDnetworkSubhorizon instead if you want to separate the network problem.')

        if w_rank == 0:
            # a list with the number of subhorizons of each process. this will help the
            # coordinator manage the messages received
            self.N_SUBHS_OF_EACH_PROC = [n_subhorizons[r] for r in range(w_size)]

            # similarly, get the number of periods in each subhorizon of each process
            n_prds_per_subh_per_proc = [
                                            n_subhorizons[r]*[int(self.T/n_subhorizons[r])]
                                                for r in range(w_size)
                                            ]

            # finally, get the periods in each subhorizon of each process
            self.PERIODS_PER_SUBH_PER_PROC = [
                                                [set(range(sum(n_prds_per_subh_per_proc[r][0:b]),
                                                    sum(n_prds_per_subh_per_proc[r][0:b])
                                                        + n_prds_per_subh_per_proc[r][b], 1))
                                                            for b in range(n_subhorizons[r])]
                                                                for r in range(w_size)
                                                ]

            # Get aggregation of forward and backward workers separately
            # the first element accounts for the coordinator
            self.AGG_FORRWARD_WORKERS = [0] + [n_subhorizons[r] for r in range(w_size)
                                                                        if r in forward_workers]
            # the first element accounts for the coordinator
            self.AGG_BACKWARD_WORKERS = [0] + [n_subhorizons[r] for r in range(w_size)
                                                                        if r in backward_workers]

        else:
            # Get aggregation of forward and backward workers separately
            # the first element accounts for the coordinator
            agg_backward_workers = [0] + [n_subhorizons[r] for r in range(w_size)
                                                                        if r in backward_workers]

        # a list with the periods in each subhorizon for this aggregation
        self.PERIODS_PER_SUBH = [
                                    set(range(int(b*int(self.T/self.N_SUBHORIZONS)),
                                        int(b*int(self.T/self.N_SUBHORIZONS))
                                            + int(self.T/self.N_SUBHORIZONS), 1))
                                                for b in range(self.N_SUBHORIZONS)
                                    ]

        # The same, but now for the general coordinator
        self.PERIODS_OF_BACKWARD_WORKERS = [[] for r in backward_workers] + [[]]
        self.PERIODS_OF_BACKWARD_WORKERS[0] = zeros((1, 1), dtype = 'int') #GenCoord

        # the following attribute is used only in the parallel strategy and, in that case, only
        # by backward workers. As the name suggests, it indicates that only the first
        # subhorizon problem is to be solved
        self.FORWARD_SEND_POINTS = []   # subhorizons in the forward process that, after
                                        # solved, guarantee that the current, possibly
                                        # partial, solution can be evaluated in at least
                                        # one of the backward workers
        if (w_size > 1) and w_rank in forward_workers:
            # iterate over the backward workers to look for compatibility
            for r in backward_workers:

                backward_subh = n_subhorizons[r] - 1    # last subhorizon in this backward worker

                # get the periods in each subhorizon of this backward process
                periods_in_subh = [
                                        set(range(int(b*int(self.T/n_subhorizons[r])),
                                            int(b*int(self.T/n_subhorizons[r]))
                                                + int(self.T/n_subhorizons[r]), 1))
                                                    for b in range(n_subhorizons[r])
                                    ]

                # what marks the compatibility of forward and backward workers is the first period
                # of the last subhorizon of the backward worker being the same as the first
                # period of one of the forward workers` subhorizon. Or, in other words, the first
                # period of the last subhorizon of the backward worker needs to be the same as the
                # period immediately subsequent to one of the forward worker`s subhorizon

                for fb in range(self.N_SUBHORIZONS):
                    if (min(periods_in_subh[backward_subh]) - max(self.PERIODS_PER_SUBH[fb])) == 1:
                        self.FORWARD_SEND_POINTS.append(fb)
                        break

        self.BACKWARD_SEND_POINTS = []

        if (w_size > 1) and w_rank in backward_workers:
            for r in [r for r in backward_workers
                                        if r != w_rank and self.N_SUBHORIZONS <= n_subhorizons[r]]:
                # self.backwardSendPoints defines what partial dual solutions are to be sent to
                # other backward workers. for partial solutions, a backward worker only sends them
                # to backward workers whose number of subhorizons is strictly more than its own
                self.BACKWARD_SEND_POINTS.extend(list(range(2, self.N_SUBHORIZONS, 1)))
                # Remember that the last subhorizon solved in the backward is always
                # sent as a complete solution to the general coordinator

        self.BACKWARD_SEND_POINTS = set(self.BACKWARD_SEND_POINTS)

        self.SYNCH_POINTS = set()   # only used if the algorithm is synchronous. these are points
                                    # at which processes have to wait until a certain message is
                                    # received. It needs to match self.backwardSendPoints and
                                    # backwardWorkersMatches of the general coordinator

        self.CUTS_TO_BE_RECVD_AT_SYNCH_POINTS = {-1: 0}

        if self.I_AM_A_BACKWARD_WORKER and not(self.ASYNCHRONOUS) and w_size > 1:
            for r in [r for r in backward_workers
                            if (
                                r != w_rank and
                                        not(solve_only_first_subhorizon[r]) and
                                                        self.N_SUBHORIZONS >= n_subhorizons[r]

                                )]:
                # the synchronization points come from backward processes whose number of
                # subhorizons is less than its own

                # first time period of the first subhorizon of backward worker r
                first_t_in_subh_2 = 0
                for subh_b2 in range(1, n_subhorizons[r], 1):
                    first_t_in_subh_2 += int(self.T/n_subhorizons[r])

                    first_subh = 0 if self.SOLVE_ONLY_1st_SUBH else 1
                    last_t_in_subh_1 = (int(self.T/self.N_SUBHORIZONS) -1
                                                if self.SOLVE_ONLY_1st_SUBH else
                                                        int(2*(self.T/self.N_SUBHORIZONS)) -1)
                                                    # minus 1 because the time starts from zero
                    for subh_b1 in range(first_subh, self.N_SUBHORIZONS - 1, 1):
                        if (last_t_in_subh_1 + 1) == first_t_in_subh_2:
                            # then, the last period in subhorizon subh_b1 of backward worker
                            # w_rank commes immediatelly before the first time period of
                            # subhorizon subh_b2 of backward worker r. thus, a subgradient from
                            # subhorizon subh_b2 of backward worker r can be used
                            # for approximating the future in subhorizon subh_b1 of
                            # backward worker w_rank
                            if self.SOLVE_ONLY_1st_SUBH:
                                self.SYNCH_POINTS.add(subh_b1)

                                if subh_b1 not in self.CUTS_TO_BE_RECVD_AT_SYNCH_POINTS:
                                    self.CUTS_TO_BE_RECVD_AT_SYNCH_POINTS[subh_b1] = 1
                                else:
                                    self.CUTS_TO_BE_RECVD_AT_SYNCH_POINTS[subh_b1] += 1
                            else:
                                # to avoid deadlocks, for processes with the same number of
                                # subhorizons, add all cuts at the end of the iteration, when
                                # all backward subhorizon problems have been solved.
                                self.SYNCH_POINTS.add(-1)
                                self.CUTS_TO_BE_RECVD_AT_SYNCH_POINTS[-1] += 1
                                                        # the '-1' key indicates that it is not
                                                        # to be added during the backward pass
                                                        # because cuts from the first and second
                                                        # subhorizons are not shared among
                                                        # these backward workers
                        last_t_in_subh_1 += int(self.T/self.N_SUBHORIZONS)

        # for each synchronization point in elf.synchronizationPoints, there might be more
        # than one cut to be received. Moreover, the cuts also might be received at varying times.
        # Therefore it is important to keep track of the cuts already received
        self.COUNT_CUTS_RECVD = {subh: {it: 0 for it in range(self.MAX_IT_DDiP)}
                                                for subh in self.CUTS_TO_BE_RECVD_AT_SYNCH_POINTS}

        self.FORWARD_WORKERS = [0] if w_size == 1 else forward_workers
        self.BACKWARD_WORKERS = [0] if w_size == 1 else backward_workers

        self.N_COMPL_VARS = -1000 # number of time-coupling variables. will be set later
        # lower and upper bounds on the time-coupling variables. will be set later
        self.LB_ON_COUPL_VARS, self.UB_ON_COUPL_VARS = ([], [])

        self.MAP ={}# will be used for retrieving variable values from a one-dimensional numpy array

        # indices of all vars in each period, continuous var in t, binary vars in t
        # indices of thermal generation in dispatch phase variables in each period,
        # indices of binary dispatch-phase status in each period
        (self.VARS_PER_PERIOD, self.CON_VARS_PER_PERIOD, self.BIN_VARS_PER_PERIOD,
                                    self.DISP_GEN_VARS_PER_PERIOD, self.BIN_DISP_VARS_PER_PERIOD) =\
                                                    ({t: [] for t in range(self.T)},
                                                        {t: [] for t in range(self.T)},
                                                            {t: [] for t in range(self.T)},
                                                                {t: [] for t in range(self.T)},
                                                                    {t: [] for t in range(self.T)})

        (self.VARS_IN_PREVIOUS_SUBHS, self.VARS_IN_PREVIOUS_AND_CURRENT_SUBHS,
                self.BIN_VARS_PER_SUBH,
                    self.VARS_PER_SUBH, self.BIN_DISP_VARS_PER_SUBH,
                                self.CON_VARS_PER_SUBH, self.DISP_GEN_VARS_PER_SUBH) =\
                                ({b: [] for b in range(self.N_SUBHORIZONS)},
                                    {b: [] for b in range(self.N_SUBHORIZONS)},
                                        {b: [] for b in range(self.N_SUBHORIZONS)},
                                            {b: [] for b in range(self.N_SUBHORIZONS)},
                                                {b: [] for b in range(self.N_SUBHORIZONS)},
                                                    {b: [] for b in range(self.N_SUBHORIZONS)},
                                                        {b: [] for b in range(self.N_SUBHORIZONS)})

        (self.CON_VARS_IN_PREVIOUS_AND_CURRENT_SUBH,
                self.BIN_VARS_IN_PREVIOUS_AND_CURRENT_SUBH) =\
                                                    ({b: [] for b in range(self.N_SUBHORIZONS)},
                                                        {b: [] for b in range(self.N_SUBHORIZONS)})

        self.GEN_CONSTRS_PER_PERIOD = {b: {t: [] for t in range(self.T)}
                                                    for b in range(self.N_SUBHORIZONS)}

        if not(self.ASYNCHRONOUS) and w_rank in backward_workers:
            self.DELAYED_CUTS = {it: {subh: {b_rank: {subhOfSending: None
                                        for subhOfSending in range(agg_backward_workers[b_rank])}
                                            for b_rank in range(len(backward_workers) + 1)}
                                                for subh in range(self.N_SUBHORIZONS)}
                                                    for it in range(self.MAX_IT_DDiP)}
