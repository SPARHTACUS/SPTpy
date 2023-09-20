"""
@author: Colonetti
"""

from enum import Enum
import logging
import time
import numpy as np

from optoptions import Package, Solver

try:
    import mip as mip
except ImportError:
    pass

try:
    import pyomo.kernel as pm
    import pyomo.environ as pe
    from pyomo.repn.standard_repn import generate_standard_repn
except ImportError:
    pass

logging.getLogger('pyomo').setLevel(logging.CRITICAL)

# choice for continuous optimization methods
class LP_Method(Enum):
    """Different methods to solve the linear programming problem. *from python-mip"""

    AUTO = 0
    """Let the solver decide which is the best method"""

    DUAL = 1
    """The dual simplex algorithm"""

    PRIMAL = 2
    """The primal simplex algorithm"""

    BARRIER = 3
    """The barrier algorithm"""

    BARRIERNOCROSS = 4
    """The barrier algorithm without performing crossover"""

# optimization status
class OptimizationStatus(Enum):
    """Status of the optimization
        *from python-mip
    """

    ERROR = -1
    """Solver returned an error"""

    OPTIMAL = 0
    """Optimal solution was computed"""

    INFEASIBLE = 1
    """The model is proven infeasible"""

    UNBOUNDED = 2
    """One or more variables that appear in the objective function are not
       included in binding constraints and the optimal objective value is
       infinity."""

    FEASIBLE = 3
    """An integer feasible solution was found during the search but the search
       was interrupted before concluding if this is the optimal solution or
       not."""

    INT_INFEASIBLE = 4
    """A feasible solution exist for the relaxed linear program but not for the
       problem with existing integer variables"""

    NO_SOLUTION_FOUND = 5
    """A truncated search was executed and no integer feasible solution was
    found"""

    LOADED = 6
    """The problem was loaded but no optimization was performed"""

    CUTOFF = 7
    """No feasible solution exists for the current cutoff"""

    OTHER = 10000

    def __eq__(self, other):
        return self.name == other.name and self.value == other.value

def _solver_status(package, solver_status, optimization_time, time_limit):
    """Return the status of the solver"""

    if package == Package.MIP:
        return solver_status

    if package == Package.PYOMO_KERNEL:
        if solver_status.solver.status is None:
            return OptimizationStatus.OTHER

        if solver_status.solver.status == pm.SolverStatus.ok:
            if solver_status.solver.termination_condition in (pm.TerminationCondition.optimal,
                                                        pm.TerminationCondition.globallyOptimal):
                return OptimizationStatus.OPTIMAL

            if ((solver_status.solver.termination_condition ==
                                                        pm.TerminationCondition.maxTimeLimit) or
                (optimization_time >= time_limit)):
                return OptimizationStatus.OTHER

            if (solver_status.solver.termination_condition in
                                                (pm.TerminationCondition.infeasible,
                                                    pm.TerminationCondition.infeasibleOrUnbounded)):
                return OptimizationStatus.INFEASIBLE

            return OptimizationStatus.OTHER

        if solver_status.solver.status == pm.SolverStatus.aborted:
            return OptimizationStatus.OTHER

    if package == Package.PYOMO_CONCRETE:
        if solver_status.termination_condition.value == 5:
            return OptimizationStatus.OPTIMAL

        if ((solver_status.termination_condition.value == 1) or (optimization_time >= time_limit)):
            return OptimizationStatus.OTHER

        if (solver_status.termination_condition.value in (9, 10)):
            return OptimizationStatus.INFEASIBLE

        return OptimizationStatus.OTHER

class peConcreteConstr:
    """Decision variable"""
    def __init__(self, model_interface, m, lhs, sign, rhs, name):

        if sign == '==':
            setattr(m, name, pe.Constraint(expr = lhs == rhs))
        elif sign == '>=':
            setattr(m, name, pe.Constraint(expr = lhs >= rhs))
        elif sign == '<=':
            setattr(m, name, pe.Constraint(expr = lhs <= rhs))
        else:
            raise ValueError(f"sign should be either ==, >= or <= and not {sign}")

        self.constr = getattr(m, name)
        self.m = m
        self.model_interface = model_interface
        self.rhs_param = rhs
        self.name = name

    @property
    def pi(self):
        """get the value of the dual variable associated with the constraint"""
        return self.model_interface._all_duals[self.constr]

    @property
    def rhs(self):
        """get the RHS of the constraint"""
        return self.rhs_param.value

    @rhs.setter
    def rhs(self, new_value):
        """change the RHS of the constraint"""
        self.rhs_param.value = new_value

class pmConstr(pm.constraint):
    """Decision variable"""
    def __init__(self, expression):
        #super keeps __init__ from the parent class
        super().__init__(expression)
        self.m = None

    @property
    def pi(self):
        """get the value of the dual variable associated with the constraint"""
        return self.m.dual[self]

class Model:
    """An optimization model"""

    def __init__(self, model_name:str = None, solver_name:str = Solver.GRB,
                                                                        package:str = Package.MIP):

        self.PACKAGE = package
        self.SOLVER_NAME = solver_name

        self.MODEL_NAME = 'm' if model_name is None else model_name

        self._rhs_params_pe = {}

        self._vars = []

        (self._all_duals, self._all_reduced_costs) = (None, None)

        self._status = 0
        self._SolCount = 0
        self._Runtime = 0

        self._VERBOSE = True
        self._PRE_PROCESS = -1
        self._MAX_MIP_GAP = 1e-3
        self._THREADS = 0
        self._LP_METHOD = LP_Method.AUTO

        self.solver_status = None

        self._update_obj_func = False

        self._time_limit = 1e12

        self._optimization_wallclock_time = 0

        self.list_name_vars_obj = []

        (self.objective_function_pyomo, self.objective_sense_pyomo) = (0, 1)

        if self.PACKAGE == Package.MIP:

            self.xsum = mip.xsum

            self.m = mip.Model(name=model_name,solver_name=solver_name.value)

            self.get_name = self.get_name_mip
            self.set_name =  self.set_name_mip
            self.get_var_x = self.get_var_x_mip
            self.get_var_type =  self.get_var_type_mip
            self.set_var_type = self.set_var_type_mip
            self.get_ub = self.get_ub_mip
            self.set_ub = self.set_ub_mip
            self.get_lb = self.get_lb_mip
            self.set_lb = self.set_lb_mip
            self.get_obj_coeff = self.get_obj_coeff_mip
            self.set_obj_coeff = self.set_obj_coeff_mip
            self.get_var_red_cost = self.get_var_red_cost_mip

        if self.PACKAGE in (Package.PYOMO_KERNEL, Package.PYOMO_CONCRETE):

            self.xsum = pe.quicksum

        if self.PACKAGE == Package.PYOMO_KERNEL:
            self.m = pm.block()

            self.m.dual = pm.suffix(direction = pm.suffix.IMPORT)
            self.m.rc = pm.suffix(direction = pm.suffix.IMPORT)

            self.m.vars = pm.variable_dict()
            self.m.constrs = pm.constraint_dict()
            self.solver = pm.SolverFactory(self.SOLVER_NAME.value)

            self.m.objective = pm.objective(expr=0, sense = pm.minimize)

            self.get_name = self.get_name_pyomo
            self.set_name =  self.set_name_pyomo
            self.get_var_x = self.get_var_x_pyomo
            self.get_var_type =  self.get_var_type_pyomo
            self.set_var_type = self.set_var_type_pyomo
            self.get_ub = self.get_ub_pyomo
            self.set_ub = self.set_ub_pyomo
            self.get_lb = self.get_lb_pyomo
            self.set_lb = self.set_lb_pyomo
            self.get_obj_coeff = self.get_obj_coeff_pyomo
            self.set_obj_coeff = self.set_obj_coeff_pyomo
            self.get_var_red_cost = self.get_var_red_cost_pyomo

        if self.PACKAGE == Package.PYOMO_CONCRETE:
            self.m = pe.ConcreteModel()

            self.m.dual = pe.Suffix(direction = pe.Suffix.IMPORT)
            self.m.rc = pe.Suffix(direction = pe.Suffix.IMPORT)

            if 'gurobi' in self.SOLVER_NAME.value:
                from pyomo.contrib.appsi.solvers import Gurobi
                self.solver = Gurobi()
            elif self.SOLVER_NAME == Solver.HiGHS:
                from pyomo.contrib.appsi.solvers import Highs
                self.solver = Highs()

            self.m.objective = pe.Objective(expr=0, sense = pe.minimize)

            self.get_name = self.get_name_mip
            self.set_name =  self.set_name_pyomo
            self.get_var_x = self.get_var_x_pyomo
            self.get_var_type =  self.get_var_type_pyomo
            self.set_var_type = self.set_var_type_pyomo_concrete
            self.get_ub = self.get_ub_pyomo
            self.set_ub = self.set_ub_pyomo
            self.get_lb = self.get_lb_pyomo
            self.set_lb = self.set_lb_pyomo
            self.get_obj_coeff = self.get_obj_coeff_pyomo_concrete
            self.set_obj_coeff = self.set_obj_coeff_pyomo_concrete
            self.get_var_red_cost = self.get_var_red_cost_pyomo_concrete

        if self.PACKAGE == Package.PYOMO_CONCRETE:
            self._pyomo_var_type_conversion = {"BINARY": pe.IntegerSet, "INTEGER": pe.IntegerSet,
                                            "B": pe.IntegerSet, "I": pe.IntegerSet,
                                            "CONTINUOUS": pe.Reals, "C": pe.Reals}

        if self.PACKAGE == Package.PYOMO_KERNEL:
            self._pyomo_var_type_conversion = {"BINARY": pm.IntegerSet, "INTEGER": pm.IntegerSet,
                                            "B": pm.IntegerSet, "I": pm.IntegerSet,
                                            "CONTINUOUS": pm.RealSet, "C": pm.RealSet}

    def optimize(self, max_seconds = 1e12):
        """optimize model subject to time limit max_seconds"""

        start = time.time()

        self._time_limit = max_seconds

        if self.PACKAGE == Package.MIP:
            self.solver_status = self.m.optimize(max_seconds = max_seconds)

            self._optimization_wallclock_time = time.time() - start

            return _solver_status(self.PACKAGE, self.solver_status,
                                    self._optimization_wallclock_time, max_seconds)

        if self.PACKAGE == Package.PYOMO_KERNEL:
            if self._update_obj_func:
                self.m.objective = pm.objective(expr=self.objective_function_pyomo,
                                                                sense = self.objective_sense_pyomo)
                self._update_obj_func = False

                self.repn = generate_standard_repn(self.m.objective)
                self.list_name_vars_obj = [var.storage_key for var in self.repn.linear_vars]

            if self.SOLVER_NAME == Solver.gurobi_persistent:
                self.solver.options["TimeLimit"] = max_seconds
            elif self.SOLVER_NAME == Solver.cplex_persistent:
                self.solver.options["timelimit"] = max_seconds
            elif self.SOLVER_NAME == Solver.mosek_persistent:
                self.solver.options["dparam.optimizer_max_time"] = max_seconds
            elif self.SOLVER_NAME == Solver.CBC:
                self.solver.options["sec"] = max_seconds

            if '_persistent' in self.SOLVER_NAME.value:
                self.solver.set_instance(self.m)

            if self.SOLVER_NAME == Solver.cplex_persistent:
                if self._LP_METHOD == LP_Method.AUTO:
                    _lp_method = 0
                elif self._LP_METHOD == LP_Method.DUAL:
                    _lp_method = 2
                elif self._LP_METHOD == LP_Method.PRIMAL:
                    _lp_method = 1
                elif self._LP_METHOD == LP_Method.BARRIER:
                    _lp_method = 4

                self.solver_status = self.solver.solve(self.m, options = {"lpmethod": _lp_method},
                                                                        tee = self._VERBOSE)
            else:
                self.solver_status = self.solver.solve(self.m, tee = self._VERBOSE)

            self._optimization_wallclock_time = time.time() - start

            return _solver_status(self.PACKAGE, self.solver_status,
                                        self._optimization_wallclock_time, max_seconds)

        if self.PACKAGE == Package.PYOMO_CONCRETE:

            (self._all_duals, self._all_reduced_costs) = (None, None)

            if self._update_obj_func:
                self.m.objective = pe.Objective(expr=self.objective_function_pyomo,
                                                                sense = self.objective_sense_pyomo)
                self._update_obj_func = False

                self.repn = generate_standard_repn(self.m.objective)
                self.list_name_vars_obj = [var.name for var in self.repn.linear_vars]

            if self.SOLVER_NAME == Solver.HiGHS:
                self.solver.highs_options = {"solver": "choose",
                                            "log_file": "Highs.log",
                                            "threads": self._THREADS,
                                            "mip_rel_gap": 1e-6,"time_limit": 1e10}

            #try:
            self.solver_status = self.solver.solve(self.m)
            #except RuntimeError:
            #self._optimization_wallclock_time = time.time() - start
            #    return OptimizationStatus.OTHER

            self._optimization_wallclock_time = time.time() - start

            try:
                self._all_duals = self.solver.get_duals()
            except RuntimeError:
                self._all_duals = None

            try:
                self._all_reduced_costs = self.solver.get_reduced_costs()
            except RuntimeError:
                self._all_reduced_costs = None

            return _solver_status(self.PACKAGE, self.solver_status,
                                        self._optimization_wallclock_time, max_seconds)

    @property
    def gap(self):
        """
            The relative gap. Given as the relative difference of the primal and dual bounds
            w.r.t. the primal bound
        """
        return (self.objective_value - self.objective_bound)/self.objective_value

    @property
    def objective_bound(self):
        """
            Best dual bound
        """

        if self.PACKAGE == Package.MIP:
            return self.m.objective_bound

        if self.PACKAGE == Package.PYOMO_KERNEL:
            return self.solver_status['Problem']._list[0]['Lower bound']

        if self.PACKAGE == Package.PYOMO_CONCRETE:
            return self.solver_status.best_objective_bound

    @property
    def objective_value(self):
        """
            Best primal bound
        """

        if self.PACKAGE == Package.MIP:
            return self.m.objective_value

        if self.PACKAGE == Package.PYOMO_KERNEL:
            return pm.value(self.m.objective)

        if self.PACKAGE == Package.PYOMO_CONCRETE:
            return self.solver_status.best_feasible_objective

    @property
    def objective(self):
        """get the objective function"""

        if self.PACKAGE == Package.MIP:
            return self.m.objective

        if self.PACKAGE == Package.PYOMO_KERNEL:
            return self.objective_function_pyomo

    @objective.setter
    def objective(self, new_objective, sense = 1):
        """set the objective function"""

        self._update_obj_func = False

        if isinstance(sense, str):
            if sense.upper() in ("MINIMIZE", "MIN"):
                sense = 1
            elif sense.upper() in ("MAXIMIZE", "MAX"):
                sense = -1
            else:
                raise ValueError("I dont understand objective function sense " + sense)

        elif isinstance(sense, int):
            if sense not in (1, -1):
                raise ValueError("arg sense must be either a str or an int. Valid choices are " +
                            "(MINIMIZE, MIN, MAXIMIZE, MAX) or (1, -1). 1 for minimization and " +
                                "-1 for maximization.")
        else:
            raise TypeError("arg sense must be either a str or an int. Valid choices are " +
                            "(MINIMIZE, MIN, MAXIMIZE, MAX) or (1, -1). 1 for minimization and " +
                                "-1 for maximization.")

        if self.PACKAGE == Package.MIP:
            self.m.objective = new_objective
            return

        if self.PACKAGE == Package.PYOMO_KERNEL:
            self.objective_function_pyomo = new_objective
            self.objective_sense_pyomo = sense

            self.repn = generate_standard_repn(self.objective_function_pyomo)
            self.list_name_vars_obj = [var.storage_key for var in self.repn.linear_vars]
            return

    @property
    def lp_method(self):
        """
            Method used for solving continuous models, including the root relaxation of MIPs, but
            not the node problems in the branch-and-bound tree
        """
        return self._LP_METHOD

    @lp_method.setter
    def lp_method(self, new_value):

        self._LP_METHOD = new_value

        if self.PACKAGE == Package.MIP:
            self.m.lp_method = new_value

        if self.PACKAGE == Package.PYOMO_KERNEL:
            # For gurobi, this changes the method used for solving continuous problems (at least as
            # long as there is more than one option), and it changes the method used for solving
            # the root relaxation in the branch-and-cut process.
            if self.SOLVER_NAME == Solver.gurobi_persistent:
                if new_value == LP_Method.AUTO:
                    self.solver.options["Method"] = -1
                elif new_value == LP_Method.DUAL:
                    self.solver.options["Method"] = 1
                elif new_value == LP_Method.PRIMAL:
                    self.solver.options["Method"] = 0
                elif new_value == LP_Method.BARRIER:
                    self.solver.options["Method"] = 2
                #-1=automatic,
                #0=primal simplex,
                #1=dual simplex,
                #2=barrier,
                #3=concurrent,
                #4=deterministic concurrent, and
                #5=deterministic concurrent simplex.

            elif self.SOLVER_NAME == Solver.cplex_persistent:
                # For cplex, lpmethod controls the algorithm used for solving continuous linear
                # models and the root relaxation
                if new_value == LP_Method.AUTO:
                    self.solver.options["lpmethod"] = 0
                elif new_value == LP_Method.DUAL:
                    self.solver.options["lpmethod"] = 2
                elif new_value == LP_Method.PRIMAL:
                    self.solver.options["lpmethod"] = 1
                elif new_value == LP_Method.BARRIER:
                    self.solver.options["lpmethod"] = 4

            elif self.SOLVER_NAME == Solver.mosek_persistent:
                #"conic","dual_simplex","free","free_simplex","intpnt","mixed_int","primal_simplex"
                # [0,        1,           2,       3,           4,         5,          6])
                if new_value == LP_Method.AUTO:
                    self.solver.options["iparam.mio_root_optimizer"] = 2
                elif new_value == LP_Method.DUAL:
                    self.solver.options["iparam.mio_root_optimizer"] = 1
                elif new_value == LP_Method.PRIMAL:
                    self.solver.options["iparam.mio_root_optimizer"] = 6
                elif new_value == LP_Method.BARRIER:
                    self.solver.options["iparam.mio_root_optimizer"] = 4

                if len({self.m.vars[v].domain_type for v in self.m.vars}) > 1:
                    # if there is more than one domain type, then there must be integer variables
                    # in the model. in this case. for mosek, parameter optimizer must be
                    # set to mixed_inter
                    self.solver.options["iparam.optimizer"] = 5

    @property
    def max_mip_gap(self):
        """
            Tolerance for the relative difference between primal and dual bounds w.r.t. the primal
            bound.
        """
        return self._MAX_MIP_GAP

    @max_mip_gap.setter
    def max_mip_gap(self, new_value):

        self._MAX_MIP_GAP = new_value

        if self.PACKAGE == Package.MIP:
            self.m.max_mip_gap = new_value
            return

        if self.PACKAGE == Package.PYOMO_KERNEL:
            if self.SOLVER_NAME == Solver.gurobi_persistent:
                self.solver.options["MIPGap"] = new_value
            elif self.SOLVER_NAME == Solver.cplex_persistent:
                self.solver.options["mip tolerances mipgap"] = new_value
            elif self.SOLVER_NAME == Solver.mosek_persistent:
                self.solver.options["dparam.mio_tol_rel_gap"] = new_value

    @property
    def threads(self):
        """
            Maximum number of threads that can be used by the solver
        """
        return self._THREADS

    @threads.setter
    def threads(self, new_value):
        """
            Set the maximum number of threads
        """
        self._THREADS = new_value

        if not(isinstance(new_value, int)):
            raise TypeError("The number of threads must be an integer. " +
                            f"The current value is {new_value} of type {type(new_value)}")

        if self.PACKAGE == Package.MIP:
            self.m.threads = new_value
            return

        if self.PACKAGE == Package.PYOMO_KERNEL:
            if self.SOLVER_NAME == Solver.mosek_persistent:
                self.solver.options["iparam.num_threads"] = new_value
                return
            self.solver.options["threads"] = new_value

    @property
    def verbose(self):
        """
            Controls the console output
        """
        return self._VERBOSE

    @verbose.setter
    def verbose(self, new_value):
        """
            Set console output: either ON (True or 1), or OFF (False or 0)
        """
        self._VERBOSE = new_value

        if self.PACKAGE == Package.MIP:
            self.m.verbose = new_value

    def remove(self, objects_to_remove):
        """remove objects in object_to_remove from the optimization model"""
        if self.PACKAGE == Package.MIP:
            self.m.remove(objects_to_remove)
            return

        if self.PACKAGE == Package.PYOMO_KERNEL:
            for objct in objects_to_remove:
                del self.m.constrs[objct.storage_key]


        if self.PACKAGE == Package.PYOMO_CONCRETE:
            for objct in objects_to_remove:
                self.m.del_component(objct)

    def reset(self):
        """reset optimization model"""
        if self.PACKAGE == Package.MIP:
            self.m.reset()
            return

        if self.PACKAGE == Package.PYOMO_KERNEL:
            return

    @property
    def vars(self) -> list:
        """return a list with all decision variables"""

        if self.PACKAGE == Package.MIP:
            return self.m.vars

        return [self.m.vars[k] for k in self.m.vars]

    def constrs(self) -> list:
        """Get all constraints in the optimization model"""

        if self.PACKAGE == Package.MIP:
            return self.m.constrs

        return [self.m.constrs[k] for k in self.m.constrs]

    @property
    def status(self):
        """
            Status of the optimization model
        """

        if self.PACKAGE == Package.MIP:
            self._status = self.m.status
            return self._status

        if self.PACKAGE == Package.PYOMO_KERNEL:
            return _solver_status(self.PACKAGE, self.solver_status,
                                            self._optimization_wallclock_time, self._time_limit)

        if self.PACKAGE == Package.PYOMO_CONCRETE:
            return _solver_status(self.PACKAGE, self.solver_status,
                                            self._optimization_wallclock_time, self._time_limit)

    def write(self, file_name):
        """Write the optimization model to a file"""

        if self.PACKAGE == Package.PYOMO_KERNEL:
            if self._update_obj_func:
                self.m.objective = pm.objective(expr=self.objective_function_pyomo,
                                                            sense = self.objective_sense_pyomo)
                self._update_obj_func = False

        if self.PACKAGE == Package.MIP:
            self.m.write(file_name)
        else:
            self.m.write(file_name, io_options = {"symbolic_solver_labels":True})

    def update(self):
        """Update the optimization model"""
        if self.PACKAGE == Package.MIP:
            self.m.solver.update()
            return

        if self._update_obj_func:
            self.m.objective = pm.objective(expr=self.objective_function_pyomo,
                                                                sense = self.objective_sense_pyomo)
            self._update_obj_func = False

    def get_name_mip(self, var):
        """variable name"""
        return var.name

    def set_name_mip(self, var, new_value):
        """variable name"""
        var.name = new_value

    def get_var_x_mip(self, var):
        """return value of the decision variable"""
        return var.x

    def get_var_type_mip(self, var):
        """return the type of the variable"""
        return var.var_type

    def set_var_type_mip(self, var, new_value):
        """change the type of the variable"""
        var.var_type = new_value

    def get_ub_mip(self, var):
        """get variable upper bound"""
        return var.ub

    def set_ub_mip(self, var, new_value):
        """set variable upper bound"""
        var.ub = new_value

    def get_lb_mip(self, var):
        """get variable lower bound"""
        return var.lb

    def set_lb_mip(self, var, new_value):
        """set variable lower bound"""
        var.lb = new_value

    def get_obj_coeff_mip(self, var):
        """return variable objective-function coefficient"""
        return var.obj

    def set_obj_coeff_mip(self, var, new_value):
        """set coefficient of the variable in the objective function"""
        var.obj = new_value

    def get_var_red_cost_mip(self, var):
        """get variable's reduced cost"""
        return var.rc

    def get_name_pyomo(self, var):
        """variable name"""
        return var.storage_key

    def set_name_pyomo(self, var, new_value):
        """variable name"""
        raise NotImplementedError("It is not possible to change the vars name in pyomo")

    def get_var_x_pyomo(self, var):
        """return value of the decision variable"""
        return var.value

    def get_var_type_pyomo(self, var):
        """return the type of the variable"""
        if var.domain_type == pm.IntegerSet and var.bounds[0] >= 0 and var.bounds[1] <= 1:
            return 'B'

        if var.domain_type == pm.RealSet:
            return 'C'

        if var.domain_type == pm.IntegerSet and not(var.bounds[0] >= 0 and var.bounds[1] <= 1):
            return 'I'

    def set_var_type_pyomo(self, var, new_value):
        """change the type of the variable"""

        if new_value in ("binary", "integer", "B", "I"):
            var_type_pyomo = pm.IntegerSet
        elif new_value in ("continuous", "C"):
            var_type_pyomo = pm.RealSet
        else:
            raise ValueError("Argument var_type should be in " +
                                            "('binary', 'integer', 'continuous', 'B', 'I', 'C')")
        var.domain_type = var_type_pyomo

    def set_var_type_pyomo_concrete(self, var, new_value):
        """change the type of the variable"""

        if new_value in ("binary", "integer", "B", "I"):
            var_type_pyomo = pe.IntegerSet
        elif new_value in ("continuous", "C"):
            var_type_pyomo = pe.Reals
        else:
            raise ValueError("Argument var_type should be in " +
                                            "('binary', 'integer', 'continuous', 'B', 'I', 'C')")
        var.domain = var_type_pyomo

    def get_ub_pyomo(self, var):
        """get variable upper bound"""
        return var.ub

    def set_ub_pyomo(self, var, new_value):
        """set variable upper bound"""
        var.ub = new_value

    def get_lb_pyomo(self, var):
        """get variable lower bound"""
        return var.lb

    def set_lb_pyomo(self, var, new_value):
        """set variable lower bound"""
        var.lb = new_value

    def get_obj_coeff_pyomo(self, var):
        """return variable objective-function coefficient"""
        if var.storage_key in self.list_name_vars_obj:
            i = self.list_name_vars_obj.index(var.storage_key)
            return self.repn.linear_coefs[i]

        return 0.000

    def get_obj_coeff_pyomo_concrete(self, var):
        """return variable objective-function coefficient"""
        if var.name in self.list_name_vars_obj:
            i = self.list_name_vars_obj.index(var.name)
            return self.repn.linear_coefs[i]

        return 0.000

    def set_obj_coeff_pyomo(self, var, new_value):
        """set coefficient of the variable in the objective function"""

        if var.storage_key in self.list_name_vars_obj:
            i = self.list_name_vars_obj.index(var.storage_key)
            self.objective_function_pyomo = (self.objective_function_pyomo
                                                    + (new_value - self.repn.linear_coefs[i])*var)
        else:
            self.objective_function_pyomo = self.objective_function_pyomo + new_value*var

        self.repn = generate_standard_repn(self.objective_function_pyomo)
        self.list_name_vars_obj = [_v.storage_key for _v in self.repn.linear_vars]

        self._update_obj_func = True

    def set_obj_coeff_pyomo_concrete(self, var, new_value):
        """set coefficient of the variable in the objective function"""

        if var.name in self.list_name_vars_obj:
            i = self.list_name_vars_obj.index(var.name)
            self.objective_function_pyomo = (self.objective_function_pyomo
                                                    + (new_value - self.repn.linear_coefs[i])*var)
        else:
            self.objective_function_pyomo = self.objective_function_pyomo + new_value*var

        self.repn = generate_standard_repn(self.objective_function_pyomo)
        self.list_name_vars_obj = [_v.name for _v in self.repn.linear_vars]

        self._update_obj_func = True

    def get_var_red_cost_pyomo(self, var):
        """get variable's reduced cost

            pyomo does not seem to really include the variable in the model if it does not appear in
            any of the constraints. in this case, the value attributed in any solution is None.
            Thus, for these cases, assigned a reduced cost of 0
        """
        return 0 if var.value is None else self.m.rc[var]

    def get_var_red_cost_pyomo_concrete(self, var):
        """get variable's reduced cost
        """
        return 0 if var.value is None else self._all_reduced_costs[var]

    def add_var(self, obj:float = 0, lb:float = 0, ub:float = 1e100,
                                                            var_type:str = 'C', name:str = 'var'):
        """Add decision variables to the optimization model
        package:        optimization package used
        m:              optimization model
        obj:            coefficient of the decision variable in the obj function. a float
        lb:             lower bound. a float
        ub:             upper bound. a float
        var_type:       either 'B', 'I' or 'C'
        name:           variable name
        """

        if self.PACKAGE == Package.MIP:
            return self.m.add_var(obj = obj, lb = lb, ub = ub, var_type = var_type, name = name)

        if self.PACKAGE == Package.PYOMO_KERNEL:
            new_var = self.m.vars[name] = pm.variable(self._pyomo_var_type_conversion[var_type],
                                                                    domain = None, lb = lb, ub = ub,
                                                                        value = None, fixed = False)
        if self.PACKAGE == Package.PYOMO_CONCRETE:
            if self.SOLVER_NAME == Solver.HiGHS:
                ub = min(ub, 1e15)
            setattr(self.m, name, pe.Var(bounds = (lb, ub),
                                        within = self._pyomo_var_type_conversion[var_type]))
            new_var = getattr(self.m, name)

        if obj > 0:
            self.objective_function_pyomo += new_var*obj

            self._update_obj_func = True

        return new_var

    def add_constr(self, expression, name:str = 'constr', lhs = None, rhs = None, sign = None):
        """
            Add a constraint to the optimization model
        """

        if self.PACKAGE == Package.MIP:
            return self.m.add_constr(expression, name = name)

        if name == 'constr':
            name = name + '_' + str(len(self.m.constrs))

        if self.PACKAGE == Package.PYOMO_KERNEL:
            if isinstance(expression, (bool, np.bool_)):
                if 'dummy_var_for_dummy_constrs' not in self.m.vars.keys():
                    self.m.vars['dummy_var_for_dummy_constrs'] = pm.variable(pm.RealSet,
                                                                domain = None, lb = 0, ub = 0,
                                                                    value = None, fixed = False)
                expression = self.m.vars['dummy_var_for_dummy_constrs'] >= 0

            constr = self.m.constrs[name] = pmConstr(expression)
            constr.m = self.m

        if self.PACKAGE == Package.PYOMO_CONCRETE:
            if lhs is not None:
                setattr(self.m, 'rhs_param_' + name, pe.Param(mutable = True))
                rhs_param = getattr(self.m, 'rhs_param_' + name)
                rhs_param.value = rhs

                constr = peConcreteConstr(self, self.m, lhs, sign, rhs_param, name)
            else:
                setattr(self.m, name, pe.Constraint(expr = expression))
                constr = getattr(self.m, name)

        return constr
