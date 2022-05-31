# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""

from mip import GUROBI, CONTINUOUS, xsum, LP_Method, Model
from mip import OptimizationStatus as OptS
from timeit import default_timer as dt
from copy import deepcopy
import numpy as np
from solver.forward import addVIBasedOnRamp

from addCompToModel.addAllComponents import addAllComp

class BD_NetworkSolver:
    '''This class is used to emulate the solver attribute of an MIP model'''

    def __init__(self, MP, SP):
        self.MP = MP
        self.SP = SP

    def set_dbl_param(self, attr: str, value: float):
        '''Set a parameter that receives a double'''
        self.MP.solver.set_dbl_param(attr, value)
        for k in self.SP.keys():
            self.SP[k].solver.set_dbl_param(attr, value)
        return()

    def set_int_param(self, attr: str, value: int):
        '''Set a parameter that receives an integer'''
        self.MP.solver.set_int_param(attr, value)
        for k in self.SP.keys():
            self.SP[k].solver.set_int_param(attr, value)
        return()

    def set_str_param(self, attr: str, value: str):
        """Set a parameter that receives a string"""
        if attr == "LogFile":
            MPname = deepcopy(value)
            SPname = deepcopy(value)
            MPname.replace(".txt", " - MP.txt")
            SPname.replace(".txt", "")
            self.MP.solver.set_str_param(attr, MPname)
            for k in self.SP.keys():
                self.SP[k].solver.set_str_param(attr, SPname + " - SP" + str(k) + ".txt")
        else:
            self.MP.solver.set_str_param(attr, value)
            for k in self.SP.keys():
                self.SP[k].solver.set_str_param(attr, value)
        return()

    def update(self):
        '''Update the optimization models'''
        self.MP.solver.update()
        for k in self.SP.keys():
            self.SP[k].solver.update()
        return()

class BD_Network:
    '''Benders decomposition applied to a subhorizon problem'''

    def __init__(self, params, redFlag, b, modelName,solver_name=GUROBI):
        self.objective_value, self.objective_bound = 1e12, -1e12

        self.gap, self.it = 100, 0

        self.constrCopyGenVars, self.copyGenVars = [], []

        self.alphaVarMP, self.dispStatMP = None, None

        self.copyOfCouplVars, self.copyOfcouplConstrs = [], []

        self.outerIteration, self.innerItBD = 0, 0

        self.alpha, self.beta = 0, 0

        self.alphaVar, self.betaVar = None, None

        # For multicut
        self.alphaIndividualSPs = {}

        self.iniLB = np.array(-1e12, dtype = 'd')
        self.iniUB = np.array(1e12, dtype = 'd')

        self.redFlag = redFlag

        self.optimalityCuts = 0
        self.tempFeasConstrs = []

        self._threads = 0
        self._verbose = 0
        self._preprocess = -1

        self.setup(params, b, modelName, solver_name)

        self.log = {'DDiPIt': [], 'inner1It': [], 'innerItBD': [], 'LB': [], 'UB': [], 'gap': [],\
                    'MPTime': [], 'SPTime': []}

    def setup(self, params, b, modelName, solver_name):
        '''Initialize the attributes'''

        self.modelName = modelName

        self.MP = Model('subhMPcontinuous_subh' + str(b), solver_name = solver_name)
        self.MP.lp_method = LP_Method.BARRIER
        if (params.solver == GUROBI):
            self.MP.solver.set_int_param("Presolve", 2)
            self.MP.solver.set_int_param("ScaleFlag", 3)
            self.MP.solver.set_dbl_param("BarConvTol", 1e-16)
        self.SP = {t: Model('subhSPnetwork_subh' + str(b) + '_time' + str(t),\
                                solver_name = solver_name) for t in params.periodsPerSubhorizon[b]}

        self.b = b

        self.fixedVars = np.zeros(params.nComplVars, dtype = 'd')
        self.bestSol = np.zeros(params.nComplVars, dtype = 'd')
        self.bestSolUB = np.zeros(params.nComplVars, dtype = 'd')

        self.params = params

        self.solver = BD_NetworkSolver(self.MP, self.SP)

    @property
    def threads(self):
        '''Get number of threads'''
        return self._threads

    @threads.setter
    def threads(self, new_n_threads):
        '''Set number of threads'''
        self.MP.threads = new_n_threads
        for key in self.SP.keys():
            self.SP[key].threads = new_n_threads
        self._threads = new_n_threads

    @property
    def objective(self):
        '''Get the objective function of the MP'''
        return self.MP.objective

    @objective.setter
    def objective(self, newObjective):
        self.MP.objective = newObjective

    @property
    def verbose(self):
        '''Get verbose'''
        return self._verbose

    @verbose.setter
    def verbose(self, new_verbose):
        '''Set verbose'''
        self.MP.verbose = new_verbose
        for key in self.SP.keys():
            self.SP[key].verbose = new_verbose
        self._verbose = new_verbose

    @property
    def preprocess(self):
        '''Get preprocess'''
        return self._preprocess

    @preprocess.setter
    def preprocess(self, new_preprocess):
        '''Set preprocess'''
        self.MP.preprocess = new_preprocess
        for key in self.SP.keys():
            self.SP[key].preprocess = new_preprocess
        self._preprocess = new_preprocess

    def add_constr(self, cnstr, name: str = ''):
        '''Add constraint to the MP'''
        self.MP.add_constr(cnstr, name = name)
        return()

    def reset(self):
        '''Reset the master problem and the subproblem'''
        self.MP.reset()
        for k in self.SP.keys():
            self.SP[k].reset()
        return()

    def lp_method(self, lp_method):
        '''Reset the master problem and the subproblem'''
        self.MP.lp_method = lp_method
        for k in self.SP.keys():
            self.SP[k].m.lp_method = lp_method
        return()

    def relax(self):
        '''Relax the integrality constraints in the master problem'''
        self.MP.relax()
        return()

    def write(self, name : str = ''):
        '''Write optimization models'''
        self.MP.write('MPnetwork_' + name + '.lp')
        self.MP.write('MPnetwork_' + name + '.mps')
        for k in self.SP.keys():
            self.SP[k].write('SPnetwork_' + str(k) + '_' + name + '.lp')
            self.SP[k].write('SPnetwork_' + str(k) + '_' + name + '.mps')
        return()

    def setCouplingVarsAndCreateAuxVars(self, couplVars, couplConstrs,\
                                        copyGenVars, constrCopyGenVars,\
                                        alphaVarSPnetwork, alpha, beta):
        '''Set the coupling variables, the objective functions and create auxiliary variables that
        will be used for estimating the SP's optimal cost'''
        self.copyOfCouplVars = couplVars
        self.copyOfcouplConstrs = couplConstrs

        self.copyGenVars = copyGenVars
        self.constrCopyGenVars = constrCopyGenVars

        self.alphaVarMP = alphaVarSPnetwork

        self.alphaVar = alpha
        self.betaVar = beta

        self.objective = self.MP.objective

        # Add the variables associated with the individual subproblems
        for sp in self.SP.keys():
            self.alphaIndividualSPs[sp] = self.MP.add_var(var_type = CONTINUOUS,\
                                                                name = f'alphaVarSP_{sp}')

        # To work with only one variable outside this class, add the following constraint
        self.MP.add_constr(self.alphaVarMP -\
                            xsum(self.alphaIndividualSPs[sp] for sp in self.SP.keys())>= 0,\
                                name = 'sumOfApproximations')
        return()

    def addAllComp(self, params, hydros, thermals, network, fixedVars, b, binVars):
        '''Add the appropriate variables and constraints to the master problem and subproblem'''

        self.thermals = thermals

        couplConstrs, couplVars, alpha, beta, alphaVarMP,\
            copyOfMPBinVars, constrOfCopyOfMPBinVars, dispStat, constrTgDisp,\
            copyGenVars, constrCopyGenVars, alphaVarSPnetwork = addAllComp(\
                                params, hydros, thermals, network, self.MP, self.MP,\
                                self.SP, b, fixedVars,\
                                BDbinaries = False, BDnetwork = True,  binVars = binVars)

        self.setCouplingVarsAndCreateAuxVars(couplVars, couplConstrs, copyGenVars,\
                                                constrCopyGenVars,\
                                                alphaVarSPnetwork, alpha, beta)

        return(couplConstrs, couplVars, alpha, beta)

    def optimize(self, max_seconds: float = 1e12, max_nodes: int = 10000):
        '''Solve the subhorizon problem with BD'''

        ini = dt()

        self.bestSol = np.zeros(self.params.nComplVars, dtype = 'd')

        self.alpha, self.beta = 1e12, 1e12

        self.objective_value, self.objective_bound = 1e12, -1e12

        if self.params.solver == GUROBI:
            # Update to make suare that the RHSs are correct
            self.MP.solver.update()

        if not(self.params.BDSubhorizonProb) and self.b > 0:

            self.MP.remove(self.tempFeasConstrs)
            self.tempFeasConstrs = []

            if self.params.solver == GUROBI:
                self.MP.solver.update()

            self.tempFeasConstrs = addVIBasedOnRamp(self.params, self.thermals, self.b,\
                            self.MP, self.copyOfcouplConstrs, self.copyOfCouplVars, self.fixedVars)

        lb, ub =np.array(max(-1e12,self.iniLB),dtype='d'), np.array(min(1e12, self.iniUB),dtype='d')

        it, gap, innerRedFlag = 0, 1e12, np.array(0, dtype = 'int')

        while (self.redFlag != 1) and (innerRedFlag != 1):

            self.log['DDiPIt'].append(self.outerIteration)
            self.log['inner1It'].append(it)
            self.log['innerItBD'].append(self.innerItBD)
            self.log['LB'].append(0)
            self.log['UB'].append(0)
            self.log['gap'].append(gap)
            self.log['MPTime'].append(0)
            self.log['SPTime'].append(0)

            self.log['MPTime'][-1] = dt()

            self.MP.reset()
            MPstatus = self.MP.optimize(max_seconds = max(max_seconds -(dt() - ini),0),\
                                                                            max_nodes = max_nodes)

            if MPstatus in (OptS.OPTIMAL, OptS.FEASIBLE):

                lb = np.array(max(lb, self.MP.objective_bound), dtype='d')

                gap = (ub - lb)/ub

                if gap <= self.params.gapInnerBDnetwork:
                    innerRedFlag = np.array(1, dtype = 'int')

                if innerRedFlag != 1:
                    fixedGen = np.array([var.x for var in self.copyGenVars], dtype = 'd')
                    fixedGen[np.where(fixedGen <= 0)] = 0 # if the hgBus for a pump is also positive

            elif (MPstatus == OptS.INFEASIBLE):
                self.MP.write('infesMP_MP.lp')
                self.MP.write('infesMP_MP.mps')

                raise Exception('The master problem of a backward subhorizon is infeasible')

            elif MPstatus  in (OptS.OTHER, OptS.NO_SOLUTION_FOUND):
                innerRedFlag = np.array(1, dtype = 'int')

                if MPstatus == OptS.NO_SOLUTION_FOUND:
                    if isinstance(self.MP.objective_bound, float):
                        lb = np.array(max(lb, self.MP.objective_bound), dtype='d')
                    self.objective_bound = lb
                    return(OptS.NO_SOLUTION_FOUND)

            else:
                innerRedFlag = np.array(1, dtype = 'int')
                raise Exception('MP not optimal: ' + MPstatus)

            self.log['MPTime'][-1] = dt() - self.log['MPTime'][-1]

            self.log['SPTime'][-1] = dt()

            if (innerRedFlag != 1) and (self.redFlag != 1):

                for i in range(len(self.copyGenVars)):
                    self.constrCopyGenVars[i].rhs = fixedGen[i]

                for k in self.SP.keys():
                    self.SP[k].reset()
                    self.SP[k].lp_method = LP_Method.BARRIER

                    SPstatus = self.SP[k].optimize(max_seconds = max(max_seconds- (dt() - ini), 0))

                    if not(SPstatus in (OptS.OPTIMAL, OptS.FEASIBLE)):
                        break

                if SPstatus in (OptS.OPTIMAL, OptS.FEASIBLE):

                    if sum([self.SP[k].objective_value for k in self.SP.keys()]) +\
                                                (self.MP.objective_value - self.alphaVarMP.x) < ub:

                        self.bestSol[self.params.varsPerSubh[self.b]] =\
                                                np.array([self.copyOfCouplVars[i].x\
                                            for i in self.params.varsPerSubh[self.b]], dtype = 'd')

                        ub = np.array(sum([self.SP[k].objective_value for k in self.SP.keys()]) +\
                                (self.MP.objective_value - self.alphaVarMP.x),dtype = 'd')

                        self.alpha = self.alphaVar.x
                        self.beta = self.betaVar.x

                elif SPstatus in (OptS.OTHER, OptS.NO_SOLUTION_FOUND):
                    innerRedFlag = np.array(1, dtype = 'int')

                else:
                    innerRedFlag = np.array(1, dtype = 'int')
                    print(f'SP not optimal: {SPstatus}', flush=True)
                    if SPstatus == OptS.INFEASIBLE:
                        self.SP.m.write('infeasSP.lp')
                        self.SP.m.write('infesSP.mps')

                        self.MP.write('infesSP_MP.lp')
                        self.MP.write('infesSP_MP.mps')

            self.log['SPTime'][-1] = dt() - self.log['SPTime'][-1]

            gap = (ub - lb)/ub

            self.log['LB'][-1] = lb
            self.log['UB'][-1] = ub
            self.log['gap'][-1] = gap

            it += 1

            if innerRedFlag != 1:
                if (it >= self.params.maxITinnerBDnetwork):
                    innerRedFlag = np.array(1, dtype = 'int')

                if gap <= self.params.gapInnerBDnetwork:
                    innerRedFlag = np.array(1, dtype = 'int')

            if innerRedFlag != 1 and (self.redFlag != 1):
                piAll = np.array([constr.pi for constr in self.constrCopyGenVars], dtype = 'd')
                for k in self.SP.keys():
                    piIndv = np.zeros(piAll.shape, dtype = 'd')
                    piIndv[self.params.genConstrsPerPeriod[self.b][k]] =\
                                                piAll[self.params.genConstrsPerPeriod[self.b][k]]

                    nonZeros = np.where(np.abs(piIndv) > 1e-6)[0]
                    constTerm = np.inner(piIndv[nonZeros], fixedGen[nonZeros])

                    lhs = xsum(piIndv[i]*self.copyGenVars[i] for i in nonZeros)

                    self.MP.add_constr(self.alphaIndividualSPs[k] >=\
                                            self.SP[k].objective_value + lhs - constTerm,\
                                                name = f'optCut_{k}_{self.optimalityCuts}')
                    self.optimalityCuts += 1

        if gap <= -1e-3:
            self.SP.m.write('Subh' + str(self.b) + '_outerIt' + str(self.outerIteration) + 'SP.lp')
            self.SP.m.write('Subh' + str(self.b) + '_outerIt' +str(self.outerIteration) + 'SP.mps')

            self.MP.write('Subh' + str(self.b) + '_outerIt' +str(self.outerIteration) + 'MP.lp')
            self.MP.write('Subh' + str(self.b) + '_outerIt' +str(self.outerIteration) + 'MP.mps')

            raise Exception('Negative gap in inner BD network')

        self.objective_value = ub
        self.objective_bound = lb

        self.gap = (ub - lb)/ub

        self.it = it

        self.iniLB = np.array(-1e12, dtype = 'd')

        if gap <= self.params.gapInnerBDnetwork or ((it >= self.params.maxITinnerBDnetwork)\
                                                                            and (ub < 1e12)):
            return(OptS.OPTIMAL)

        return(OptS.OTHER)
