# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""
from math import ceil
from timeit import default_timer as dt
from copy import deepcopy
import numpy as np
from mip import GUROBI, SearchEmphasis, xsum, LP_Method, Model
from mip import OptimizationStatus as OptS
from solver.BDnetwork import BD_Network
from solver.forward import addVIBasedOnRamp

from addCompToModel.addAllComponents import addAllComp

def solveContinuousRelaxation(params, continuousRelaxation, contRelax_couplConstrs,\
                            contRelax_betaVar, betaVar, contRelax_alphaVar, alphaVar,
                            max_seconds, ini, fixedVars, b):
    '''Solve the continuous relaxation of the current subhorizon'''

    lb = -1e12
    innerRedFlag = np.array(0, dtype = 'int')

    # Fix the decisions of previous subhorizons
    for t in [t for t in range(params.T) if t < min(params.periodsPerSubhorizon[b])]:
        for i in set(params.varsPerPeriod[t]):
            contRelax_couplConstrs[i].rhs = fixedVars[i]

    contRelax_betaVar.obj = betaVar.obj
    contRelax_alphaVar.obj = alphaVar.obj

    CRstatus = continuousRelaxation.optimize(max_seconds = max(max_seconds-(dt() - ini),0))

    if CRstatus in (OptS.OPTIMAL, OptS.FEASIBLE):
        lb = np.array(max(lb, continuousRelaxation.objective_bound), dtype='d')

    elif CRstatus in (OptS.OTHER, OptS.NO_SOLUTION_FOUND):
        innerRedFlag = np.array(1, dtype = 'int')

    else:
        innerRedFlag = np.array(1, dtype = 'int')
        print(f'Continuous relaxation not optimal: {CRstatus}', flush=True)
        if (CRstatus == OptS.INFEASIBLE):
            np.savetxt('infeasRefSol.csv', fixedVars, fmt = '%.4f')
            continuousRelaxation.write('CR' + str(b) + '.lp')
            continuousRelaxation.write('CR' + str(b) + '.mps')
            raise Exception('The CR is infeasible')

    return(lb, innerRedFlag)

class BDsolver:
    '''This class is used to emulate the solver attribute of an MIP model'''

    def __init__(self, MP, SP):
        self.MP = MP
        self.SP = SP

    def set_dbl_param(self, attr: str, value: float):
        '''Set a parameter that receives a double'''
        self.MP.solver.set_dbl_param(attr, value)
        self.SP.solver.set_dbl_param(attr, value)
        return()

    def set_int_param(self, attr: str, value: int):
        '''Set a parameter that receives an integer'''
        self.MP.solver.set_int_param(attr, value)
        self.SP.solver.set_int_param(attr, value)
        return()

    def set_str_param(self, attr: str, value: str):
        """Set a parameter that receives a string"""
        if attr == "LogFile":
            MPname = value.replace(".txt", " - MP.txt")
            SPname = value.replace(".txt", " - SP.txt")
            self.MP.solver.set_str_param(attr, MPname)
            self.SP.solver.set_str_param(attr, SPname)
        else:
            self.MP.solver.set_str_param(attr, value)
            self.SP.solver.set_str_param(attr, value)
        return()

class BD:
    '''Benders decomposition applied to a subhorizon problem'''

    def __init__(self, params, redFlag, thermals, hydros, b, modelName,solver_name=GUROBI):

        self.objective_value, self.objective_bound = 1e12, -1e12

        self.gap, self.it = 100, 0

        self.constrCopyVars, self.copyVars, self.copyBinVars = [], [], []

        self.alphaVarMP, self.dispStatMP = None, None

        self.copyOfCouplVars, self.copyOfcouplConstrs = [], []

        self.constrTgDispSP = {}

        self.outerIteration = 0

        self.alpha, self.beta = 0, 0

        self.alphaVar, self.betaVar = None, None

        self.iniLB = np.array(-1e12, dtype = 'd')

        self.fixedVars = np.zeros(params.nComplVars, dtype = 'd')
        self.bestSol = np.zeros(params.nComplVars, dtype = 'd')
        self.bestSolUB = np.zeros(params.nComplVars, dtype = 'd')

        self.redFlag = redFlag

        self.optimalityCuts, self.tempFeasConstrs, self.trustRegion = [], [], []

        self.lp_method = LP_Method.BARRIER

        self._threads = 0
        self._verbose = 0
        self._preprocess = -1

        self.solveFirstSubhWithReg = True  # only used in self.b == 0. if it is True and
                                    # regularization is used, then the first subhorizon is solved
                                    # with a trust-region constraint

        self.contRelax_couplConstrs, self.contRelax_couplVars = [], []
        self.contRelax_betaVar, self.contRelax_alphaVar = None, None
        #############

        self.setup(params, thermals, hydros, b, modelName, solver_name)

        self.log = {'DDiPIt': [], 'inner1It': [], 'inner2It': [], 'LB': [], 'UB': [], 'gap': [],\
                    'MPTime': [], 'SPTime': []}

    def setup(self, params, thermals, hydros, b, modelName, solver_name):
        '''Initialize the attributes'''

        self.MP = Model(f'subhBinMP_subh{b}', solver_name = solver_name)
        self.MP.max_mip_gap = params.gapInnerBDMP
        self.MP.lp_method = LP_Method.DUAL
        self.MP.SearchEmphasis = SearchEmphasis.OPTIMALITY

        if params.BDnetworkSubhorizon:
            self.SP = BD_Network(params,self.redFlag,b,f'backwardSubh{b}',solver_name=params.solver)
        else:
            self.SP = Model(f'subhCompleteSP_subh{b}', solver_name = solver_name)
            self.SP.lp_method = self.lp_method

        self.b = b

        self.params = params
        self.thermals = thermals

        self.constrTgDispSP ={(g,t): None for g in range(len(thermals.id)) for t in range(params.T)}

        self.solver = BDsolver(self.MP, self.SP)

    @property
    def threads(self):
        '''Get number of threads'''
        return self._threads

    @threads.setter
    def threads(self, new_n_threads):
        '''Set number of threads'''
        self.MP.threads = new_n_threads
        self.SP.threads = new_n_threads
        self._threads = new_n_threads

    @property
    def verbose(self):
        '''Get verbose'''
        return self._verbose

    @verbose.setter
    def verbose(self, new_verbose):
        '''Set verbose'''
        self.MP.verbose = new_verbose
        self.SP.verbose = new_verbose
        self._verbose = new_verbose

    @property
    def preprocess(self):
        '''Get preprocess'''
        return self._preprocess

    @preprocess.setter
    def preprocess(self, new_preprocess):
        '''Set preprocess'''
        self.MP.preprocess = new_preprocess
        self.SP.preprocess = new_preprocess
        self._preprocess = new_preprocess

    def add_constr(self, cnstr, name : str = ''):
        '''Add constraint to the SP'''
        self.SP.add_constr(cnstr, name = name)
        return()

    def add_constr_MP(self, cnstr, name : str = ''):
        '''Add constraint to the SP'''
        self.MP.add_constr(cnstr, name = name)
        return()

    def write(self, name : str = ''):
        '''Write optimization models'''
        self.MP.write('MP_' + name + '.lp')
        self.MP.write('MP_' + name + '.mps')
        self.SP.write('SP_' + name + '.lp')
        self.SP.write('SP_' + name + '.mps')

        return()

    def reset(self):
        '''Reset the master problem and the subproblem'''
        self.MP.reset()
        self.SP.reset()
        return()

    def relax(self):
        '''Relax the integrality constraints in the master problem'''
        self.MP.relax()
        return()

    def addAllComp(self, params, hydros, thermals, network, fixedVars, b, binVars):
        '''Add the appropriate variables and constraints to the master problem and subproblem'''

        couplConstrs, couplVars, alpha, beta, alphaVarMP, copyOfMPBinVars,\
                constrOfCopyOfMPBinVars, dispStat, constrTgDisp,\
                    copyGenVars, constrCopyGenVars, alphaVarSPnetwork = addAllComp(\
                                        params, hydros, thermals, network,\
                                        self.MP,\
                                        self.SP.MP if params.BDnetworkSubhorizon else self.SP,\
                                        self.SP.SP if params.BDnetworkSubhorizon else self.SP, b,\
                                        fixedVars,\
                                        BDbinaries = True,\
                                        BDnetwork = params.BDnetworkSubhorizon,\
                                        binVars = binVars)

        self.dispStatMP = dispStat

        self.copyVars = copyOfMPBinVars
        self.constrCopyVars = constrOfCopyOfMPBinVars
        self.constrTgDispSP = constrTgDisp
        self.copyOfCouplVars = couplVars
        self.copyOfcouplConstrs = couplConstrs

        self.alphaVarMP = alphaVarMP

        self.alphaVar = alpha
        self.betaVar = beta

        if params.BDnetworkSubhorizon:
            self.SP.setCouplingVarsAndCreateAuxVars(couplVars, couplConstrs,\
                                                    copyGenVars, constrCopyGenVars,\
                                                    alphaVarSPnetwork, alpha, beta)

        self.objective = self.SP.MP.objective if params.BDnetworkSubhorizon else self.SP.objective

        return(couplConstrs, couplVars, alpha, beta)

    def optimize(self, max_seconds: float = 1e12, max_nodes: int = 100000):
        '''Solve the subhorizon problem with BD'''

        ini = dt()

        fixedVars = deepcopy(self.fixedVars)
        self.bestSol = deepcopy(self.fixedVars)

        self.alpha, self.beta = 1e12, 1e12

        if self.b == 0 and self.outerIteration > 1:
            self.MP.remove(self.tempFeasConstrs + self.trustRegion)
        else:
            self.MP.remove(self.tempFeasConstrs + self.optimalityCuts + self.trustRegion)
            self.optimalityCuts = []

        self.tempFeasConstrs, self.trustRegion = [], []

        if self.params.solver == GUROBI:
            # Update to make suare that the RHSs are correct
            self.MP.solver.update()

        self.tempFeasConstrs = addVIBasedOnRamp(self.params, self.thermals, self.b,\
                                    self.MP, self.copyOfcouplConstrs, self.copyVars, self.fixedVars)

        lb = max(-1e12, self.iniLB) if (self.b == 0) else -1e12
        ub = np.array(1e12, dtype = 'd')

        it, gap = 0, 1e12

        innerRedFlag = np.array(0, dtype = 'int')

        fixedBinaryVars = np.zeros(len(self.constrCopyVars), dtype = 'd')

        while (self.redFlag != 1) and (innerRedFlag != 1):

            self.log['DDiPIt'].append(self.outerIteration)
            self.log['inner1It'].append(it)
            self.log['inner2It'].append(0)
            self.log['LB'].append(0)
            self.log['UB'].append(0)
            self.log['gap'].append(gap)
            self.log['MPTime'].append(0)
            self.log['SPTime'].append(0)

            self.log['MPTime'][-1] = dt()

            self.MP.reset()
            MPstatus = self.MP.optimize(max_seconds = max(max_seconds-(dt()-ini),0))

            if MPstatus in (OptS.OPTIMAL, OptS.FEASIBLE):

                lb = np.array(max(lb, self.MP.objective_bound), dtype='d')

                gap = (ub - lb)/ub

                if gap <= self.params.gapInnerBD:
                    innerRedFlag = np.array(1, dtype = 'int')

                if innerRedFlag != 1:

                    fixedBinaryVars = np.array([var.x for var in self.copyVars], dtype = 'd')

                    fixedBinaryVars[np.where(fixedBinaryVars <= 0.5)] = 0
                    fixedBinaryVars[np.where(fixedBinaryVars > 0.5)] = 1

            elif MPstatus in (OptS.OTHER, OptS.NO_SOLUTION_FOUND):
                innerRedFlag = np.array(1, dtype = 'int')

            else:
                innerRedFlag = np.array(1, dtype = 'int')
                print(f'MP not optimal: {MPstatus}', flush=True)
                if (MPstatus == OptS.INFEASIBLE):
                    np.savetxt('infeasRefSol.csv', self.bestSolUB, fmt = '%.4f')
                    self.MP.write('MP' + str(self.b) + '.lp')
                    self.MP.write('MP' + str(self.b) + '.mps')
                    raise Exception('The MP is infeasible')

            self.log['MPTime'][-1] = dt() - self.log['MPTime'][-1]

            self.log['SPTime'][-1] = dt()

            if innerRedFlag != 1:
                for i in range(len(self.constrCopyVars)):
                    self.constrCopyVars[i].rhs = fixedBinaryVars[i]

                if self.params.BDnetworkSubhorizon:
                    self.SP.outerIteration = self.outerIteration
                    self.SP.innerItBD = it

                self.SP.reset()
                self.SP.lp_method = LP_Method.BARRIER

                SPstatus = self.SP.optimize(max_seconds=max(max_seconds - (dt() - ini),0))

                if SPstatus == OptS.OPTIMAL:
                    SPobjVal = self.SP.objective_value
                    SPobjBound = self.SP.objective_bound

                    if not(self.params.BDnetworkSubhorizon):
                        sol = np.array([self.copyOfCouplVars[i].x\
                                            for i in self.params.varsPerSubh[self.b]], dtype = 'd')
                    else:
                        sol = self.SP.bestSol[self.params.varsPerSubh[self.b]]

                    fixedVars[self.params.varsPerSubh[self.b]] = sol[:]

                    fixedVars[self.params.varsPerSubh[self.b]] = np.maximum(fixedVars[\
                                                                self.params.varsPerSubh[self.b]],\
                                                                self.params.lbOnCouplVars[\
                                                                self.params.varsPerSubh[self.b]])

                    fixedVars[self.params.varsPerSubh[self.b]] = np.minimum(fixedVars[\
                                                                self.params.varsPerSubh[self.b]],\
                                                                self.params.ubOnCouplVars[\
                                                                self.params.varsPerSubh[self.b]])

                    if (SPobjVal + (self.MP.objective_value - self.alphaVarMP.x)) < ub:

                        ub = np.array(SPobjVal + (self.MP.objective_value - self.alphaVarMP.x),\
                                                                                    dtype = 'd')

                        self.bestSol[self.params.varsPerSubh[self.b]] = fixedVars[\
                                                                    self.params.varsPerSubh[self.b]]

                        self.alpha = self.alphaVar.x if not(self.params.BDnetworkSubhorizon) else\
                                                                                    self.SP.alpha
                        self.beta = self.betaVar.x if not(self.params.BDnetworkSubhorizon) else\
                                                                                    self.SP.beta

                elif SPstatus  in (OptS.OTHER, OptS.NO_SOLUTION_FOUND):
                    innerRedFlag = np.array(1, dtype = 'int')

                else:
                    innerRedFlag = np.array(1, dtype = 'int')
                    print(f'SP not optimal: {SPstatus}', flush=True)
                    if SPstatus == OptS.INFEASIBLE:
                        if self.params.BDnetworkSubhorizon:
                            self.SP.write('infeasSP')
                        else:
                            self.SP.write('infeasSP.lp')
                            self.SP.write('infesSP.mps')

                        self.MP.write('infesSP_MP.lp')
                        self.MP.write('infesSP_MP.mps')
                        raise Exception('The SP is infeasible')

            gap = (ub - lb)/ub

            self.log['SPTime'][-1] = dt() - self.log['SPTime'][-1]

            self.log['LB'][-1] = lb
            self.log['UB'][-1] = ub
            self.log['gap'][-1] = gap

            it += 1

            if innerRedFlag != 1:
                if (it >= self.params.maxITinnerBD):
                    innerRedFlag = np.array(1, dtype = 'int')

                if gap <= self.params.gapInnerBD:
                    innerRedFlag = np.array(1, dtype = 'int')

            if innerRedFlag != 1 and (self.redFlag != 1):
                pi = np.array([var.pi for var in self.constrCopyVars], dtype = 'd')

                nonZeros = np.where(np.abs(pi) > 0)[0]
                constTerm = np.inner(pi, fixedBinaryVars)

                lhs = xsum(pi[i]*self.copyVars[i] for i in nonZeros)

                self.optimalityCuts.append(self.MP.add_constr(self.alphaVarMP >=\
                                            SPobjBound + lhs - constTerm,\
                                                name ='optCut_' + str(len(self.optimalityCuts))))
        if gap <= -1e-3:
            self.MP.write('MP.lp')
            self.SP.write('SP.lp')
            raise Exception('Negative gap in inner BD')

        self.objective_value = ub
        self.objective_bound = lb

        self.gap = (ub - lb)/ub

        self.it = it

        self.iniLB = np.array(-1e12, dtype = 'd')

        self.solveFirstSubhWithReg = True

        if gap <= self.params.gapInnerBD or ((it >= self.params.maxITinnerBD) and (ub < 1e12)):
            return(OptS.OPTIMAL)

        return(OptS.OTHER)
