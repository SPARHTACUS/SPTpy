# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""
import mpi4py
mpi4py.rc.thread_level = 'single'
import csv
import itertools
import os
from timeit import default_timer
from mpi4py import MPI
import numpy as np

from readAndLoadData.loadData import loadData
from solver.mainSolver import runSolver
from postOptimization import postOptimization
from solver.write import writeDDiPdata

wComm = MPI.COMM_WORLD
wRank = wComm.Get_rank()
wSize = wComm.Get_size()
host = MPI.Get_processor_name()

def main(expList, expNames):
    '''main function'''

    rootFolder = os.path.abspath(os.path.join(__file__ , "../..")).replace("\\","/")

    params, hydros, thermals, network, fComm, fRank, fSize, bComm, bRank, bSize\
                                    = loadData(rootFolder, wComm, wSize, wRank, expList, expNames)

    wComm.Barrier()
    params.start = default_timer()
    params.lastTime = params.start + params.timeLimit
    wComm.Barrier()

    ###########################################################################
    if wRank == 0:
        print('###############################################################')
        print('###################  Overview of the system  ##################')
        print("System: " + params.ps + "\tcase: " + params.case)
        print(f"Planning horizon in hours: {params.T*params.discretization}")
        print(f"Time steps: {params.T}")
        print(f"{len(thermals.id)} thermal plants with installed capacity "\
                                            f"{sum(thermals.maxP)*params.powerBase:.0f} MW")
        print(f"{len(hydros.id)} hydropower plants with installed capacity "\
                                            f"{sum(hydros.plantMaxPower)*params.powerBase:.0f} MW")
        print("Total installed capacity (MW): " + \
                        f"{(sum(thermals.maxP) + sum(hydros.plantMaxPower))*params.powerBase:.0f}")
        print(f"Buses: {len(network.busID)}")
        print(f"AC transmission lines: {len(network.AClineFromTo)}")
        print(f"DC links: {len(network.DClinkFromTo)}")
        print(f"Peak net load (MW): {np.max(np.sum(network.load,axis=1))*params.powerBase:.0f}")
        if wSize == 1:
            print(f"Subhorizons: {params.nSubhorizons}" + "\n" +\
                                    f"Periods in each subhorizon: {params.periodsPerSubhorizon}")
        print(f'The total number of processes is {wSize}')
        if wSize > 1:
            print(f'Forward processes: {params.forwardWorkers}')
            print(f'Backward processes: {params.backwardWorkers}')
        else:
            print('Forward processes: [0]')
            print('Backward processes: [0]')
        if params.asynchronous:
            print('Asynchronous optimization')
        else:
            print('Synchronous optimization')
        print(f'Solver:\t{params.solver}')
        print(flush = True)

    f = open(params.outputFolder + '/params - ' + params.ps + ' - case ' +\
                        str(params.case) + ' - rank ' + str(wRank) + '.csv', 'w', encoding="utf-8")
    f.write('attribute;value\n')

    doNotPrint = ['lbOnCouplVars', 'ubOnCouplVars', 'map', 'varsPerPeriod', 'conVarsPerPeriod',\
                    'binVarsPerPeriod', 'dispGenVarsPerPeriod', 'binDispVarsPerPeriod',
                        'varsInPreviousSubhs', 'varsInPreviousAndCurrentSubh', 'binVarsPerSubh',\
                            'varsPerSubh', 'binDispVarsPerSubh',\
                                'conVarsPerSubh', 'dispGenVarsPerSubh', 'genConstrsPerPeriod',\
                                    'contVarsInPreviousAndCurrentSubh',\
                                        'binVarsInPreviousAndCurrentSubh', 'periodsOfBackwardWs',\
                                            'delayedCuts']
    for attr in dir(params):
        if attr[-2:] != '__' and attr not in doNotPrint:
            if isinstance(getattr(params, attr), np.ndarray):
                f.write(attr + ';' + str(list(getattr(params, attr)))+ '\n')
            else:
                f.write(attr + ';' + str(getattr(params, attr)) + '\n')
    f.close()
    del f

    if False:
        # use this to read a vector of values for the coupling variables, compute the remainder
        # of the variables based on this vector, and write the results
        assert wSize == 1, 'This feature should not be used with more than 1 process'

        print('\n\nCompute complete solution for vector not found on-the-fly\n\n', flush = True)

        fromFolder = 'D:/paper_parallelCBC/output_synch_cbc/SIN/case ' + str(params.case) + '/' +\
                                                                        params.expName + '/'
        bestSol = np.loadtxt(fromFolder +'bestSolutionFound - SIN - case '+str(params.case)+'.csv',\
                                                                    dtype = 'd', delimiter = ';')
        f = open(fromFolder + 'convergence - SIN - case ' + str(params.case) + '.csv',\
                                                                        'r', encoding='utf-8')
        reader = csv.reader(f, delimiter = ';')
        row = next(reader)  # header
        while True:
            try:
                row = next(reader)  # header
            except StopIteration:
                break
            ub = float(row[2])*params.scalObjF

        params.outputFolder =\
                params.outputFolder[:params.outputFolder[:len(params.outputFolder) - 1].rfind('/')]\
                    + '/' + params.expName + '/'

        if not(os.path.isdir(params.outputFolder)):
            os.makedirs(params.outputFolder)

        np.savetxt(params.outputFolder + 'bestSolutionFound - '+\
                                    params.ps + ' - case ' + str(params.case) + '.csv',\
                                    bestSol, fmt = '%.12f', delimiter=';')
    else:
        bestSol, ub, pLog, subhorizonInfo, backwardInfo = runSolver(\
                                    params, hydros, thermals, network,\
                                    wComm, wRank, wSize, fComm, fRank, fSize, bComm, bRank, bSize)

        writeDDiPdata(params, pLog, subhorizonInfo, backwardInfo, wRank)

    if wRank == 0 and ub < 1e12:
        print('\n\nPrint the results\n\n', flush = True)
        postOptimization(params, thermals, hydros, network, bestSol, ub)

    wComm.Barrier()

    return()

if __name__ == '__main__':

    experiments = {'case': [str(i) for i in range(1, 21, 1)],\
                    'nSubhorizonsPerProcess': [[1, 8, 16, 24,\
                                        2, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 24, 24, 24]],\
                    'forwardWs': [[1, 2, 3]],\
                    'backwardWs': [list(range(4, 20, 1))],\
                    'solveOnlyFirstSubhorizonInput': [4*[False] + [True] + 20*[False]],\
                    'trials': [0, 1, 2, 3, 4]}

    experiment = {k: None for k in experiments}

    expID = 0

    keys, values = zip(*experiments.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for perm in permutations_dicts:
        for key in experiments:
            experiment[key] = perm[key]

        # Run the experiment
        expName = 'exp' + str(expID) + '_trial' + str(experiment['trials'])

        if (1 < wSize) and\
                    (wSize != (1 + len(experiment['forwardWs']) + len(experiment['backwardWs']))):
            tw = len(experiment['forwardWs']) + len(experiment['backwardWs'])
            raise ValueError('Number of processes does not match number of ' +\
                            f'forward and backward workers. For {tw} workers, ' +\
                            f'there should be {tw + 1} processes.')

        main(experiment, expName)

        expID += 1

        wComm.Barrier()
