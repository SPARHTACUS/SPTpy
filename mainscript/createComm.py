# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""
import numpy as np
from mpi4py import MPI

def createComm(params, worldComm, rankWorld, sizeWorld):
    '''Split the world communicator into primal and dual communicators
    Also create the windows for one-sided communications'''

    ### Windows
    # Best solution
    if sizeWorld > 1:
        params.winBestSol = MPI.Win.Create(np.zeros(params.nComplVars, dtype = 'd'),\
                                        1, info = MPI.INFO_NULL, comm = worldComm)
    # The red flag, upper bound, and lower bound windows are created in mainSolver.py

    #### Forward communicator
    if ((rankWorld in params.forwardWorkers) or (rankWorld == 0)):
        color = 0
    else:
        color = MPI.UNDEFINED

    fComm = worldComm.Split(color = color, key = rankWorld)

    if ((rankWorld in params.forwardWorkers) or (rankWorld == 0)):
        fRank = fComm.Get_rank()
        fSize = fComm.Get_size()
    else:
        fRank = -1e3
        fSize = -1e3
    ############################################################################

    #### Backward communicator
    if ((rankWorld in params.backwardWorkers) or (rankWorld == 0)):
        color = 0
    else:
        color = MPI.UNDEFINED

    bComm = worldComm.Split(color = color, key = rankWorld)

    if ((rankWorld in params.backwardWorkers) or (rankWorld == 0)):
        bRank = bComm.Get_rank()
        bSize = bComm.Get_size()
    else:
        bRank = -1e3
        bSize = -1e3
    ############################################################################

    return(fComm, fRank, fSize, bComm, bRank, bSize)
