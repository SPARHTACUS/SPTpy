# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""

import csv
import locale
from copy import deepcopy
from timeit import default_timer as dt
from time import sleep
from mpi4py import MPI
import numpy as np
from solver.write import writeEventTracker

locale.setlocale(locale.LC_ALL, 'en_US.utf-8')

def sendMessageToBackward(params, backwardWorkerOfForward, fwPackageRecv, evaluatedSols,\
                        forwardSols, bwPackage, indicesOfSols, backwardWsAssigned,\
                        indicesOfSolsNotSent, availableBackwardWorkers, eventTracker, bComm):
    '''Send an iterate to be evaluated in the backward step
    First, check if the partial solution up to forward's subhorizon fwPackageRecv[src][4]
    (last subhorizon solved) matches the beginning of the last subhorizon of backward worker bw'''

    solsSentToAll = set() # solutions sent to one backward process of each aggregation

    for i in indicesOfSolsNotSent:
        forwardSrc = indicesOfSols[i][1]
        forwardSrcIt = indicesOfSols[i][2]
        # Get the aggregations of the backward workers to whom the solution was already sent
        aggOfBWAssigned = [params.aggBackwardWorkers[bWorker] for bWorker in\
                                                    backwardWsAssigned[forwardSrc, forwardSrcIt]]
        for bw, _ in [k for k in backwardWorkerOfForward[forwardSrc]\
                                                    if (k[1] == int(fwPackageRecv[forwardSrc][4]))]:
            # the above comprehension is a list of the backward workers who can received the
            # primal solution from forward worker forwardSrc up to the forward worker's
            # fwPackageRecv[forwardSrc][4] subhorizon
            if (params.aggBackwardWorkers[bw] not in aggOfBWAssigned)\
                                                            and (bw in availableBackwardWorkers):
                # Only send the current solution to backward worker that is available and whose
                # aggregation is different than that of the other backward workers to whom
                # the was sent

                bwPackage = np.array([0, forwardSrc,forwardSrcIt,int(fwPackageRecv[forwardSrc][4]),\
                                    0, 0], dtype = 'd')
                # bwPackage = np.array([redFlag, forward source fRank,\
                #       forward source iteration, forward source subhorizon, 0, 0], dtype = 'd')

                # Then send the partial solution to bw
                evaluatedSols[bw][:] = forwardSols[forwardSrc][:]
                bComm.Isend([bwPackage, MPI.DOUBLE], dest = bw, tag = 3)
                bComm.Isend([evaluatedSols[bw], MPI.DOUBLE], dest = bw, tag = 30)

                availableBackwardWorkers.remove(bw)

                backwardWsAssigned[forwardSrc, forwardSrcIt].append(bw)

                aggOfBWAssigned.append(params.aggBackwardWorkers[bw])

                if set(params.aggBackwardWorkers) == set(aggOfBWAssigned):
                    solsSentToAll.add(i)

                eventTracker.append(('primalSolSent', '54', ' ', ' ',\
                        ' ', bw, forwardSrc, ' ',\
                            int(fwPackageRecv[forwardSrc][3]), int(fwPackageRecv[forwardSrc][4]),\
                                ' ', ' ', dt() - params.start))

    solsSentToAll = list(solsSentToAll)
    solsSentToAll.sort(reverse = True)
    for i in solsSentToAll:
        indicesOfSolsNotSent.remove(i)

    return()

def sendMessageToForward(params, it, backwardSource, bwPackageRecv, buffSentForward,\
                        objValuesRelaxs, subgrads, evaluatedSols,
                        availableForwardWorkers, eventTracker, fComm, fSize):
    '''Send message to forward worker'''

    it, subh = int(bwPackageRecv[backwardSource][3]), int(bwPackageRecv[backwardSource][4])
    # subh is 0 if subhorizon 0 was also solved, and 1 otherwise. 'it' is the iteration counter
    # of the backward worker

    if subh == 0:
        return()

    for fRank in [i for i in range(1, fSize, 1)\
                    if params.aggBackwardWorkers[backwardSource] <= params.aggForwardWorkers[i]]:
        # Send to all forward workers, even if they are not available
        # but only send it to those forward worker whose number of subhorizons is more than
        # or equal to the number of subhorizon in the source backward worker
        buffSentForward[('package', backwardSource, fRank, it, subh)] = [np.array([backwardSource]\
                                        + list(bwPackageRecv[backwardSource]), dtype = 'd'), None]
        buffSentForward[('objVals', backwardSource, fRank, it, subh)] =\
                                                    [deepcopy(objValuesRelaxs[backwardSource]),None]
        buffSentForward[('subgrad', backwardSource, fRank, it, subh)] =\
                                                    [deepcopy(subgrads[backwardSource]), None]
        buffSentForward[('evalSol', backwardSource, fRank, it, subh)] =\
                                                    [deepcopy(evaluatedSols[backwardSource]), None]

        # Send the newly received subgradients to another backward worker
        buffSentForward[('package', backwardSource, fRank, it, subh)][1] =\
                    fComm.Isend([buffSentForward[('package', backwardSource, fRank, it, subh)][0],\
                                                        MPI.DOUBLE], dest = fRank, tag = 10)

        buffSentForward[('objVals', backwardSource, fRank, it, subh)][1] =\
                    fComm.Isend([buffSentForward[('objVals', backwardSource, fRank, it, subh)][0],\
                                                        MPI.DOUBLE], dest = fRank, tag = 11)

        buffSentForward[('subgrad', backwardSource, fRank, it, subh)][1] =\
                    fComm.Isend([buffSentForward[('subgrad', backwardSource, fRank, it, subh)][0],\
                                                        MPI.DOUBLE], dest = fRank, tag = 12)

        buffSentForward[('evalSol', backwardSource, fRank, it, subh)][1] =\
                    fComm.Isend([buffSentForward[('evalSol', backwardSource, fRank, it, subh)][0],\
                                                        MPI.DOUBLE], dest = fRank, tag = 13)

        eventTracker.append(('dualSolSent', '105', ' ', ' ',\
                                backwardSource, ' ', ' ', fRank,\
                                    it, subh,\
                                        ' ', ' ', dt() - params.start))

    # Recall that the forward worker will only start working once backward information from a
    # backward worker with the same aggregation as his has been received by him
    newBusyWorkers = []
    for fRank in availableForwardWorkers:
        if params.aggBackwardWorkers[backwardSource] == params.aggForwardWorkers[fRank]:
            newBusyWorkers.append(fRank)

    for fRank in newBusyWorkers:
        availableForwardWorkers.remove(fRank)

    return()

def recvMessageFromForward(params, redFlag, ub, lb, bestSol, status, fwPackageRecvTemp,\
                            fwPackageRecv, forwardSols, indicesOfSols, backwardWsAssigned,\
                            indicesOfSolsNotSent, availableForwardWorkers,\
                            eventTracker, pLog, wSize, fComm):
    '''Receive a message from a forward worker'''

    # get the message's source and the message's tag
    fRank, tg = status.Get_source(), status.Get_tag()

    if tg == 29:
        lbRecv = np.array([1e12, -1e6, -1e6], dtype = 'd')
        fComm.Recv([lbRecv, MPI.DOUBLE], source = fRank, tag = 29)
        if lbRecv[0] > lb:
            lb = np.array(lbRecv[0], dtype = 'd')
            print('Lower bound updated to ' + locale.currency(lb/params.scalObjF, grouping = True)+\
                    f'. The gap is now {100*(ub - lb)/ub:.4f}%. Source fRank: {fRank}', flush=True)

            params.winLB.Lock(rank = 0)
            params.winLB.Put([lb, MPI.DOUBLE], target_rank = 0)
            params.winLB.Unlock(rank = 0)

            eventTracker.append(('LBupdated', '141', ' ', lb,\
                                    ' ', ' ', fRank, ' ',\
                                        int(lbRecv[1]), int(lbRecv[2]),\
                                            ' ', ' ', dt() - params.start))

    elif tg == 21:
        fComm.Recv([fwPackageRecvTemp[fRank], MPI.DOUBLE], source = fRank, tag = 21)

        # Check if the newly received solution gives a better lower bound
        if (fwPackageRecvTemp[fRank][2]) > lb:
            lb = np.array(fwPackageRecvTemp[fRank][2], dtype = 'd')
            print('Lower bound updated to ' + locale.currency(lb/params.scalObjF, grouping = True)+\
                    f'. The gap is now {100*(ub - lb)/ub:.4f}%. Source fRank: {fRank}', flush=True)

            params.winLB.Lock(rank = 0)
            params.winLB.Put([lb, MPI.DOUBLE], target_rank = 0)
            params.winLB.Unlock(rank = 0)

            eventTracker.append(('LBupdated', '157', ' ', lb,\
                            ' ', ' ', fRank, ' ',\
                                int(fwPackageRecvTemp[fRank][3]), int(fwPackageRecvTemp[fRank][4]),\
                                    ' ', ' ', dt() - params.start))

            pLog['lb'].append(lb)
            pLog['ub'].append(ub)
            pLog['gap'].append((ub - lb)/ub)
            pLog['runTimeForward'].append(dt() - params.start)
            pLog['runTimeBackward'].append(0)

            if ((ub - lb)/ub <= params.relGapDDiP) and params.asynchronous:
                redFlag = np.array(1, dtype = 'int')
                for r in range(wSize):
                    params.winRedFlag.Lock(rank = r)
                    params.winRedFlag.Put([redFlag, MPI.INT], target_rank = r)
                    params.winRedFlag.Unlock(rank = r)

    elif tg == 22:
        fwPackageRecv[fRank][:] = fwPackageRecvTemp[fRank][:]
        fComm.Recv([forwardSols[fRank], MPI.DOUBLE], source = fRank, tag = 22)
        # There is enough information to send the received partial solution to backward workers

        # int(fwPackageRecv[fRank][3]) is the forward's iteration counter
        # int(fwPackageRecv[fRank][4]) is the forward's subhorizon
        indicesOfSols.append((len(indicesOfSols), fRank,\
                                        int(fwPackageRecv[fRank][3]), int(fwPackageRecv[fRank][4])))

        if (fRank, int(fwPackageRecv[fRank][3])) not in backwardWsAssigned.keys():
            backwardWsAssigned[fRank, int(fwPackageRecv[fRank][3])] = []

        goodCandidate = True
        for i in indicesOfSolsNotSent:
            if indicesOfSols[i][1] == fRank:
                # then, forward worker src has just sent a more updated solution. thus, replace
                # the old solution with this new one
                goodCandidate = False
                indicesOfSolsNotSent[indicesOfSolsNotSent.index(i)] = len(indicesOfSols) - 1
                break

        if goodCandidate:
            indicesOfSolsNotSent.append(len(indicesOfSols) - 1)

        if ((int(fwPackageRecv[fRank][4]) + 1) == params.aggForwardWorkers[fRank]):
            eventTracker.append(('primalSolRecv', '212',\
                                fwPackageRecv[fRank][5] + fwPackageRecv[fRank][6], ' ',\
                                    ' ', ' ', fRank, ' ',\
                                        int(fwPackageRecv[fRank][3]), int(fwPackageRecv[fRank][4]),\
                                            ' ', ' ', dt() - params.start))
        else:
            eventTracker.append(('primalSolRecv', '217', ' ', ' ',\
                                    ' ', ' ', fRank, ' ',\
                                        int(fwPackageRecv[fRank][3]), int(fwPackageRecv[fRank][4]),\
                                            ' ', ' ', dt() - params.start))

        # First, check if the solution received is complete, then
        # check if the newly received solution gives a better upper bound
        if ((int(fwPackageRecv[fRank][4]) + 1) == params.aggForwardWorkers[fRank]) and\
            (fwPackageRecv[fRank][5] + fwPackageRecv[fRank][6]) < ub:
            ub = np.array(fwPackageRecv[fRank][5] + fwPackageRecv[fRank][6], dtype = 'd')
            bestSol[:] = forwardSols[fRank][:]
            print('Upper bound updated to ' + locale.currency(ub/params.scalObjF, grouping = True)+\
                    f'. The gap is now {100*(ub - lb)/ub:.4f}%. Source fRank: {fRank}', flush=True)

            params.winBestSol.Lock(rank = 0)
            params.winBestSol.Put([bestSol, MPI.DOUBLE], target_rank = 0)
            params.winBestSol.Unlock(rank = 0)

            params.winUB.Lock(rank = 0)
            params.winUB.Put([ub, MPI.DOUBLE], target_rank = 0)
            params.winUB.Unlock(rank = 0)

            if ((ub - lb)/ub <= params.relGapDDiP) and params.asynchronous:
                redFlag = np.array(1, dtype = 'int')
                for r in range(wSize):
                    params.winRedFlag.Lock(rank = r)
                    params.winRedFlag.Put([redFlag, MPI.INT], target_rank = r)
                    params.winRedFlag.Unlock(rank = r)

            eventTracker.append(('UBupdated', '245', ub, ' ',\
                                ' ', ' ', fRank, ' ',\
                                    int(fwPackageRecv[fRank][3]), int(fwPackageRecv[fRank][4]),\
                                        ' ', ' ', dt() - params.start))

            pLog['lb'].append(lb)
            pLog['ub'].append(ub)
            pLog['gap'].append((ub - lb)/ub)
            pLog['runTimeForward'].append(dt() - params.start)
            pLog['runTimeBackward'].append(0)

        if (int(fwPackageRecv[fRank][4]) + 1) == params.aggForwardWorkers[fRank]:
            availableForwardWorkers.append(fRank)

    else:
        print(f'Im the GenCoord, Ive got a message with tag {tg} from {fRank}')
        print('And I dont know what to do with it. Line 89.', flush = True)

    return(redFlag, ub, lb, bestSol)

def recvMessageFromBackward(params, redFlag, ub, lb, status, allInfoFromBackwardWorker,\
                backwardWorkersMatches, evaluatedSols, bwPackageRecv, objValuesRelaxs, subgrads,\
                buffSentBackward, availableBackwardWorkers, eventTracker, pLog, wSize, bComm):
    '''Receive message from a backward worker'''

    # get the message's source and the message's tag
    bRank, tg = status.Get_source(), status.Get_tag()

    enoughInfoReceived = False # this will be true only when a complete solution has been received,
                            # i.e, once a backward process has solved all of its subproblems

    if tg == 35:
        # partial information
        bComm.Recv([bwPackageRecv[bRank], MPI.DOUBLE], source = bRank, tag = tg)
        allInfoFromBackwardWorker[bRank] += 1
    elif tg == 36:
        # partial information
        bComm.Recv([subgrads[bRank], MPI.DOUBLE], source = bRank, tag = tg)
        allInfoFromBackwardWorker[bRank] += 1
    elif tg == 37:
        # partial information
        bComm.Recv([objValuesRelaxs[bRank], MPI.DOUBLE], source = bRank, tag = tg)
        allInfoFromBackwardWorker[bRank] += 1
        eventTracker.append(('partialDualSolutionRecv', '288', ' ', ' ',\
                                    bRank, ' ', ' ', ' ',\
                                        int(bwPackageRecv[bRank][3]), int(bwPackageRecv[bRank][4]),\
                                            ' ', ' ', dt() - params.start))
    elif tg == 38:
        # first part of a complete dual solution
        bComm.Recv([bwPackageRecv[bRank], MPI.DOUBLE], source = bRank, tag = tg)
    elif tg == 39:
        # second part of a complete dual solution
        bComm.Recv([subgrads[bRank], MPI.DOUBLE], source = bRank, tag = tg)
    elif tg == 40:
        # third and last part of a complete dual solution
        bComm.Recv([objValuesRelaxs[bRank], MPI.DOUBLE], source = bRank, tag = tg)
        enoughInfoReceived = True
        if params.asynchronous:
            # if asynchronous optimization is used, the backward worker becomes available
            #immediately. otherwise, it is only available again at the begining of the next it
            availableBackwardWorkers.append(bRank)

        if int(bwPackageRecv[bRank][4]) == 0:
            # int(bwPackageRecv[src][4]) is the subhorizon's index. then, if it is the first
            # subhorizon
            eventTracker.append(('completeDualSolutionRecv', '310',\
                                ' ', bwPackageRecv[bRank][2],\
                                    bRank, ' ', ' ', ' ',\
                                        int(bwPackageRecv[bRank][3]), int(bwPackageRecv[bRank][4]),\
                                            ' ', ' ', dt() - params.start))
        else:
            eventTracker.append(('completeDualSolutionRecv', '316',\
                                ' ', ' ',\
                                    bRank, ' ', ' ', ' ',\
                                        int(bwPackageRecv[bRank][3]), int(bwPackageRecv[bRank][4]),\
                                            ' ', ' ', dt() - params.start))
    else:
        print(f'Im the GenCoord, Ive got a message with tag {tg} from {bRank}' + '\n' +
                                    'And I dont know what to do with it. Line 71.', flush = True)

    # The following is used for sending info to the backward workers
    if enoughInfoReceived or allInfoFromBackwardWorker[bRank] == 3:
        # get the iteration (from the source), and the index of the subhorizon problem
        it, subh = int(bwPackageRecv[bRank][3]), int(bwPackageRecv[bRank][4])

        # if it is the first subhorizon (i.e., subh == 0), then, update the lower bound, if possible
        if (subh == 0) and (bwPackageRecv[bRank][2]) > lb:
            lb = np.array(bwPackageRecv[bRank][2], dtype = 'd')
            print('Lower bound updated to ' + locale.currency(lb/params.scalObjF, grouping = True)+\
                    f'. The gap is now {100*(ub - lb)/ub:.4f}%. Source bRank: {bRank}', flush=True)

            params.winLB.Lock(rank = 0)
            params.winLB.Put([lb, MPI.DOUBLE], target_rank = 0)
            params.winLB.Unlock(rank = 0)

            eventTracker.append(('LBupdated', '340', ' ', lb,\
                                bRank, ' ', ' ', ' ',\
                                    ' ', ' ', ' ', ' ', dt() - params.start))

            if ((ub - lb)/ub <= params.relGapDDiP) and params.asynchronous:
                redFlag = np.array(1, dtype = 'int')
                for r in range(wSize):
                    params.winRedFlag.Lock(rank = r)
                    params.winRedFlag.Put([redFlag, MPI.INT], target_rank = r)
                    params.winRedFlag.Unlock(rank = r)

            pLog['lb'].append(lb)
            pLog['ub'].append(ub)
            pLog['gap'].append((ub - lb)/ub)
            pLog['runTimeForward'].append(dt() - params.start)
            pLog['runTimeBackward'].append(0)

        # There is enough information to send the received partial solution to backward workers
        allInfoFromBackwardWorker[bRank] = 0
        if subh > 0:
            for bRankDest, _ in [k for k in backwardWorkersMatches[bRank]\
                                                        if k[1] == int(bwPackageRecv[bRank][4])]:
                buffSentBackward[('package', bRank, bRankDest, it, subh)] = [np.array([bRank]+\
                                                    list(bwPackageRecv[bRank]), dtype = 'd'), None]
                buffSentBackward[('objVals', bRank, bRankDest, it, subh)] =\
                                                            [deepcopy(objValuesRelaxs[bRank]), None]
                buffSentBackward[('subgrad', bRank, bRankDest, it, subh)] =\
                                                            [deepcopy(subgrads[bRank]), None]
                buffSentBackward[('evalSol', bRank, bRankDest, it, subh)] =\
                                                            [deepcopy(evaluatedSols[bRank]), None]

                # Send the newly received subgradients to another backward worker
                buffSentBackward[('package', bRank, bRankDest, it, subh)][1] =\
                        bComm.Isend([buffSentBackward[('package', bRank, bRankDest, it, subh)][0],\
                                                            MPI.DOUBLE], dest = bRankDest, tag = 31)

                buffSentBackward[('objVals', bRank, bRankDest, it, subh)][1] =\
                        bComm.Isend([buffSentBackward[('objVals', bRank, bRankDest, it, subh)][0],\
                                                            MPI.DOUBLE], dest = bRankDest, tag = 32)

                buffSentBackward[('subgrad', bRank, bRankDest, it, subh)][1] =\
                        bComm.Isend([buffSentBackward[('subgrad', bRank, bRankDest, it, subh)][0],\
                                                            MPI.DOUBLE], dest = bRankDest, tag = 33)

                buffSentBackward[('evalSol', bRank, bRankDest, it, subh)][1] =\
                        bComm.Isend([buffSentBackward[('evalSol', bRank, bRankDest, it, subh)][0],\
                                                            MPI.DOUBLE], dest = bRankDest, tag = 34)
                eventTracker.append(('partialDualSolutionSent', '387', ' ', ' ',\
                                        bRank, bRankDest, ' ', ' ',\
                                            it, subh,\
                                                ' ', ' ', dt() - params.start))

    return(enoughInfoReceived, bRank, lb)

def forwardAndBackwardMatches(params, fSize) -> dict:
    '''Each forward worker has its own sending points, which may or may not
    be shared by others. Once the sending points are reached, the partial
    solutions can be sent to the appropriate backward workers

    The following subhorizon matches forward and backward workers: to which
    backward worker each forward worker can send partial solutions?'''

    #### only used if synchronous optimization is used
    # in the synchronous optimization, every backward worker receives primal solutions always from
    # the same forward worker
    matchedBackwards = {i: False for i in params.backwardWorkers\
                                                            if not(params.firstSubhorizonFlags[i])}
    # similarly, forward workers always send primal solutions to the same backward workers
    # make sure that, for the same number of subhorizons, the same forward worker does not send
    # solutions to more than one backward worker
    nSubhsOfMatchedBackwards = {fw: set() for fw in range(fSize)}
    ####

    backwardWorkerOfForward = {fw: [] for fw in range(fSize)}

    for fw in params.forwardWorkers:
        fRank = params.forwardWorkers.index(fw) + 1# The workers start from index 1, and their
                                    # respective indices in fComm is based on their ranks in wComm

        for bw in [bw for bw in params.backwardWorkers if not(params.firstSubhorizonFlags[bw])\
                                                                and not(matchedBackwards[bw])]:
            if params.asynchronous or\
                        (params.nSubhsOfEachProcess[bw] not in nSubhsOfMatchedBackwards[fRank]):
                bRank = params.backwardWorkers.index(bw) + 1# index of the backward process in the
                                                            # list of backward process only

                backwardSubh=params.nSubhsOfEachProcess[bw] - 1# last subhorizon for backward worker
                firstPeriodInLastSubh = min(params.periodsPerSubhOfEachProcess[bw][backwardSubh])

                for fb in range(params.nSubhsOfEachProcess[fw]):
                    if (firstPeriodInLastSubh-max(params.periodsPerSubhOfEachProcess[fw][fb]))==1:
                        backwardWorkerOfForward[fRank].append((bRank, fb))
                        nSubhsOfMatchedBackwards[fRank].add(params.nSubhsOfEachProcess[bw])
                        if not(params.asynchronous):
                            matchedBackwards[bw] = True
                        break

                if (bRank, params.nSubhsOfEachProcess[fw]-1) not in backwardWorkerOfForward[fRank]:
                    # always include forward's last subhorizon because, at this point,
                    # a complete solution is available
                    backwardWorkerOfForward[fRank].append((bRank, params.nSubhsOfEachProcess[fw]-1))
                    nSubhsOfMatchedBackwards[fRank].add(params.nSubhsOfEachProcess[bw])
                    if not(params.asynchronous):
                        matchedBackwards[bw] = True

    with open(params.outputFolder + 'backwardWorkerOfForward - ' + \
            params.ps + ' - case ' + str(params.case)+'.csv','w',encoding='ISO-8859-1') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in backwardWorkerOfForward.items():
            writer.writerow([key, value])

    return(backwardWorkerOfForward)

def backwardMatches(params, bSize) -> dict:
    ''' Do the same thing with the backward processes'''
    backwardWorkersMatches = {bw: [] for bw in range(bSize)}

    for bw1 in [i for i in params.backwardWorkers if not(params.firstSubhorizonFlags[i])]:
        bRank1 = params.backwardWorkers.index(bw1) + 1
        for bw2 in [i for i in params.backwardWorkers if (i != bw1) and\
                            (params.nSubhsOfEachProcess[bw1] <= params.nSubhsOfEachProcess[i])]:
            # check if the subgradients from backward worker bw1 can be added to one of the
            # subhorizon problems of backward worker bw2

            # bw2 is the index in the list of all processes
            # bRank2 is the index in the list of only the general coord + backward process (bcomm)
            bRank2 = params.backwardWorkers.index(bw2) + 1
            firstSubh = 0 if params.firstSubhorizonFlags[bw2] else 1
            lastSubh = 1 if params.firstSubhorizonFlags[bw2] else params.nSubhsOfEachProcess[bw2] -1
            for subhOfB2 in range(firstSubh, lastSubh, 1):
                lastPeriodInSubh = max(params.periodsPerSubhOfEachProcess[bw2][subhOfB2])

                for subhOfB1 in range(1, params.nSubhsOfEachProcess[bw1], 1):
                    # Check if the first period in subhorizon subhOfB1 of backward worker bw1
                    # (the sending end) comes immediately after the last period of subhorizon
                    # subhOfB2 of backward worker bw2 (the receiving end)
                    if (min(params.periodsPerSubhOfEachProcess[bw1][subhOfB1])-lastPeriodInSubh)==1:
                        backwardWorkersMatches[bRank1].append((bRank2, subhOfB1))
                        # Then, the cut derived from subhorizon subhOfB1 of backward worker bRank1
                        # can be added to subhorizon subhOfB2 of backward worker bRank bRank2

    with open(params.outputFolder + 'backwardWorkersMatches - ' + params.ps + ' - case ' +\
                str(params.case) + '.csv', 'w', encoding = 'ISO-8859-1') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in backwardWorkersMatches.items():
            writer.writerow([key, value])

    return(backwardWorkersMatches)

def genCoord(params, redFlag, ub, lb, it, bestSol, pLog, wRank, wSize, fComm, fSize, bComm, bSize):
    '''The general coordinator receives primal solutions from the
        forward workers and subgradients from the backwards workers.
        The general coordinator is also responsible for checking the convergence of the method'''

    iniCheck = dt()

    availableForwardWorkers = []                                                    # as in fComm
    # from 1 to len(params.backwardWorkers) as in bComm
    availableBackwardWorkers = list(range(1, bSize, 1))

    eventTracker = []       # a logger

    indicesOfSols = []      # a list of the primal solutions received from forward workers

    backwardWsAssigned = {} # for each candidate solution from a forward process with given
                            # iteration (src, it), backwardWsAssigned[src, it]
                            # gives the indices of backward workers (bRank) to whom the solution
                            # was sent to be evaluated.

    indicesOfSolsNotSent = [] # indices of solutions yet to be sent to backward workers

    forwardSols = [np.zeros(params.nComplVars, dtype = 'd') for i in range(fSize)]

    evaluatedSols = [np.zeros(params.nComplVars, dtype = 'd') for i in range(bSize)]

    backwardWorkerOfForward = forwardAndBackwardMatches(params, fSize)
    backwardWorkersMatches = backwardMatches(params, bSize)

    buffSentBackward, buffSentForward = {}, {}

    status = MPI.Status()

    fwPackageRecvTemp = [np.array([0, 0, 0, 0, 0, 0, 0], dtype = 'd') for i in range(fSize)]
    fwPackageRecv = [np.array([0, 0, 0, 0, 0, 0, 0], dtype = 'd') for i in range(fSize)]
    forwardSols = [np.zeros(params.nComplVars, dtype = 'd') for i in range(fSize)]

    bwPackageRecv = [np.array([0, 0, 0, 0, 0], dtype = 'd') for r in range(bSize)]
    objValuesRelaxs = [[]] + [np.zeros(params.periodsOfBackwardWs[r].shape[0], dtype = 'd')\
                                                                for r in range(1, bSize, 1)]
    subgrads = [[]] + [np.zeros((params.periodsOfBackwardWs[r].shape[0],\
                                params.nComplVars), dtype = 'd') for r in range(1, bSize, 1)]

    allInfoFromBackwardWorker = [0 for i in range(bSize)]

    bwPackage = np.array([redFlag, 0, 0, 0, 0, 0], dtype = 'd')

    iterCounter = 0                                         # used only in the synchronous algorithm
    allReceivedFromForward, allReceivedFromBackward = False, False# used only in the synchronous alg
    countFinishedBackwardWs = 0

    # sort the backward workers in ascending order according to the number of subhorizons
    bwSubhs = [params.nSubhsOfEachProcess[bw] for bw in params.backwardWorkers]
    bwsSorted = []
    for i in range(1, len(params.backwardWorkers) + 1, 1):
        bwsSorted.append(bwSubhs.index(min(bwSubhs)) + 1)
        bwSubhs[bwSubhs.index(min(bwSubhs))] = i*1e12

    while (redFlag != 1):
        messageFromForward, messageFromBackward, enoughInfoReceivedBackward = False, False, False

        if (fComm.Iprobe(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)):
            messageFromForward = True
            redFlag, ub, lb, bestSol = recvMessageFromForward(params, redFlag, ub, lb, bestSol,\
                                            status, fwPackageRecvTemp, fwPackageRecv,\
                                            forwardSols, indicesOfSols, backwardWsAssigned,\
                                            indicesOfSolsNotSent, availableForwardWorkers,\
                                            eventTracker, pLog, wSize, fComm)

            if not(params.asynchronous) and\
                                    (len(availableForwardWorkers) == len(params.forwardWorkers)):
                # then the forward worker has finished solving all of its subhorizon problems
                allReceivedFromForward = True

        if (redFlag != 1) and not(messageFromForward) and\
                        (bComm.Iprobe(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)):
            messageFromBackward = True
            enoughInfoReceivedBackward, backwardSrc, lb = recvMessageFromBackward(params, redFlag,\
                                            ub, lb, status, allInfoFromBackwardWorker,\
                                            backwardWorkersMatches, evaluatedSols,\
                                            bwPackageRecv, objValuesRelaxs, subgrads,\
                                            buffSentBackward, availableBackwardWorkers,\
                                            eventTracker, pLog, wSize, bComm)

        if (redFlag != 1) and enoughInfoReceivedBackward:
            countFinishedBackwardWs += 1
            if params.asynchronous:
                # send the dual solution to forward workers
                sendMessageToForward(params, it, backwardSrc, bwPackageRecv, buffSentForward,\
                                objValuesRelaxs, subgrads, evaluatedSols, availableForwardWorkers,\
                                eventTracker, fComm, fSize)
            else:
                # the solution is only sent if all backward workers have finished their works
                if countFinishedBackwardWs == len(params.backwardWorkers):
                    # then all backward workers have finished their works
                    for bWRKR in bwsSorted:
                        # send all dual solutions to the forward worker
                        sendMessageToForward(params, it, bWRKR, bwPackageRecv, buffSentForward,\
                                objValuesRelaxs, subgrads, evaluatedSols, availableForwardWorkers,\
                                eventTracker, fComm, fSize)

                    allReceivedFromBackward = True

                elif countFinishedBackwardWs > len(params.backwardWorkers):
                    raise ValueError('There is something wrong. countFinishedBackwardWs should not'\
                                    +' be greater than len(params.backwardWorkers)')

        if (redFlag !=1) and len(indicesOfSolsNotSent)>0 and len(availableBackwardWorkers)>0:
            sendMessageToBackward(params, backwardWorkerOfForward, fwPackageRecv,evaluatedSols,\
                            forwardSols, bwPackage, indicesOfSols, backwardWsAssigned,\
                            indicesOfSolsNotSent, availableBackwardWorkers, eventTracker, bComm)

        if (redFlag != 1) and not(messageFromForward) and not(messageFromBackward):
            sleep(0.01)

        # Check if the time limit has been reached
        if (redFlag != 1) and (dt() >= params.lastTime):
            eventTracker.append(('timeLimitReached', '585', ' ', ' ',\
                                    ' ', ' ', ' ', ' ',\
                                        iterCounter, ' ',\
                                            ' ', ' ', dt() - params.start))
            redFlag = np.array(1, dtype = 'int')
            for r in range(wSize):
                params.winRedFlag.Lock(rank = r)
                params.winRedFlag.Put([redFlag, MPI.INT], target_rank = r)
                params.winRedFlag.Unlock(rank = r)

        if (dt() - iniCheck) >= 30:
            # delete buffers that were already sent
            keysToDeleteBackward = [k for k, item in buffSentBackward.items() if item[1].Test()]
            keysToDeleteForward = [k for k, item in buffSentForward.items() if item[1].Test()]

            for k in keysToDeleteBackward:
                del buffSentBackward[k]

            for k in keysToDeleteForward:
                del buffSentForward[k]

            iniCheck = dt()

        if not(params.asynchronous) and allReceivedFromForward and allReceivedFromBackward:
            eventTracker.append(('EndOfIteration', '609', ' ', ' ',\
                                    ' ', ' ', ' ', ' ',\
                                        iterCounter, ' ', ' ', ' ', dt() - params.start))

            iterCounter += 1

            if ((ub - lb)/ub <= params.relGapDDiP) or (iterCounter > params.maxItDDiP):
                redFlag = np.array(1, dtype = 'int')
                for r in range(wSize):
                    params.winRedFlag.Lock(rank = r)
                    params.winRedFlag.Put([redFlag, MPI.INT], target_rank = r)
                    params.winRedFlag.Unlock(rank = r)

            availableBackwardWorkers = list(range(1, bSize, 1))

            allReceivedFromForward, allReceivedFromBackward = False, False

            countFinishedBackwardWs = 0

    writeEventTracker(params, eventTracker, wRank)

    print('UB: ' + locale.currency(ub/params.scalObjF, grouping = True) +\
            '\tLB: ' + locale.currency(lb/params.scalObjF, grouping = True) +\
                f'\tGap (%): {100*(ub - lb)/ub:.4f}', flush = True)

    return(bestSol, ub, lb, (ub - lb)/ub, redFlag, buffSentBackward, buffSentForward)
