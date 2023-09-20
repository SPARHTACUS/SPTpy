# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""
from copy import deepcopy
import numpy as np

def update_load(params, network):
    """
        update the load numpy array
    """

    # Update the load
    network.NET_LOAD = deepcopy(network.NET_LOAD[[b for b in range(network.NET_LOAD.shape[1])
                                                            if b in network.BUS_HEADER.values()],:])

    for b, bus in enumerate(network.BUS_ID):
        network.BUS_HEADER[bus] = b

def _del_line(network, line_to_del:int):
    """
        delete a transmission line from the network

        line_to_del is the line ID of the line to be deleted
    """
    network.LINE_ID.remove(line_to_del)
    del network.LINE_F_T[line_to_del]
    del network.LINE_FLOW_UB[line_to_del]
    del network.LINE_X[line_to_del]
    del network.ACTIVE_BOUNDS[line_to_del]
    del network.ACTIVE_UB[line_to_del]
    del network.ACTIVE_LB[line_to_del]
    del network.ACTIVE_UB_PER_PERIOD[line_to_del]
    del network.ACTIVE_LB_PER_PERIOD[line_to_del]

def _del_bus(network, bus_to_del:int):
    """
        delete a bus from the network

        bus_to_del is the bus ID of the bus to be deleted
    """
    del network.LINES_FROM_BUS[bus_to_del]
    del network.LINES_TO_BUS[bus_to_del]
    del network.LINKS_FROM_BUS[bus_to_del]
    del network.LINKS_TO_BUS[bus_to_del]
    del network.BUS_NAME[bus_to_del]
    network.BUS_ID.remove(bus_to_del)
    del network.BUS_HEADER[bus_to_del]

def del_end_of_line_buses(network, buses_no_injections):
    """
        Delete no load and no generation buses connected to a single line
    """

    for bus in buses_no_injections:
        if (len(network.LINES_FROM_BUS[bus]) + len(network.LINES_TO_BUS[bus])) <= 1:

            for l in (network.LINES_FROM_BUS[bus] + network.LINES_TO_BUS[bus]):
                if not(network.LINE_F_T[l][0] == bus):
                    # Remove the line from the buses connected to 'bus'
                    network.LINES_FROM_BUS[network.LINE_F_T[l][0]].remove(l)

                elif not(network.LINE_F_T[l][1] == bus):
                    # Remove the line from the buses connected to 'bus'
                    network.LINES_TO_BUS[network.LINE_F_T[l][1]].remove(l)

                _del_line(network, l)

            _del_bus(network, bus)

def del_mid_point_buses(params, network, buses_no_injections):
    """
        Delete buses with no generation and no load connected only to two lines
    """

    header = {}
    header['From (ID)'] = 0
    header['From (Name)'] = 1
    header['To (ID)'] = 2
    header['To (Name)'] = 3
    header['Cap'] = 4
    header['Reac'] = 5

    for bus in buses_no_injections:
        if (len(network.LINES_FROM_BUS[bus]) + len(network.LINES_TO_BUS[bus])) == 2:

            # Add a new transmission line
            buses_of_new_connec = []
            cap, admt, reac = 1e12, 0, 0

            for l in (network.LINES_FROM_BUS[bus] + network.LINES_TO_BUS[bus]):
                if (not(network.LINE_F_T[l][0] == bus) and
                                        not(network.LINE_F_T[l][0] in buses_of_new_connec)):
                    buses_of_new_connec.append(network.LINE_F_T[l][0])
                    # Remove the line from the buses connected to 'bus'
                    network.LINES_FROM_BUS[network.LINE_F_T[l][0]].remove(l)

                elif (not(network.LINE_F_T[l][1] == bus) and
                                        not(network.LINE_F_T[l][1] in buses_of_new_connec)):
                    buses_of_new_connec.append(network.LINE_F_T[l][1])
                    # Remove the line from the buses connected to 'bus'
                    network.LINES_TO_BUS[network.LINE_F_T[l][1]].remove(l)

                cap = min(cap, network.LINE_FLOW_UB[l])
                reac += 1/network.LINE_X[l]

                _del_line(network, l)

            admt = 1/reac

            # Add the new line. Note that this new line adopts the key l from the last deleted line
            buses_of_new_connec.sort()

            row = [str(buses_of_new_connec[0]),
                    str(network.BUS_NAME[buses_of_new_connec[0]]),
                        str(buses_of_new_connec[1]),
                            str(network.BUS_NAME[buses_of_new_connec[1]]),
                                str(cap*params.POWER_BASE), str(1/admt)]

            network.add_new_AC_line(params, row, header)

            _del_bus(network, bus)

def reduce_system(params, hydros, thermals, network):
    """
        Build the DC network model
    """

    done, it, n_buses_deleted, n_lines_deleted = False, 0, 0, 0

    # initial number of buses and initial number of transmission lines
    ini_n_buses, ini_n_lines = len(network.BUS_ID), len(network.LINE_ID)

    #### Set of candidate buses to be deleted
    buses_no_injections = (set(network.BUS_ID)
                        - ({bus for h in hydros.ID for bus in hydros.UNIT_BUS[h].values()}
                                    | set(set(bus for g in thermals.ID for bus in thermals.BUS[g])))
                        - {network.BUS_ID[b]
                            for b in set(np.where(np.abs(network.NET_LOAD) > 0)[1])}
                        - set(network.REF_BUS_ID)
                        )

    while not(done):
        it += 1

        del_end_of_line_buses(network, buses_no_injections)

        update_load(params, network)

        buses_no_injections = (set(network.BUS_ID)
                        - ({bus for h in hydros.ID for bus in hydros.UNIT_BUS[h].values()}
                                    | set(set(bus for g in thermals.ID for bus in thermals.BUS[g])))
                        - {network.BUS_ID[b]
                            for b in set(np.where(np.abs(network.NET_LOAD) > 0)[1])}
                        - set(network.REF_BUS_ID)
                        )

        del_mid_point_buses(params, network, buses_no_injections)

        update_load(params, network)

        buses_no_injections = (set(network.BUS_ID)
                        - ({bus for h in hydros.ID for bus in hydros.UNIT_BUS[h].values()}
                                    | set(set(bus for g in thermals.ID for bus in thermals.BUS[g])))
                        - {network.BUS_ID[b]
                            for b in set(np.where(np.abs(network.NET_LOAD) > 0)[1])}
                        - set(network.REF_BUS_ID)
                        )

        if len([bus for bus in buses_no_injections
            if (len(network.LINES_FROM_BUS[bus]) + len(network.LINES_TO_BUS[bus])) <= 2]) == 0:
            # then there is no other bus to delete
            done = True

    f_n_buses, f_n_lines = len(network.BUS_ID), len(network.LINE_ID)
    n_buses_deleted, n_lines_deleted = (ini_n_buses - f_n_buses), (ini_n_lines - f_n_lines)

    print('\n\n\n' + f"{it} iterations were performed")
    print(f"{n_buses_deleted} buses and {n_lines_deleted} lines were removed" + "\n\n\n")
