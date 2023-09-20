"""
@author: Colonetti
"""
from time import time
from copy import deepcopy
import numpy as np

from network import add_new_parallel_line

def _remove_many_connect_buses(params, network, thermals, hydros, bus_to_del:int):
    """
        delete buses whose lines are not binding
    """

    # get all lines connected to this bus
    lines_connected_to_bus = network.LINES_FROM_BUS[bus_to_del]+network.LINES_TO_BUS[bus_to_del]

    buses_connected = [
                        bus for l in lines_connected_to_bus for bus in network.LINE_F_T[l]
                                if bus != bus_to_del
                    ]

    buses_connected.sort()

    all_buses = buses_connected + [bus_to_del]
    all_buses.sort()

    new_connections = [(bus_1, bus_2) for b_idx_1, bus_1 in enumerate(buses_connected)
                                                    for bus_2 in buses_connected[b_idx_1+1:]]

    A = np.zeros((len(lines_connected_to_bus), len(all_buses)), dtype = 'int')
    for l_idx, l in enumerate(lines_connected_to_bus):
        A[l_idx, all_buses.index(network.LINE_F_T[l][1])] = 1
        A[l_idx, all_buses.index(network.LINE_F_T[l][0])] = -1

    Y = np.diag(np.array([1/network.LINE_X[l] for l in lines_connected_to_bus], dtype = 'd'))
    B = np.matmul(np.transpose(A), np.matmul(Y, A))

    b_ext_idxs = [all_buses.index(bus_to_del)]

    b_front_idxs = [all_buses.index(bus) for bus in buses_connected]
    b_front_idxs.sort()

    B_front_ext = B[b_front_idxs, :][:, b_ext_idxs]

    B_ext_ext_inv = np.linalg.inv(B[b_ext_idxs, :][:, b_ext_idxs])

    B_front_front_new = B[b_front_idxs, :][:, b_front_idxs] - np.matmul(B_front_ext,
                                                            np.matmul(B_ext_ext_inv,
                                                                B[b_ext_idxs, :][:, b_front_idxs]))

    B_ext_impact = -1*np.matmul(B_front_ext, B_ext_ext_inv)

    for b_idx, bus in enumerate(buses_connected):
        _reassign_injections(hydros, thermals, network, bus_to_del, bus, B_ext_impact[b_idx][0])

    for connec in new_connections:
        existing_paral_line = None  #in case there is already a line between buses_of_new_connection

        # check if there is already a line between buses in buses_of_new_connection
        lines_btw_buses = [l for l in network.LINE_ID if network.LINE_F_T[l] == connec]
        if len(lines_btw_buses) > 0:
            assert len(lines_btw_buses) == 1
            existing_paral_line = lines_btw_buses[0]

        if existing_paral_line is None:
            network.LINES_FROM_BUS[connec[0]].append(max(network.LINE_ID) + 1)
            network.LINES_TO_BUS[connec[1]].append(max(network.LINE_ID) + 1)

        network.add_new_line(params, max(network.LINE_ID) + 1,
                                connec[0], connec[1],
                                    -1/B_front_front_new[buses_connected.index(connec[0]),
                                        buses_connected.index(connec[1])], 0, 0, 0,
                                        99999, 99999,
                                            0, 0, 0, 0, 0)

    for l in lines_connected_to_bus:
        network.LINES_FROM_BUS[network.LINE_F_T[l][0]].remove(l)
        network.LINES_TO_BUS[network.LINE_F_T[l][1]].remove(l)

    _del_lines(network, lines_connected_to_bus)

    del network.LINES_FROM_BUS[bus_to_del]
    del network.LINES_TO_BUS[bus_to_del]

    update_load_and_network(params, network, thermals, hydros,
                                                [network.BUS_ID.index(bus_to_del)], [bus_to_del])

def _del_lines(network, list_of_lines:list):
    """
        list_of_lines is a list of lines to be deleted from network
    """

    for l in list_of_lines:
        if (network.LINE_F_T[l][0] in network.LINES_FROM_BUS) and\
                                            (l in network.LINES_FROM_BUS[network.LINE_F_T[l][0]]):
            network.LINES_FROM_BUS[network.LINE_F_T[l][0]].remove(l)
        if (network.LINE_F_T[l][1] in network.LINES_TO_BUS) and\
                                            (l in network.LINES_TO_BUS[network.LINE_F_T[l][1]]):
            network.LINES_TO_BUS[network.LINE_F_T[l][1]].remove(l)

        del network.LINE_ID[network.LINE_ID.index(l)]
        del network.LINE_F_T[l]
        del network.LINE_FLOW_UB[l]
        del network.LINE_FLOW_LB[l]
        del network.LINE_X[l]
        del network.ACTIVE_BOUNDS[l]
        del network.ACTIVE_UB[l]
        del network.ACTIVE_LB[l]
        del network.ACTIVE_UB_PER_PERIOD[l]
        del network.ACTIVE_LB_PER_PERIOD[l]

def _reassign_injections(hydros, thermals, network, bus:int, new_bus:int, bus_coeff:float):
    """
    if a bus 'bus' is removed from the grid but there are either active or passive power injections
    to it, then these injections need to be reassigned to a new bus 'new_bus'
    """
    #### Remove old bus and add elements to the new bus
    if np.max(np.abs(network.NET_LOAD[network.BUS_HEADER[bus], :]) > 0):
        network.NET_LOAD[network.BUS_HEADER[new_bus], :] = np.add(
                                            network.NET_LOAD[network.BUS_HEADER[new_bus], :],
                                            bus_coeff*network.NET_LOAD[network.BUS_HEADER[bus], :])

    hydro_units = [(h, u) for h in hydros.ID for u in hydros.UNIT_ID[h]
                                                                if bus in hydros.UNIT_BUS[h][u]]
    for (h, u) in hydro_units:
        if new_bus not in hydros.UNIT_BUS[h][u]:
            hydros.UNIT_BUS[h][u].append(new_bus)
            hydros.UNIT_BUS_COEFF[h][u].update(
                                            {new_bus: bus_coeff*hydros.UNIT_BUS_COEFF[h][u][bus]})
        else:
            hydros.UNIT_BUS_COEFF[h][u][new_bus] += bus_coeff*hydros.UNIT_BUS_COEFF[h][u][bus]

    thermal_units = [g for g in thermals.UNIT_NAME.keys() if bus in thermals.BUS[g]]
    for g in thermal_units:
        if new_bus not in thermals.BUS[g]:
            thermals.BUS[g].append(new_bus)
            thermals.BUS_COEFF[g].update({new_bus: bus_coeff*thermals.BUS_COEFF[g][bus]})
        else:
            thermals.BUS_COEFF[g][new_bus] += bus_coeff*thermals.BUS_COEFF[g][bus]

def update_load_and_network(params, network, thermals, hydros,
                                indices_of_buses_to_delete:list,
                                buses_to_delete:list):
    """Buses and/lines have been deleted. Update the the network object"""

    # The buses to be kept are
    indices_of_buses_to_keep = [b for b in range(len(network.BUS_ID))
                                                        if b not in indices_of_buses_to_delete]

    indices_of_buses_to_keep.sort()
    # Update the load
    network.NET_LOAD = deepcopy(network.NET_LOAD[indices_of_buses_to_keep, :])

    for bus in buses_to_delete:
        del network.BUS_NAME[bus]
        network.BUS_ID.remove(bus)

    for bus in buses_to_delete:
        if bus in network.REF_BUS_ID:
            # delete this bus from the reference buses set and choose a new one to replace it
            network.REF_BUS_ID.remove(bus)
            for bus_2 in network.BUS_ID:
                if bus_2 not in network.REF_BUS_ID:
                    network.REF_BUS_ID.append(bus_2)
                    break

    for bus in buses_to_delete:
        hydro_units = [(h, u) for h in hydros.ID for u in hydros.UNIT_ID[h]
                                                                if bus in hydros.UNIT_BUS[h][u]]
        for k in hydro_units:
            hydros.UNIT_BUS[k[0]][k[1]].remove(bus)
            del hydros.UNIT_BUS_COEFF[k[0]][k[1]][bus]

        thermal_units = [u for u in thermals.UNIT_NAME.keys() if bus in thermals.BUS[u]]
        for g in thermal_units:
            thermals.BUS[g].remove(bus)
            del thermals.BUS_COEFF[g][bus]

    network.BUS_HEADER = {bus: b for (b, bus) in enumerate(network.BUS_ID)}

def _del_end_of_line_buses(network, buses_no_load_no_gen,
                                        buses_to_delete, index_of_buses_to_delete, lines_deleted):
    """Delete buses with no load and no generation connected to a single line"""

    for bus in buses_no_load_no_gen:
        if (((len(network.LINES_FROM_BUS[bus]) + len(network.LINES_TO_BUS[bus])) <= 1) and
            (len(network.LINKS_FROM_BUS[bus]) + len(network.LINKS_TO_BUS[bus]) == 0)):
            buses_to_delete.append(bus)
            index_of_buses_to_delete.append(network.BUS_ID.index(bus))

            for l in (network.LINES_FROM_BUS[bus] + network.LINES_TO_BUS[bus]):
                if not(network.LINE_F_T[l][0] == bus):
                    # Remove the line from the other bus connected to 'bus'
                    network.LINES_FROM_BUS[network.LINE_F_T[l][0]].remove(l)

                    if bus in network.REF_BUS_ID:
                        index_del_bus = network.REF_BUS_ID.index(bus)
                        network.REF_BUS_ID[index_del_bus] = network.LINE_F_T[l][0]

                elif not(network.LINE_F_T[l][1] == bus):
                    # Remove the line from the other bus connected to 'bus'
                    network.LINES_TO_BUS[network.LINE_F_T[l][1]].remove(l)

                    if bus in network.REF_BUS_ID:
                        index_del_bus = network.REF_BUS_ID.index(bus)
                        network.REF_BUS_ID[index_del_bus] = network.LINE_F_T[l][1]

                _del_lines(network, [l])

                lines_deleted += 1

            if bus in network.REF_BUS_ID:
                # just in case there was no lines connecting this bus to the rest of the network
                network.REF_BUS_ID.remove(bus)

            del network.LINES_FROM_BUS[bus]
            del network.LINES_TO_BUS[bus]

    return lines_deleted

def _del_mid_point_buses(params, network, buses_no_load_no_gen,
                                    buses_to_delete, index_of_buses_to_delete, lines_deleted):
    """Delete buses with no generation and no load connected only to two lines"""

    for bus in buses_no_load_no_gen:
        if ((len(network.LINES_FROM_BUS[bus] + network.LINES_TO_BUS[bus]) == 2) and
            (len(network.LINKS_FROM_BUS[bus]) + len(network.LINKS_TO_BUS[bus]) == 0)):
            buses_to_delete.append(bus)
            index_of_buses_to_delete.append(network.BUS_ID.index(bus))

            # Add a new transmission line
            buses_of_new_connection = []
            cap_ub, cap_lb = np.array(params.T*[1e12]), np.array(params.T*[-1e12])
            cap_emerg_ub, cap_emerg_lb = np.array(params.T*[1e12]), np.array(params.T*[-1e12])
            reactance, resistance, shut_conductance, shunt_suscep = 0, 0, 0, 0

            active_bounds_of_old_lines = False

            buses_of_new_connection = [bus2 for l in (network.LINES_FROM_BUS[bus]
                                                    + network.LINES_TO_BUS[bus])
                                                    for bus2 in network.LINE_F_T[l] if bus2 != bus]
            # Add the new line. Note that this new line adopts the key l from the last deleted line
            buses_of_new_connection.sort()

            for l in (network.LINES_FROM_BUS[bus] + network.LINES_TO_BUS[bus]):
                if (not(network.LINE_F_T[l][0] == bus) and
                                            not(network.LINE_F_T[l][0] in buses_of_new_connection)):
                    buses_of_new_connection.append(network.LINE_F_T[l][0])
                    # Remove the line from the buses connected to 'bus'
                    network.LINES_FROM_BUS[network.LINE_F_T[l][0]].remove(l)

                elif (not(network.LINE_F_T[l][1] == bus) and
                                            not(network.LINE_F_T[l][1] in buses_of_new_connection)):
                    buses_of_new_connection.append(network.LINE_F_T[l][1])
                    # Remove the line from the buses connected to 'bus'
                    network.LINES_TO_BUS[network.LINE_F_T[l][1]].remove(l)

                if (network.LINE_F_T[l][0] == buses_of_new_connection[0] or
                        network.LINE_F_T[l][1] == buses_of_new_connection[1]):
                    cap_ub = np.min((cap_ub, network.LINE_FLOW_UB[l]), axis = 0)
                    cap_lb = np.max((cap_lb, network.LINE_FLOW_LB[l]), axis = 0)
                else:
                    cap_ub = np.min((cap_ub, -1*network.LINE_FLOW_LB[l]), axis = 0)
                    cap_lb = np.max((cap_lb, -1*network.LINE_FLOW_UB[l]), axis = 0)

                reactance += network.LINE_X[l]

                active_bounds_of_old_lines =max(active_bounds_of_old_lines,network.ACTIVE_BOUNDS[l])

                _del_lines(network, [l])

                lines_deleted += 1

            if bus in network.REF_BUS_ID:
                index_del_bus = network.REF_BUS_ID.index(bus)
                network.REF_BUS_ID[index_del_bus] = buses_of_new_connection[0]

            #### Check if connection already exists
            found = False
            for l2 in [l2 for l2 in network.LINES_FROM_BUS[buses_of_new_connection[0]]
                                        if network.LINE_F_T[l2][1] == buses_of_new_connection[1]]:
                # Then the line already exists
                found = True

                (_, network.LINE_X[l2], _1, _2,
                    _3, _4,
                    network.LINE_FLOW_UB[l2], network.LINE_FLOW_LB[l2],
                    _5, _6) = add_new_parallel_line(
                                            resistance, reactance, shut_conductance, shunt_suscep,
                                            cap_ub, cap_lb, cap_emerg_ub, cap_emerg_lb,
                                            0, network.LINE_X[l2], 0, 0,
                                            network.LINE_FLOW_UB[l2], network.LINE_FLOW_LB[l2],
                                            network.LINE_FLOW_UB[l2],
                                            network.LINE_FLOW_LB[l2])

                network.ACTIVE_BOUNDS[l2] = max(active_bounds_of_old_lines,
                                                    network.ACTIVE_BOUNDS[l2])
                network.ACTIVE_UB[l2] = max(active_bounds_of_old_lines, network.ACTIVE_UB[l2])
                network.ACTIVE_LB[l2] = max(active_bounds_of_old_lines, network.ACTIVE_LB[l2])
                network.ACTIVE_UB_PER_PERIOD[l2] = {t: max(active_bounds_of_old_lines,
                                                            network.ACTIVE_UB_PER_PERIOD[l2][t])
                                                                        for t in range(params.T)}
                network.ACTIVE_LB_PER_PERIOD[l2] = {t: max(active_bounds_of_old_lines,
                                                            network.ACTIVE_LB_PER_PERIOD[l2][t])
                                                                        for t in range(params.T)}
                break

            if not(found):
                network.LINE_ID.append(l)
                network.LINE_F_T[l] = (buses_of_new_connection[0], buses_of_new_connection[1])
                network.LINE_FLOW_UB[l] = cap_ub
                network.LINE_FLOW_LB[l] = cap_lb
                network.LINE_X[l] = reactance
                network.LINES_FROM_BUS[buses_of_new_connection[0]].append(l)
                network.LINES_TO_BUS[buses_of_new_connection[1]].append(l)

                network.ACTIVE_BOUNDS[l] = active_bounds_of_old_lines
                network.ACTIVE_UB[l] = active_bounds_of_old_lines
                network.ACTIVE_LB[l] = active_bounds_of_old_lines
                network.ACTIVE_UB_PER_PERIOD[l] = {t: active_bounds_of_old_lines
                                                                        for t in range(params.T)}
                network.ACTIVE_LB_PER_PERIOD[l] = {t: active_bounds_of_old_lines
                                                                        for t in range(params.T)}
                lines_deleted -= 1

            del network.LINES_FROM_BUS[bus]
            del network.LINES_TO_BUS[bus]

    return lines_deleted

def _remove_n_connections_buses(params, network, hydros, thermals):
    """
        delete buses with no active power injections connected exactly to three lines
    """

    buses_connected_to_active_lines = set()
    for l in [l for l in network.LINE_ID if network.ACTIVE_BOUNDS[l]]:
        buses_connected_to_active_lines.add(network.LINE_F_T[l][0])
        buses_connected_to_active_lines.add(network.LINE_F_T[l][1])

    buses_cannot_be_del = (
                            buses_connected_to_active_lines
                                            | {bus for bus in network.BUS_ID
                                            if (len(network.LINKS_FROM_BUS[bus]
                                                    + network.LINKS_TO_BUS[bus]) > 0)}
                            )

    def _get_cand_buses(network):
        """
            get buses with injections that can be deleted
        """

        cand_buses = []

        n_connections_of_buses = {bus: len(network.LINES_FROM_BUS[bus] + network.LINES_TO_BUS[bus])
                                        for bus in set(network.BUS_ID)
                                            - buses_cannot_be_del}

        for nc in range(0, MAX_NUMBER_OF_CONNEC + 1, 1):
            for bus in [bus for bus in set(network.BUS_ID) - buses_cannot_be_del
                                                            if (n_connections_of_buses[bus] == nc)]:
                cand_buses.append(bus)

        return cand_buses

    MAX_NUMBER_OF_CONNEC, ADMIT_THRESHOLD = params.MAX_NUMBER_OF_CONNECTIONS, 0

    total_cand_buses =  0
    total_removal = 0

    ini = time()
    cand_buses = _get_cand_buses(network)
    total_cand_buses += time() - ini

    while len(cand_buses) > 0:

        lines_connected_to_bus = (network.LINES_FROM_BUS[cand_buses[0]]
                                    + network.LINES_TO_BUS[cand_buses[0]]
                                    )

        if len(lines_connected_to_bus) == 0:
            # in case there is no line connected to this bus
            cand_buses = cand_buses[1:]
            continue

        if len(lines_connected_to_bus) > 5:
            buses_connected = [
                                bus for l in lines_connected_to_bus for bus in network.LINE_F_T[l]
                                        if bus != cand_buses[0]
                            ]

            buses_connected.sort()

            new_connections = [[bus_1, bus_2] for b_idx_1, bus_1 in enumerate(buses_connected)
                                                        for bus_2 in buses_connected[b_idx_1+1:]]

            par_lines = 0

            for connec in new_connections:
                # check if there is already a line between buses in buses_of_new_connection
                lines_btw_buses = [l for l in network.LINE_ID
                                                if network.LINE_F_T[l] == (connec[0], connec[1])]
                if len(lines_btw_buses) > 0:
                    par_lines += 1

            if len(new_connections) - par_lines <= 1:
                ini = time()
                _remove_many_connect_buses(params, network, thermals, hydros, cand_buses[0])
                total_removal += time() - ini

        else:
            ini = time()
            _remove_many_connect_buses(params, network, thermals, hydros, cand_buses[0])
            total_removal += time() - ini

        cand_buses = cand_buses[1:]

def reduce_network(params, hydros, thermals, network):
    """
        try to remove unnecessary lines and buses
    """

    ini_time = time()

    ini_buses, ini_lines = len(network.BUS_ID), len(network.LINE_ID)

    MAX_IT = 20

    it = 0

    buses_rm_1, lines_rm_1 = 0, 0
    buses_rm_2, lines_rm_2 = 0, 0
    buses_rm_3, lines_rm_3 = 0, 0
    buses_rm_4, lines_rm_4 = 0, 0

    total_time_no_injection = 0
    total_time_end_line_w_inj = 0
    total_time_mid_point_w_inj = 0
    total_time_many_connec = 0

    while (it < MAX_IT):

        initial_number_of_buses, initial_number_of_lines = len(network.BUS_ID), len(network.LINE_ID)

        # this loop is only used to make sure that the updated network will be checked in each
        # of the functions

        ini_n_inj = time()

        #### Set of candidate buses to be deleted
        buses_no_load_no_gen = (set(network.BUS_ID)
                                - network.get_renewable_gen_buses(thermals, hydros)
                                - network.get_load_buses(thermals, hydros)
                                - network.get_gen_buses(thermals, hydros))

        buses_to_delete, indices_of_buses_to_delete = [], []
        if params.MAX_NUMBER_OF_CONNECTIONS >= 1:
            _del_end_of_line_buses(network, buses_no_load_no_gen,
                                    buses_to_delete, indices_of_buses_to_delete, 0)
        buses_no_load_no_gen = buses_no_load_no_gen - set(buses_to_delete)

        if params.MAX_NUMBER_OF_CONNECTIONS >= 2:
            _del_mid_point_buses(params, network, buses_no_load_no_gen,
                                    buses_to_delete, indices_of_buses_to_delete, 0)

        update_load_and_network(params, network, thermals, hydros,
                                                indices_of_buses_to_delete,
                                                buses_to_delete)

        buses_rm_1 += initial_number_of_buses - len(network.BUS_ID)
        lines_rm_1 += initial_number_of_lines - len(network.LINE_ID)

        total_time_no_injection += time() - ini_n_inj


        #### now remove buses with injections
        ini_rm_end_line_buses = time()

        ini_b, ini_l = len(network.BUS_ID), len(network.LINE_ID)

        if params.MAX_NUMBER_OF_CONNECTIONS >= 1:
            _remove_end_of_line_buses_with_injections(params, hydros, thermals, network)

        buses_rm_2 += ini_b - len(network.BUS_ID)
        lines_rm_2 += ini_l - len(network.LINE_ID)

        total_time_end_line_w_inj += time() - ini_rm_end_line_buses

        ini_rm_mid_point = time()

        ini_b, ini_l = len(network.BUS_ID), len(network.LINE_ID)

        if params.MAX_NUMBER_OF_CONNECTIONS >= 2:
            _remove_mid_point_buses_with_injs(params, hydros, thermals, network)

        buses_rm_3 += ini_b - len(network.BUS_ID)
        lines_rm_3 += ini_l - len(network.LINE_ID)

        total_time_mid_point_w_inj += time() - ini_rm_mid_point


        ini_rm_m_c = time()

        ini_b, ini_l = len(network.BUS_ID), len(network.LINE_ID)
        if params.MAX_NUMBER_OF_CONNECTIONS >= 1:
            _remove_n_connections_buses(params, network, hydros, thermals)

        buses_rm_4 += ini_b - len(network.BUS_ID)
        lines_rm_4 += ini_l - len(network.LINE_ID)

        total_time_many_connec += time() - ini_rm_m_c

        it += 1

        if initial_number_of_buses == len(network.BUS_ID):
            break

    end_buses, end_lines = len(network.BUS_ID), len(network.LINE_ID)

    print(f'\n\n\n{ini_buses - end_buses} buses and {ini_lines - end_lines} lines were removed' +
                    f" in {it} iterations and {time() - ini_time:,.2f} seconds")
    print(f"No injection buses:\t\t{buses_rm_1} buses and {lines_rm_1} lines removed" +
                                    f" in {total_time_no_injection:,.2f} seconds.")
    print(f"End-line buses with injection:\t{buses_rm_2} buses and {lines_rm_2} lines removed" +
                                    f" in {total_time_end_line_w_inj:,.2f} seconds.")
    print(f"Mid-point buses:\t\t{buses_rm_3} buses and {lines_rm_3} lines removed" +
                                    f" in {total_time_mid_point_w_inj:,.2f} seconds.")
    print(f"Buses with many connections:\t{buses_rm_4} buses and {lines_rm_4} lines removed" +
                                    f" in {total_time_many_connec:,.2f} seconds.\n\n",
                                                                                flush = True)

def del_end_of_line_buses_and_reassign_injection(network, thermals, hydros, candidate_buses):
    """
        End-of-line buses are those connected to a single power line.
        Delete these buses and move their power injections to the neighbouring bus
    """

    for bus in candidate_buses:

        # The elements connected to the bus to be deleted must be
        # relocated to a new bus
        new_bus = -1e12
        for l in (network.LINES_FROM_BUS[bus] + network.LINES_TO_BUS[bus]):
            if not(network.LINE_F_T[l][0] == bus):
                # Remove the line from the buses connected to 'bus'
                network.LINES_FROM_BUS[network.LINE_F_T[l][0]].remove(l)
                new_bus = network.LINE_F_T[l][0]
            elif not(network.LINE_F_T[l][1] == bus):
                # Remove the line from the buses connected to 'bus'
                network.LINES_TO_BUS[network.LINE_F_T[l][1]].remove(l)
                new_bus = network.LINE_F_T[l][1]

            _del_lines(network, [l])

        del network.LINES_FROM_BUS[bus]
        del network.LINES_TO_BUS[bus]

        _reassign_injections(hydros, thermals, network, bus, new_bus, 1.00)

def _remove_end_of_line_buses_with_injections(params, hydros, thermals, network):
    """Even buses with power injection (either positive or negative) can be removed from the
    network without any damage to the representation as long as the maximum injection is at
    most equal to the capacity of the line connecting such bus to the rest of the network"""

    buses_cannot_be_del = (
                            set(network.REF_BUS_ID)|
                            {bus for (bus, v) in network.LINKS_FROM_BUS.items() if len(v) > 0} |
                            {bus for (bus, v) in network.LINKS_TO_BUS.items() if len(v) > 0}
                            )
                            #| {bus for h in hydros.ID for u in hydros.UNIT_ID[h] for bus in hydros.UNIT_BUS[h][u]}

                            #| {bus for g in thermals.UNIT_NAME.keys() for bus in thermals.BUS[g]}
        #                    | {bus for bus in network.BUS_ID
         #                   if np.max(np.abs(network.NET_LOAD[network.BUS_HEADER[bus]][:])) > 0}
    def _add_artificial_sec_constr(params, network, bus, l):
        """
            add a security constraint to limit the net injection of elements being reassigned
            to a new bus
        """

        constr_id = str(bus) + '_' + str(l)
        for t in range(params.T):
            if not(t in network.SEC_CONSTRS):
                network.SEC_CONSTRS[t] = {}

            network.SEC_CONSTRS[t][constr_id] = {
                            'name': 'art_bus_' + str(bus)+ '_' + str(t),
                            'net load': network.NET_LOAD[network.BUS_HEADER[bus], t],
                            'participants': {},
                            'participants_factors': {'thermals': {}, 'hydros': {}},
                            'LB': (network.LINE_FLOW_LB[l][t]
                                                            if bus == network.LINE_F_T[l][0] else
                                                                -1*network.LINE_FLOW_UB[l][t]),
                            'UB': (network.LINE_FLOW_UB[l][t]
                                                            if bus == network.LINE_F_T[l][0] else
                                                                -1*network.LINE_FLOW_LB[l][t])}

            if t > 0:
                network.SEC_CONSTRS[t][constr_id]['participants'] =\
                                                network.SEC_CONSTRS[0][constr_id]['participants']
                network.SEC_CONSTRS[t][constr_id]['participants_factors'] =\
                                        network.SEC_CONSTRS[0][constr_id]['participants_factors']
            else:
                network.SEC_CONSTRS[t][constr_id]['participants'] = {'thermals':[],'hydros':[]}

                network.SEC_CONSTRS[t][constr_id]['participants']['hydros'] +=\
                                            [(h, u) for h in hydros.ID for u in hydros.UNIT_ID[h]
                                                                    if bus in hydros.UNIT_BUS[h][u]]

                network.SEC_CONSTRS[t][constr_id]['participants']['thermals'] +=\
                                    [g for g in thermals.UNIT_NAME.keys() if bus in thermals.BUS[g]]

                network.SEC_CONSTRS[t][constr_id]['participants_factors']['thermals'] =\
                            {g: thermals.BUS_COEFF[g][bus]
                            for g in network.SEC_CONSTRS[t][constr_id]['participants']['thermals']}

                network.SEC_CONSTRS[t][constr_id]['participants_factors']['hydros'] = {}

                if len(network.SEC_CONSTRS[t][constr_id]['participants']['hydros']) > 0:

                    network.SEC_CONSTRS[t][constr_id]['participants_factors']['hydros'].update(
                                                    {(h, u): hydros.UNIT_BUS_COEFF[h][u][bus]
                                                        for (h, u) in network.SEC_CONSTRS[t]
                                                            [constr_id]['participants']['hydros']})

    def _get_candidate_buses(params, network, thermals, hydros):
        """
            get a set of buses that can be delete
        """

        # get the subset of buses connected to the system through a single line
        single_line_buses = {bus for bus in set(network.BUS_ID) - buses_cannot_be_del
                            if len(network.LINES_FROM_BUS[bus] + network.LINES_TO_BUS[bus]) <= 1}

        # Get the minimum and maximum loads of each bus during the planning horizon
        min_load = {bus: np.min(network.NET_LOAD[b, :]) for b, bus in enumerate(network.BUS_ID)
                                                                    if bus in single_line_buses}
        max_load = {bus: np.max(network.NET_LOAD[b, :]) for b, bus in enumerate(network.BUS_ID)
                                                                    if bus in single_line_buses}

        # Remember that net loads (power withdraws) are positive in network.NET_LOAD,
        # while net generation in NET_LOAD is negative.

        max_gen_of_bus = {bus: 0 for bus in network.BUS_ID}

        #### Set of candidate buses to be deleted
        candidate_buses = []

        threshold_line_limit = (99999.00/params.POWER_BASE)

        for bus in single_line_buses:
            lines_connected = network.LINES_FROM_BUS[bus] + network.LINES_TO_BUS[bus]

            # if there is a single line connecting bus to the network, and it is not a DC link

            for g in [g for g in thermals.UNIT_NAME.keys() if bus in thermals.BUS[g]]:
                max_gen_of_bus[bus] += thermals.BUS_COEFF[g][bus]*thermals.MAX_P[g]

            for (h, u) in [(h, u) for h in hydros.ID for u in hydros.UNIT_ID[h]
                            if hydros.TURB_OR_PUMP[h] != 'Pump' and bus in hydros.UNIT_BUS[h][u]]:
                max_gen_of_bus[bus] += hydros.UNIT_BUS_COEFF[h][u][bus]*hydros.UNIT_MAX_P[h][u]

            for (h, u) in [(h, u) for h in hydros.ID for u in hydros.UNIT_ID[h]
                            if hydros.TURB_OR_PUMP[h] == 'Pump' and bus in hydros.UNIT_BUS[h][u]]:
                max_load[bus] += hydros.UNIT_BUS_COEFF[h][u][bus]*(
                                                        sum(hydros.UNIT_MAX_TURB_DISCH[h].values())
                                                            *hydros.PUMP_CONVERSION_FACTOR[h]
                                                                        )

            for l in lines_connected:
                if (
                    not(network.ACTIVE_BOUNDS[l])
                        or
                        ((np.min(network.LINE_FLOW_UB[l]) >= threshold_line_limit)
                        and (np.max(network.LINE_FLOW_LB[l]) <= - threshold_line_limit))
                        or
                            (abs(max_load[bus]) <=
                        min(np.min(network.LINE_FLOW_UB[l]), -1*np.max(network.LINE_FLOW_LB[l]))
                        and abs(-1*min_load[bus] + max_gen_of_bus[bus])
                        <= min(np.min(network.LINE_FLOW_UB[l]),-1*np.max(network.LINE_FLOW_LB[l])))
                    ):
                    # either the load has limitless capacity or its possible most
                    # negative power injection (largest possible load) and most positive
                    # power injection (largest possible generation) are both within its capacity
                    candidate_buses.append(bus)
                else:
                    if (abs(max_load[bus]) >
                        min(np.min(network.LINE_FLOW_UB[l]), -1*np.max(network.LINE_FLOW_LB[l]))
                        or abs(-1*min_load[bus] + max_gen_of_bus[bus])
                        > min(np.min(network.LINE_FLOW_UB[l]), -1*np.max(network.LINE_FLOW_LB[l]))):

                        _add_artificial_sec_constr(params, network, bus, l)

                        candidate_buses.append(bus)

        return candidate_buses

    candidate_buses = _get_candidate_buses(params, network, thermals, hydros)

    while len(candidate_buses) > 0:
        #### Delete end-of-line buses

        del_end_of_line_buses_and_reassign_injection(network, thermals, hydros, candidate_buses)

        update_load_and_network(params, network, thermals, hydros,
                                            [network.BUS_ID.index(bus) for bus in candidate_buses],
                                            candidate_buses)

        candidate_buses = _get_candidate_buses(params, network, thermals, hydros)

def _remove_mid_bus_with_inj(params, network, thermals, hydros,
                            buses_deleted,
                            bus):
    """
        remove a mid-point bus `bus` that has injections connected to it
    """

    # by definition, the bus with the smallest ID will be the 'from bus' for the new line
    # and, the bus with the largest ID is the bus where the power injections will be reassigned
    # to and will be the 'to bus' for the new line

    if len(network.LINES_FROM_BUS[bus]) == 1 and len(network.LINES_TO_BUS[bus]) == 1:
        # the bus that is connected to the bus being deleted by
        # a line that 'comes out' of bus 'bus'
        bus_connected_line_from = [bus2 for l in network.LINES_FROM_BUS[bus]
                                    for bus2 in network.LINE_F_T[l] if bus2 != bus][0]

        # the bus that is connected to the bus being deleted by
        # a line that 'goes to' bus 'bus'
        bus_connected_line_to = [bus2 for l in network.LINES_TO_BUS[bus]
                                    for bus2 in network.LINE_F_T[l] if bus2 != bus][0]
    else:
        if len(network.LINES_FROM_BUS[bus]) == 2:
            bus_connected_line_from = [bus2
                                    for bus2 in network.LINE_F_T[network.LINES_FROM_BUS[bus][0]]
                                        if bus2 != bus][0]
            bus_connected_line_to = [bus2
                                    for bus2 in network.LINE_F_T[network.LINES_FROM_BUS[bus][1]]
                                        if bus2 != bus][0]
        elif len(network.LINES_TO_BUS[bus]) == 2:
            bus_connected_line_from = [bus2
                                    for bus2 in network.LINE_F_T[network.LINES_TO_BUS[bus][0]]
                                        if bus2 != bus][0]
            bus_connected_line_to = [bus2
                                    for bus2 in network.LINE_F_T[network.LINES_TO_BUS[bus][1]]
                                        if bus2 != bus][0]
        else:
            # then one of the lines have been deleted here. this bus can be deleted using
            # remove_end_of_line_buses_with_injections
            return

    if bus_connected_line_from < bus_connected_line_to:
        # in this case, the new line will 'come out' of bus bus_connected_line_from and will
        # go to bus bus_connected_line_to. Also, the injections previously in the bus being
        # deleted will be reassigned to bus bus_connected_line_to

        # moreover, the new line will assume the line ID line_1. thus, line_1 must be the
        # line 'coming out' of the bus with the smallest ID
        buses_of_new_connection = [bus_connected_line_from, bus_connected_line_to]
    else:
        buses_of_new_connection = [bus_connected_line_to, bus_connected_line_from]

    assert buses_of_new_connection[0] != buses_of_new_connection[1]

    if len(set(buses_of_new_connection)) == 1:
        raise ValueError("buses_of_new_connection should contain two buses")

    existing_paral_line = None # in case there is already a line between buses_of_new_connection

    # check if there is already a line between buses in buses_of_new_connection
    lines_btw_buses = [l for l in network.LINE_ID
                if network.LINE_F_T[l] == (buses_of_new_connection[0], buses_of_new_connection[1])
                or network.LINE_F_T[l] == (buses_of_new_connection[1], buses_of_new_connection[0])]
    if len(lines_btw_buses) > 0:
        assert len(lines_btw_buses) == 1
        existing_paral_line = lines_btw_buses[0]

    line_1 = [l for l in network.LINES_FROM_BUS[bus] + network.LINES_TO_BUS[bus]
                                            if buses_of_new_connection[0] in network.LINE_F_T[l]][0]
    line_2 = [l for l in network.LINES_FROM_BUS[bus] + network.LINES_TO_BUS[bus]
                                            if buses_of_new_connection[1] in network.LINE_F_T[l]][0]

    line_id_to_keep, line_id_to_del = line_1, line_2

    if ((network.ACTIVE_BOUNDS[line_id_to_keep] and network.ACTIVE_BOUNDS[line_id_to_del]) or
        ((bus in {bus_2 for g in thermals.UNIT_NAME.keys() for bus_2 in thermals.BUS[g]} |
                {bus_2 for h in hydros.ID for u in hydros.UNIT_ID[h]
                                                        for bus_2 in hydros.UNIT_BUS[h][u]})
            and (network.ACTIVE_BOUNDS[line_id_to_keep] or network.ACTIVE_BOUNDS[line_id_to_del]))):
        # it might happen that a candidate bus at first consider to be deleted is connected
        # to two lines that are possibly binding. this might happen if one bus 'to the left' of the
        # current candidate bus was deleted and one of its lines was binding. then, the new line
        # now connected to the current bus is now possibly binding.
        return

    #### now the bus can be surely deleted
    buses_deleted.append(bus)

    for bus2 in buses_of_new_connection:
        if line_1 in network.LINES_FROM_BUS[bus2]:
            network.LINES_FROM_BUS[bus2].remove(line_1)
        if line_1 in network.LINES_TO_BUS[bus2]:
            network.LINES_TO_BUS[bus2].remove(line_1)
        if line_2 in network.LINES_FROM_BUS[bus2]:
            network.LINES_FROM_BUS[bus2].remove(line_2)
        if line_2 in network.LINES_TO_BUS[bus2]:
            network.LINES_TO_BUS[bus2].remove(line_2)

    pf =-(1/network.LINE_X[line_id_to_keep])/((1/network.LINE_X[line_1])+(1/network.LINE_X[line_2]))
    coeffs_new_buses = {buses_of_new_connection[0]: -pf, buses_of_new_connection[1]: (1 + pf)}

    addition_to_cap = 0

    if np.max(np.abs(network.NET_LOAD[network.BUS_HEADER[bus], :])) != 0:
        if network.ACTIVE_BOUNDS[line_id_to_keep]:
            addition_to_cap = (coeffs_new_buses[buses_of_new_connection[0]]*
                                                    network.NET_LOAD[network.BUS_HEADER[bus], :])
        elif network.ACTIVE_BOUNDS[line_id_to_del]:
            addition_to_cap = (coeffs_new_buses[buses_of_new_connection[1]]*
                                                    network.NET_LOAD[network.BUS_HEADER[bus], :])

        network.NET_LOAD[network.BUS_HEADER[buses_of_new_connection[1]], :] = np.add(
                            network.NET_LOAD[network.BUS_HEADER[buses_of_new_connection[1]], :],
                            coeffs_new_buses[buses_of_new_connection[1]]*
                                network.NET_LOAD[network.BUS_HEADER[bus], :])

        network.NET_LOAD[network.BUS_HEADER[buses_of_new_connection[0]], :] = np.add(
                                network.NET_LOAD[network.BUS_HEADER[buses_of_new_connection[0]], :],
                                coeffs_new_buses[buses_of_new_connection[0]]*
                                    network.NET_LOAD[network.BUS_HEADER[bus], :])

    if network.ACTIVE_BOUNDS[line_id_to_keep]:
        if buses_of_new_connection[0] == network.LINE_F_T[line_id_to_keep][0]:

            network.ACTIVE_UB[line_id_to_keep] = network.ACTIVE_UB[line_id_to_keep]
            network.ACTIVE_LB[line_id_to_keep] = network.ACTIVE_LB[line_id_to_keep]

            network.ACTIVE_UB_PER_PERIOD[line_id_to_keep] = {t:
                                                    network.ACTIVE_UB_PER_PERIOD[line_id_to_keep][t]
                                                            for t in range(params.T)}
            network.ACTIVE_LB_PER_PERIOD[line_id_to_keep] = {t:
                                                    network.ACTIVE_LB_PER_PERIOD[line_id_to_keep][t]
                                                            for t in range(params.T)}

            network.LINE_FLOW_UB[line_id_to_keep] =\
                                    network.LINE_FLOW_UB[line_id_to_keep] - addition_to_cap
            network.LINE_FLOW_LB[line_id_to_keep] =\
                                    network.LINE_FLOW_LB[line_id_to_keep] - addition_to_cap
        else:
            # the flows are now in the inverse direction wrt to the old line
            old_active_ub, old_active_lb = (network.ACTIVE_UB[line_id_to_keep],
                                                network.ACTIVE_LB[line_id_to_keep])
            network.ACTIVE_UB[line_id_to_keep] = old_active_lb
            network.ACTIVE_LB[line_id_to_keep] = old_active_ub

            (old_active_ub_per_period,
                        old_active_lb_per_period) = (network.ACTIVE_UB_PER_PERIOD[line_id_to_keep],
                                                    network.ACTIVE_LB_PER_PERIOD[line_id_to_keep])
            network.ACTIVE_UB_PER_PERIOD[line_id_to_keep] = {t: old_active_lb_per_period[t]
                                                            for t in range(params.T)}
            network.ACTIVE_LB_PER_PERIOD[line_id_to_keep] = {t: old_active_ub_per_period[t]
                                                            for t in range(params.T)}

            old_ub = network.LINE_FLOW_UB[line_id_to_keep][:]
            old_lb = network.LINE_FLOW_LB[line_id_to_keep][:]
            network.LINE_FLOW_UB[line_id_to_keep] = -1*old_lb - addition_to_cap
            network.LINE_FLOW_LB[line_id_to_keep] = -1*old_ub - addition_to_cap

    elif network.ACTIVE_BOUNDS[line_id_to_del]:
        if buses_of_new_connection[1] == network.LINE_F_T[line_id_to_del][1]:
            # then the positive flow direction wrt the to-bus of the new connection is the same
            # i.e., if the new to-bus had a line 'going to' it, then the new line is also 'going to'
            #(buses_of_new_connection[0])--  -->>(bus_to_del)--  -->>(buses_of_new_connection[1])

            network.ACTIVE_UB[line_id_to_keep] = network.ACTIVE_UB[line_id_to_del]
            network.ACTIVE_LB[line_id_to_keep] = network.ACTIVE_LB[line_id_to_del]

            network.ACTIVE_UB_PER_PERIOD[line_id_to_keep] = {t:
                                                    network.ACTIVE_UB_PER_PERIOD[line_id_to_del][t]
                                                            for t in range(params.T)}
            network.ACTIVE_LB_PER_PERIOD[line_id_to_keep] = {t:
                                                    network.ACTIVE_LB_PER_PERIOD[line_id_to_del][t]
                                                            for t in range(params.T)}

            network.LINE_FLOW_UB[line_id_to_keep] = network.LINE_FLOW_UB[line_id_to_del] +\
                                                                                    addition_to_cap
            network.LINE_FLOW_LB[line_id_to_keep] = network.LINE_FLOW_LB[line_id_to_del] +\
                                                                                    addition_to_cap
        else:
            # the flow wrt to the new to-bus has been reversed.
            # in this case, the lower bound on the old line of bus buses_of_new_connection[1]
            # being deleted now becomes the upper bound of the new line.
            # this is simply because the old lower bound limits how much power can 'go to'
            # bus buses_of_new_connection[1]
            #(buses_of_new_connection[0])--  -->>(bus_to_del)<<--  --(buses_of_new_connection[1])

            old_active_ub, old_active_lb = (network.ACTIVE_UB[line_id_to_del],
                                                network.ACTIVE_LB[line_id_to_del])
            network.ACTIVE_UB[line_id_to_keep] = old_active_lb
            network.ACTIVE_LB[line_id_to_keep] = old_active_ub

            (old_active_ub_per_period,
                        old_active_lb_per_period) = (network.ACTIVE_UB_PER_PERIOD[line_id_to_del],
                                                    network.ACTIVE_LB_PER_PERIOD[line_id_to_del])
            network.ACTIVE_UB_PER_PERIOD[line_id_to_keep] = {t: old_active_lb_per_period[t]
                                                            for t in range(params.T)}
            network.ACTIVE_LB_PER_PERIOD[line_id_to_keep] = {t: old_active_ub_per_period[t]
                                                            for t in range(params.T)}

            old_ub = network.LINE_FLOW_UB[line_id_to_del][:]
            old_lb = network.LINE_FLOW_LB[line_id_to_del][:]
            network.LINE_FLOW_UB[line_id_to_keep] = -1*old_lb + addition_to_cap
            network.LINE_FLOW_LB[line_id_to_keep] = -1*old_ub + addition_to_cap

    else:
        # there was no active flow for these two lines
        raise ValueError("This function is meant for buses connected to a single active line")

    network.LINE_F_T[line_id_to_keep] = (buses_of_new_connection[0], buses_of_new_connection[1])
    network.LINE_X[line_id_to_keep] = network.LINE_X[line_1] + network.LINE_X[line_2]
    network.LINES_FROM_BUS[buses_of_new_connection[0]].append(line_id_to_keep)
    network.LINES_TO_BUS[buses_of_new_connection[1]].append(line_id_to_keep)

    network.ACTIVE_BOUNDS[line_id_to_keep] = (network.ACTIVE_BOUNDS[line_id_to_keep] or
                                                    network.ACTIVE_BOUNDS[line_id_to_del])

    if existing_paral_line is not None:
        (_, network.LINE_X[line_id_to_keep], _1, _2, _3, _4,
                            network.LINE_FLOW_UB[line_id_to_keep],
                            network.LINE_FLOW_LB[line_id_to_keep],
                                _5, _6)= add_new_parallel_line(
                                                    0,
                                                    network.LINE_X[line_id_to_keep],
                                                    0, 0,
                                                    network.LINE_FLOW_UB[line_id_to_keep],
                                                    network.LINE_FLOW_LB[line_id_to_keep],
                                                    network.LINE_FLOW_UB[line_id_to_keep],
                                                    network.LINE_FLOW_LB[line_id_to_keep],
                                                    0,
                                                    network.LINE_X[existing_paral_line],
                                                    0, 0,
                                                    network.LINE_FLOW_UB[existing_paral_line],
                                                    network.LINE_FLOW_LB[existing_paral_line],
                                                    network.LINE_FLOW_UB[existing_paral_line],
                                                    network.LINE_FLOW_LB[existing_paral_line])

        network.ACTIVE_BOUNDS[line_id_to_keep] = max(network.ACTIVE_BOUNDS[existing_paral_line],
                                                        network.ACTIVE_BOUNDS[line_id_to_keep])
        network.ACTIVE_UB[line_id_to_keep] = max(network.ACTIVE_UB[existing_paral_line],
                                                    network.ACTIVE_UB[line_id_to_keep])
        network.ACTIVE_LB[line_id_to_keep] = max(network.ACTIVE_LB[existing_paral_line],
                                                    network.ACTIVE_LB[line_id_to_keep])
        network.ACTIVE_UB_PER_PERIOD[line_id_to_keep] = {t: max(
                                            network.ACTIVE_UB_PER_PERIOD[existing_paral_line][t],
                                            network.ACTIVE_UB_PER_PERIOD[line_id_to_keep][t])
                                                                for t in range(params.T)}
        network.ACTIVE_LB_PER_PERIOD[line_id_to_keep] = {t: max(
                                            network.ACTIVE_LB_PER_PERIOD[existing_paral_line][t],
                                            network.ACTIVE_LB_PER_PERIOD[line_id_to_keep][t])
                                                                for t in range(params.T)}

    _del_lines(network, [line_id_to_del])

    del network.LINES_FROM_BUS[bus]
    del network.LINES_TO_BUS[bus]

    if existing_paral_line is not None:
        _del_lines(network, [existing_paral_line])

def _remove_mid_point_buses_with_injs(params, hydros, thermals, network):
    """
    Buses with injections connected to the network by two lines can be removed by reformulating
    the power flow between the two terminals connected to the bus being removed
    Thus, one bus and one line are removed
    """

    def _get_cand_buses(network, buses_cannot_be_del):
        """get mid-point buses with injections that can be deleted"""
        candidate_buses = []

        # the bus must be connected to two lines
        # exactly one of the lines connected to the bus can be binding
        # if a bus is removed, none of its immediate neighbouring buses can be removed
        candidate_buses = [bus for bus in set(network.BUS_ID) - buses_cannot_be_del
                        if len(network.LINES_FROM_BUS[bus] + network.LINES_TO_BUS[bus]) == 2
                        and (len([l for l in network.LINES_FROM_BUS[bus] +
                                    network.LINES_TO_BUS[bus] if network.ACTIVE_BOUNDS[l]]) == 1)
                        and (len(network.LINKS_FROM_BUS[bus] + network.LINKS_TO_BUS[bus]) == 0)]

        return candidate_buses

    MAX_IT = 10

    gen_buses = network.get_gen_buses(thermals, hydros)

    buses_cannot_be_del = set(network.REF_BUS_ID) | gen_buses

    candidate_buses = _get_cand_buses(network, buses_cannot_be_del)

    it = 0

    while it < MAX_IT and len(candidate_buses) > 0:

        buses_deleted = []

        for bus in candidate_buses:
            _remove_mid_bus_with_inj(params, network, thermals, hydros,
                                                        buses_deleted,
                                                        bus)

        update_load_and_network(params, network, thermals, hydros,
                                            [network.BUS_ID.index(bus) for bus in buses_deleted],
                                                buses_deleted)

        candidate_buses = _get_cand_buses(network, buses_cannot_be_del)

        it += 1
