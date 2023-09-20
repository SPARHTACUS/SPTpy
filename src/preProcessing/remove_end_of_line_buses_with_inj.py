# -*- coding: utf-8 -*-
"""
@author: Colonetti
"""

import numpy as np
from preProcessing.reduce_system import del_mid_point_buses, update_load, _del_bus, _del_line

def del_end_of_line_bus_and_reassign_inj(network, thermals, hydros, candidateBuses):
    """End-of-line buses are those connected to a single power line.
   Delete these buses and move their power injections to the neighbouring bus"""

    for bus in [bus for bus in candidateBuses if ((len(network.LINES_FROM_BUS[bus])
                                                        + len(network.LINES_TO_BUS[bus])) <= 1)]:

        # The elements connected to the bus to be deleted must be relocated to a new bus
        for l in network.LINES_FROM_BUS[bus] + network.LINES_TO_BUS[bus]:
            if not(network.LINE_F_T[l][0] == bus):
                # Remove the line from the buses connected to 'bus'
                network.LINES_FROM_BUS[network.LINE_F_T[l][0]].remove(l)
                new_bus = network.LINE_F_T[l][0]
            elif not(network.LINE_F_T[l][1] == bus):
                # Remove the line from the buses connected to 'bus'
                network.LINES_TO_BUS[network.LINE_F_T[l][1]].remove(l)
                new_bus = network.LINE_F_T[l][1]

            _del_line(network, l)

        #### Remove old bus and add elements to the new bus
        if np.max(np.abs(network.NET_LOAD[:, network.BUS_HEADER[bus]])) > 0:
            network.NET_LOAD[:, network.BUS_HEADER[new_bus]] = np.add(
                                                    network.NET_LOAD[:, network.BUS_HEADER[new_bus]],
                                                    network.NET_LOAD[:, network.BUS_HEADER[bus]])

        for h in [h for h in hydros.ID if bus in hydros.UNIT_BUS[h].values()]:
            # Change buses of units
            for u in hydros.UNIT_ID[h]:
                if hydros.UNIT_BUS[h][u] == bus:
                    hydros.UNIT_BUS[h][u] = new_bus

        for g in [g for g in thermals.ID if bus == thermals.BUS[g]]:
            thermals.BUS[g] = new_bus

        _del_bus(network, bus)

def remove_end_of_line_buses_with_inj(params, hydros, thermals, network):
    """
        End-of-line network buses with injections can be reallocated as long as its maximum
        injection is less than the line's capacity
    """

    def _get_candidate_buses(network, hydros, thermals) -> set:
        """
            get the buses connected to the system through a single line
            1 - these buses cannot be the end-point of any DC link
            2 - they cannot be reference buses
            3 - the maximum power injection at these particular buses must not exceed the capacity
            of the respective lines connecting them to the system
        """
        candidate_buses = set()
        for bus in [bus for bus in network.BUS_ID
                        if len(network.LINKS_FROM_BUS[bus] + network.LINKS_TO_BUS[bus]) == 0
                            and len(network.LINES_FROM_BUS[bus] + network.LINES_TO_BUS[bus]) <= 1]:

            min_load = min(np.min(network.NET_LOAD[network.BUS_HEADER[bus],:]), 0)
            max_load = max(np.max(network.NET_LOAD[network.BUS_HEADER[bus],:]), 0)

            for h in [h for h in hydros.ID if hydros.TURB_OR_PUMP[h] == 'Pump'
                                                                and bus in hydros.UNIT_BUS[h]]:

                max_load += (sum(hydros.UNIT_MAX_TURB_DISCH[h])*hydros.PUMP_CONVERSION_FACTOR[h])

            max_gen = (sum(thermals.MAX_P[g] for g in thermals.ID if thermals.BUS[g] == bus)
                        + sum(hydros.UNIT_MAX_P[h][u]
                                for h in hydros.ID
                                    for u in hydros.UNIT_ID[h]
                                        if hydros.UNIT_BUS[h][u] == bus)
                        )

            for l in network.LINES_FROM_BUS[bus] + network.LINES_TO_BUS[bus]:
                if (
                    (network.LINE_FLOW_UB[l] >= 99999/params.POWER_BASE)
                        or
                        (((-abs(-min_load + max_gen) >= -1*network.LINE_FLOW_UB[l])
                            and (abs(-min_load + max_gen) <= network.LINE_FLOW_UB[l]))
                        and
                            ((-abs(max_load) >= -1*network.LINE_FLOW_UB[l])
                                and (abs(max_load) <= network.LINE_FLOW_UB[l])))):

                    candidate_buses.add(bus)

        return candidate_buses

    ini_n_buses, ini_n_lines = len(network.BUS_ID), len(network.LINE_ID)

    done, it = False, 0

    candidate_buses = _get_candidate_buses(network, hydros, thermals)

    while not(done):
        it += 1

        #### Delete end-of-line buses
        del_end_of_line_bus_and_reassign_inj(network, thermals, hydros, candidate_buses)

        update_load(params, network)

        buses_no_injections = (
                            set(network.BUS_ID)
                            - ({bus for h in hydros.ID for bus in hydros.UNIT_BUS[h].values()}
                                                                    | set(thermals.BUS.values()))
                            - {network.BUS_ID[b] for b in np.where(np.abs(network.NET_LOAD) > 0)[1]}
                            - set(network.REF_BUS_ID)
                            - set([network.LINK_F_T[l][0] for l in network.LINK_F_T] +
                            [network.LINK_F_T[l][1] for l in network.LINK_F_T])
                            )

        del_mid_point_buses(params, network, buses_no_injections)

        update_load(params, network)

        candidate_buses = _get_candidate_buses(network, hydros, thermals)

        if len(candidate_buses) == 0:
            done = True

    f_n_buses, f_n_lines = len(network.BUS_ID), len(network.LINE_ID)
    n_buses_deleted, n_lines_deleted = (ini_n_buses - f_n_buses), (ini_n_lines - f_n_lines)

    print('\n\n\n')
    print(f'{it} iterations were performed')
    print(f'{n_buses_deleted} buses and {n_lines_deleted} lines were removed')
    print('\n\n\n', flush = True)
