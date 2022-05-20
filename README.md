[![Sparhtacus](https://sparhtacus.com/wp-content/uploads/2020/12/spt-hzG.png)](https://sparhtacus.com/sobre/)

Modelo computacional open-source para unit commitment hidrotérmico em Python. 

Programa open-source distribuído sob [licença MIT](https://github.com/SPARHTACUS/SPTpy/blob/main/LICENSE.md). 

# PDDiP (P)arallel (D)ual (D)ynamic (i)nteger (P)rogramming

This is the source code of the PDDiP algorithm.

All input and output data are in .csv. We use Case 1 of our test system called 'SIN' to describe the input and output data below.

## **Input data**

**Network**: (network - SIN.csv) contains the buses and transmission lines, along with their reactances in p.u. in a 100-MW base and limits in MW

**Power plants**: (powerPlants - SIN.csv) hydro reservoirs and thermal generating units. The cascade configurations for the reservoirs are given in this file, as well as the default bounds on reservoir volumes in hm<sup>3</sup> and spillage in (m<sup>3</sup>/s). For the thermal generating units, the bounds in generation are given in MW, the ramping limits are in MW/h, the default generation costs are in `$`/(MWh/h), and the minimum up and down-times are given in hours. The cost of load curtailment is given in `$`/(MWh/h).

**Data of hydro generating units**: (dataOfGeneratingUnits - SIN.csv). In this work, we aggregate the generating units of the hydro plants and we neglect all forbidden zones. Nonetheless, data of turbine discharge, pumping rates and connection buses are taken from the data of hydro generating units given in this file.

**Start-up and shut-down trajectories of thermal generating units**: (trajectories - SIN.csv) given in MW for each step of the trajectory.

**Gross load**: (gross load - SIN - case 1.csv) contains the buses' load in MW in each period of the planning horizon

**Renewable generation**: (renewable generation - SIN - case 1.csv) renewable generation in MW at each bus of the network and period

**Inflows to reservoirs**: (inflows - SIN - case 1.csv) incremental inflows in m<sup>3</sup>/s to each reservoir in the system in all periods

**Piecewise linear approximation of the hydropower function**: (aggregated_3Dim - SIN - case 1 - HPF without binaries.csv) approximations to the hydropower functions of the hydro plants in terms of reservoir volume, plant's turbine discharge and the plant's spillage. The coefficients of turbine discharge and spillage are given in MW/(m<sup>3</sup>/s), while that of reservoir volumes are given in MW/(hm<sup>3</sup>). The constant term is given in MW.

**Cost-to-go function**: (cost-to-go function - SIN - case 1.csv) is a piecewise linear function of reservoir volumes at the end of the planning horizon that estimates the future cost of operating the system. The coefficients of the reservoir volumes are given in $/(hm<sup>3</sup>); the constant term is in $.

**Bounds on generation of groups of hydro plants**: (bounds on generation of groups of hydro plants - SIN - case 1.csv) bounds (upper, lower or equalities) for the combined generation of groups of hydro plants

**Bounds on generation of groups of thermal plants**: (bounds on generation of groups of thermal units - SIN - case 1.csv) bounds (upper, lower or equalities) for the combined generation of groups of thermal generating units

**Initial reservoir volumes**: (initial reservoir volumes - SIN - case 1.csv) reservoir volumes in (hm<sup>3</sup>) immediately before the beginning of the planning horizon

**Previous water discharges**: (previous water discharges of hydro plants - SIN - case 1.csv) total discharges of reservoirs in m<sup>3</sup>/s in periods before the beginning of the planning horizon

**Initial state of thermal generating units**: (initialStateOfThermalPowerPlants - SIN - case 1.csv) states of the thermal generating units immediately before the beginning of the planning horizon. The generation is given in MW; the unit might had been ON (1), or (OFF); the amount of time in that previous state is given in hours.

**Reset bounds on reservoir volumes**: (initialStateOfThermalPowerPlants - SIN - case 1.csv) these bounds might change over time due, for instance, to regulatory reasons or flood concerns. The new bounds are given in hm<sup>3</sup>.

**Reset generation costs**: (reset generation costs of thermal units - SIN - case 1.csv) the costs might change from case to case due to variation in fuel costs. The costs are given in $/(MWh/h).

## **Output data**

**Final results**: (final results - SIN - case 1.csv) stores the present cost, future cost (as given by the cost-to-go function), and total cost associated with the best solution found.

**Convergence**: (convergence - SIN - case 1.csv) gives the step-by-step improvements of lower and upper bound in the pddip

**Best solution found**: (bestSolutionFound - SIN - case 1.csv) the best solution found in the pddip. It is a one-dimension vector of the values of the time-coupling variables in this solution.

**Decisions for the hydro plants**: (hydro decisions - SIN - case 1.csv) reservoir volumes (hm<sup>3</sup>), turbine discharge (hm<sup>3</sup>), pumping (m<sup>3</sup>/s), water bypass (m<sup>3</sup>/s), spillage (m<sup>3</sup>/s) and generation (MW) for each reservoir in each period of the planning horizon.

**Reservoir volumes**: (reservoir volume of hydro plants - SIN - case 1.csv) decisions for the reservoir volumes (hm<sup>3</sup>) over the planning horizon.

**Turbine discharge**: (total turbine discharge of hydro plants - SIN - case 1.csv) turbine discharge in m<sup>3</sup>/s.

**Pumped water**: (pumped water - SIN - case 1.csv) pumped water in m<sup>3</sup>/s only for the reservoir with such capability.

**Water bypass**: (water transfer of hydro plants - SIN - case 1.csv) water bypass in m<sup>3</sup>/s again only for the reservoir with such capability.

**Spillage**: (spillage of hydro plants - SIN - case 1.csv) spillage in m<sup>3</sup>/s for all reservoirs and time periods

**Decisions for the thermal generating units**: (thermal decisions - SIN - case 1.csv) shows the binary decisions (dispatch status, start-up, and shut-down), generation in the dispatch phase (MW), and total generation (MW) for all thermal generating units in each period.

**Total generation of hydro plants and thermal units, and global supply-demand balance at each period**: (readableSolution - SIN - case 1.csv) summarizes the generations over the time periods.

**Network's slack variables**: (network slack variables - SIN - case 1.csv) gives the values of buses' load curtailment and generation surplus 

**Thermal units' slack variables**: (slack variables of max and min gen - SIN - case 1.csv) contains the values of the slack variables (violations) associated with the constraints on the generation of groups of thermal units.

**Reservoir volumes' slack variables**: (slack variables of reservoir volumes - SIN - case 1.csv) contains the values of violations of the coupling constraints of reservoir volumes.

**Variables - Part 1**: (variables - Part 1 - SIN - case 1.csv) stores the values of all variables (except for bus angles and transmission lines' flows).

**Variables - Part 2** (variables - Part 2 - Angles and Flows - SIN - case 1.csv) the values of all bus angles and transmission lines' flows.
