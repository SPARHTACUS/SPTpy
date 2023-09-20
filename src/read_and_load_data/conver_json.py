"""
@author: Colonetti
"""

import json
import os

def convert_json(params, case_name, json_file_path:str,
                                    min_gen_cut_MW:float = 1,
                                        scaling_factor:float = None,
                                            deficit_cost:float = None):
    """
        convert the json file to csvs
    """

    def _get_cost_(cost:list = None):
        if len(cost) == 1:
            return cost[-1]
        return cost[-1]

    if not(os.path.isdir(params.IN_DIR)):
        os.makedirs(params.IN_DIR)

    if not(os.path.isdir(params.IN_DIR + 'case ' + str(params.CASE) + '/')):
        os.makedirs(params.IN_DIR + 'case ' + str(params.CASE) + '/')

    # Opening JSON file
    f = open(json_file_path)

    data = json.load(f) # returns JSON object as a dictionary

    f.close()
    del f

    if scaling_factor is None:
        scaling_factor = max(min(_get_cost_(gen["Production cost curve ($)"])
                                        for gen in data["Generators"].values()
                                            if _get_cost_(gen["Production cost curve ($)"]) > 0), 1)

    if deficit_cost is None:
        deficit_cost = (max(_get_cost_(gen["Production cost curve ($)"])
                            + _get_cost_(gen["Startup costs ($)"])
                                for gen in data["Generators"].values()) + 1000)/scaling_factor

    f = open(params.IN_DIR + "powerPlants - " + case_name + ".csv",
                            'w', encoding = 'ISO-8859-1')
    f.write('<BEGIN>;\n')
    f.write('<Hydro plants>;\n')
    f.write("ID;Name;Minimum reservoir volume (hm3);Maximum reservoir volume (hm3);")
    f.write("Name of downriver reservoir;Water travelling time (h);")
    f.write("Run-of-river plant? TRUE or FALSE;Minimum forebay level (m);")
    f.write("Maximum forebay level (m);")
    f.write("Maximum spillage (m3/s);Basin;Influence of spillage on the HPF? Yes or No;")
    f.write("Maximum spillage - HPF;Downriver plant of bypass discharge;")
    f.write("Maximum bypass discharge (m3/s);Water travel time in the bypass process (h);")
    f.write("Downriver reservoir of pump units;Upriver reservoir of pump units;")
    f.write("Water travel time in pumping (h);Comments\n")
    f.write('</Hydro plants>;\n')
    f.write('<Thermal plants>;\n')
    f.write("ID;Name;Minimum power output (MW);Maximum power output (MW);")
    f.write("Unitary linear cost ($/MW);Ramp-up limit (MW/h);Ramp-down limit (MW/h);")
    f.write("Minimum up-time (h);Minimum down-time (h);Bus id;Constant cost ($);Start-up cost ($);")
    f.write("Shut-down cost ($);Comments\n")

    i = 0
    for g, gen in data["Generators"].items():
        f.write(g.replace("g","") + ";")
        f.write(g + ";")
        if gen["Production cost curve (MW)"][0] < min_gen_cut_MW:
            f.write("0;")
        else:
            f.write(str(gen["Production cost curve (MW)"][0]) + ";")
        f.write(str(gen["Production cost curve (MW)"][-1]) + ";")
        f.write(str(_get_cost_(gen["Production cost curve ($)"])/scaling_factor) + ";")
        f.write(str(gen["Ramp up limit (MW)"]) + ";")
        f.write(str(gen["Ramp down limit (MW)"]) + ";")
        if (gen["Production cost curve (MW)"][0] < min_gen_cut_MW):
            f.write("0;")
            f.write("0;")
        else:
            f.write(str(gen["Minimum uptime (h)"]) + ";")
            f.write(str(gen["Minimum downtime (h)"]) + ";")
        f.write(gen["Bus"].replace("b","") + ";")
        f.write("0;")
        if (gen["Minimum uptime (h)"] <= 1 and gen["Minimum downtime (h)"] <= 1
                and gen["Production cost curve (MW)"][0] < min_gen_cut_MW):
            f.write("0;")
        else:
            f.write(str(_get_cost_(gen["Startup costs ($)"])/scaling_factor) + ";")
        f.write("0;")
        f.write("\n")
        i += 1
    f.write("</Thermal plants>\n")
    f.write("<Deficit cost>\n")
    f.write("Deficit cost  in ($/(MWh/h))\n")
    f.write(str(deficit_cost) + "\n")
    f.write("</Deficit cost>\n")

    f.write("</END>")
    f.close()
    del f

    f = open(params.IN_DIR + 'case ' + str(params.CASE) + '/'
                "initial states of thermal units - " + case_name + " - case "+params.CASE+".csv",
                'w', encoding = 'ISO-8859-1')
    f.write("<BEGIN>\n")
    f.write("<Thermal plants>\n")
    f.write("ID;Name;Generation in time t = -1 in MW;")
    f.write("State in t = -1. Either 1, if up, or 0, if down;")
    f.write("Start-up trajectory (TRUE or FALSE);Shut-down trajectory (TRUE or FALSE);")
    f.write("Number of hours (> 0) in the state of t = -1\n")
    i = 0
    for g, gen in data["Generators"].items():
        f.write(g.replace("g","") + ";")
        f.write(g + ";")
        f.write(str(gen["Initial power (MW)"]) + ";")
        if float(gen["Initial status (h)"]) <= 0:
            f.write("0;FALSE;FALSE;")
            f.write(str(-1*int(gen["Initial status (h)"])) + ";")
        else:
            f.write("1;FALSE;FALSE;")
            f.write(str(int(gen["Initial status (h)"])) + ";")
        f.write("\n")
        i += 1
    f.write("</Thermal plants>\n")
    f.write("</END>")
    f.close()
    del f

    f = open(params.IN_DIR + "network - " + case_name + ".csv", 'w', encoding = 'ISO-8859-1')
    ref_defined = False
    f.write("<BEGIN>\n")
    f.write("<Buses>\n")
    f.write("ID;Name;Reference bus;Base voltage (kV);Area;Subsystem market - Name;")
    f.write("Subsystem market - ID\n")

    for bus in data["Buses"].keys():
        f.write(bus.replace("b", "") + ";")
        f.write("Bus" + str(bus.replace("b", "")) + ";")
        if not(ref_defined):
            f.write("Ref;")
            ref_defined = True
        else:
            f.write(";")
        f.write("45;")
        f.write("1;")
        f.write("sys1;")
        f.write("1\n")
    f.write("</Buses>\n")
    f.write("<AC Transmission lines>\n")
    f.write("From (ID);From (Name);To (ID);To (Name);Line rating (MW);")
    f.write("Reactance (p.u.) - 100-MVA base\n")
    i = 0
    for l, line in data["Transmission lines"].items():
        b_from_id = int(line["Source bus"].replace("b", ""))
        b_from_name = line["Source bus"].replace("b", "Bus")
        b_to_id = int(line["Target bus"].replace("b", ""))
        b_to_name = line["Target bus"].replace("b", "Bus")
        f.write(str(b_from_id) + ';')
        f.write(str(b_from_name) + ';')
        f.write(str(b_to_id) + ';')
        f.write(str(b_to_name) + ';')
        if "Normal flow limit (MW)" in line.keys():
            f.write(str(line["Normal flow limit (MW)"]) + ';')
        else:
            f.write('99999;')
        f.write(str(1/(line["Susceptance (S)"]/100)) + ';')
        f.write("\n")
        i += 1
    f.write("</AC Transmission lines>\n")
    f.write("</END>\n")
    f.write("<DC Links>\n")
    f.write("From (ID);From (Name);To (ID);To (Name);Rating (MW)\n")
    f.write("</DC Links>")

    f.close()
    del f


    f = open(params.IN_DIR + 'case ' + str(params.CASE) + '/'+
                "gross load - " + case_name + " - case " + params.CASE + ".csv",
                                                                    'w', encoding = 'ISO-8859-1')
    f.write('<BEGIN>\n')
    f.write('Bus/Hour;')
    for t in range(36):
        f.write(str(t) + ';')

    f.write("\n")

    for b_name, load in data["Buses"].items():
        f.write("Bus" + str(b_name.replace("b", "")) + ";")
        if isinstance(load["Load (MW)"], list):
            for t in range(data["Parameters"]["Time horizon (h)"]):
                f.write(str(load["Load (MW)"][t]) + ";")
        else:
            for t in range(data["Parameters"]["Time horizon (h)"]):
                f.write(str(load["Load (MW)"]) + ";")
        f.write("\n")
    f.write("</END>")
    f.close()
    del f
