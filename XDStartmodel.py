## coding=utf-8
## --------------------------------------------------------------------------------
##  "THE BEER-WARE LICENSE" (Revision 43):
##  <lkrause@gwdg.de> wrote this file. As long as you retain this notice you
##  can do whatever you want with this stuff. If we meet some day, and you think
##  this stuff is worth it, you can buy me a beer in return.
##  last change: 28-10-2014
##
##  README:
##  This script establishes "rfree-ready" XD2006 starting models (WIGLed, high 
##  res. atom repositioning and hydrogen atoms set on Qpeak positions) input: a
##  fully converged shelxl structure model (.res!) with AFIXed and properly
##  named hydrogen atoms, no AFIX 1 is needed! 1st step: WIGL with constrained
##  hydrogen atoms, a must! This is the only way to WIGL the hydrogen atom
##  positions (step_1_WIGL)
##   - it seems that there is a bug in shelxl2014 concerning WIGL and special
##    positions, carefully check the results in these cases!
##  2nd step: high res. atom positioning, with REMed hydrogen atoms (step_2_HR)
##  3rd step: cycles of low res. Qpeak search / H atom placement (step_3_QS)
##  4th step: run XDini
##
##  REMARK ON DISORDER:
##  disordered hydrogen atoms have to follow this setup
##  part != 0 disordered part starts here
##  part == 0 disorder ends here
##  hydrogen atoms following a part != 0 will be excluded from the q-peak
##  search/assignment and are being reinserted UNCHANGED; Check their SOC and
##  VIBRATION constraints!
##  the disordered part in start.res should look like this:
##
##  ...
##  C5    1    0.416139    0.488323    0.831158    11.00000    0.02319    0.02714 =
##             0.01577    -0.00189     0.00277      0.01702
##  PART 1
##  AFIX 137
##  H5A   2    0.405072    0.569926    0.827686    10.50000   -1.50000
##  H5B   2    0.481783    0.510239    0.857216    10.50000   -1.50000
##  H5C   2    0.320583    0.413530    0.843063    10.50000   -1.50000
##  AFIX 0
##  PART 2
##  AFIX 137
##  H5D   2    0.475207    0.590649    0.833991    10.50000   -1.50000
##  H5E   2    0.437564    0.449150    0.860206    10.50000   -1.50000
##  H5F   2    0.320935    0.460023    0.831727    10.50000   -1.50000
##  AFIX 0
##  PART 0
##  ...
##
##  REMARK ON WIGL:
##  the SPEC command is needed in some cases where problems with atoms on special
##  positions occur during the WIGL step. It should not do any harm, however,
##  the WIGL_spec value can be adjusted! SPEC will be removed after the WIGL is
##  completed.
##
##  ADDITIONAL INFO:
##  if this script is run by another program the job/input-file names for XD
##  (e.g. XDJob) and XL (e.g. start) can be directly passed to this script add
##  the plain names as arguments when calling this script (no further indicators
##  are required) ".res" is assumed, enter the XLName WITHOUT the file extension!
##  usage: XDStart.py XDName XLName
##
##  -------------------------------------------------------------------------------

####################################################
##                    imports                     ##
####################################################
from collections import OrderedDict
from math import sin, cos, pi, sqrt
from os import listdir, devnull, path, remove
from re import sub, match, compile, search
from shutil import copyfile
from subprocess import call
from sys import argv, exc_info, version_info
import numpy as np


####################################################
##                                                ##
##              change values here                ##
##                                                ##
####################################################
defaults = {
    ## PATH FORMAT: "c:/bn/SXTL/xl.exe" or "c:\\bn\\SXTL\\xl.exe"
    "SHELXL_EXE":       "shelxl",    ## path to the SHELXL executable
    "XDINI_EXE":        "xdini", ## due to library dependencies the xd software package must be properly installed with all environmental variables set up

    "Reattempts":       2,       ## [   2] if the start model creation fails try (Reattempts) again

    "WIGL_val":         0.20,    ## [0.20] WIGL atom positions by -+ this value
    "WIGL_spec":        0.50,    ## [0.50] keep atoms from wigling off a special position, or move them back

    "LS_WIGL":          25,      ## [  25] number of L.S. cycles used in the WIGL step (use a large number to make sure the refinement converges!)
    "LS_HR":            10,      ## [  10] number of L.S. cycles used for the high resolution atom repositioning
    "LS_QS":            1,       ## [   1] number of L.S. cycles used for the Qpeak search

    "QS_plan_add":      0,       ## [   0] add to the PLAN starting value (number of hydrogen atoms)
    "QS_plan_frac":     0.10,    ## [0.10] increase the PLAN value by a fraction of the total number of hydrogen atoms after each cycle
    "QS_plan_frac_lim": 1.20,    ## [1.20] PLAN value to start increasing the distance threshold (X times the number of H-atoms)
    "QS_plan_frac_add": 0.10,    ## [0.10] as the PLAN value exceeds the threshold QS_plan_frac_lim it is increased by this value

    "QS_dthresh":       0.10,    ## [0.10] starting value of the distance threshold (matching of Qpeak/Hatom positions, good starting value is 0.20 when WIGL -0.20 0.20 is used!)
    "QS_dthresh_add":   0.02,    ## [0.02] as the PLAN value exceeds the threshold (QS_mult*Hatoms) the distance threshold is increased by this value
    "QS_dthresh_uplim": 0.24,    ## [0.24] upper limit for the distance before low resolution (SHEL) is increased

    "QS_shel_lower":    0.90,    ## [0.90] lower SHEL value for the Qpeak search
    "QS_shel_decr":     0.05,    ## [0.05] decrease lower SHEL by this value if PLAN and dist. thresh. get too large (triggered by QS_dthresh_uplim)
    "QS_shel_lim":      0.70,    ## [0.70] if SHEL exceeds QS_shel_lim the startmodel creation is aborted!

    "DEBUG_QS_MATCH":   False,   ## DEBUG: additional output for QS cycles (print all Qpeak - Hatom matches and distances)
    "DEBUG_QS_SKIP":    False,   ## DEBUG: additional output for QS cycles (print distances from every Hatom to every Qpeak)
    "DEBUG_TRACEBACK":  False,   ## DEBUG: error traceback
}


####################################################
##                     notes                      ##
####################################################
##error handling: raise SystemExit(0,1), 0 = continue, 1 = abort


####################################################
##                                                ##
##        don't change values below here          ##
##                                                ##
####################################################
FNULL = open(devnull, "w")

## Python version check:
if version_info[0] != 2 or version_info[1] != 7:
    print "Your python version is {}.{}.{} Please use version 2.7.x!".format(version_info[0], version_info[1], version_info[2])
    raise SystemExit(1)


####################################################
##                    classes                     ##
####################################################
class Atom(object):
    def __init__(self):
        self.name = None
        self.sfac = None
        self.sof = None
        self.Uij = None
        self.x_frac = None
        self.y_frac = None
        self.z_frac = None
        self.x_cart = None
        self.y_cart = None
        self.z_cart = None
        self.coord = None
        self.assigned = False

    def assign(self, name, sfac, x, y, z, sof, Uij):
        self.name = name
        self.sfac = sfac
        self.sof = sof
        self.Uij = Uij
        self.x_frac = x
        self.y_frac = y
        self.z_frac = z
        self.x_cart = a * x + b * cos(gamma * pi / 180) * y + c * cos(beta * pi / 180) * z
        self.y_cart = b * sin(gamma * pi / 180) * y + c * ((cos(alpha * pi / 180) - cos(beta * pi / 180) * cos(gamma * pi / 180)) / sin(gamma * pi / 180)) * z
        self.z_cart = c * (sqrt(1 - cos(alpha * pi / 180) ** 2 - cos(beta * pi / 180) ** 2 - cos(gamma * pi / 180) ** 2 + 2 * cos(alpha * pi / 180) * cos(beta * pi / 180) * cos(gamma * pi / 180)) / sin(gamma * pi / 180)) * z
        self.coord = np.array([self.x_cart, self.y_cart, self.z_cart])


def main(defaults):
    vars = defaults.copy()
    ####################################################
    ##                 file handler                   ##
    ####################################################
    resfound = False
    res_name = None
    for entry in listdir("."):
        if entry == "shelx.res":
            print "ERROR: shelx.res found, please rename your shelx.* input files to start.*"
            raise SystemExit(1)
        if entry == "shelx.ins":
            print "ERROR: shelx.ins found, skipping startmodel creation!"
            raise SystemExit(0)
        if ".res" in entry and not resfound:
            res_name = entry
        if entry == "start.res" or entry == "rfree.res":
            res_name = entry
            resfound = True
    if not res_name:
        print "ERROR: no .res file found!"
        raise SystemExit(1)


    ####################################################
    ##            parse system arguments              ##
    ####################################################
    try:
        xd_name = argv[1]
    except IndexError:
        xd_name = raw_input("xd name: ")
    try:
        res_name = argv[2] + ".res"
    except IndexError:
        res_name = raw_input("enter filename [{}]: ".format(res_name)) or res_name


    ####################################################
    ############           NOWIGL          #############
    ####################################################
    try:
        if argv[3].upper() == 'NOWIGL':
            vars['WIGL_val'] = 0.0
            print ' NOWIGL'
    except IndexError:
        pass
    ####################################################
    ####################################################
    ####################################################


    if not search("\.res$", res_name):
        res_name += ".res"
    hkl_name = sub(".res", ".hkl", res_name)
    try:
        with open(res_name) as res_file:
            start_res = res_file.readlines()
    except IOError:
        raw_input(" error: {} not found!".format(res_name))
        raise SystemExit(1)


    ####################################################
    ##    extract informations /check for disorder    ##
    ####################################################
    sfac_list = []
    atom_list = []
    disorder_dict = {}
    cell_exp = compile("^CELL\s+")
    sfac_exp = compile("^SFAC\s+")
    atom_exp = compile("\w+\d+\w*\s+\d+\s+(-*\d+\.\d+\s+){5,}")
    disorder = False
    for line in start_res:
        if sfac_exp.search(line):
            sfac_list = line.split()
        elif search("^PART\s+", line.upper()):
            if line.split()[1] != "0":
                disorder = True
            else:
                disorder = False
        elif atom_exp.search(line):
            if not disorder:
                atom_list.append(line.split())
            else:
                try:
                    disorder_dict[atom_list[-1][0]].append(line.split())
                except KeyError:
                    disorder_dict[atom_list[-1][0]] = line.split()
        elif cell_exp.search(line):
            global a, b, c, alpha, beta, gamma
            a, b, c, alpha, beta, gamma = [float(j) for j in line.split()[2:]]
    sfac_dict = {value: key for (key, value) in enumerate(sfac_list)}

    if disorder_dict.keys():
        print " ! Disorder: {}".format(" ".join(disorder_dict.keys()))


    ####################################################
    ##           fill the dicts with data             ##
    ####################################################
    hydro = OrderedDict()
    # qpeak = OrderedDict()#redundant at this point, yet useful for debugging!
    other = OrderedDict()


    ####################################################
    ## assign Atoms/Qpeaks/Hydrogen to the class atom ##
    ##     and fill the corresponding dictionnary     ##
    ####################################################
    #
    for i in list(atom_list):
        try:
            i[1] = int(i[1])
            i[2:7] = [float(j) for j in i[2:7]]
            if search("^Q\d+", i[0]):
                # ignore Qpeaks in this step!
                # qpeak[i[0]] = Atom()
                # qpeak[i[0]].assign(i[0],i[1],i[2],i[3],i[4],i[5],i[6])
                atom_list.remove(i)
            elif i[1] == sfac_dict["H"]:
                hydro[i[0]] = Atom()
                hydro[i[0]].assign(i[0], i[1], i[2], i[3], i[4], i[5], i[6])
                atom_list.remove(i)
            else:
                other[i[0]] = Atom()
                other[i[0]].assign(i[0], i[1], i[2], i[3], i[4], i[5], i[6])
                atom_list.remove(i)
        except ValueError:
            pass  # not an atom, raises exception at: i[1] = int(i[1])
        except KeyError:
            pass  # no H atoms at all, raises exception at: i[1] == sfac_dict["H"]


    ####################################################
    ##                                                ##
    ##                      WIGL                      ##
    ##                                                ##
    ####################################################
    ##              setup .ins for WIGL               ##
    ####################################################
    unwanted_wigl = ["LIST", "L.S.", "WIGL", "PLAN", "FMAP", "WGHT", "PART", "BOND", "SPEC"]
    header_wigl = "LIST 6\nL.S. {0} -1\nWIGL -{1} {1}\nSPEC {2}\nPLAN 0\nFMAP 2\nWGHT 0.0\n".format(vars["LS_WIGL"], vars["WIGL_val"], vars["WIGL_spec"])
    with open("step_1_WIGL.ins", "w") as wigl_ins:
        for line in start_res:
            if [True for i in unwanted_wigl if i in line.upper()]:
                pass
            elif "FVAR" in line:
                wigl_ins.write(header_wigl + line)
            elif match("HKLF\s+4", line):
                wigl_ins.write("HKLF 4\nEND")
                break
            else:
                wigl_ins.write(line)


    ####################################################
    ##                  start WIGL                    ##
    ####################################################
    print "\n> WIGL atom positions (step_1_WIGL)"
    copyfile(hkl_name, "step_1_WIGL.hkl")
    call([vars["SHELXL_EXE"], "step_1_WIGL"], stdout=FNULL)


    ####################################################
    ##                                                ##
    ##                atom positioning                ##
    ##                                                ##
    ####################################################
    ##      setup .ins for the atom positioning       ##
    ####################################################
    unwanted_atpos_ = ["SHEL", "LIST", "L.S.", "PLAN", "FMAP", "WGHT", "AFIX", "SPEC"]
    header_atpos = "SHEL 0.6 0.1\nLIST 6\nL.S. {} -1\nPLAN 0\nFMAP 2\n    WGHT 0.0\n".format(vars["LS_HR"])
    with open("step_1_WIGL.res") as wigl_res:
        read_wigl = wigl_res.readlines()
    with open("step_2_HR.ins", "w") as atpos_ins:
        for line in read_wigl:
            if [True for i in unwanted_atpos_ if i in line.upper()]:
                pass
            elif "FVAR" in line:
                atpos_ins.write(header_atpos + line)
            elif match("^HKLF\s+4", line):
                atpos_ins.write("HKLF 4\nEND")
                break
            else:
                atpos_ins.write(sub("^H", "REM H", line))


    ####################################################
    ##             start atom positioning             ##
    ####################################################
    print "\n> high res. atom repositioning (step_2_HR)"
    copyfile(hkl_name, "step_2_HR.hkl")
    call([vars["SHELXL_EXE"], "step_2_HR"], stdout=FNULL)


    ####################################################
    ##                                                ##
    ##                  Qpeak search                  ##
    ##                                                ##
    ####################################################
    ##                  init values                   ##
    ####################################################
    copyfile(hkl_name, "step_3_QS.hkl")
    qs_cycle    = 0
    qpeak_exp   = compile("Q\d+\w*\s+\d+\s+(-*\d+\.\d+\s+)+")  # we need to re-read the Qpeaks after each qs_cycle!
    unwanted_qs = ["SHEL", "LIST", "L.S.", "PLAN", "FMAP", "WGHT"]


    ####################################################
    ##            read new atom positions             ##
    ####################################################
    with open("step_2_HR.res") as atpos_res:
        read_atpos = atpos_res.readlines()


    ####################################################
    ##           start Qpeak search loop              ##
    ####################################################
    while True:
        qs_cycle += 1


        ####################################################
        ##        setup .ins for the Qpeak search         ##
        ####################################################
        with open("step_3_QS.ins", "w") as QS_ins:
            plan = len(hydro) + vars["QS_plan_add"]
            header_qs   = "SHEL 999 {}\nLIST 6\nL.S. {} -1\nPLAN {}\nFMAP 2\nWGHT 0.0\n".format(vars["QS_shel_lower"], vars["LS_QS"], plan)
            for line in read_atpos:
                if [True for i in unwanted_qs if i in line.upper()]:
                    pass
                elif "FVAR" in line:
                    QS_ins.write(header_qs + line + "AFIX 1\n")
                elif match("HKLF\s+4", line):
                    QS_ins.write("AFIX 0\nHKLF 4\nEND")
                    break
                else:
                    QS_ins.write(line)


        ####################################################
        ##                start SHELXL                    ##
        ####################################################
        call([vars["SHELXL_EXE"], "step_3_QS"], stdout=FNULL)


        ####################################################
        ##                 weed Qpeaks                    ##
        ####################################################
        with open("step_3_QS.res") as qs_res:
            read_qs = qs_res.readlines()

        all_qpeaks = []
        for line in read_qs:
            if qpeak_exp.search(line):
                all_qpeaks.append(line.split())

        qpeaks = OrderedDict()  # make new empty dict, dismiss all the Qpeaks from previous loop!
        for i in list(all_qpeaks):
            try:
                i[1] = int(i[1])
                i[2:7] = [float(j) for j in i[2:7]]
                if search("^Q\d+", i[0]):
                    qpeaks[i[0]] = Atom()
                    qpeaks[i[0]].assign(i[0], i[1], i[2], i[3], i[4], i[5], i[6])
                    all_qpeaks.remove(i)
            except ValueError:
                pass

        ####################################################
        ##             link Qpeaks/Hydrogens              ##
        ####################################################
        if plan > len(hydro) * vars["QS_plan_frac_lim"]:  # increase in every subsequent loop!
            vars["QS_plan_frac_lim"] += vars["QS_plan_frac_add"]
            vars["QS_dthresh"] += vars["QS_dthresh_add"]

        assigned = 0
        for h in hydro:
            hydro[h].assigned = False
            for q in qpeaks:
                dist = round(float(np.sqrt(sum(i ** 2 for i in (hydro[h].coord - qpeaks[q].coord)))), 4)
                if dist <= vars["QS_dthresh"]:  # qpeak[q].assigned == False: -> fails! max. res. affects Qpeak order
                    if vars["DEBUG_QS_MATCH"]:
                        print "  {:<5}- {:<4}: {:6.4f} <".format(h, q, dist)
                    assigned += 1
                    hydro[h].assigned = True
                    # qpeak[q].assigned = True# -> fails! changes in the max. resolution affect the Qpeak order
                    hydro[h].Xcor = qpeaks[q].x_frac
                    hydro[h].Ycor = qpeaks[q].y_frac
                    hydro[h].Zcor = qpeaks[q].z_frac
                else:
                    if vars["DEBUG_QS_SKIP"]:
                        print "  {:<5}- {:<4}: {:6.4f}".format(h, q, dist)


        ####################################################
        ##      adjust the  limits and print results      ##
        ####################################################
        print '\n> searching Qpeaks (step_3_QS): cycle {:>2}\n  assigned: {}/{}\n  missing: {}'.format(qs_cycle,assigned,len(hydro),[hydro[h].name for h in hydro if hydro[h].assigned == False])
        if assigned == len(hydro):
            print '  hydrogen atoms successfully reassigned with:\n  PLAN {}, a distance threshold of {:>4.2f} Ang.\n  and a maximum resolution of {:>4.2f} Ang.!'.format(plan, vars["QS_dthresh"], vars["QS_shel_lower"])
            global retry
            retry = vars["Reattempts"]
            break
        elif vars["QS_shel_lower"] < vars["QS_shel_lim"]:
            raise Exception("Hydrogen reassignment failed!")
        elif vars["QS_dthresh"] > vars["QS_dthresh_uplim"]:
            vars["QS_shel_lower"] -= vars["QS_shel_decr"]
            print "  Decreasing lower resolution limit to {:>4.2f} Ang!".format(vars["QS_shel_lower"])
        else:
            if int(vars["QS_plan_frac"] * plan) == 0:
                vars["QS_plan_add"] += 1
            else:
                vars["QS_plan_add"] += int(vars["QS_plan_frac"] * plan)
            print "  PLAN {} and distance threshold of {:>4.2f} Ang.!".format(plan, vars["QS_dthresh"])


    ####################################################
    ##                                                ##
    ##                   final output                 ##
    ##                                                ##
    ####################################################
    ##            prepare files for XD2006            ##
    ####################################################
    with open("shelx.ins", "w") as shelx_ins:
        unwanted_xd = ["TITL", "L.S.", "PLAN", "AFIX", "LIST", "FMAP", "WGHT", "SHEL", r"Q\d+", "BOND", "ACTA", "SIZE"]  # remove unwanted SHELXL cards
        for line in read_qs:
            try:
                if [True for i in unwanted_xd if search(i, line.upper())]:
                    continue
                elif "CELL" in line:
                    shelx_ins.write("TITL {}\n".format(xd_name) + line)
                elif match("REM H\d+", line) and hydro[line.split()[1]].assigned:
                    i = line.split()
                    shelx_ins.write("{:<5} {}   {:> 8.6f}   {:> 8.6f}   {:> 8.6f}   {:> 8.5f}  {:> 10.6f}\n".format( hydro[i[1]].name, hydro[i[1]].sfac, hydro[i[1]].Xcor, hydro[i[1]].Ycor, hydro[i[1]].Zcor, hydro[i[1]].sof, hydro[i[1]].Uij))
                elif "REM" in line or match("^\s*\n", line):
                    continue
                else:
                    shelx_ins.write(line)
            except KeyError:  # disordered Hydrogen!
                shelx_ins.write(sub("^REM ", "", line))
    copyfile(hkl_name, "shelx.hkl")

    if path.isfile("shelx.ins"):
        print "\nshelx.ins written!"
    else:
        raise Exception("ERROR writing shelx.ins!")


    ####################################################
    ##                   run XDINI                    ##
    ####################################################
    print "\n> running XDINI"
    call([vars["XDINI_EXE"], xd_name, "shelx", "scm"], stdout=FNULL)
    if path.isfile("xd.inp"):
    #    remove("xd.hkl")
        print "\nxd.mas and xd.inp written!"
    else:
        remove("shelx.ins")
        raise Exception("ERROR writing XD files!")


####################################################
##                                                ##
##                   __main__                     ##
##        manage retries and bug tracebacks       ##
##                                                ##
####################################################
retry = defaults["Reattempts"]
if __name__ == "__main__":
    while retry >= 0:
        try:
            main(defaults)
            raise SystemExit(0)
        except Exception,args:
            print "\n> {}\n> Retrying ({})!".format(args,retry)
            retry -= 1
            if defaults["DEBUG_TRACEBACK"]:
                raise
            continue
    else:
        raise SystemExit(1)
