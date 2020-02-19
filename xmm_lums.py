"""
This code measures upper luminosities for different models when supplied with ra and dec coordinates and a radius.
Allows us to get an estimate of the luminosity of a source without a detection. Copyright XMM Cluster Survey
"""
import json
import os
import subprocess
import sys
import warnings
from copy import deepcopy
from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call
from subprocess import DEVNULL as DNULL

import imageio
import matplotlib.pyplot as plt
import pandas as pd
import xspec as x
from astropy import wcs
from astropy.cosmology import Planck15
from astropy.io import fits
from numpy import sqrt, linspace, meshgrid, empty, uint8, frombuffer, ndenumerate, pi, zeros
from tqdm import tqdm

warnings.simplefilter('ignore', wcs.FITSFixedWarning)


def validate_config(conf_dict):
    """
    Function to make sure the inputs from the config file are present, and the correct types.
    :param conf_dict: Dictionary from reading of configuration json.
    :return:
    """
    def model_parser(m_dict):
        # Declare a new dictionary to keep everything in the same order
        new_m_dict = {}
        for par in deepcopy(m_dict):
            if isinstance(m_dict[par], list) and par.lower() != "nh":
                if len(m_dict[par]) != 3:
                    sys.exit("Model parameters that are lists must have three elements; start, end, number of steps")
                else:
                    new_m_dict[par] = linspace(m_dict[par][0], m_dict[par][1], m_dict[par][2])

            # This just ensures nH is formatted in the way the code expects
            elif isinstance(m_dict[par], list) and par.lower() == "nh":
                if len(m_dict[par]) != 3:
                    sys.exit("Model parameters that are lists must have three elements; start, end, number of steps")
                else:
                    new_m_dict["nH"] = linspace(m_dict[par][0], m_dict[par][1], m_dict[par][2])

            elif par.lower() == "nh" and not isinstance(m_dict[par], list):
                new_m_dict["nH"] = m_dict[par]

            else:
                new_m_dict[par] = m_dict[par]

        return new_m_dict

    required_head = ["sample_csv", "generate_combined", "force_rad", "xmm_data_path", "xmm_reg_path",
                     "back_outer_factor", "id_col", "ra_col", "dec_col", "produce_plots", "models", "allowed_cores"]

    missing = False
    for el in required_head:
        if el not in conf_dict.keys():
            print("You are missing the {arg} argument!".format(arg=el))
            missing = True
    if missing:
        sys.exit(1)

    if os.path.exists(conf_dict['sample_csv']):
        conf_dict["samp_path"] = os.path.abspath("/".join(conf_dict["sample_csv"].split("/")[:-1])) + "/"
    else:
        sys.exit('That sample csv does not exist!')

    if not isinstance(conf_dict["generate_combined"], bool):
        sys.exit('generate_combined should be of type bool')

    if not isinstance(conf_dict["force_rad"], bool):
        sys.exit("force_rad should be of type bool")

    if not isinstance(conf_dict["xmm_data_path"], str):
        sys.exit("xmm_data_path should be of type int!")
    elif not os.path.exists(conf_dict["xmm_data_path"]):
        sys.exit("That xmm_data_path doesn't exist!")
    elif conf_dict["xmm_data_path"][-1] != "/":
        conf_dict["xmm_data_path"] += "/"
    if "sftp" in conf_dict["xmm_data_path"]:
        if os.path.exists("xmm_data"):
            os.remove("xmm_data")
        subprocess.call("ln -s {} xmm_data".format(conf_dict["xmm_data_path"]), shell=True)
        conf_dict["xmm_data_path"] = os.path.abspath("xmm_data") + "/"

    if not isinstance(conf_dict["xmm_reg_path"], str):
        sys.exit("xmm_reg_path should be of type int!")
    elif not os.path.exists(conf_dict["xmm_reg_path"]):
        sys.exit("That xmm_reg_path doesn't exist!")
    elif conf_dict["xmm_reg_path"][-1] != "/":
        conf_dict["xmm_reg_path"] += "/"
    if "sftp" in conf_dict["xmm_reg_path"]:
        if os.path.exists("xmm_reg"):
            os.remove("xmm_reg")
        subprocess.call("ln -s {} xmm_reg".format(conf_dict["xmm_reg_path"]), shell=True)
        conf_dict["xmm_reg_path"] = os.path.abspath("xmm_reg") + "/"

    if not isinstance(conf_dict["back_outer_factor"], (float, int)):
        sys.exit("back_outer_factor should be either a float or an integer")

    if not isinstance(conf_dict["id_col"], str):
        sys.exit("id_col must be a string")

    if not isinstance(conf_dict["ra_col"], str):
        sys.exit("ra_col must be a string")

    if not isinstance(conf_dict["dec_col"], str):
        sys.exit("dec_col must be a string")

    if not isinstance(conf_dict["produce_plots"], bool):
        sys.exit("produce_plots must be boolean!")

    if not isinstance(conf_dict["models"], dict):
        sys.exit("models must be supplied as a dictionary, even if you are only using one")
    else:
        print("IF YOU WANT A MODEL PARAMETER TO BE READ FROM THE SAMPLE, PUT IT AS -1")
        print("PARAMETERS MUST BE IN THE ORDER THAT XSPEC EXPECTS THEM")
        for entry in conf_dict["models"]:
            conf_dict["models"][entry] = model_parser(conf_dict["models"][entry])

    if not isinstance(conf_dict["allowed_cores"], int):
        sys.exit("allowed_cores should be an integer value.")

    return conf_dict


def command_stack_maker(conf_dict, obj_id, obs_id, file_locs, src_r, excl_r, save_dir, ins, obj_type, for_comb=False):
    centre = src_r[:2]
    radius = src_reg[2]

    if obj_type == "ext":
        ext = "yes"
    elif obj_type == "pnt":
        ext = "no"
    else:
        sys.exit("Object type isn't recognised, use ext or pnt")

    cmds = []
    # Creates a temporary directory inside the destination - as multiple SAS processes in parallel can interfere with
    # one another's temporary files
    cmd = "mkdir {dest}/{o}_{ins}_temp".format(dest=save_dir, ins=ins, o=obj_id)
    cmds.append(cmd)

    # Changes directory to the location above the temporary folder
    cmd = "cd {dest};export SAS_ODF={odf_path}".format(dest=save_dir, odf_path=file_locs["odf"])
    cmds.append(cmd)

    # If we're running on Kraken, for instance, with a newer version of SAS then we need new CCF files
    if update_ccf:
        cmd = 'if [ ! -e "ccf.cif" ]; then cifbuild calindexset="ccf.cif"; fi'
        cmds.append(cmd)
        cmd = "cp ccf.cif {o}_{ins}_temp/".format(ins=ins, o=obj_id)
        cmds.append(cmd)
        cmd = "cd {o}_{ins}_temp".format(ins=ins, o=obj_id)
        cmds.append(cmd)
        # Standard stop, SAS needs calibration files for some things like ARFGEN
        cmd = "export SAS_CCF='ccf.cif'"
        cmds.append(cmd)

    else:
        cmd = "cd {o}_{ins}_temp".format(ins=ins, o=obj_id)
        cmds.append(cmd)
        # Standard stop, SAS needs calibration files for some things like ARFGEN
        cmd = "export SAS_CCF={ccf}".format(ccf=file_locs["ccf"])
        cmds.append(cmd)

    # Construct spatial regions to NOT include (to remove sources)
    reg_expr = "((X,Y) IN circle({cenx},{ceny}, {rad}))"
    excl_expr = " &&! ".join([reg_expr.format(cenx=entry[0], ceny=entry[1], rad=entry[2]) for entry in excl_r])
    if excl_expr != "":
        excl_expr = " &&! " + excl_expr

    evt_path = file_locs["e" + ins.lower() + "_evts"]

    expression = "expression='#XMMEA_{iflag} && (FLAG .eq. 0) && ((X,Y) IN circle({cenx},{ceny}, {rad})){e_expr}'" \
        .format(iflag="E" + ins[0].upper(), cenx=centre[0], ceny=centre[1], rad=radius, e_expr=excl_expr)

    back_expression = "expression='#XMMEA_{iflag} && (FLAG .eq. 0) && ((X,Y) IN annulus({cenx},{ceny}, {in_rad}, " \
                      "{out_rad})){e_expr}'".format(iflag="E" + ins[0].upper(), cenx=centre[0], ceny=centre[1],
                                                    in_rad=radius*1.05, out_rad=radius*conf_dict["back_outer_factor"],
                                                    e_expr=excl_expr)

    all_but_expression = "expression='#XMMEA_{iflag} && " \
                         "(FLAG .eq. 0){e_expr}'".format(iflag="E" + ins[0].upper(),  cenx=centre[0], ceny=centre[1],
                                                         in_rad=radius*1.05, e_expr=excl_expr,
                                                         out_rad=radius*conf_dict["back_outer_factor"])

    # Defining paths for all products, also allows to check they haven't already been generated.
    spec_path = obj_id + "_" + obs_id + '_' + ins + '_spec.fits'
    rmf_path = obj_id + "_" + obs_id + '_' + ins + '.rmf'
    arf_path = obj_id + "_" + obs_id + '_' + ins + '.arf'
    back_spec_path = obj_id + "_" + obs_id + "_" + ins + "_back_spec.fits"
    grp_path = obj_id + "_" + obs_id + '_' + ins + '_grp.fits'
    cutout_reg_path = obj_id + "_" + obs_id + '_' + ins + '_cutouts.fits'

    # Constructs all commands related to spectrum generation.
    if not os.path.exists(save_dir + "/" + grp_path):
        # Different channel limits for EPN and EMOS instruments
        if ins.lower() == "pn":
            if for_comb:
                spec_lim = 11999
                added_term = " acceptchanrange=yes"
            else:
                spec_lim = 20479
                added_term = ""
        else:
            spec_lim = 11999
            added_term = ""

        cmd = "evselect table={evts} withspectrumset=yes spectrumset={spfile} energycolumn=PI spectralbinsize=5 " \
              "withspecranges=yes specchannelmin=0 specchannelmax={ulim} {exp}"\
            .format(evts=evt_path, spfile=spec_path, ulim=spec_lim, exp=expression)
        cmds.append(cmd)

        if not for_comb:
            cmd = "rmfgen rmfset={rmfile} spectrumset={spfile} detmaptype=flat extendedsource={ty}"\
                .format(rmfile=rmf_path, spfile=spec_path, ty=ext)
        else:
            # This is for the sake of epicspeccombine, RMF for all three instruments need the same binning
            cmd = "rmfgen rmfset={rmfile} spectrumset={spfile} detmaptype=flat extendedsource={ty} " \
                  "withenergybins=yes energymin=0.1 energymax=12.0 nenergybins=2400{add}".format(rmfile=rmf_path,
                                                                                                 spfile=spec_path,
                                                                                                 add=added_term, ty=ext)
        cmds.append(cmd)

        cmd = "arfgen spectrumset={spfile} arfset={arfile} withrmfset=yes rmfset={rmfile} badpixlocation={evts} " \
              "extendedsource={ty} detmaptype=flat setbackscale=yes".format(evts=evt_path, spfile=spec_path,
                                                                            rmfile=rmf_path, arfile=arf_path, ty=ext)
        cmds.append(cmd)

        # For the background
        cmd = "evselect table={evts} withspectrumset=yes spectrumset={spfile} energycolumn=PI spectralbinsize=5 " \
              "withspecranges=yes specchannelmin=0 specchannelmax={ulim} {exp}".format(evts=evt_path,
                                                                                       spfile=back_spec_path,
                                                                                       ulim=spec_lim,
                                                                                       exp=back_expression)
        cmds.append(cmd)

        cmd = "evselect table={evts} imageset={img} xcolumn=X ycolumn=Y ximagebinsize=87 " \
              "yimagebinsize=87 squarepixels=yes ximagesize=512 yimagesize=512 imagebinning=binSize " \
              "ximagemin=3649 ximagemax=48106 withxranges=yes yimagemin=3649 yimagemax=48106 " \
              "withyranges=yes {exp}" \
            .format(evts=evt_path, img=cutout_reg_path, exp=all_but_expression)
        cmds.append(cmd)

        cmd = "specgroup groupedset={grpfile} spectrumset={spfile} arfset={arfile} rmfset={rmfile} minSN=1 " \
              "oversample=3 backgndset={bspec}".format(grpfile=grp_path, spfile=spec_path, arfile=arf_path,
                                                       rmfile=rmf_path, bspec=back_spec_path)
        cmds.append(cmd)

        # Delete ccf.cif and then move everything up to the main directory for this ObsID
        cmd = "rm ccf.cif;mv * ../"
        cmds.append(cmd)

        # Change directory to the main directory
        cmd = "cd ../"
        cmds.append(cmd)

        # Remove the temporary folder
        cmd = "rm -r {o}_{ins}_temp".format(ins=ins, o=obj_id)
        cmds.append(cmd)

        return ';'.join(cmds)
    return ""


def coords_rad_regions(conf_dict, obs, im_path, ra, dec, rad, force_user_rad=False):
    def reg_extract(reg_string):
        reg_string = reg_string.split("(")[-1].split(")")[0]
        reg_x, reg_y, reg_x_rad, reg_y_rad, reg_ang = [float(el) for el in reg_string.split(",")]
        # -1 because XAPA starts counting at 1, not 0 like Python
        return reg_x-1, reg_y-1, reg_x_rad, reg_y_rad, reg_ang

    def calc_sep(reg_x, reg_y):
        return sqrt((reg_x - cen_pix[0])**2 + (reg_y - cen_pix[1])**2)

    def pix_to_sky(pix_x, pix_y, pix_r):
        edge_sky = pix_sky_wcs.all_pix2world(pix_x+pix_r, pix_y, 0)
        cen_sky = pix_sky_wcs.all_pix2world(pix_x, pix_y, 0)
        sky_x = float(cen_sky[0])
        sky_y = float(cen_sky[1])
        sky_r = abs(edge_sky[0] - cen_sky[0])
        return sky_x, sky_y, sky_r

    # Loads in an XCS generated image to help with the WCS coordinates transformations
    im_fits = fits.open(im_path)
    im_head = im_fits[0].header
    # Reads in WCS for ra,dec->pixel and pixel->skycoords
    deg_pix_wcs = wcs.WCS(im_head)
    pix_sky_wcs = wcs.WCS(im_head, key='L')
    im_fits.close()
    # Convert passed ra and dec to image pixel coordinates
    cen_pix = deg_pix_wcs.all_world2pix(ra, dec, 0)

    # Have to search XAPA regions for any sources that need to be removed from background spectrum generation
    with open(conf_dict["xmm_reg_path"] + obs + "/final_class_regions_REDO.reg") as reggy:
        reg_lines = reggy.readlines()
    if "global" in reg_lines[0]:
        reg_lines = reg_lines[1:]

    reg_summary = [reg_extract(line) for line in reg_lines]
    separations = [calc_sep(reg[0], reg[1]) for reg in reg_summary]

    poss_source_reg = []
    i = separations.index(min(separations))
    # Identifies possible source region, but only if its a point source
    if separations[i] < reg_summary[i][2] == reg_summary[i][3]:
        poss_source_reg.append(i)

    source_reg = None
    if len(poss_source_reg) == 1 and not force_user_rad:
        source_reg = reg_summary.pop(poss_source_reg[0])
        separations.pop(poss_source_reg[0])
    elif len(poss_source_reg) == 0:
        onwards.write("No XAPA region, closest is {0} pixels away, "
                      "setting radius to {1} arcsec.".format(round(min(separations), 2), rad))
    elif len(poss_source_reg) == 1 and force_user_rad:
        onwards.write("Forcing the radius to {} arcseconds".format(rad))
        separations.pop(poss_source_reg[0])
        source_reg = None

    # TODO Make this do extended sources properly, so they can be properly subtracted (though it would probs be a
    #  rare circumstance

    if source_reg is None:
        # If no matching XAPA region was found, we use the values passed into the function
        # Converts arcseconds to degrees
        rad /= 3600
        edge_pix = deg_pix_wcs.all_world2pix(ra + rad, dec, 0)
        pix_rad = abs(cen_pix[0] - edge_pix[0])
    else:
        # If a matchin XAPA region WAS found, the central point and radius are used instead
        cen_pix = [source_reg[0], source_reg[1]]
        pix_rad = source_reg[2]
    source_sky = pix_to_sky(*cen_pix, pix_rad)

    # Now going to find the sources within some arbitrary large radius (50 pixels?), just to make sure they're excluded
    sources_within_lim = [i for i, sep in enumerate(separations) if sep < pix_rad+50]
    exclude_sky = [pix_to_sky(*reg_summary[i][:3]) for i in sources_within_lim]

    return source_sky, exclude_sky


def run_sas(cmd, pass_shell, pass_stdout, pass_stderr, conf_dict):
    sp = conf_dict["samp_path"]
    try:
        if "epicspeccombine" not in cmd:
            group_cmd = [el for el in cmd.split(";") if "group" in el][0]
            spec_file = group_cmd.split("spectrumset=")[-1].split(" ")[0]
            back_spec_file = group_cmd.split("backgndset=")[-1].split(" ")[0]
            arf_file = group_cmd.split("arfset=")[-1].split(" ")[0]
            rmf_file = group_cmd.split("rmfset=")[-1].split(" ")[0]
        else:
            combine_cmd = [el for el in cmd.split(";") if "combine" in el][0]
            spec_file = combine_cmd.split('filepha="')[-1].split('"')[0]
            back_spec_file = combine_cmd.split('filebkg="')[-1].split('"')[0]
            resp_file = combine_cmd.split('filersp="')[-1].split('"')[0]
        obs_id = spec_file.split("_")[1]
        obj_id = spec_file.split("_")[0]
        ins = spec_file.split("_")[2]

        # call(cmd, shell=pass_shell, stdout=pass_stdout, stderr=pass_stderr)
        with open(sp + "xmm_spectra/{o}/{oi}_{i}_sasgen.log".format(o=obs_id, oi=obj_id, i=ins), 'w') as loggy:
            call(cmd, shell=pass_shell, stderr=loggy, stdout=loggy)

        try:
            spec = fits.open(conf_dict["samp_path"] + "xmm_spectra/{o}/".format(o=obs_id) + spec_file, mode="update")
            spec[1].header["BACKFILE"] = sp + "xmm_spectra/{o}/".format(o=obs_id) + back_spec_file
            if "epicspeccombine" not in cmd:
                spec[1].header["ANCRFILE"] = sp + "xmm_spectra/{o}/".format(o=obs_id) + arf_file
                spec[1].header["RESPFILE"] = sp + "xmm_spectra/{o}/".format(o=obs_id) + rmf_file
            else:
                spec[1].header["RESPFILE"] = sp + "xmm_spectra/{o}/".format(o=obs_id) + resp_file
                spec.flush()
            spec.flush()
        except FileNotFoundError:
            tqdm.write("{} does not exist!".format(spec_file))

    except IndexError:
        call(cmd, shell=pass_shell, stderr=pass_stderr, stdout=pass_stdout)


def combine_stack_maker(obj_id, obs_id, evt_files, save_dir):
    cmds = []
    # Combined file path definition
    comb_spec_path = obj_id + "_" + obs_id + '_spec_comb.fits'
    comb_back_spec_path = obj_id + "_" + obs_id + "_back_spec_comb.fits"
    comb_resp_path = obj_id + "_" + obs_id + "_resp_comb.fits"

    if not os.path.exists(save_dir + "/" + comb_spec_path):
        # Changes directory to the temporary folder
        cmd = "cd {dest}".format(dest=save_dir)
        cmds.append(cmd)

        # Standard stop, SAS needs calibration files for some things like ARFGEN
        cmd = "export SAS_CCF={ccf}".format(ccf=evt_files["ccf"])
        cmds.append(cmd)

        spec_list = []
        back_list = []
        arf_list = []
        rmf_list = []
        for ins in ["PN", "MOS1", "MOS2"]:
            spec_list.append(obj_id + "_" + obs_id + '_' + ins + '_spec.fits')
            back_list.append(obj_id + "_" + obs_id + "_" + ins + "_back_spec.fits")
            rmf_list.append(obj_id + "_" + obs_id + '_' + ins + '.rmf')
            arf_list.append(obj_id + "_" + obs_id + '_' + ins + '.arf')

        cmd = 'epicspeccombine pha="{sl}" bkg="{bl}" rmf="{rl}" arf="{al}" filepha="{so}" filebkg="{bo}" ' \
              'filersp="{ro}"'.format(sl=" ".join(spec_list), bl=" ".join(back_list), rl=" ".join(rmf_list),
                                      al=" ".join(arf_list), so=comb_spec_path, bo=comb_back_spec_path,
                                      ro=comb_resp_path)
        cmds.append(cmd)
        return ";".join(cmds)
    return ""


def sas_pool(stack, desc, c_dict, cpu_count=2):
    stack = [el for el in stack if el != ""]
    if len(stack) > 0:
        with Pool(cpu_count) as pool:
            r = list(tqdm(pool.imap_unordered(partial(run_sas, pass_shell=True, pass_stdout=DNULL, pass_stderr=DNULL,
                                                      conf_dict=c_dict), stack), total=len(stack), desc=desc))


def nh_lookup(ra, dec):
    """
    Uses HEASOFT to lookup hydrogen column density for given coordinates.
    :param ra:
    :param dec:
    :return:
    """
    heasoft_cmd = 'nh 2000 {ra} {dec}'.format(ra=ra, dec=dec)
    proc = subprocess.Popen(heasoft_cmd, stdout=subprocess.PIPE, shell=True)
    output = proc.stdout.read().decode("utf-8")
    lines = output.split('\n')
    value = lines[-3].split(' ')[-1]

    return float(value)


def ecfs_calc(parameters, obj_id, obs_id, save_dir, ins):
    def check_response_status(rsp_file):
        if not os.path.exists(rsp_file):
            use_it = False
        elif os.path.exists(rsp_file) and "rmf" in rsp_file:
            use_it = True
        elif os.path.exists(rsp_file) and "arf" in rsp_file:
            arf_opened = fits.open(rsp_file)
            if arf_opened[1].data["SPECRESP"].sum() == 0:
                use_it = False
            else:
                use_it = True
            arf_opened.close()
        else:
            sys.exit("How did you get here??")

        return use_it

    os.chdir(save_dir)
    fake_it_exp = 500000  # In seconds
    for_plotting = {}

    # Runs through the passed models and their parameters. If parameters are lists, all combinations will be constructed
    model_dfs = {}
    for m in parameters:
        for_plotting[m] = {"energies": [], "rates": [], "folded_model": []}

        model_pars = parameters[m]
        if "redshift" in model_pars and not isinstance(model_pars["redshift"], (float, int)):
            sys.exit("Ranges of redshift are not allowed!")

        # Calculates the number of rows in the parameter space for this model/parameter selection
        num_rows = 1
        for par in model_pars:
            if not isinstance(model_pars[par], (float, int)):
                num_rows *= len(model_pars[par])

        # Empty parameter space numpy array
        p_space = empty((num_rows, len(model_pars.keys())), dtype="float32")
        for mesh_ind, mesh_col in enumerate(meshgrid(*model_pars.values())):
            p_space[:, mesh_ind] = mesh_col.flatten()

        # Convenient dataframe with the parameter space stored in a dictionary.
        model_dfs[m] = pd.DataFrame(columns=list(parameters[m].keys()), data=p_space)

    rmf_path = obj_id + "_" + obs_id + '_' + ins + '.rmf'
    run_inst_rmf = check_response_status(rmf_path)
    arf_path = obj_id + "_" + obs_id + '_' + ins + '.arf'
    run_inst_arf = check_response_status(arf_path)

    for m_df in model_dfs:
        if not os.path.exists("{0}_{1}_{2}_ecfs.csv".format(obj_id, model, ins)) and (run_inst_rmf and run_inst_arf):
            current_df = model_dfs[m_df].copy()
            current_df = current_df.assign(ph_per_sec_lowen=-1, flux_lowen=-1, ph_per_sec_highen=-1, flux_highen=-1,
                                           ph_per_sec_wideen=-1, flux_wideen=-1)
            for par_ind, par_comb in model_dfs[m_df].iterrows():
                x_mod = x.Model(m_df)
                x_mod.setPars(*par_comb.astype(float).values)

                fake_settings = x.FakeitSettings(arf=arf_path, response=rmf_path, exposure=fake_it_exp,
                                                 fileName="temp.fak")
                x.AllData.fakeit(1, fake_settings)
                fake_spec = x.AllData(1)
                # pyXSPEC insists on writing a spec file for fakeit, so I immediately delete it.
                os.remove(fake_settings.fileName)

                fake_spec.ignore("**-0.5 10.0-**")
                x.Fit.perform()

                x.Plot.device = '/null'
                x.Plot.xAxis = "keV"
                x.Plot('data')
                for_plotting[m_df]["energies"].append(x.Plot.x())
                for_plotting[m_df]["rates"].append(x.Plot.y())
                for_plotting[m_df]["folded_model"].append(x.Plot.model())

                # x_mod.setPars(0)
                # x.AllModels.calcFlux("0.5 2.0 err")
                x.AllModels.calcFlux("0.5 2.0")
                lowen_flux = fake_spec.flux[0]
                # x.AllModels.calcFlux("2.0 10.0 err")
                x.AllModels.calcFlux("2.0 10.0")
                highen_flux = fake_spec.flux[0]

                # Have to use an ignore to get a count rate for the energy range I care about
                fake_spec.ignore("**-0.5 2.0-**")
                # the 0th element of rate is the background subtracted rate, but doesn't matter -> no background!
                current_df.loc[par_ind, "ph_per_sec_lowen"] = fake_spec.rate[0]
                current_df.loc[par_ind, "flux_lowen"] = lowen_flux

                # Now reset the ignore to ignore nothing
                fake_spec.notice("all")
                # And now ignore to get the high energy range
                fake_spec.ignore("**-2.0 10.0-**")
                current_df.loc[par_ind, "ph_per_sec_highen"] = fake_spec.rate[0]
                current_df.loc[par_ind, "flux_highen"] = highen_flux

                # Now reset the ignore to ignore nothing
                fake_spec.notice("all")
                # And now ignore to get the energy range used in the paper
                fake_spec.ignore("**-0.5 8.0-**")
                current_df.loc[par_ind, "ph_per_sec_wideen"] = fake_spec.rate[0]
                current_df.loc[par_ind, "flux_wideen"] = highen_flux

                x.AllData.clear()
                x.AllModels.clear()
            model_dfs[m_df] = current_df
        elif not run_inst_rmf and not run_inst_arf:
            model_dfs[m_df] = None
            for_plotting[m_df] = None
        else:
            try:
                model_dfs[m_df] = pd.read_csv("{0}_{1}_{2}_ecfs.csv".format(obj_id, model, ins), header="infer")
            except FileNotFoundError:
                model_dfs[m_df] = None
            for_plotting[m_df] = None

    return for_plotting, model_dfs


def fake_spec_plots(for_plotting, save_dir, obj_id):
    inst_list = list(for_plotting.keys())
    model_list = list(for_plotting[inst_list[0]].keys())
    split_models = {mod: {ins: {"energies": [], "rates": [], "folded_model": []} for ins in inst_list} for mod in
                    model_list}

    # This data wrangling bit is horrible, I probably should have just iterated instrument inside of ecfs_calc, but
    # here we are
    for ins in for_plotting:
        for m in model_list:
            split_models[m][ins]["energies"] = for_plotting[ins][m]["energies"]
            split_models[m][ins]["rates"] = for_plotting[ins][m]["rates"]
            split_models[m][ins]["folded_model"] = for_plotting[ins][m]["folded_model"]

    # Takes us to the place where the results csv is saved, this does add a lot of reading from disk, but the files and
    # sample sizes will be so small I don't care about the penalty
    os.chdir(save_dir)

    # This sets up the pyplot stuff for factor plots of all the models
    factor_fig, ax_arr = plt.subplots(1, len(model_list), figsize=(len(model_list)*10.5, 10))

    for m, subplot in ndenumerate(ax_arr):
        subplot.minorticks_on()
        subplot.tick_params(axis='both', direction='in', which='both', top=True, right=True)
        subplot.grid(linestyle='dotted', linewidth=1)
        subplot.axis(option="tight")
    factor_fig.suptitle("Conversion Factors", fontsize=14, y=0.95, x=0.51)

    # Now we start plotting
    for m_ind, m in enumerate(model_list):
        images = []
        pn_output_df = pd.read_csv("{0}_{1}_{2}_ecfs.csv".format(obj_id, m, "PN"), header="infer")
        mos2_output_df = pd.read_csv("{0}_{1}_{2}_ecfs.csv".format(obj_id, m, "MOS2"), header="infer")

        # Makes plots of the conversion factor for PN and MOS1 - reading in which isn't great but oh well
        # First need to calculate the factors
        pn_output_df["factor_lowen"] = pn_output_df["flux_lowen"] / pn_output_df["ph_per_sec_lowen"]
        pn_output_df["factor_highen"] = pn_output_df["flux_highen"] / pn_output_df["ph_per_sec_highen"]

        mos2_output_df["factor_lowen"] = mos2_output_df["flux_lowen"] / mos2_output_df["ph_per_sec_lowen"]
        mos2_output_df["factor_highen"] = mos2_output_df["flux_highen"] / mos2_output_df["ph_per_sec_highen"]

        maximum = max(mos2_output_df["factor_lowen"].max(), mos2_output_df["factor_highen"].max(),
                      pn_output_df["factor_lowen"].max(), pn_output_df["factor_highen"].max()) * 1.2
        minimum = min(mos2_output_df["factor_lowen"].min(), mos2_output_df["factor_highen"].min(),
                      pn_output_df["factor_lowen"].min(), pn_output_df["factor_highen"].min()) * 0.8

        try:
            factor_ax = ax_arr[m_ind]
        except TypeError:
            factor_ax = ax_arr

        factor_ax.plot([minimum, maximum], [minimum, maximum], color="red", linestyle="dashed", label="One to One")
        factor_ax.plot(mos2_output_df["factor_lowen"], pn_output_df["factor_lowen"], "+", label="0.50-2.00keV")
        factor_ax.plot(mos2_output_df["factor_highen"], pn_output_df["factor_highen"], "x", label="2.00-10.00keV")
        factor_ax.legend(loc="best")

        factor_ax.set_xlabel("MOS2 Conversion Factor")
        factor_ax.set_ylabel("PN Conversion Factor")
        factor_ax.set_xlim(minimum, maximum)
        factor_ax.set_ylim(minimum, maximum)
        factor_ax.set_title(m)

        p_space = pn_output_df.drop(["ph_per_sec_lowen", "flux_lowen", "ph_per_sec_highen", "flux_highen",
                                     "factor_lowen", "factor_highen", "flux_wideen", "ph_per_sec_wideen",
                                     "pred_lowen_flux", "pred_highen_flux", "pred_lowen_lum", "pred_highen_lum"],
                                    axis=1)
        p_cols = p_space.columns.values
        # Finds maximum value of PN fake spectrum fitted model, then makes it 10% larger, for plot limit
        max_y = max([max(entry) for entry in split_models[m]["PN"]["folded_model"]]) * 1.1
        for i in range(0, len(split_models[m]["PN"]["energies"])):
            fig = plt.figure(99, figsize=(8, 6))
            ax = plt.gca()
            ax.minorticks_on()
            ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)
            ax.grid(linestyle='dotted', linewidth=1)
            ax.axis(option="tight")
            ax.set_xlabel("Energy (keV)", fontsize=10)
            ax.set_ylabel(r"Normalised Counts s$^{-1} $keV$^{-1}$", fontsize=10)
            ax.set_ylim(0.001, max_y)
            ax.set_yscale("log")
            t_str = "{} Fake Spectrum - ".format(m) + " ".join(["{0}={1}".format(c, '{:.2f}'.format(
                round(p_space.loc[i, c], 2))) for c in p_cols])
            ax.set_title(t_str, fontsize=10)

            for ins in for_plotting:
                plt.plot(split_models[m][ins]["energies"][i], split_models[m][ins]["folded_model"][i], label=ins)

            plt.legend(loc="upper right")
            # To convert the figure to a numpy array and allow it to be saved to a gif by mimsave
            fig.canvas.draw()
            image_from_plot = frombuffer(fig.canvas.tostring_rgb(), dtype=uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(image_from_plot)
            plt.close(99)

        imageio.mimsave("{0}_{1}.gif".format(obj_id, m), images, fps=8)

    factor_fig.savefig("{}_factors.png".format(obj_id), bbox_inches='tight')
    plt.close("all")


def photon_rate_command(conf_dict, file_locs, obj_id, src_region, excl_regions, save_dir, ins, en, obj_type):
    out_name = "{0}_{1}_{2}_output.txt".format(obj_id, "e" + ins.lower(), en)
    cmds = []

    if obj_type == "ext":
        psf_model = "EXTENDED"
    elif obj_type == "pnt":
        # TODO Ask Paul which PSF models I should be using
        psf_model = "ELLBETA"
    else:
        psf_model = None
        sys.exit("How on earth did you get here?")

    if not os.path.exists(save_dir + "/" + out_name):
        centre = src_region[:2]
        radius = src_reg[2]

        # Creates a temporary directory inside the destination - multiple SAS processes in parallel can interfere with
        # one another's temporary files
        cmd = "mkdir {dest}/{o}_{i}_{e}_temp".format(dest=save_dir, o=obj_id, i=ins, e=en)
        cmds.append(cmd)

        # Changes directory to the temporary folder
        # Changes directory to the location above the temporary folder
        cmd = "cd {dest};export SAS_ODF={odf_path}".format(dest=save_dir, odf_path=file_locs["odf"])
        cmds.append(cmd)

        # If we're running on Kraken, for instance, with a newer version of SAS then we need new CCF files
        if update_ccf:
            cmd = 'if [ ! -e "ccf.cif" ]; then cifbuild calindexset="ccf.cif"; fi'
            cmds.append(cmd)
            cmd = "cp ccf.cif {o}_{ins}_{e}_temp/".format(ins=ins, o=obj_id, e=en)
            cmds.append(cmd)
            cmd = "cd {o}_{ins}_{e}_temp/".format(ins=ins, o=obj_id, e=en)
            cmds.append(cmd)
            # Standard stop, SAS needs calibration files for some things like ARFGEN
            cmd = "export SAS_CCF='ccf.cif'"
            cmds.append(cmd)

        else:
            cmd = "cd {o}_{ins}_{e}_temp/".format(ins=ins, o=obj_id, e=en)
            cmds.append(cmd)
            # Standard stop, SAS needs calibration files for some things like ARFGEN
            cmd = "export SAS_CCF={ccf}".format(ccf=file_locs["ccf"])
            cmds.append(cmd)

        # Construct spatial regions to NOT include (to remove sources)
        cir_reg = "((X,Y) IN circle({cenx},{ceny}, {rad}))"
        src_expr = cir_reg.format(cenx=centre[0], ceny=centre[1], rad=radius)
        bck_expr = "((X,Y) IN annulus({cenx},{ceny}, " \
                   "{in_rad}, {out_rad}))".format(cenx=centre[0], ceny=centre[1],
                                                  out_rad=radius * conf_dict["back_outer_factor"], in_rad=radius*1.05)
        excl_expr = " &&! ".join([cir_reg.format(cenx=ent[0], ceny=ent[1], rad=ent[2]) for ent in excl_regions])
        if excl_expr != "":
            excl_expr = " &&! " + excl_expr

        expression = "'{s_expr}{e_expr}'".format(e_expr=excl_expr, s_expr=src_expr)
        b_expression = "'{b_expr}{e_expr}'".format(b_expr=bck_expr, e_expr=excl_expr)

        im_name = "{0}_{1}_im".format("e"+ins.lower(), en)
        exp_name = "{0}_{1}_expmap".format("e"+ins.lower(), en)
        # ulsig 0.997 is 3 sigma
        cmd = "eregionanalyse imageset='{mi}' exposuremap='{me}' srcexp={e} backexp={be} ulsig=0.997 " \
              "withoutputfile=yes output={o_name} psfmodel={psf}".format(mi=file_locs[im_name], be=b_expression,
                                                                         me=file_locs[exp_name], e=expression,
                                                                         o_name=out_name, psf=psf_model)
        cmds.append(cmd)

        # Move everything up to the main directory for this ObsID
        cmd = "mv * ../"
        cmds.append(cmd)

        # Change directory to the main directory
        cmd = "cd ../"
        cmds.append(cmd)

        # Remove the temporary folder
        cmd = "rm -r {o}_{i}_{e}_temp".format(o=obj_id, i=ins, e=en)
        cmds.append(cmd)

    return ";".join(cmds)


def interpret_cr_file(output_path):
    with open(output_path, 'r') as out:
        output_lines = out.readlines()
    cnt_rate = [entry for entry in output_lines if "upper limit" in entry][0]
    cnt_rate = float(cnt_rate.split("c/r: ")[-1].split(" c/s")[0])

    return cnt_rate


def calc_lum(ins, model_dfs, obj_id, save_dir):
    os.chdir(save_dir)
    l_en_output = save_dir + "/{s}_e".format(s=obj_id) + ins.lower() + "_lowen_output.txt"
    h_en_output = save_dir + "/{s}_e".format(s=obj_id) + ins.lower() + "_highen_output.txt"
    l_en_rate = interpret_cr_file(l_en_output)
    h_en_rate = interpret_cr_file(h_en_output)

    l_en_flux_col = "pred_lowen_flux"
    h_en_flux_col = "pred_highen_flux"
    l_en_lum_col = "pred_lowen_lum"
    h_en_lum_col = "pred_highen_lum"
    for m_df in model_dfs:
        if not os.path.exists("{0}_{1}_{2}_ecfs.csv".format(obj_id, m_df, ins)):
            model_dfs[m_df].loc[:, l_en_flux_col] = (model_dfs[m_df]["flux_lowen"] /
                                                     model_dfs[m_df]["ph_per_sec_lowen"]) * l_en_rate
            model_dfs[m_df].loc[:, h_en_flux_col] = (model_dfs[m_df]["flux_highen"] /
                                                     model_dfs[m_df]["ph_per_sec_highen"]) * h_en_rate

            model_dfs[m_df][l_en_lum_col] = flux_to_lum(model_dfs[m_df][l_en_flux_col], model_dfs[m_df]["redshift"])
            model_dfs[m_df][h_en_lum_col] = flux_to_lum(model_dfs[m_df][h_en_flux_col], model_dfs[m_df]["redshift"])
            model_dfs[m_df].to_csv("{0}_{1}_{2}_ecfs.csv".format(obj_id, m_df, ins), index=False)

    return model_dfs


def flux_to_lum(flux, redshift):
    lum_dist = Planck15.luminosity_distance(redshift).to("cm")
    return (4 * pi * lum_dist.value**2) * flux


def calc_comb_lum(all_dfs, save_dir, obj_id):
    inst_list = list(all_dfs.keys())
    model_list = list(all_dfs[inst_list[0]].keys())
    split_models = {mod: {ins: None for ins in inst_list} for mod in model_list}

    en_bands = ["lowen", "highen"]
    cols = []
    for en in en_bands:
        cols.append("comb_{e}_ph_rate".format(e=en))
        cols.append("comb_{e}_flux".format(e=en))
    cols += ["comb_pred_lowen_flux", "comb_pred_lowen_lum", "comb_pred_highen_flux", "comb_pred_highen_lum"]

    # This data wrangling bit is horrible, I probably should have just iterated instrument inside of ecfs_calc, but
    # here we are
    for ins in all_dfs:
        for m in model_list:
            split_models[m][ins] = all_dfs[ins][m]

    comb_pred = {}
    for m in split_models:
        if not os.path.exists(save_dir + "/{s}_{m}_comb_pred.csv".format(s=obj_id, m=m)):
            empty_dat = zeros((list(split_models[m].values())[0].shape[0], len(cols)))
            comb_lum_df = pd.DataFrame(columns=cols, data=empty_dat)
            rates = {en: 0 for en in en_bands}
            for ins in split_models[m]:
                for en in en_bands:
                    comb_lum_df["comb_{e}_ph_rate".format(e=en)] += split_models[m][ins]["ph_per_sec_{e}".format(e=en)]
                    # Using an average flux when adding them together
                    comb_lum_df["comb_{e}_flux".format(e=en)] += (split_models[m][ins]["flux_{e}".format(e=en)] /
                                                                  len(split_models[m]))

                    o_file = save_dir + "/{s}_e{i}_{e}_output.txt".format(s=obj_id, i=ins.lower(), e=en)
                    rates[en] += interpret_cr_file(o_file)

            for en in en_bands:
                f = (comb_lum_df["comb_{e}_flux".format(e=en)] / comb_lum_df["comb_{e}_ph_rate".format(e=en)]) \
                    * rates[en]
                comb_lum_df["comb_pred_{e}_flux".format(e=en)] = f

                # Just take the redshifts from the last instrument accessed, they will be the same for all
                lum = flux_to_lum(comb_lum_df["comb_pred_{e}_flux".format(e=en)], split_models[m][ins]["redshift"])
                comb_lum_df["comb_pred_{e}_lum".format(e=en)] = lum

            comb_lum_df.to_csv(save_dir + "/{s}_{m}_comb_pred.csv".format(s=obj_id, m=m), index=False)
        else:
            comb_lum_df = pd.read_csv(save_dir + "/{s}_{m}_comb_pred.csv".format(s=obj_id, m=m), header="infer")

        comb_pred[m] = comb_lum_df

    return comb_pred


def lum_plots(for_plotting, combined_for_plotting, save_dir, obj_id):
    inst_list = list(for_plotting.keys())
    model_list = list(for_plotting[inst_list[0]].keys())
    split_models = {mod: {ins: None for ins in inst_list} for mod in model_list}

    # This data wrangling bit is horrible, I probably should have just iterated instrument inside of ecfs_calc, but
    # here we are
    for ins in for_plotting:
        for m in model_list:
            split_models[m][ins] = for_plotting[ins][m]

    en_bands = ["lowen", "highen"]  # Will add wideen to this at some point, 0.5-8.0keV like in the paper

    for m in model_list:
        if not os.path.exists("{sav}/{o}_{mod}_pred_lum_dist.png".format(sav=save_dir, o=obj_id, mod=m)):
            lum_fig, ax_arr = plt.subplots(1, len(en_bands), figsize=(len(en_bands) * 12, 10))

            for m_ind, subplot in ndenumerate(ax_arr):
                subplot.minorticks_on()
                subplot.tick_params(axis='both', direction='in', which='both', top=True, right=True)
                subplot.grid(linestyle='dotted', linewidth=1)
                subplot.axis(option="tight")
            lum_fig.suptitle(r"Predicted Luminosity Distributions", fontsize=14, y=0.95, x=0.51)

            # Plot the distributions on histograms
            for en_ind, en in enumerate(en_bands):
                cur_ax = ax_arr[en_ind]
                # cur_ax.set_xscale("log")
                cur_ax.set_title(en)
                cur_ax.set_xlabel("L$_x$")
                cur_ax.set_ylabel("Density")
                for ins in inst_list:
                    cur_ax.hist(split_models[m][ins]["pred_{}_lum".format(en)],
                                bins="auto", label="{0} {1}".format(m, ins), alpha=0.7, density=True)
                cur_ax.hist(combined_for_plotting[m]["comb_pred_{e}_lum".format(e=en)], bins="auto",
                            label="{0} {1}".format(m, "Combined"), alpha=0.7, density=True)
                cur_ax.legend(loc="best")
            plt.savefig("{sav}/{o}_{mod}_pred_lum_dist.png".format(sav=save_dir, o=obj_id, mod=m),
                        bbox_inches='tight')
            plt.close("all")


def run_ecfs(conf_dict):
    # Ugly method but works because I planned to do multithreading of this bit
    x.Fit.statMethod = "cstat"
    x.Fit.query = "yes"
    x.Xset.chatter = 0

    insts = ["PN", "MOS1", "MOS2"]
    ecfs_onwards = tqdm(total=len(ecfs_iterables) * len(insts), desc="Generating ECFS Files")
    for le_pars in ecfs_iterables:
        all_inst_spec = {inst: None for inst in insts}
        all_inst_dfs = {inst: None for inst in insts}
        for inst in insts:
            try:
                ecfs_onwards.write("{o} - {i}".format(o=le_pars[2], i=inst))
                spectra, conv_dfs = ecfs_calc(*le_pars, inst)
                if None not in spectra.values():
                    all_inst_spec[inst] = spectra
                else:
                    all_inst_spec.pop(inst)

                for entry in deepcopy(conv_dfs):
                    if conv_dfs[entry] is None:
                        conv_dfs.pop(entry)
                if len(conv_dfs) != 0:
                    conv_dfs = calc_lum(inst, conv_dfs, le_pars[1], le_pars[-1])
                    all_inst_dfs[inst] = conv_dfs
                else:
                    all_inst_dfs.pop(inst)

            except Exception as e:
                ecfs_onwards.write(str(e))
                # Sin of broad exception because pyXSPEC doesn't behave like a Python module
                all_inst_spec.pop(inst)
                all_inst_dfs.pop(inst)
            x.AllData.clear()
            x.AllModels.clear()

            ecfs_onwards.update(1)
        if "MOS2" in list(all_inst_spec.keys()) and "PN" in list(all_inst_spec.keys()) and conf_dict["produce_plots"]:
            fake_spec_plots(all_inst_spec, le_pars[-1], le_pars[1])

        combined_lum = calc_comb_lum(all_inst_dfs, le_pars[-1], le_pars[1])
        if conf_dict["produce_plots"]:
            lum_plots(all_inst_dfs, combined_lum, le_pars[-1], le_pars[1])
    ecfs_onwards.close()


def validate_samp_file(conf_dict):
    samp = pd.read_csv(conf_dict["sample_csv"], header="infer", dtype={config["id_col"]: str, "OBSID": str})

    for entry in conf_dict:
        if "col" in entry and conf_dict[entry] not in samp.columns.values:
            sys.exit("{e} column is missing from the sample csv!".format(e=conf_dict[entry]))

    if "type" not in samp.columns.values:
        sys.exit("Please add a type column to your sample, with values of ext or pnt")

    if "rad" not in samp.columns.values:
        sys.exit("Please add a rad column to your sample, with extraction radii in arcseconds")
    return samp


if __name__ == "__main__":
    if sys.version_info[0] < 3:
        sys.exit("You cannot run this tool with Python 2.x")
    elif sys.version_info[1] < 6:
        # warnings.warn("Be wary, before Python 3.6 dictionaries aren't necessarily ordered")
        sys.exit("Be wary, before Python 3.6 dictionaries aren't necessarily ordered")

    required_args = ["Configuration JSON"]
    if len(sys.argv) != len(required_args) + 1:
        print('Please pass the following arguments: ')
        print('{}'.format(", ".join(required_args)))
        sys.exit(1)
    elif os.path.exists(sys.argv[1]):
        config_file = sys.argv[1]
        with open(sys.argv[1], 'r') as conf:
            config = json.load(conf)
        config = validate_config(config)
    else:
        sys.exit('That config file does not exist!')

    xmm_samp = validate_samp_file(config)
    if not os.path.exists(config["samp_path"] + "/xmm_spectra"):
        os.mkdir(config["samp_path"] + "/xmm_spectra")

    if os.environ["SAS_PATH"].lower().split("/sas_")[-1].split(".")[0] == "17":
        update_ccf = True
    else:
        update_ccf = False
    print("You're using SAS version {v}".format(v=os.environ["SAS_PATH"].lower().split("/sas_")[-1].split(".")[0]))

    sasgen_stack = []
    combine_stack = []
    le_flux_stack = []
    ecfs_iterables = []
    onwards = tqdm(desc="Preparing SAS commands for candidates", total=len(xmm_samp))
    for ind, row in xmm_samp.iterrows():
        for x_id in row["OBSID"].split(","):
            s_dir = config["samp_path"] + "xmm_spectra/{o}".format(o=x_id)

            # This is for the ECFS effort later on
            interim = []
            # Uses HEASOFT to look up nH, then converts to form that xspec wants
            nH = nh_lookup(row[config["ra_col"]], row[config["dec_col"]]) / 10e+21
            par_copy = deepcopy(config["models"])
            for model in list(config["models"].keys()):
                if "nH" in par_copy[model]:
                    par_copy[model]["nH"] = nH
                if "redshift" in par_copy[model]:
                    par_copy[model]["redshift"] = row["redshift"]

            interim.append(par_copy)
            interim.append(row[config["id_col"]])
            interim.append(x_id)
            interim.append(s_dir)

            if not os.path.exists(s_dir):
                os.mkdir(s_dir)

            src = config["xmm_data_path"] + x_id + "/"
            src = os.path.abspath(src) + "/"
            f_loc = {"epn_lowen_im": "{s}images/{o}-0.50-2.00keV-pn_merged_img.fits".format(s=src, o=x_id),
                     "emos1_lowen_im": "{s}images/{o}-0.50-2.00keV-mos1_merged_img.fits".format(s=src, o=x_id),
                     "emos2_lowen_im": "{s}images/{o}-0.50-2.00keV-mos2_merged_img.fits".format(s=src, o=x_id),
                     "epn_evts": "{s}eclean/pn_exp1_clean_evts.fits".format(s=src, o=x_id),
                     "emos1_evts": "{s}eclean/mos1_exp1_clean_evts.fits".format(s=src, o=x_id),
                     "emos2_evts": "{s}eclean/mos2_exp1_clean_evts.fits".format(s=src, o=x_id),
                     "odf": "{s}odf/".format(s=src), "ccf": "{s}odf/ccf.cif".format(s=src),
                     "reg_file": "{r}{o}/final_class_regions_REDO.reg".format(r=config["xmm_reg_path"], o=x_id),
                     "epn_highen_im": "{s}images/{o}-2.00-10.00keV-pn_merged_img.fits".format(s=src, o=x_id),
                     "emos1_highen_im": "{s}images/{o}-2.00-10.00keV-mos1_merged_img.fits".format(s=src, o=x_id),
                     "emos2_highen_im": "{s}images/{o}-2.00-10.00keV-mos2_merged_img.fits".format(s=src, o=x_id),
                     "epn_lowen_expmap": "{s}images/{o}-0.50-2.00keV-pn_merged_expmap.fits".format(s=src, o=x_id),
                     "emos1_lowen_expmap": "{s}images/{o}-0.50-2.00keV-mos1_merged_expmap.fits".format(s=src, o=x_id),
                     "emos2_lowen_expmap": "{s}images/{o}-0.50-2.00keV-mos2_merged_expmap.fits".format(s=src, o=x_id),
                     "epn_highen_expmap": "{s}images/{o}-2.00-10.00keV-pn_merged_expmap.fits".format(s=src, o=x_id),
                     "emos1_highen_expmap": "{s}images/{o}-2.00-10.00keV-mos1_merged_expmap.fits".format(s=src, o=x_id),
                     "emos2_highen_expmap": "{s}images/{o}-2.00-10.00keV-mos2_merged_expmap.fits".format(s=src, o=x_id)}

            # Input radius in arcseconds, 15 is about PSF size so good place to start.
            try:
                src_reg, excl_reg = coords_rad_regions(config, x_id, f_loc["epn_lowen_im"], row[config["ra_col"]],
                                                       row[config["dec_col"]], row["rad"], config["force_rad"])
                # Makes sure there are no problems before adding to the list that will be run
                ecfs_iterables.append(interim)
            except (FileNotFoundError, ValueError, KeyError):
                break

            for instrument in ["PN", "MOS1", "MOS2"]:
                command = command_stack_maker(config, row[config["id_col"]], x_id, f_loc, src_reg, excl_reg, s_dir,
                                              instrument, row["type"], for_comb=False)
                sasgen_stack.append(command)
                for energy in ["lowen", "highen"]:
                    command = photon_rate_command(config, f_loc, row[config["id_col"]], src_reg, excl_reg, s_dir,
                                                  instrument, energy, row["type"])
                    le_flux_stack.append(command)

            command = combine_stack_maker(row[config["id_col"]], x_id, f_loc, s_dir)
            combine_stack.append(command)

        onwards.update(1)
    onwards.close()

    sas_pool(sasgen_stack, "Generating Spectra", config, cpu_count=config["allowed_cores"])
    if config["generate_combined"]:
        sas_pool(combine_stack, "Combining Spectra", config, cpu_count=config["allowed_cores"])
    sas_pool(le_flux_stack, "Measuring upper limit photon rates", config, cpu_count=22)
    run_ecfs(config)








