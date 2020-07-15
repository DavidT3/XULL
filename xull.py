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
from subprocess import DEVNULL as DNULL
from subprocess import call, Popen, PIPE

import imageio
import matplotlib.pyplot as plt
import pandas as pd
import xspec as x
from astropy import units as u
from astropy.units import UnitConversionError
from astropy import wcs
from astropy.cosmology import Planck15
from astropy.io import fits
from numpy import sqrt, linspace, meshgrid, empty, uint8, frombuffer, ndenumerate, pi, percentile, isnan, nan
from scipy.stats import truncnorm
from tqdm import tqdm

warnings.simplefilter('ignore', wcs.FITSFixedWarning)
# TODO Make the energy bands completely general, and passable in through the config file
# TODO Generalise Cosmology


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

    required_head = ["sample_csv", "generate_combined", "force_rad", "xmm_data_path", "xmm_reg_path", "produce_plots",
                     "back_outer_factor", "id_col", "ra_col", "dec_col", "rad_col", "rad_unit", "models",
                     "allowed_cores", "instruments", "conf_level"]

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

    if not isinstance(conf_dict["xmm_reg_path"], str):
        sys.exit("xmm_reg_path should be of type int!")
    elif not os.path.exists(conf_dict["xmm_reg_path"]):
        sys.exit("That xmm_reg_path doesn't exist!")
    elif conf_dict["xmm_reg_path"][-1] != "/":
        conf_dict["xmm_reg_path"] += "/"

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
        print("REDSHIFT IS READ FROM THE SAMPLE, AND NH IS READ FROM HEASOFT, VALUES IN MODEL WILL BE DISCARDED\n")
        print("PARAMETERS MUST BE IN THE ORDER THAT XSPEC EXPECTS THEM")
        for entry in conf_dict["models"]:
            conf_dict["models"][entry] = model_parser(conf_dict["models"][entry])

    if not isinstance(conf_dict["allowed_cores"], int):
        sys.exit("allowed_cores should be an integer value.")

    if not isinstance(conf_dict["conf_level"], int):
        sys.exit("Please an integer percentage (e.g. 90) for conf_level")

    if not isinstance(conf_dict["rad_col"], str):
        sys.exit("rad_col must be a string")

    if not isinstance(conf_dict["rad_unit"], str):
        sys.exit("rad_unit must be a string representation of a distance unit, i.e. kpc or Mpc")
    elif conf_dict["rad_unit"] == "mpc":
        sys.exit("rad_col is case sensitive, mpc is milliparsecs...")
    else:
        try:
            basic_quan = u.Quantity(1, conf_dict["rad_unit"])
            basic_quan.to("m")
        except ValueError as le_error:
            if isinstance(le_error, UnitConversionError) and conf_dict["rad_unit"] not in ["arcsecond", "arcsec"]:
                sys.exit(str(le_error))
            elif isinstance(le_error, UnitConversionError) and conf_dict["rad_unit"] in ["arcsecond", "arcsec"]:
                print("arcsecond is not a valid length unit in astropy, but this code accounts for that\n")
            else:
                sys.exit(conf_dict["rad_unit"] + " does not appear to be a valid Astropy unit.")

    if not isinstance(conf_dict["redshift_col"], str):
        sys.exit("redshift_col must be a string")

    if not isinstance(conf_dict["type_col"], str):
        sys.exit("type_col must be a string")

    if not isinstance(conf_dict["xmm_obsid_col"], str):
        sys.exit("xmm_obsid_col must be a string")

    if not isinstance(conf_dict["instruments"], list):
        sys.exit("instruments must be a list of strings, even if you're only generating for one")
    elif len(conf_dict["instruments"]) == 0:
        sys.exit("You've passed an empty list for the instruments parameter")
    else:
        ins_check = [choice for choice in conf_dict["instruments"] if choice not in ["PN", "MOS1", "MOS2"]]
        if len(ins_check) != 0:
            print("You've passed illegal instruments: {}".format(", ".join(ins_check)))
            sys.exit("The only legal choices for instrument are PN, MOS1, and MOS2")

    return conf_dict


def command_stack_maker(conf_dict, obj_id, obs_id, file_locs, src_r, excl_r, save_dir, ins, obj_type, for_comb, z):
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
    low_en_im_path = obj_id + "_" + obs_id + '_' + ins + '_lowen_im.fits'
    high_en_im_path = obj_id + "_" + obs_id + '_' + ins + '_highen_im.fits'
    low_en_expmap_path = obj_id + "_" + obs_id + '_' + ins + '_lowen_expmap.fits'
    high_en_expmap_path = obj_id + "_" + obs_id + '_' + ins + '_highen_expmap.fits'

    file_locs["e" + ins.lower() + "_lowen_im"] = s_dir + "/" + low_en_im_path
    file_locs["e" + ins.lower() + "_lowen_expmap"] = s_dir + "/" + low_en_expmap_path
    file_locs["e" + ins.lower() + "_highen_im"] = s_dir + "/" + high_en_im_path
    file_locs["e" + ins.lower() + "_highen_expmap"] = s_dir + "/" + high_en_expmap_path

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

        cmd = "backscale spectrumset={sp} badpixlocation={ev} withbadpixcorr=yes".format(sp=back_spec_path, ev=evt_path)
        cmds.append(cmd)

        cmd = "evselect table={evts} imageset={img} xcolumn=X ycolumn=Y ximagebinsize=87 " \
              "yimagebinsize=87 squarepixels=yes ximagesize=512 yimagesize=512 imagebinning=binSize " \
              "ximagemin=3649 ximagemax=48106 withxranges=yes yimagemin=3649 yimagemax=48106 " \
              "withyranges=yes {exp}" \
            .format(evts=evt_path, img=cutout_reg_path, exp=all_but_expression)
        cmds.append(cmd)

        # Generating images and exposure maps appropriate for the redshift of the object in question
        z_lowen = [int(limit / (z + 1)) for limit in [500, 2000]]
        z_highen = [int(limit / (z + 1)) for limit in [2000, 10000]]

        # Copying the expression from XCS energy limited images
        if "PN" in ins:
            z_expr = "expression='#XMMEA_EP && (PATTERN <= 4) && (FLAG .eq. 0) && (PI in [{l}:{u}])'"
        elif "MOS" in ins:
            z_expr = "expression='#XMMEA_EM && (PATTERN <= 12) && (FLAG .eq. 0) && (PI in [{l}:{u}])'"
        else:
            sys.exit("wtf is this instrument?")

        # Lowen image with limits redshifted
        cmd = "evselect table={evts} imageset={img} xcolumn=X ycolumn=Y ximagebinsize=87 " \
              "yimagebinsize=87 squarepixels=yes ximagesize=512 yimagesize=512 imagebinning=binSize " \
              "ximagemin=3649 ximagemax=48106 withxranges=yes yimagemin=3649 yimagemax=48106 " \
              "withyranges=yes {exp}".format(evts=evt_path, img=low_en_im_path,
                                             exp=z_expr.format(l=z_lowen[0], u=z_lowen[1]))
        cmds.append(cmd)

        # Highen image with limits redshifted
        cmd = "evselect table={evts} imageset={img} xcolumn=X ycolumn=Y ximagebinsize=87 " \
              "yimagebinsize=87 squarepixels=yes ximagesize=512 yimagesize=512 imagebinning=binSize " \
              "ximagemin=3649 ximagemax=48106 withxranges=yes yimagemin=3649 yimagemax=48106 " \
              "withyranges=yes {exp}".format(evts=evt_path, img=high_en_im_path,
                                             exp=z_expr.format(l=z_highen[0], u=z_highen[1]))
        cmds.append(cmd)

        # Lowen expmap with limits redshifted
        cmd = "eexpmap eventset={evts} imageset={img} expimageset={eimg} withdetcoords=no withvignetting=yes " \
              "attitudeset={att} pimin={l} pimax={u}".format(evts=evt_path, img=low_en_im_path, eimg=low_en_expmap_path,
                                                             att=file_locs["e" + ins.lower() + "_att"],
                                                             l=z_lowen[0], u=z_lowen[1])
        cmds.append(cmd)

        # Highen expmap with limits redshifted
        cmd = "eexpmap eventset={evts} imageset={img} expimageset={eimg} withdetcoords=no withvignetting=yes " \
              "attitudeset={att} pimin={l} pimax={u}".format(evts=evt_path, img=high_en_im_path,
                                                             eimg=high_en_expmap_path,
                                                             att=file_locs["e" + ins.lower() + "_att"],
                                                             l=z_highen[0], u=z_highen[1])
        cmds.append(cmd)

        cmd = "specgroup groupedset={grpfile} spectrumset={spfile} arfset={arfile} rmfset={rmfile} mincounts=5 " \
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

        return ';'.join(cmds), file_locs
    return "", file_locs


def coords_rad_regions(conf_dict, obs, im_path, ra, dec, rad, target_type, force_user_rad, save_dir, obj_id):
    def reg_extract(reg_string):
        reg_x, reg_y, reg_x_rad, reg_y_rad, reg_ang = [float(el) for el in
                                                       reg_string.split("(")[-1].split(")")[0].split(",")]
        if "#" in reg_string:
            reg_colour = reg_string.split(" # ")[-1].split("color=")[-1].strip("\n")
            if reg_colour == "green":
                reg_type = "ext"
            elif reg_colour == "magenta":
                reg_type = "ext_psf"
            elif reg_colour == "blue":
                reg_type = "ext_pnt_cont"
            elif reg_colour == "cyan":
                reg_type = "ext_run1_cont"
            elif reg_colour == "yellow":
                reg_type = "ext_less_ten_counts"
        else:
            reg_type = "pnt"

        # -1 because XAPA starts counting at 1, not 0 like Python
        return reg_x-1, reg_y-1, reg_x_rad, reg_y_rad, reg_ang, reg_type

    def calc_sep(reg_x, reg_y):
        return sqrt((reg_x - cen_pix[0])**2 + (reg_y - cen_pix[1])**2)

    def pix_to_sky(pix_x, pix_y, pix_r):
        edge_sky = pix_sky_wcs.all_pix2world(pix_x+pix_r, pix_y, 0)
        cen_sky = pix_sky_wcs.all_pix2world(pix_x, pix_y, 0)
        sky_x = float(cen_sky[0])
        sky_y = float(cen_sky[1])
        sky_r = abs(edge_sky[0] - cen_sky[0])
        return sky_x, sky_y, sky_r

    def new_reg_sep_list(summary, seps, source_inds):
        if source_inds is None:
            return summary, seps
        else:
            new_reg_summary = [entry for entry_ind, entry in enumerate(summary) if entry_ind not in source_inds]
            new_seps = [entry for entry_ind, entry in enumerate(seps) if entry_ind not in source_inds]
            return new_reg_summary, new_seps

    def create_xull_reg_file(source, excluded_sources):
        reg_line = "physical ; circle({x},{y},{r})"
        source_line = reg_line.format(x=source[0], y=source[1], r=source[2]) + " # color=green"
        xull_reg_lines = [reg_line.format(x=entry[0], y=entry[1], r=entry[2]) for entry in excluded_sources]
        all_reg_lines = "\n".join(["global color=red"] + xull_reg_lines + [source_line])
        with open(save_dir + "/{}_xull_source+excl.reg".format(obj_id), 'w') as new_reggy:
            new_reggy.write(all_reg_lines)

    def read_xull_reg_file(f_path):
        with open(f_path, 'r') as xull_reggy:
            lines = xull_reggy.readlines()[1:]

        read_in_source = None
        read_in_excl = []
        for line in lines:
            parsed_info = [float(el) for el in line.split("(")[-1].split(")")[0].split(",")]
            if "green" in line:
                read_in_source = parsed_info
            else:
                read_in_excl.append(parsed_info)

        return read_in_source, read_in_excl

    if not os.path.exists(save_dir + "/{}_xull_source+excl.reg".format(obj_id)):
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
        max_reg_radii = [max(reg[2:4]) for reg in reg_summary]
        separations = [calc_sep(reg[0], reg[1]) for reg in reg_summary]

        # Finds if any of the target coords are inside a circle with the max radius of the XAPA region
        # So elliptical XAPA regions become circular
        poss_source_reg = []
        for sep_ind, sep in enumerate(separations):
            if reg_summary[sep_ind][-1] in ["pnt", "ext_psf"] and sep <= max_reg_radii[sep_ind]:
                poss_source_reg.append(sep_ind)
            elif reg_summary[sep_ind][-1] not in ["pnt", "ext_psf"] and sep <= (max_reg_radii[sep_ind] * 0.5):
                poss_source_reg.append(sep_ind)

        if len(poss_source_reg) == 0:
            deg_min_sep = abs(deg_pix_wcs.all_pix2world(cen_pix[0] + min(separations), cen_pix[1], 0)[0] - ra) * 3600
            onwards.write("No XAPA region, closest is {0} arcseconds away, "
                          "setting radius to {1} arcsec.".format(round(deg_min_sep, 2), round(rad, 2)))
            source_reg = None
        elif len(poss_source_reg) == 1:
            # Nesting the if statements here was just easier to read
            # Checks if target is point source and if XAPA region is point source
            if target_type == "pnt" and reg_summary[poss_source_reg[0]][-1] in ["pnt", "ext_psf"] \
                    and not force_user_rad:
                onwards.write("Target point source is in XAPA point source, setting source radius to XAPA region "
                              "radius.")
                source_reg = reg_summary.pop(poss_source_reg[0])
                reg_summary, separations = new_reg_sep_list(reg_summary, separations, poss_source_reg)
            elif target_type == "pnt" and reg_summary[poss_source_reg[0]][-1] in ["pnt", "ext_psf"] and force_user_rad:
                onwards.write("Target point source is in XAPA point source, but forcing using sample radius.")
                source_reg = None
                reg_summary, separations = new_reg_sep_list(reg_summary, separations, poss_source_reg)
            elif target_type == "pnt" and reg_summary[poss_source_reg[0]][-1] not in ["pnt", "ext_psf"]:
                raise ValueError("Target point source inside an extended source, using sample radius, this target will "
                                 "not be processed")

            elif target_type == "ext" and reg_summary[poss_source_reg[0]][-1] != "pnt" and not force_user_rad:
                onwards.write("Target extended source coordinates are within 50% of the radius of a XAPA extended "
                              "source, using XAPA source.")
                source_reg = reg_summary.pop(poss_source_reg[0])
                reg_summary, separations = new_reg_sep_list(reg_summary, separations, poss_source_reg)
            elif target_type == "ext" and reg_summary[poss_source_reg[0]][-1] != "pnt" and force_user_rad:
                onwards.write("Target extended source inside a XAPA extended source, using sample radius.")
                source_reg = None
                reg_summary, separations = new_reg_sep_list(reg_summary, separations, poss_source_reg)

            elif target_type == "ext" and reg_summary[poss_source_reg[0]][-1] == "pnt":
                onwards.write("Target extended source coordinates match with a XAPA point source, using sample radius")
                reg_summary, separations = new_reg_sep_list(reg_summary, separations, poss_source_reg)
                source_reg = None

            else:
                source_reg = None
                onwards.write("{0} target, {1} match, force_user_rad={2}".format(target_type,
                                                                                 reg_summary[poss_source_reg[0]][-1],
                                                                                 force_user_rad))
                sys.exit("How did you get here?")

        elif len(poss_source_reg) == 2:
            # This doesn't cover every possible combination, because I suspect that the only way this will come up is if
            # I'm looking for a extended source which XAPA also has a point source region inside
            # Grabbing source which is closest to target coordinates
            if separations[poss_source_reg[0]] < separations[poss_source_reg[1]]:
                min_sep_ind = poss_source_reg[0]
                other_match = poss_source_reg[1]
            else:
                min_sep_ind = poss_source_reg[1]
                other_match = poss_source_reg[0]

            if target_type == "ext" and reg_summary[min_sep_ind][-1] == "pnt" \
                    and reg_summary[other_match][-1] != "pnt" and not force_user_rad:
                onwards.write("The extended target has matched to two XAPA regions, the closest is a point source, "
                              "the furthest is extended - will use the extended region")
                source_reg = reg_summary.pop(other_match)
                reg_summary, separations = new_reg_sep_list(reg_summary, separations, poss_source_reg)
            elif target_type == "ext" and reg_summary[min_sep_ind][-1] == "pnt" \
                    and reg_summary[other_match][-1] != "pnt" and force_user_rad:
                onwards.write("The extended target has matched to two XAPA regions, the closest is a point source, "
                              "the furthest is extended - using radius from sample")
                source_reg = None
                reg_summary, separations = new_reg_sep_list(reg_summary, separations, poss_source_reg)

            elif target_type == "ext" and reg_summary[min_sep_ind][-1] != "pnt" \
                    and reg_summary[other_match][-1] == "pnt" and not force_user_rad:
                onwards.write("The point target has matched to two XAPA regions, the closest is an extended source, "
                              "the furthest is a point - will use the extended region")
                source_reg = reg_summary.pop(min_sep_ind)
                reg_summary, separations = new_reg_sep_list(reg_summary, separations, poss_source_reg)
            elif target_type == "ext" and reg_summary[min_sep_ind][-1] != "pnt" \
                    and reg_summary[other_match][-1] == "pnt" and force_user_rad:
                onwards.write("The point target has matched to two XAPA regions, the closest is an extended source, "
                              "the furthest is a point - using radius from sample")
                source_reg = None
                reg_summary, separations = new_reg_sep_list(reg_summary, separations, poss_source_reg)
            else:
                raise ValueError("Target point source inside an extended source, using sample radius, this target will "
                                 "not be processed")

        else:
            sys.exit("The target coordinates appear to be in {} different XAPA regions, email "
                     "david.turner@sussex.ac.uk and tell him he needs to fix his matching "
                     "code".format(len(poss_source_reg)))

        if source_reg is None:
            # If no matching XAPA region was found, we use the values passed into the function
            # Converts arcseconds to degrees
            rad /= 3600
            edge_pix = deg_pix_wcs.all_world2pix(ra + rad, dec, 0)
            pix_rad = abs(cen_pix[0] - edge_pix[0])
        else:
            # If a matching XAPA region WAS found, the central point and radius are used instead
            cen_pix = [source_reg[0], source_reg[1]]
            pix_rad = source_reg[2]
        source_sky = pix_to_sky(*cen_pix, pix_rad)

        # Now going to find the sources within some arbitrary large radius (50 pixels?), to make sure they're excluded
        sources_within_lim = [i for i, sep in enumerate(separations) if sep < pix_rad+50]
        exclude_sky = [pix_to_sky(*reg_summary[i][:2], reg_summary[i][3]) for i in sources_within_lim]
        create_xull_reg_file(source_sky, exclude_sky)
        
    else:
        source_sky, exclude_sky = read_xull_reg_file(save_dir + "/{}_xull_source+excl.reg".format(obj_id))

    return source_sky, exclude_sky


def run_sas(cmd, pass_shell, pass_stdout, pass_stderr, conf_dict):
    sas_version = conf_dict["sas_version"]
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
        log_name = sp + "xmm_spectra_sas{v}/{o}/{oi}_{i}_sasgen.log".format(o=obs_id, oi=obj_id, i=ins, v=sas_version)
        with open(log_name, 'w') as loggy:
            call(cmd, shell=pass_shell, stderr=loggy, stdout=loggy)

        try:
            sp_path = conf_dict["samp_path"] + "xmm_spectra_sas{v}/{o}/".format(o=obs_id, v=sas_version) + spec_file
            spec = fits.open(sp_path, mode="update")
            spec[1].header["BACKFILE"] = sp + "xmm_spectra_sas{v}/{o}/".format(o=obs_id, v=sas_version) + back_spec_file
            if "epicspeccombine" not in cmd:
                spec[1].header["ANCRFILE"] = sp + "xmm_spectra_sas{v}/{o}/".format(o=obs_id, v=sas_version) + arf_file
                spec[1].header["RESPFILE"] = sp + "xmm_spectra_sas{v}/{o}/".format(o=obs_id, v=sas_version) + rmf_file
            else:
                spec[1].header["RESPFILE"] = sp + "xmm_spectra_sas{v}/{o}/".format(o=obs_id, v=sas_version) + resp_file
                spec.flush()
            spec.flush()
        except FileNotFoundError:
            tqdm.write("{} does not exist!".format(spec_file))

    except IndexError:
        if "eregion" in cmd:
            im_path = cmd.split("imageset='")[-1].split("'")[0]
            obj_id, obs_id, ins, en = im_path.split("/")[-1].split("_")[0:4]
            log_name = sp + "xmm_spectra_sas{v}/{o}/{oi}_{i}_{e}_eregionanalyse.log".format(o=obs_id, oi=obj_id, i=ins,
                                                                                            v=sas_version, e=en)
            with open(log_name, 'w') as loggy:
                call(cmd, shell=pass_shell, stderr=loggy, stdout=loggy)
        else:
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


def ecfs_calc(parameters, obj_id, obs_id, save_dir, ins, make_plots, redshift):
    def check_response_status(rsp_file):
        if not os.path.exists(rsp_file):
            use_it = False
        elif os.path.exists(rsp_file) and "rmf" in rsp_file:
            use_it = True
        elif os.path.exists(rsp_file) and "arf" in rsp_file:
            arf_opened = fits.open(rsp_file)
            nan_arr = isnan(arf_opened[1].data["SPECRESP"])
            if arf_opened[1].data["SPECRESP"].sum() == 0 or True in nan_arr:
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
        if not os.path.exists("{0}_{1}_ecf_lum_table.csv".format(obj_id, model)) and (run_inst_rmf and run_inst_arf):
            current_df = model_dfs[m_df].copy()
            current_df = current_df.assign(ph_per_sec_lowen=-1, flux_lowen=-1, ph_per_sec_highen=-1, flux_highen=-1)
            for par_ind, par_comb in model_dfs[m_df].iterrows():
                x_mod = x.Model(m_df)
                x_mod.setPars(*par_comb.astype(float).values)

                fake_settings = x.FakeitSettings(arf=arf_path, response=rmf_path, exposure=fake_it_exp,
                                                 fileName="temp.fak")
                x.AllData.fakeit(1, fake_settings)
                fake_spec = x.AllData(1)
                # pyXSPEC insists on writing a spec file for fakeit, so I immediately delete it.
                os.remove(fake_settings.fileName)

                fake_spec.ignore("**-0.3 10.0-**")
                x.Fit.perform()

                if make_plots:
                    x.Plot.device = '/null'
                    x.Plot.xAxis = "keV"
                    x.Plot('data')
                    for_plotting[m_df]["energies"].append(x.Plot.x())
                    for_plotting[m_df]["rates"].append(x.Plot.y())
                    for_plotting[m_df]["folded_model"].append(x.Plot.model())

                # Now to find source frame limits
                z_lowen = [limit / (redshift + 1) for limit in [0.5, 2.0]]
                z_highen = [limit / (redshift + 1) for limit in [2.0, 10.0]]
                # Count rate measurements have to come before I zero the absorption, as we'll be measuring absorbed c/r

                # Have to use an ignore to get a count rate for the energy range I care about
                fake_spec.ignore("**-{l} {u}-**".format(l=z_lowen[0], u=z_lowen[1]))
                # the 0th element of rate is the background subtracted rate, but doesn't matter -> no background!
                lowen_rate = fake_spec.rate[0]

                # Now reset the ignore to ignore nothing
                fake_spec.notice("all")
                # And now ignore to get the high energy range
                fake_spec.ignore("**-{l} {u}-**".format(l=z_highen[0], u=z_highen[1]))
                highen_rate = fake_spec.rate[0]

                """# THIS ISN'T USED AT THE MOMENT AND SHOULD PROBABLY BE CHUCKED OUT
                # Now reset the ignore to ignore nothing
                fake_spec.notice("all")
                # And now ignore to get the energy range used in the paper
                fake_spec.ignore("**-0.5 8.0-**")
                current_df.loc[par_ind, "ph_per_sec_wideen"] = fake_spec.rate[0]
                current_df.loc[par_ind, "flux_wideen"] = highen_flux"""

                fake_spec.notice("all")
                # Sort of janky way of finding of one of nH absorption models is being used, definitely not rigorous
                if "abs" in m_df:
                    # TODO Make this check the parameter dict, and find position based on that
                    # This should zero nH, which means the calculated fluxes will be unabsorbed (according to Paul)
                    x_mod.setPars(0)

                x.AllModels.calcFlux("{l} {u}".format(l=z_lowen[0], u=z_lowen[1]))
                lowen_flux = fake_spec.flux[0]
                x.AllModels.calcFlux("{l} {u}".format(l=z_highen[0], u=z_highen[1]))
                highen_flux = fake_spec.flux[0]

                current_df.loc[par_ind, "ph_per_sec_lowen"] = lowen_rate
                current_df.loc[par_ind, "flux_lowen"] = lowen_flux

                current_df.loc[par_ind, "ph_per_sec_highen"] = highen_rate
                current_df.loc[par_ind, "flux_highen"] = highen_flux

                x.AllData.clear()
                x.AllModels.clear()
            model_dfs[m_df] = current_df
        elif not run_inst_rmf and not run_inst_arf:
            model_dfs[m_df] = None
            for_plotting[m_df] = None
        else:
            try:
                model_dfs[m_df] = pd.read_csv("{0}_{1}_ecf_lum_table.csv".format(obj_id, model, ins), header="infer")
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

    # Now we start plotting
    for m_ind, m in enumerate(model_list):
        if not os.path.exists("{0}_{1}.gif".format(obj_id, m)):
            images = []
            output_df = pd.read_csv("{0}_{1}_ecf_lum_table.csv".format(obj_id, m), header="infer")

            for col_ind, col in enumerate(output_df.columns.values):
                if "ph_per_sec" in col:
                    break
            p_space = output_df.drop(output_df.columns.values[col_ind-1:], axis=1)
            p_cols = p_space.columns.values
            # Finds maximum value of the fake spectrum fitted models, then makes it 10% larger, for plot limit
            # max_y = max([max(entry) for entry in split_models[m]["PN"]["folded_model"]]) * 1.1
            max_y = max([max(entry) for inst in split_models[m] for entry in split_models[m][inst]["folded_model"]])*1.1

            ins = list(split_models[m].keys())[0]
            for i in range(0, len(split_models[m][ins]["energies"])):
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

            imageio.mimsave("{0}_{1}.gif".format(obj_id, m), images, fps=7)
        plt.close("all")


def photon_rate_command(conf_dict, file_locs, obj_id, src_region, excl_regions, save_dir, ins, en, obj_type):
    out_name = "{0}_{1}_{2}_output.txt".format(obj_id, "e" + ins.lower(), en)
    cmds = []

    if obj_type == "ext":
        psf_model = "EXTENDED"
    elif obj_type == "pnt":
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
    up_cnt_rate = [entry for entry in output_lines if "upper limit" in entry][0]
    up_cnt_rate = float(up_cnt_rate.split("c/r: ")[-1].split(" c/s")[0])

    back_sub_rate_line = [entry for entry in output_lines if "Bckgnd subtracted source c/r" in entry][0]
    back_sub_rate = float(back_sub_rate_line.split("c/r: ")[-1].split(" +/-")[0])
    back_sub_rate_err = float(back_sub_rate_line.split("+/- ")[-1])

    exp_line = [entry for entry in output_lines if "exposure time" in entry][0]
    exp_time = float(exp_line.split("exposure time: ")[-1])

    return {"up_lim": up_cnt_rate, "bsub_rate": back_sub_rate, "bsub_rate_err": back_sub_rate_err,
            "exp_time": exp_time}


def calc_lums(all_dfs, save_dir, obj_id, conf_level, z_col):
    def flux_to_lum(flux, redshift):
        lum_dist = Planck15.luminosity_distance(redshift).to("cm")
        return (4 * pi * lum_dist.value ** 2) * flux

    # For the time being I will only care about count rate uncertainties, not any measurement uncertainties on the ECF
    def cnt_rate_dist(factor, redshift, cnt_rate, cnt_rate_err, cnt_rate_low_lim):
        lum_distributions = []
        lum_med = []
        lum_pl = []
        lum_mi = []
        for i, z in enumerate(redshift):
            # Kathy thought using 1/exp_time as lower limit was cheating, so it'll be truncated at c/r of zero now
            # Requires an upper limit, so I give it an absurdly large count rate which is likely impossible
            try:
                samples = truncnorm.rvs((0 - cnt_rate) / cnt_rate_err, (10000000-cnt_rate)/cnt_rate_err,
                                        loc=cnt_rate, scale=cnt_rate_err, size=10000)
            except RuntimeError:
                samples = truncnorm.rvs(0, (10000000 - cnt_rate) / cnt_rate_err,
                                        loc=cnt_rate, scale=cnt_rate_err, size=10000)
            lum_distribution = flux_to_lum(factor[i]*samples, z)
            med = percentile(lum_distribution, 50)
            pl_err = percentile(lum_distribution, 50 + (conf_level/2)) - med
            mi_err = med - percentile(lum_distribution, 50 - (conf_level/2))
            lum_med.append(med)
            lum_pl.append(pl_err)
            lum_mi.append(mi_err)
            lum_distributions += list(lum_distribution)

        whole_med = percentile(lum_distributions, 50)
        whole_pl = percentile(lum_distributions, 50 + (conf_level/2)) - whole_med
        whole_mi = whole_med - percentile(lum_distributions, 50 - (conf_level/2))

        return lum_med, lum_pl, lum_mi, lum_distributions, whole_med, whole_pl, whole_mi

    en_bands = ["lowen", "highen"]
    lum_dfs = {}
    inst_list = list(all_dfs.keys())
    model_list = list(all_dfs[inst_list[0]].keys())
    split_models = {mod: {ins: None for ins in inst_list} for mod in model_list}
    # This data wrangling bit is horrible, I probably should have just iterated instrument inside of ecfs_calc, but
    # here we are
    for ins in all_dfs:
        for m in model_list:
            split_models[m][ins] = all_dfs[ins][m]

    for m_df in deepcopy(split_models):
        if not os.path.exists("{0}_{1}_ecf_lum_table.csv".format(obj_id, m_df)) \
                or not os.path.exists("{0}_{1}_comb_pred.csv".format(obj_id, m_df)):
            # Here I shall setup a summary Lx dataframe, where ULx is the upper limit luminosity for each model, and
            # MLx is the median luminosity for each model based on MC-ing the count rate error
            mod_cols = [n for n in split_models[m_df][inst_list[0]].columns.values if "pred" not in n and
                        "lum" not in n]
            cols = mod_cols.copy()
            glob_mod_cols = []

            rates = {}
            for en in en_bands:
                for ins in deepcopy(split_models[m_df]):
                    try:
                        out_file = save_dir + "/{s}_e{i}_{e}_output.txt".format(s=obj_id, i=ins.lower(), e=en)
                        rates["{i}_{e}".format(i=ins, e=en)] = interpret_cr_file(out_file)

                        # Split out like so we can do more informative errors about eregionanalyse failures
                        r = rates["{i}_{e}".format(i=ins, e=en)]
                        if r["exp_time"] == 0:
                            split_models[m_df].pop(ins)
                            onwards.write("{0} - {1} - {2} dropped because exposure time is 0.".format(obj_id, m_df, ins))
                            continue
                        if r["up_lim"] <= 0:
                            split_models[m_df].pop(ins)
                            onwards.write("{0} - {1} - {2} dropped because upper limit is <= 0".format(obj_id, m_df, ins))
                            continue
                        if r["bsub_rate_err"] <= 0 and r["up_lim"] <= 0:
                            split_models[m_df].pop(ins)
                            onwards.write("{0} - {1} - {2} dropped because background subtracted "
                                          "rate uncertainty is <= 0 and upper limit is <= 0".format(obj_id, m_df, ins))
                            continue
                        else:
                            rates["{i}_{e}".format(i=ins,
                                                   e=en)]["min_rate"] = 1/rates["{i}_{e}".format(i=ins,
                                                                                                 e=en)]["exp_time"]
                    except FileNotFoundError:
                        split_models[m_df].pop(ins)
                        onwards.write("{0} {1} {2} eregionanalyse file is missing!".format(obj_id, m_df, ins))

            for en in en_bands:
                for ins in split_models[m_df]:
                    cols.append("{i}_{e}_ULx".format(i=ins, e=en))
                    cols.append("{i}_{e}_MLx".format(i=ins, e=en))
                    glob_mod_cols.append("{i}_{e}_MLx".format(i=ins, e=en))
                    cols.append("{i}_{e}_MLx+".format(i=ins, e=en))
                    glob_mod_cols.append("{i}_{e}_MLx+".format(i=ins, e=en))
                    cols.append("{i}_{e}_MLx-".format(i=ins, e=en))
                    glob_mod_cols.append("{i}_{e}_MLx-".format(i=ins, e=en))
                glob_mod_cols.append("all_{e}_MLx".format(e=en))
                glob_mod_cols.append("all_{e}_MLx+".format(e=en))
                glob_mod_cols.append("all_{e}_MLx-".format(e=en))

            model_pars_lum_summary = pd.DataFrame(columns=cols)
            global_lum_summary = pd.DataFrame(columns=glob_mod_cols)
            inst_list = list(split_models[m_df].keys())
            if len(inst_list) == 0:
                continue

            for mod_col in mod_cols:
                model_pars_lum_summary.loc[:, mod_col] = split_models[m_df][inst_list[0]][mod_col]

            for en in en_bands:
                all_ins_distribution = []
                for ins in split_models[m_df]:
                    r = rates["{}_{}".format(ins, en)]
                    df = split_models[m_df][ins]
                    ecf = df["flux_{}".format(en)] / df["ph_per_sec_{}".format(en)]
                    f = ecf * r["up_lim"]
                    model_pars_lum_summary.loc[:, "{i}_{e}_ECF".format(i=ins, e=en)] = ecf
                    model_pars_lum_summary.loc[:, "{i}_{e}_ULx".format(i=ins, e=en)] = flux_to_lum(f, df[z_col])
                    if r["bsub_rate_err"] != 0:
                        mlx, mlx_pl, mlx_mi, whole_dist, whole_mlx, whole_mlx_pl, whole_mlx_mi \
                            = cnt_rate_dist(ecf, df[z_col], r["bsub_rate"], r["bsub_rate_err"], r["min_rate"])
                        all_ins_distribution += whole_dist

                        model_pars_lum_summary.loc[:, "{i}_{e}_MLx".format(i=ins, e=en)] = mlx
                        model_pars_lum_summary.loc[:, "{i}_{e}_MLx+".format(i=ins, e=en)] = mlx_pl
                        model_pars_lum_summary.loc[:, "{i}_{e}_MLx-".format(i=ins, e=en)] = mlx_mi

                        global_lum_summary.loc[0, "{i}_{e}_MLx".format(i=ins, e=en)] = whole_mlx
                        global_lum_summary.loc[0, "{i}_{e}_MLx+".format(i=ins, e=en)] = whole_mlx_pl
                        global_lum_summary.loc[0, "{i}_{e}_MLx-".format(i=ins, e=en)] = whole_mlx_mi
                    else:
                        model_pars_lum_summary.loc[:, "{i}_{e}_MLx".format(i=ins, e=en)] = nan
                        model_pars_lum_summary.loc[:, "{i}_{e}_MLx+".format(i=ins, e=en)] = nan
                        model_pars_lum_summary.loc[:, "{i}_{e}_MLx-".format(i=ins, e=en)] = nan

                        global_lum_summary.loc[0, "{i}_{e}_MLx".format(i=ins, e=en)] = nan
                        global_lum_summary.loc[0, "{i}_{e}_MLx+".format(i=ins, e=en)] = nan
                        global_lum_summary.loc[0, "{i}_{e}_MLx-".format(i=ins, e=en)] = nan

                all_inst_med = percentile(all_ins_distribution, 50)
                all_inst_pl = percentile(all_ins_distribution, 50 + (conf_level/2)) - all_inst_med
                all_inst_mi = all_inst_med - percentile(all_ins_distribution, 50 - (conf_level/2))
                global_lum_summary.loc[0, "all_{e}_MLx".format(e=en)] = all_inst_med
                global_lum_summary.loc[0, "all_{e}_MLx+".format(e=en)] = all_inst_pl
                global_lum_summary.loc[0, "all_{e}_MLx-".format(e=en)] = all_inst_mi

            model_pars_lum_summary.to_csv("{0}_{1}_ecf_lum_table.csv".format(obj_id, m_df), index=False)
            global_lum_summary.to_csv("{0}_{1}_comb_pred.csv".format(obj_id, m_df), index=False)

        else:
            model_pars_lum_summary = pd.read_csv("{0}_{1}_ecf_lum_table.csv".format(obj_id, m_df), header="infer")
            global_lum_summary = pd.read_csv("{0}_{1}_comb_pred.csv".format(obj_id, m_df), header="infer")

        lum_dfs[m_df] = {"par_table": model_pars_lum_summary, "marg_pred": global_lum_summary}

    return lum_dfs


def lum_plots(lum_dfs, save_dir, obj_id, chosen_insts):
    en_bands = ["lowen", "highen"]  # Will add wideen to this at some point, 0.5-8.0keV like in the paper

    for m in lum_dfs:
        conf_insts = []
        for ins in chosen_insts:
            for col in lum_dfs[m]["par_table"].columns.values:
                if ins in col:
                    conf_insts.append(ins)
        # Set removes any duplicates, then sorting ensures the same order which should mean colours are more consistent
        conf_insts = list(set(conf_insts))
        conf_insts.sort()
        if not os.path.exists("{sav}/{o}_{mod}_pred_lum_dist.png".format(sav=save_dir, o=obj_id, mod=m)):
            lum_fig, ax_arr = plt.subplots(2, len(en_bands), figsize=(len(en_bands)*8, 16), sharex="col", sharey="col")

            for m_ind, subplot in ndenumerate(ax_arr):
                subplot.minorticks_on()
                subplot.tick_params(axis='both', direction='in', which='both', top=True, right=True)
                subplot.grid(linestyle='dotted', linewidth=1)
                subplot.axis(option="tight")
            lum_fig.suptitle(r"Predicted Luminosity Distributions", fontsize=14, y=0.91, x=0.51)

            # Plot the distributions on histograms
            for en_ind, en in enumerate(en_bands):
                for l_type_ind, l_type in enumerate(["ULx", "MLx"]):
                    cur_ax = ax_arr[l_type_ind, en_ind]
                    if l_type == "ULx":
                        cur_ax.set_title(en + " Upper Limit Luminosities")
                    elif l_type == "MLx":
                        cur_ax.set_title(en + " Median Luminosities")
                    cur_ax.set_xlabel("L$_x$")
                    cur_ax.set_ylabel("Density")
                    colours = []
                    for ins in conf_insts:
                        if l_type == "ULx":
                            cur_ax.hist(lum_dfs[m]["par_table"]["{i}_{e}_ULx".format(i=ins, e=en)], bins="auto",
                                        label="{0} {1}".format(m, ins), alpha=0.7, density=True)
                        elif l_type == "MLx":
                            mlx_hist = cur_ax.hist(lum_dfs[m]["par_table"]["{i}_{e}_MLx".format(i=ins, e=en)],
                                                   bins="auto", label="{0} {1}".format(m, ins), alpha=0.7,
                                                   density=True)
                            colours.append(mlx_hist[2][0].get_facecolor())

                    # Unfortunately have to do two loops of the same parameters because the hists set the ylims
                    for ins_ind, ins in enumerate(conf_insts):
                        if l_type == "MLx":
                            y_pos = cur_ax.get_ylim()[1] * ((ins_ind+1)/6)
                            colour = colours[ins_ind]
                            cur_ax.plot(lum_dfs[m]["marg_pred"]["{i}_{e}_MLx".format(i=ins, e=en)], y_pos, "o",
                                        color=colour, alpha=0.7)  # label="{0} {1} Marg MLx".format(m, ins)

                            cur_ax.errorbar(lum_dfs[m]["marg_pred"]["{i}_{e}_MLx".format(i=ins, e=en)], y_pos,
                                            xerr=[lum_dfs[m]["marg_pred"]["{i}_{e}_MLx-".format(i=ins, e=en)],
                                                  lum_dfs[m]["marg_pred"]["{i}_{e}_MLx+".format(i=ins, e=en)]],
                                            color=colour, alpha=0.7)

                    if l_type == "MLx":
                        y_pos = cur_ax.get_ylim()[1] * 0.7
                        all_line = cur_ax.plot(lum_dfs[m]["marg_pred"]["all_{e}_MLx".format(i=ins, e=en)], y_pos, "d",
                                               alpha=0.7, label="{} Combined".format(m))
                        cur_ax.errorbar(lum_dfs[m]["marg_pred"]["all_{e}_MLx".format(i=ins, e=en)], y_pos,
                                        xerr=[lum_dfs[m]["marg_pred"]["all_{e}_MLx-".format(i=ins, e=en)],
                                              lum_dfs[m]["marg_pred"]["all_{e}_MLx+".format(i=ins, e=en)]],
                                        color=all_line[0].get_color(), alpha=0.7)
                    cur_ax.legend(loc="best")

            plt.savefig("{sav}/{o}_{mod}_pred_lum_dist.png".format(sav=save_dir, o=obj_id, mod=m),
                        bbox_inches='tight')
            plt.close("all")


def run_ecfs(conf_dict, red_list):
    # TODO The way this handles previously run files is a hot mess, and should be redone when I can be bothered
    glob_stacks = {}
    for m in conf_dict["models"]:
        glob_stacks[m] = pd.DataFrame(columns=[conf_dict["id_col"], conf_dict["xmm_obsid_col"]])

    # Ugly method but works because I planned to do multithreading of this bit
    x.Fit.statMethod = "cstat"
    x.Fit.query = "yes"
    x.Xset.chatter = 0

    insts = conf_dict["instruments"]
    ecfs_onwards = tqdm(total=len(ecfs_iterables) * len(insts), desc="Generating ECFS Files")
    for counter, le_pars in enumerate(ecfs_iterables):
        all_inst_spec = {inst: None for inst in insts}
        all_inst_dfs = {inst: None for inst in insts}
        for inst in insts:
            try:
                ecfs_onwards.write("{ob} - {o} - {i}".format(ob=le_pars[1], o=le_pars[2], i=inst))
                spectra, conv_dfs = ecfs_calc(*le_pars, inst, conf_dict["produce_plots"], red_list[counter])
                if None not in spectra.values():
                    all_inst_spec[inst] = spectra
                else:
                    all_inst_spec.pop(inst)

                for entry in deepcopy(conv_dfs):
                    if conv_dfs[entry] is None:
                        conv_dfs.pop(entry)
                if len(conv_dfs) != 0:
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

        # This IndexError exception is for when the ECF calculation has failed (normally due to missing SAS products
        try:
            lum_info = calc_lums(all_inst_dfs, le_pars[-1], le_pars[1], conf_dict["conf_level"],
                                 conf_dict["redshift_col"])
            for m in lum_info:
                df_row = pd.DataFrame(columns=[conf_dict["id_col"], conf_dict["xmm_obsid_col"]],
                                      data=[[le_pars[1], le_pars[2]]])
                for col in lum_info[m]["par_table"].columns.values:
                    if "ULx" in col or "ECF" in col:
                        # Median value across all the model parameters
                        df_row.loc[0, col] = percentile(lum_info[m]["par_table"][col], 50)
                df_row = pd.concat([df_row, lum_info[m]["marg_pred"]], axis=1, sort=False)

                glob_stacks[m] = pd.concat([glob_stacks[m], df_row], axis=0, sort=False)
            if conf_dict["produce_plots"]:
                try:
                    lum_plots(lum_info, le_pars[-1], le_pars[1], conf_dict["instruments"])
                    fake_spec_plots(all_inst_spec, le_pars[-1], le_pars[1])
                except (FileNotFoundError, IndexError) as e:
                    # onwards.write(str(e))
                    pass
        except IndexError:
            pass

    return glob_stacks


def validate_samp_file(conf_dict):
    def rad_to_arcseconds(rads, rad_unit, redshifts):
        # This returns the equivelant to 1 radian of seperation at a given redshift
        ang_diam_dists = Planck15.angular_diameter_distance(redshifts)
        # So here we're finding how many radians our given radius is, then converting to arcseconds
        ang_rads = ((rads * u.Unit(rad_unit)) / ang_diam_dists).decompose().value * (180 / pi) * 3600
        return ang_rads

    samp = pd.read_csv(conf_dict["sample_csv"], header="infer", dtype={config["id_col"]: str, "OBSID": str})

    for entry in conf_dict:
        if "col" in entry and conf_dict[entry] not in samp.columns.values:
            sys.exit("{e} column is missing from the sample csv!".format(e=conf_dict[entry]))

    if "OBSID" not in samp.columns.values:
        sys.exit("OBSID column is missing from the sample csv!")

    len_allowed = len(samp[samp[conf_dict["type_col"]] == "ext"]) + len(samp[samp[conf_dict["type_col"]] == "pnt"])
    if not len_allowed == len(samp):
        sys.exit("Only ext and pnt are allowed in the type column")

    if config["rad_unit"] not in ["arcsecond", "arcsec"]:
        new_rads = rad_to_arcseconds(samp[config["rad_col"]].values, config["rad_unit"],
                                     samp[config["redshift_col"]].values)
        samp[conf_dict["rad_col"]] = new_rads
        conf_dict["rad_unit"] = "arcsec"

    return samp, conf_dict


if __name__ == "__main__":
    # Throws errors if you've not a new enough version of Python
    if sys.version_info[0] < 3:
        sys.exit("You cannot run this tool with Python 2.x")
    elif sys.version_info[1] < 6:
        # warnings.warn("Be wary, before Python 3.6 dictionaries aren't necessarily ordered")
        sys.exit("Be wary, before Python 3.6 dictionaries aren't necessarily ordered")

    # Shouts at the user and tells them what they need to pass, if they don't give any arguments
    required_args = ["Configuration JSON"]
    if len(sys.argv) != len(required_args) + 1:
        print('Please pass the following arguments: ')
        print('{}'.format(", ".join(required_args)))
        sys.exit(1)
    elif os.path.exists(sys.argv[1]):
        config_file = sys.argv[1]
        with open(sys.argv[1], 'r') as conf:
            config = json.load(conf)
        # Goes through the entries and makes sure they're all as they should be, in terms of datatype etc.
        config = validate_config(config)
    else:
        sys.exit('That config file does not exist!')

    # Used to check the SAS_PATH, but there's no guarantee it will be formatted the same across
    #  computers
    # Now we check that SAS is present, then call a terminal command that reports version
    #  and parse the output
    if "SAS_DIR" not in os.environ:
        raise ImportError("Can't find SAS_DIR environment variable, please install SAS.")
    else:
        # This way, the user can just import the SAS_VERSION from this utils code
        sas_out, sas_err = Popen("sas --version", stdout=PIPE, stderr=PIPE, shell=True).communicate()
        sas_v = sas_out.decode("UTF-8").strip("]\n").split('-')[-1]

    if sas_v != "14":
        # SAS 14 is what we use on Apollo, so don't have to regenerate CCFs
        update_ccf = False
    else:
        # Anything else will trigger the regeneration of CCFs, even if its an unexpected result,
        #  means it fails safe and will keep going but err on the side of caution
        update_ccf = True
    print("You're using SAS version {v}".format(v=sas_v))
    print("")
    config["sas_version"] = sas_v

    # Checks that the sample file is correctly structured, and has all the headers it should, also converts radii
    xmm_samp, config = validate_samp_file(config)
    # New directory to store all the SAS products and other files
    if not os.path.exists(config["samp_path"] + "/xmm_spectra_sas{}".format(sas_v)):
        os.mkdir(config["samp_path"] + "/xmm_spectra_sas{}".format(sas_v))

    sasgen_stack = []
    combine_stack = []
    le_flux_stack = []
    ecfs_iterables = []
    redshift_list = []
    onwards = tqdm(desc="Preparing SAS commands for candidates", total=len(xmm_samp))
    for ind, row in xmm_samp.iterrows():
        for x_id in row["OBSID"].split(","):
            s_dir = config["samp_path"] + "xmm_spectra_sas{v}/{o}".format(o=x_id, v=sas_v)

            # This is for the ECFS effort later on
            interim = []
            # Uses HEASOFT to look up nH, then converts to form that xspec wants
            nH = nh_lookup(row[config["ra_col"]], row[config["dec_col"]]) / 10e+21
            par_copy = deepcopy(config["models"])
            for model in list(config["models"].keys()):
                if "nH" in par_copy[model]:
                    par_copy[model]["nH"] = nH

                if config["redshift_col"] in par_copy[model]:
                    par_copy[model][config["redshift_col"]] = row[config["redshift_col"]]
                elif "redshift" in par_copy[model] or "z" in par_copy[model]:
                    onwards.write("Please make the redshift entry in your model the same as your choice for "
                                  "redshift_col config parameter.\nIf you're using a model with no redshift but have "
                                  "named a non-redshift parameter 'z' in your model config, please change it to "
                                  "something else!")
                    sys.exit()

            interim.append(par_copy)
            interim.append(row[config["id_col"]])
            interim.append(x_id)
            interim.append(s_dir)

            if not os.path.exists(s_dir):
                os.mkdir(s_dir)

            src = config["xmm_data_path"] + x_id + "/"
            src = os.path.abspath(src) + "/"
            # Here we set up the paths of all the important files, the im and expmap paths will be modified by
            # command_stack_maker
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
                     "emos2_highen_expmap": "{s}images/{o}-2.00-10.00keV-mos2_merged_expmap.fits".format(s=src, o=x_id),
                     "epn_att": "{s}epchain/P{o}OBX000ATTTSR0000.FIT".format(s=src, o=x_id),
                     "emos1_att": "{s}emchain/P{o}OBX000ATTTSR0000.FIT".format(s=src, o=x_id),
                     "emos2_att": "{s}emchain/P{o}OBX000ATTTSR0000.FIT".format(s=src, o=x_id)}

            # Input radius in arcseconds
            try:
                src_reg, excl_reg = coords_rad_regions(config, x_id, f_loc["epn_lowen_im"], row[config["ra_col"]],
                                                       row[config["dec_col"]], row[config["rad_col"]],
                                                       row[config["type_col"]], config["force_rad"], s_dir,
                                                       row[config["id_col"]])
                # Makes sure there are no problems before adding to the list that will be run
                ecfs_iterables.append(interim)
                redshift_list.append(row[config["redshift_col"]])
            except (FileNotFoundError, ValueError, KeyError) as error:
                onwards.write(str(error))
                break

            for instrument in config["instruments"]:
                command, f_loc = command_stack_maker(config, row[config["id_col"]], x_id, f_loc, src_reg, excl_reg,
                                                     s_dir, instrument, row[config["type_col"]],
                                                     config["generate_combined"], row[config["redshift_col"]])
                sasgen_stack.append(command)
                for energy in ["lowen", "highen"]:
                    command = photon_rate_command(config, f_loc, row[config["id_col"]], src_reg, excl_reg, s_dir,
                                                  instrument, energy, row[config["type_col"]])
                    le_flux_stack.append(command)

            command = combine_stack_maker(row[config["id_col"]], x_id, f_loc, s_dir)
            combine_stack.append(command)

        onwards.update(1)
    onwards.close()

    sas_pool(sasgen_stack, "Generating SAS Products", config, cpu_count=config["allowed_cores"])
    if config["generate_combined"]:
        sas_pool(combine_stack, "Combining Spectra", config, cpu_count=config["allowed_cores"])
    sas_pool(le_flux_stack, "Measuring upper limit photon rates", config, cpu_count=config["allowed_cores"])
    model_global_dfs = run_ecfs(config, redshift_list)

    for m in model_global_dfs:
        model_global_dfs[m].to_csv(config["samp_path"] +
                                   "xmm_spectra_sas{0}/{1}_lums.csv".format(config["sas_version"], m), index=False)

