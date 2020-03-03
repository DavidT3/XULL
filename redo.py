"""
This code is a supplement to XULL, and will fit to a sample of spectra, calculating their true ECF values.
Allows us to get an estimate of the luminosity of a source without a detection. Copyright XMM Cluster Survey
"""
import sys
import os
import json
from copy import deepcopy
from tqdm import tqdm
import subprocess
import pandas as pd
import xspec as x


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
            if isinstance(m_dict[par], list):
                sys.exit("Model parameters passed into REDO are no allowed to be lists, you must provide a single "
                         "estimate as a start point for the fit.")
            elif par.lower() == "nh":
                new_m_dict["nH"] = m_dict[par]
            else:
                new_m_dict[par] = m_dict[par]
        return new_m_dict

    required_head = ["sample_csv", "id_col", "ra_col", "dec_col", "models"]

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

    if not isinstance(conf_dict["id_col"], str):
        sys.exit("id_col must be a string")

    if not isinstance(conf_dict["ra_col"], str):
        sys.exit("ra_col must be a string")

    if not isinstance(conf_dict["dec_col"], str):
        sys.exit("dec_col must be a string")

    if not isinstance(conf_dict["models"], dict):
        sys.exit("models must be supplied as a dictionary, even if you are only using one")
    else:
        print("REDSHIFT IS READ FROM THE SAMPLE, AND NH IS READ FROM HEASOFT, VALUES IN MODEL WILL BE DISCARDED\n")
        # print("PARAMETERS MUST BE IN THE ORDER THAT XSPEC EXPECTS THEM")
        for entry in conf_dict["models"]:
            conf_dict["models"][entry] = model_parser(conf_dict["models"][entry])

    if not isinstance(conf_dict["redshift_col"], str):
        sys.exit("redshift_col must be a string")

    if not isinstance(conf_dict["type_col"], str):
        sys.exit("type_col must be a string")

    if not isinstance(conf_dict["xmm_obsid_col"], str):
        sys.exit("xmm_obsid_col must be a string")

    return conf_dict


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


def ecfs_calc(file, model_name, parameters, obj_id, obs_id, save_dir, ins, redshift):
    os.chdir(save_dir)
    x_mod = x.Model(model_name)
    x_mod.setPars(*list(parameters.values()))
    x_mod(3).frozen = False
    if "abs" in model_name:
        # This should zero nH, which means the calculated fluxes will be unabsorbed (according to Paul)
        x_mod(1).frozen = True

    spectrum = x.Spectrum(file)
    spectrum.ignore("**-0.3 10.0-**")
    x.Fit.perform()

    # Now to find source frame limits
    z_lowen = [limit / (redshift + 1) for limit in [0.5, 2.0]]
    z_highen = [limit / (redshift + 1) for limit in [2.0, 10.0]]
    # Count rate measurements have to come before I zero the absorption, as we'll be measuring absorbed c/r

    # Luminosity errors also have to come before I zero the absorption
    x.AllModels.calcLumin("{l} {u} {red} err".format(l=z_lowen[0], u=z_lowen[1], red=redshift))
    lowen_lx_p = (spectrum.lumin[2] * 1e+44) - (spectrum.lumin[0] * 1e+44)
    lowen_lx_m = (spectrum.lumin[0] * 1e+44) - (spectrum.lumin[1] * 1e+44)

    x.AllModels.calcLumin("{l} {u} {red} err".format(l=z_highen[0], u=z_highen[1], red=redshift))
    highen_lx_p = (spectrum.lumin[2] * 1e+44) - (spectrum.lumin[0] * 1e+44)
    highen_lx_m = (spectrum.lumin[0] * 1e+44) - (spectrum.lumin[1] * 1e+44)

    # Have to use an ignore to get a count rate for the energy range I care about
    spectrum.ignore("**-{l} {u}-**".format(l=z_lowen[0], u=z_lowen[1]))
    # the 0th element of rate is the background subtracted rate, but doesn't matter -> no background!
    lowen_rate = spectrum.rate[0]

    # Now reset the ignore to ignore nothing
    spectrum.notice("all")
    # And now ignore to get the high energy range
    spectrum.ignore("**-{l} {u}-**".format(l=z_highen[0], u=z_highen[1]))
    highen_rate = spectrum.rate[0]

    fitted_pars = [x_mod(par_ind).values[0] for par_ind in range(1, x_mod.nParameters+1)]
    spectrum.notice("all")
    # Sort of janky way of finding of one of nH absorption models is being used, definitely not rigorous
    if "abs" in model_name:
        # This should zero nH, which means the calculated fluxes will be unabsorbed (according to Paul)
        x_mod.setPars(0)

    x.AllModels.calcFlux("{l} {u}".format(l=z_lowen[0], u=z_lowen[1]))
    lowen_flux = spectrum.flux[0]
    x.AllModels.calcLumin("{l} {u} {red} err".format(l=z_lowen[0], u=z_lowen[1], red=redshift))
    lowen_lx = spectrum.lumin[0] * 1e+44

    x.AllModels.calcFlux("{l} {u}".format(l=z_highen[0], u=z_highen[1]))
    highen_flux = spectrum.flux[0]
    x.AllModels.calcLumin("{l} {u} {red} err".format(l=z_highen[0], u=z_highen[1], red=redshift))
    highen_lx = spectrum.lumin[0] * 1e+44

    x.AllData.clear()
    x.AllModels.clear()

    return [lowen_rate, lowen_flux, highen_rate, highen_flux], fitted_pars, [lowen_lx, lowen_lx_m, lowen_lx_p], \
           [highen_lx, highen_lx_m, highen_lx_p]


if __name__ == "__main__":
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

    # This isn't particularly universal for computers other than Kraken, but figures out which version of SAS is loaded
    sas_v = os.environ["SAS_PATH"].lower().split("/sas_")[-1].split(".")[0]
    if sas_v == "17":
        # Newer SAS than Apollo means we have to regenerate all the CCF files before ARF/RMFGEN
        update_ccf = True
    elif sas_v == "14":
        update_ccf = False
    else:
        sys.exit("Don't recognise the SAS version you're using, talk to David he can fix this")
    print("You're using SAS version {v}".format(v=os.environ["SAS_PATH"].lower().split("/sas_")[-1].split(".")[0]))
    print("")
    config["sas_version"] = sas_v

    # Checks that the sample file is correctly structured, and has all the headers it should, also converts radii
    xmm_samp = pd.read_csv(config["sample_csv"], header="infer", dtype={config["xmm_obsid_col"]: str,
                                                                        config["id_col"]: str})
    # New directory to store all the SAS products and other files
    if not os.path.exists(config["samp_path"] + "/xmm_spectra_sas{}".format(sas_v)):
        sys.exit("You have to run xull.py first to generate the SAS products.")

    x.Fit.statMethod = "cstat"
    x.Fit.query = "yes"
    x.Xset.chatter = 0
    ecfs_table_dict = {}
    for model in list(config["models"].keys()):
        ecfs_table_dict[model] = pd.DataFrame(columns=[config["id_col"], config["xmm_obsid_col"]])

    onwards = tqdm(desc="Fitting Spectra", total=len(xmm_samp))
    for ind, row in xmm_samp.iterrows():
        for x_id in row["OBSID"].split(","):
            s_dir = config["samp_path"] + "xmm_spectra_sas{v}/{o}".format(o=x_id, v=sas_v)
            group_spec = [el for el in os.listdir(s_dir) if "grp" in el]
            instruments = [el.split("_grp")[0].split("_")[-1] for el in group_spec]

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

                ecf_table_row = pd.DataFrame(columns=[config["id_col"], config["xmm_obsid_col"]],
                                             data=[[row[config["id_col"]], x_id]])
                for ins_ind, instrument in enumerate(instruments):
                    spec_file = group_spec[ins_ind]
                    try:
                        ecf_components, fit_pars, low_lx, high_lx = ecfs_calc(spec_file, model, par_copy[model],
                                                                              row[config["id_col"]], x_id, s_dir,
                                                                              instrument, row[config["redshift_col"]])
                        lowen_ecf = ecf_components[1]/ecf_components[0]
                        highen_ecf = ecf_components[3]/ecf_components[2]
                        ecf_table_row.loc[0, instrument + "_lowen_ECF"] = lowen_ecf
                        ecf_table_row.loc[0, instrument + "_highen_ECF"] = highen_ecf
                        ecf_table_row.loc[0, instrument + "_lowen_Lx"] = low_lx[0]
                        ecf_table_row.loc[0, instrument + "_lowen_Lx_m"] = low_lx[1]
                        ecf_table_row.loc[0, instrument + "_lowen_Lx_p"] = low_lx[2]
                        ecf_table_row.loc[0, instrument + "_highen_Lx"] = high_lx[0]
                        ecf_table_row.loc[0, instrument + "_highen_Lx_m"] = high_lx[1]
                        ecf_table_row.loc[0, instrument + "_highen_Lx_p"] = high_lx[2]

                        ins_fit_pars = [instrument + "_" + p for p in list(par_copy[model].keys())]
                        for new_par_ind, new_par_name in enumerate(ins_fit_pars):
                            ecf_table_row.loc[0, new_par_name] = fit_pars[new_par_ind]

                    except Exception as error:
                        onwards.write("{i} - " + str(error))

                ecfs_table_dict[model] = pd.concat([ecfs_table_dict[model], ecf_table_row], axis=0, sort=False)
        onwards.update(1)

    savey = config["samp_path"] + "/xmm_spectra_sas{}".format(sas_v)
    for model in ecfs_table_dict:
        ecfs_table_dict[model].to_csv("{0}/{1}_true_ECFs.csv".format(savey, model), index=False)



