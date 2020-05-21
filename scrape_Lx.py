"""
This code scrapes Lx and ECF values for a specific instance of a model given to XULL. Copyright XMM Cluster Survey
"""
import pandas as pd
import os
import sys


if __name__ == "__main__":
    # Pass the path of the XMM Spectra directory generated for the XULL you're interested in
    if len(sys.argv) != 2:
        sys.exit("Please pass the xmm_spectra_sas{{version}} path created by XULL")
    elif not os.path.exists(sys.argv[1]):
        sys.exit("The path you passed does not exist!")
    else:
        top_level_path = os.path.abspath(sys.argv[1]) + "/"

    os.chdir(top_level_path)
    files_in_dir = [el for el in os.listdir(".") if "_lums.csv" in el]
    models = [el.split("_lums")[0] for el in files_in_dir]

    print("Here are your model options:")
    for m_ind, m in enumerate(models):
        print("{})".format(m_ind), m)

    m_choice = int(input("Select a model using its number: "))
    if m_choice > len(models)-1:
        sys.exit("That wasn't a valid option")

    obsid_dirs = [el for el in os.listdir(".") if len(el) == 10 and "." not in el]

    found_valid_file = False
    count = 0
    while not found_valid_file and count < len(obsid_dirs):
        obsid_files = [el for el in os.listdir(obsid_dirs[count]) if "{}_ecf_lum_table".format(models[m_choice]) in el]
        if len(obsid_files) > 0:
            found_valid_file = True
        else:
            count += 1

    ecf_table = pd.read_csv("{o}/{f}".format(o=obsid_dirs[count], f=obsid_files[0]))
    for col_ind, col in enumerate(ecf_table.columns.values):
        if "ph_per" in col:
            break

    pars = ecf_table.columns.values[:col_ind]
    give_choices = []
    for par in pars:
        if len(set(ecf_table[par].values)) != 1:
            give_choices.append(par)

    print("The following parameter columns have multiple values in them; {}\n".format(", ".join(give_choices)))
    v_choices = {}
    for par in give_choices:
        print("Your choices for {} are:".format(par))
        for val_ind, val in enumerate(ecf_table[par].values):
            print("{})".format(val_ind), val)
        v_choice = int(input("Select a value using its number: "))
        if v_choice > len(ecf_table[par].values)-1:
            sys.exit("That was not a valid choice")
        print("")
        v_choices[par] = ecf_table[par].values[v_choice]

    to_cut = ecf_table.copy()
    for par in v_choices:
        to_cut = to_cut[to_cut[par] == v_choices[par]]

    file_index = to_cut.index[0]
    del to_cut

    id_col = pd.read_csv("{}_lums.csv".format(models[m_choice]), header="infer").columns.values[0]

    scraped_df = pd.DataFrame(columns=[id_col] + list(ecf_table.columns.values))
    for obs_dir in obsid_dirs:
        obsid_files = [el for el in os.listdir(obs_dir) if "{}_ecf_lum_table".format(models[m_choice]) in el]
        for f in obsid_files:
            temp_f = pd.read_csv(obs_dir + "/" + f, header="infer")
            temp_f.insert(0, column=id_col, value=f.split("_")[0])
            scraped_df = scraped_df.append(temp_f.iloc[file_index])

    name = ""
    for par in v_choices:
        name += par+str(v_choices[par])+"_"

    scraped_df.to_csv("{m}_{n}scraped.csv".format(m=models[m_choice], n=name), index=False)




