from astropy.io import fits
from astropy import wcs
import sys
import pandas as pd
from tqdm import tqdm

up_lim_path = "/run/user/1000/gvfs/sftp:host=apollo/mnt/pact/dt237/pag22/vernon_clusters/upper_limit/"
spec_lx_path = "/run/user/1000/gvfs/sftp:host=apollo/mnt/pact/dt237/pag22/apec_test/clusters/"


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Please pass path of sample and upper or spec depending on if upper lims or spectral")

    sample = pd.read_csv(sys.argv[1], header="infer", dtype={"OBSID": str, "objid": str})
    mod_samp = pd.DataFrame(columns=list(sample.columns.values).append("rad"))

    onwards = tqdm(desc="Looking up regions", total=len(sample))
    for ind, row in sample.iterrows():
        if sys.argv[2] == "upper":
            try:
                full_path = up_lim_path + row["OBSID"] + "/"
                im_fits = fits.open(full_path + "{}-0.50-2.00keV-pn_merged_img.fits".format(row["OBSID"]))
                im_head = im_fits[0].header
                # Reads in WCS for ra,dec->pixel and pixel->skycoords
                deg_pix_wcs = wcs.WCS(im_head)
                pix_sky_wcs = wcs.WCS(im_head, key='L')
                im_fits.close()

                reg_file = full_path + "{obs}_rm_{objid}_phys.reg".format(obs=row["OBSID"], objid=row["objid"])
                with open(reg_file, 'r') as reg:
                    lines = reg.readlines()

                sky_x, sky_y, sky_r = [float(el) for el in lines[1].split("(")[-1].split(")")[0].split(",")]
                pix_x, pix_y = pix_sky_wcs.all_world2pix(sky_x, sky_y, 1)
                pix_x_off, pix_y_off = pix_sky_wcs.all_world2pix(sky_x+sky_r, sky_y, 1)
                pix_r = abs(pix_x_off - pix_x)

                deg_x, deg_y = deg_pix_wcs.all_pix2world(pix_x, pix_y, 1)
                deg_x_off, deg_y_off = deg_pix_wcs.all_pix2world(pix_x+pix_r, pix_y, 1)
                deg_r = abs(deg_x_off - deg_x) * 60*60
                row["rad"] = deg_r
                mod_samp = mod_samp.append(row)
            except (FileNotFoundError, KeyError):
                pass
        elif sys.argv[2] == "spec":
            reg_path = spec_lx_path + row["name"] + "-{}".format(round(row["redshift"], 3)) + "/spec/source.reg"
            with open(reg_path, 'r') as reggy:
                lines = reggy.readlines()

            rad = float(lines[0].split("ellipse(")[-1].split(")")[0].split(",")[2].split("'")[0]) * 60
            row["rad"] = rad
            mod_samp = mod_samp.append(row)

        onwards.update(1)

    mod_samp.to_csv(sys.argv[1], index=False)
