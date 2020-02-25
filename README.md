# David's handy how-to guide

## Dependencies
* Python 3.6+
* The usual suspects; numpy, scipy, and matplotlib.
* imageio
* Astropy
* TQDM
* pyXSPEC - can be a bit of a pain
* pandas

## Expected sample format
The input must be a csv with the following information:
* Some form of unique object identifier e.g. MEM_MATCH_ID, SDSS Object ID.
* An XMM ObsID (or several of them separated by commas but still passed as a string).
* Right-ascension and declination columns.
* A redshift column.
* A radius column, values can be in arcsecond or any valid astropy distance unit.
* Column describing the source type, allowed to have values of either ext or pnt.

This file MUST have a header, but you can give the columns any names you want, you just have to tell the code what 
they are in the config file.

If you don't care about the redshift of your objects, set the values in your column to zero, so rest frame values will 
be calculated.

## Structuring the config file
The configuration file is passed to the script when you call it, and gives the code a lot of information that it needs.

**FYI: In json files boolean variables must be all lowercase, and lists can only be defined with square brackets** 

It must be a json with the following entries:
* sample_csv - The path to the csv describing your sample
* xmm_data_path - Where all the XMM ObsID directories live
* xmm_reg_path - The path to the XAPA id_results directory
* id_col - Name of the object id column in your sample
* xmm_obsid_col - Name of the ObsID column in your sample
* ra_col - Name of the RA column in your sample
* dec_col - Name of the dec column in your sample
* redshift_col - Name of the redshift column in your sample
* type_col - Name of the type column in your sample
* rad_col - Name of the radius column in your sample
* rad_unit - The units of the radius column, e.g. arcsec, arcsecond, kpc, Mpc.
* generate_combined - At this time should always be false, defines whether to generate combined spectra from all 
three cameras
* force_rad - A boolean option, if true the code will always use the radius provided in your sample csv. 
If false, it will use the radius of a matching XAPA region, if one exists.
* allowed_cores - How many cores the code is allowed to use at once, be careful on Apollo, as this code doesn't 
currently submit jobs, so it should be limited to the number allocated to your interactive session.
* conf_level - Desired confidence level for the median luminosity calculations, 90 for instance.
* produce_plots - Boolean option, this should be set to false if you're running in a headless screen.
* back_outer_factor - Scaling factor applied to object radius to find outer radius of the background annulus.
* models - One of the most important, this defines the models that you wish to use to calculate your various predicted 
luminosities. This is the most complicated so an example is given below.

Here is an example model definition:
```json
"models": {"tbabs*apec": {"nh": -1, "kT": [1, 10, 15], "abund": 0.3, "redshift": -1, "Norm": 1},
           "tbabs*zpowerlw": {"nh": -1, "PhoIndex": [-3, 3, 10], "redshift": -1, "Norm": 1}}
```

The name of the model (i.e. tbabs*apec) must be that of a valid XSPEC model, AND THE PARAMETERS YOU GIVE IT *MUST* BE 
IN THE ORDER EXPECTED BY XSPEC - OTHERWISE EVERYTHING IS LIKELY TO GO WRONG. 

Any nH parameter should be named nh or nH, 
and will then be looked up for the RA and DEC of an individual object (hence why nH is -1 in the above example). 

Any redshift parameter should be given the same as you gave the redshift column in your sample, as the code will fetch 
the value from there. 

If you wish to explore a range of values for a parameter that isn't nH or redshift, you can pass a 
list [start, stop, num_steps], so for instance the tbabs*apec model will try 15 different values of kT between 
1keV and 10keV.

As seen above, you can also get the code to try multiple models, simply pass them as another member of the models 
dictionary.

## Running the code

```python
python xmm_lums.py sample/test_paul_clusters/xull_config.json5
```

## Outputs
The main output files can be found at the end of the run in the xmm_spectra_sas{version} folder. 
There will be one for each model you included, and it will be named {model}_lums.csv. Hopefully all the headers are 
self-explanatory, but one thing to note is that the ULx value for a given energy and instrument is the median value of 
all those measured for the different instances of your chosen model.

Otherwise, the output files for each object can be found in the ObsID folders:
* {id}_{instrument}\_{energy}_output.txt files are the eregionanalyse results.
* {id}_{instrument}_sasgen.log files are logs of the SAS product generation stage.
* {id}_{model}_ecf_lum_table.csv is the equivalent of the old XCS ECF files, with all the values for different model 
instances.
* {id}_{model}_comb_pred.csv are basically the separate rows of the top level output file.
* {id}_{model}.gif is only generated if produce_plots is true, and is a gif of the different model instances.
* {id}_{model}_pred_lum_dist.png is only generated if produce_plots is true, and is a plot of the result distributions.