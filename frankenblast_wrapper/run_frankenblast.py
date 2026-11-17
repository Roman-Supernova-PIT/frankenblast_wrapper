#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:36:33 2026

@author: hmfausey
"""

# Imports
import importlib
import dustmaps.sfd
import os
import error_estimation
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from frankenblast.classes import Transient, Host
from sedpy.observate import load_filters
from frankenblast.new_roman_lsst_fit_host_sed import fit_host  # , maggies_to_asinh
import IO_functions as IO
import time
import fsps
import prospect.sources.ssp_basis as ssp_basis

ssp_basis.fsps = fsps

os.environ["SPS_HOME"] = "/Users/hfausey/fsps/fsps"
os.environ["SBIPP_ROOT"] = "/Users/hfausey/frankenblast_data/data/SBI"
os.environ["SBIPP_PHOT_ROOT"] = "/Users/hfausey/frankenblast_data/data/sbipp_phot"
os.environ["SBIPP_TRAINING_ROOT"] = "/Users/hfausey/frankenblast_data/data/sbi_training_sets"
os.environ["SED_OUTPUT_ROOT"] = "/Users/hfausey/frankenblast_wrapper_special/src/frankenblast_wrapper/my_files/results/roman_deep_errfloor_comissioning"


# All filters and names
frankenblast_trained_wide = load_filters(
    [
        "lsst_u",
        "lsst_g",
        "lsst_r",
        "lsst_i",
        "lsst_z",
        "lsst_y",
        "roman_R062_wide",
        "roman_Z087_wide",
        "roman_Y106_wide",
        "roman_J129_wide",
        "roman_H158_wide",
    ]
)
frankenblast_trained_deep = load_filters(
    [
        "lsst_u",
        "lsst_g",
        "lsst_r",
        "lsst_i",
        "lsst_z",
        "lsst_y",
        "roman_Z087_deep",
        "roman_Y106_deep",
        "roman_J129_deep",
        "roman_H158_deep",
        "roman_F184_deep",
    ]
)

frankenblast_roman_only_wide = load_filters(
    [
        "roman_R062_wide",
        "roman_Z087_wide",
        "roman_Y106_wide",
        "roman_J129_wide",
        "roman_H158_wide",
    ]
)
frankenblast_roman_only_deep = load_filters(
    [
        "roman_Z087_deep",
        "roman_Y106_deep",
        "roman_J129_deep",
        "roman_H158_deep",
        "roman_F184_deep",
    ]
)

frankenblast_subaru_wide = load_filters(
    [
        "hsc_g",
        "hsc_r2",
        "hsc_i2",
        "roman_R062_wide",
        "roman_Z087_wide",
        "roman_Y106_wide",
        "roman_J129_wide",
        "roman_H158_wide",
    ]
)
frankenblast_subaru_deep = load_filters(
    [
        "hsc_g",
        "hsc_r2",
        "hsc_i2",
        "roman_Z087_deep",
        "roman_Y106_deep",
        "roman_J129_deep",
        "roman_H158_deep",
        "roman_F184_deep",
    ]
)


frankenblast_names_wide = np.array(
    [
        "lsst_u",
        "lsst_g",
        "lsst_r",
        "lsst_i",
        "lsst_z",
        "lsst_y",
        "R062",
        "Z087",
        "Y106",
        "J129",
        "H158",
    ]
)
frankenblast_names_deep = np.array(
    [
        "lsst_u",
        "lsst_g",
        "lsst_r",
        "lsst_i",
        "lsst_z",
        "lsst_y",
        "Z087",
        "Y106",
        "J129",
        "H158",
        "F184",
    ]
)

frankenblast_roman_only_names_wide = np.array(["R062", "Z087", "Y106", "J129", "H158"])
frankenblast_roman_only_names_deep = np.array(["Z087", "Y106", "J129", "H158", "F184"])

subaru_names_wide = np.array(
    ["hsc_g", "hsc_r2", "hsc_i2", "R062", "Z087", "Y106", "J129", "H158"]
)
subaru_names_deep = np.array(
    ["hsc_g", "hsc_r2", "hsc_i2", "Z087", "Y106", "J129", "H158", "F184"]
)


def run(
    diaobject_id,
    provenance_id,
    galID,
    field,
    RA,
    Dec,
    magnitudes,
    magerrs,
    LSST_data,
    subaru_data,
    redshift,
    filetype="parquet",
):
    """Runs frankenblast and saves results.

    Loads in photometry along with galaxy, field, and external data information,
    reformats inputs, identifies the correct frankenblast model, and then runs
    frankenblast. It packages results (sometimes with corrected uncertainties)
    and saves parameters to the database and to an output parquet file.

    Parameters
    ----------
    diaobject_id : UUID
        The identifier associated with the supernova
    provenance_id : UUID
        Identifier for host galaxy photometry
    galID : int
        The galaxy identifier/id number
    field : {'wide', 'deep'}
        Which roman field the galaxy was observed with
    RA : float
        Right ascension of galaxy
    Dec: float
        Declination of galaxy
    magnitudes : array_like
        All host photometry
    magerrs : array_like
        All host photometry uncertainties (1-sigma)
    LSST_data : bool
        Whether LSST data is included
    subaru_data : bool
        Whether Subaru data is included
    redshift : float
        Host galaxy redshift (if known. Can be None if unknown)
    filetype : {'parquet', 'asdf'}
        Whether output file should be saved in parquet or asdf

    Returns
    -------
    out : None
        Host galaxy parameter estimates are saved to the database or to a
        parquet file.
    """

    ##This if/else block is determining which trained model it needs to pull
    # from depending on what filters are being used, if LSST or subaru data is
    # available etc.
    
    print(LSST_data, subaru_data)
    
    if LSST_data:
        if field == "wide":

            trained_filters = frankenblast_trained_wide
            filter_names = frankenblast_names_wide

            if redshift == None:
                train_fname = "simple_lsst_wide"

            elif redshift != None:
                train_fname = "simple_lsst_wide_zfixed"
            else:
                raise Exception("Redshift error")

        elif field == "deep":

            trained_filters = frankenblast_trained_deep
            filter_names = frankenblast_names_deep

            if redshift == None:
                train_fname = "simple_lsst_deep"

            elif redshift != None:
                train_fname = "simple_lsst_deep_zfixed"
            else:
                raise Exception("Redshift error")
        else:
            raise Exception("Field error")

    elif subaru_data:

        if field == "wide":

            trained_filters = frankenblast_subaru_wide
            filter_names = subaru_names_wide

            if redshift == None:
                train_fname = "simple_subaru_wide"
            elif redshift != None:
                train_fname = "simple_subaru_wide_zfixed"
            else:
                raise Exception("Redshift error")

        elif field == "deep":

            trained_filters = frankenblast_subaru_deep
            filter_names = subaru_names_deep

            if redshift == None:
                train_fname = "simple_subaru_deep"

            elif redshift != None:
                train_fname = "simple_subaru_deep_zfixed"
            else:
                raise Exception("Redshift error")
        else:
            raise Exception("Field error")

    elif (not LSST_data) and (not subaru_data):
        print("Were here!")
        if np.sum(np.isnan(magnitudes)) > 2:
            print("Not enough photometry bands for estimate. Skipping...")
            return None

        if field == "wide":
            trained_filters = frankenblast_roman_only_wide
            filter_names = frankenblast_roman_only_names_wide
            if redshift == None:
                train_fname = "simple_roman_wide"
                print(train_fname)
            elif redshift != None:
                train_fname = "simple_roman_wide_zfixed"
            else:
                raise Exception("Redshift error")

        elif field == "deep":

            trained_filters = frankenblast_roman_only_deep
            filter_names = frankenblast_roman_only_names_deep
            print(frankenblast_roman_only_deep, frankenblast_roman_only_names_deep)

            if redshift == None:
                train_fname = "simple_roman_deep_errfloor"
            elif redshift != None:
                train_fname = "simple_roman_deep_zfixed_errfloor"
            else:
                raise Exception("Redshift error")

        else:
            raise Exception("Field error")

    else:
        raise Exception("LSST/Subaru/Roman data error")

    ##These paramters are set for the current frankenblast models
    redshift_prior = "FastUniform 0-3"
    sfr_model = 4
    gal_age_model = "tage"
    AGN_info = False

    ##Create transient for frankenblast input
    transient_class = Transient(
        name=str(galID),
        coordinates=SkyCoord(ra=RA * u.deg, dec=Dec * u.deg, frame="icrs"),
        transient_redshift=redshift,
        milkyway_dust_reddening=None,  # Initially None
    )

    ##Create the host galaxy for frankenblast
    best_host = Host(
        name=transient_class.name,
        redshift=transient_class.transient_redshift,
        coordinates=SkyCoord(ra=RA * u.deg, dec=Dec * u.deg, frame="icrs"),
    )

    ##Set the transients host galaxy
    transient_class.host = best_host

    ##Array to keep track of photometry
    all_phot = []

    ##Convert to maggies and magerrs
    maggies = 10 ** (-0.4 * magnitudes)
    maggies_err = 10 ** (-0.4 * magnitudes) - 10 ** (-0.4 * (magnitudes + magerrs))
    
    print(trained_filters, filter_names)
    
    ##Fill `all_phot` with relevant info
    for i in range(len(filter_names)):
        dataframe = {
            "magnitude": magnitudes[i],
            "magnitude_error": magerrs[i],
            "flux": maggies[i],
            "flux_error": maggies_err[i],
        }

        all_phot.append(dataframe)

    ##Set the photometry and filters
    transient_class.host_photometry = all_phot
    transient_class.host_phot_filters = trained_filters

    ##Make sure these are set. I don't know why it is so finicky... but it is
    SBIPP_ROOT = os.environ["SBIPP_ROOT"]
    SBIPP_PHOT_ROOT = os.environ["SBIPP_PHOT_ROOT"]
    SBIPP_TRAINING_ROOT = os.environ["SBIPP_TRAINING_ROOT"]
    SED_OUTPUT_ROOT = os.environ["SED_OUTPUT_ROOT"]

    ##Set up SBI parameters
    sbi_params = {
        "anpe_fname_global": f"{SBIPP_ROOT}/SBI_model_{train_fname}_global.pt",  # trained sbi model
        "train_fname_global": f"{SBIPP_PHOT_ROOT}/sbi_{train_fname}_global.h5",  # training set
        "nhidden": 500,  # architecture of the trained density estimator
        "nblocks": 15,  # architecture of the trained density estimator
    }

    ##Initialize fsps - is this needed?
    sp = fsps.StellarPopulation()

    aperture_type = "global"

    ##Define output files
    hfile = f"{SED_OUTPUT_ROOT}/{transient_class.name}/{transient_class.name}_{aperture_type}.h5"
    root = f"{SED_OUTPUT_ROOT}/{transient_class.name}/"

    ##Run SBI
    start = time.time()

    ##Run frankenblast
    flag = fit_host(
        transient_class,
        sbi_params=sbi_params,
        fname=train_fname,
        all_filters=trained_filters,
        extinction_correct=True,
        sbipp=True,
        aperture_type=aperture_type,
        aperture=transient_class.global_apertures,
        save=True,
    )

    ##Keep track of time
    end = time.time()
    length = end - start

    print("It took", length / 60, "minutes to run SBIPP!")
    print("Host SED fit completed.")

    ##Handle data and uncertainties

    """NEED TO SET W SOMEWHERE ONCE WE HAVE A GOOD HANDNLE ON IT"""
    ##The fit needs to have happened to save values and update uncertainties
    if os.path.exists(hfile):
        output_data = error_estimation.get_error(galID, hfile, root, train_fname, w=1)

        # Save off the metadata for this specific SN+galaxy+photometry+fit+version+etc.
        if filetype == "parquet":
            IO.save_meta_parquet(
                diaobject_id,
                provenance_id,
                LSST_data,
                subaru_data,
                redshift_prior,
                sfr_model,
                gal_age_model,
                AGN_info,
                filter_names,
                magnitudes,
                magerrs,
                root,
            )
            IO.save_data_parquet(output_data, root)

        elif filetype == "asdf":
            IO.save_all_asdf(
                output_data,
                diaobject_id,
                provenance_id,
                LSST_data,
                subaru_data,
                redshift_prior,
                sfr_model,
                gal_age_model,
                AGN_info,
                filter_names,
                magnitudes,
                magerrs,
                root,
            )
        elif filetype == "database":
            # Blah blah save to database
            raise Exception(("No database yet"))
        else:
            raise Exception("Invalid file type for saving results")

    return None
