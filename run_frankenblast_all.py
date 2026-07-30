#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:36:33 2026

@author: hmfausey
"""

# Imports
import importlib

# %load_ext autoreload
# %autoreload 2

import dustmaps.sfd

dustmaps.sfd.fetch()

import os

# Training sets

##All of these are going to be changed once everything ends up...well...whereever
# its going

os.environ["SPS_HOME"] = "/Users/hfausey/fsps/fsps"
os.environ["SBIPP_ROOT"] = "/Users/hfausey/frankenblast_data/data/SBI"
os.environ["SBIPP_PHOT_ROOT"] = "/Users/hfausey/frankenblast_data/data/sbipp_phot"
os.environ["SBIPP_TRAINING_ROOT"] = (
    "/Users/hfausey/frankenblast_data/data/sbi_training_sets"
)
os.environ["SED_OUTPUT_ROOT"] = "/Users/hfausey/frankenblast_data/data/sed_output"

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from frankenblast.classes import Transient, Host
from sedpy.observate import load_filters
from frankenblast.new_roman_lsst_fit_host_sed import fit_host #, maggies_to_asinh
import time

import fsps
import prospect.sources.ssp_basis as ssp_basis


ssp_basis.fsps = fsps

##Set up arrays of filter sets being used in different cases

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


def run(galID, field, RA, Dec, magnitudes, magerrs, LSST_data, subaru_data, redshift):

    ##This function is the main wrapper for frankenblast

    ##This if/else block is determining which trained model it needs to pull from
    # depending on what filters are being used, if LSST or subaru data is available
    # etc.

    if LSST_data:
        if field == "wide":

            trained_filters = frankenblast_trained_wide
            filter_names = frankenblast_names_wide

            if redshift == None:
                train_fname = "zfree_RR_RZYJH"

            elif redshift != None:
                train_fname = "zfix_RR_RZYJH"
            else:
                raise Exception("Redshift error")

        elif field == "deep":
            trained_filters = frankenblast_trained_deep
            filter_names = frankenblast_names_deep

            if redshift == None:
                train_fname = "zfree_RR_ZYJHF"

            elif redshift != None:
                train_fname = "zfix_RR_ZYJHF"
            else:
                raise Exception("Redshift error")
        else:
            raise Exception("Field error")

    elif subaru_data:

        if field == "wide":

            trained_filters = frankenblast_subaru_wide
            filter_names = subaru_names_wide

            if redshift == None:
                train_fname = "subaru_wide"
            elif redshift != None:
                raise Exception("Model currently unavailable")
            else:
                raise Exception("Redshift error")

        elif field == "deep":

            trained_filters = frankenblast_subaru_deep
            filter_names = subaru_names_deep

            if redshift == None:
                train_fname = "subaru_deep"

            elif redshift != None:
                raise Exception("Model currently unavailable")
            else:
                raise Exception("Redshift error")
        else:
            raise Exception("Field error")

    elif (not LSST_data) and (not subaru_data):

        if field == "wide":
            trained_filters = frankenblast_roman_only_wide
            filter_names = frankenblast_roman_only_names_wide

            if redshift == None:
                train_fname = "simple_roman_RZYJH"
            elif redshift != None:
                raise Exception("Model currently unavailable")
            else:
                raise Exception("Redshift error")

        elif field == "deep":

            trained_filters = frankenblast_roman_only_deep
            filter_names = frankenblast_roman_only_names_deep

            if redshift == None:
                train_fname = "simple_roman_ZYJHF"
            elif redshift != None:
                raise Exception("Model currently unavailable")
            else:
                raise Exception("Redshift error")

        else:
            raise Exception("Field error")

    else:
        raise Exception("LSST/Subaru/Roman data error")
    
    # Create transient
    transient_class = Transient(
        name=str(galID),
        coordinates=SkyCoord(ra=RA * u.deg, dec=Dec * u.deg, frame="icrs"),
        transient_redshift=redshift,
        milkyway_dust_reddening=None,  # Initially None
    )

    # Skipping dust reddening stuff for now

    best_host = Host(
        name=transient_class.name,
        redshift=transient_class.transient_redshift,
        coordinates=SkyCoord(ra=RA * u.deg, dec=Dec * u.deg, frame="icrs"),
    )

    # Host
    transient_class.host = best_host

    # Mask any null/none magnitudes
    mask = ~np.isnan(magnitudes)

    ##In practice I assume there will be missing data, but using simulated data
    # it is unlikely and may need to be checked
    if not (mask.all() == True):
        print("Be careful, some filters are nan, does this seem right?")

    all_phot = []

    # Convert to maggies and magerrs
    maggies = 10 ** (-0.4 * magnitudes)
    maggies_err = 10 ** (-0.4 * magnitudes) - 10 ** (-0.4 * (magnitudes + magerrs))

    for i in range(len(filter_names)):
        dataframe = {
            "magnitude": magnitudes[i],
            "magnitude_error": magerrs[i],
            "flux": maggies[i],
            "flux_error": maggies_err[i],
        }

        all_phot.append(dataframe)

    transient_class.host_photometry = all_phot
    transient_class.host_phot_filters = trained_filters

    # Make sure these are set. I don't know why it is so finicky, but it is
    SBIPP_ROOT = os.environ["SBIPP_ROOT"]
    SBIPP_PHOT_ROOT = os.environ["SBIPP_PHOT_ROOT"]
    SBIPP_TRAINING_ROOT = os.environ["SBIPP_TRAINING_ROOT"]
    SED_OUTPUT_ROOT = os.environ["SED_OUTPUT_ROOT"]

    # Set up SBI params
    sbi_params = {
        "anpe_fname_global": f"{SBIPP_ROOT}/SBI_model_{train_fname}_global.pt",  # trained sbi model
        "train_fname_global": f"{SBIPP_PHOT_ROOT}/sbi_{train_fname}_global.h5",  # training set
        "nhidden": 500,  # architecture of the trained density estimator
        "nblocks": 15,  # architecture of the trained density estimator
    }

    sp = fsps.StellarPopulation()

    # Run SBI
    start = time.time()

    # This line below actually runs frankenblast
    flag = fit_host(
        transient_class,
        sbi_params=sbi_params,
        fname=train_fname,
        all_filters=trained_filters,
        extinction_correct=False,
        sbipp=True,
        aperture_type="global",
        aperture=transient_class.global_apertures,
        save=True,
    )

    # Keep track of time
    end = time.time()
    length = end - start

    print("It took", length / 60, "minutes to run SBIPP!")
    print("Host SED fit completed.")
