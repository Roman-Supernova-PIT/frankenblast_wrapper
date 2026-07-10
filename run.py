#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:33:51 2026

@author: hmfausey
"""

import os

import pickle
import pandas as pd
import multiprocessing as mp
from functools import partial
import numpy as np
import random
import h5py

import run_frankenblast_all
import IO_functions as IO
import plot_sed

os.environ["SPS_HOME"] = "/Users/hfausey/fsps/fsps"
os.environ["SBIPP_ROOT"] = "/Users/hfausey/frankenblast_data/data/SBI"
os.environ["SBIPP_PHOT_ROOT"] = "/Users/hfausey/frankenblast_data/data/sbipp_phot"
os.environ["SBIPP_TRAINING_ROOT"] = (
    "/Users/hfausey/frankenblast_data/data/sbi_training_sets"
)
os.environ["SED_OUTPUT_ROOT"] = "/Users/hfausey/frankenblast_data/data/sed_output"
import fsps

# from run_prospector import get_host_properties
from my_files.sim_photometry import pull_sims

import logging
import re

all_data = None
# filter_set = None
fsps = None


def fit_one_galaxy(
    ID,
    field,
    sim_testing_set,
    provenance_id,
    diaobject_id,
    redshift_prior,
    sfr_model,
    gal_age_model,
    filetype,
    test_file=None,
    test_theta=None,
):
    """Right now this is using the simulated data. this will need to be changed"""
    """We are passing the test data file path in this function, but eventually that will
       need to be removed"""

    global all_data
    # global filter_set
    import fsps

    import logging
    import re

    # pull simulated data

    root = os.environ["SED_OUTPUT_ROOT"] + f"/{ID}/"
    hfile = os.environ["SED_OUTPUT_ROOT"] + f"/{ID}/{ID}_global.h5"

    # NEED AGN INFO FROM PULL SIMS OR EQUIVALENT

    # ID WILL PROBABLY BE REPLACED WITH PROVENANCE ID IN THE PULL SIMS OR EQUIVALENT FUNCTION
    (
        galID,
        RA,
        Dec,
        filters,
        magnitudes,
        magerrs,
        LSST_data,
        subaru_data,
        redshift,
        AGN_info,
    ) = pull_sims(
        ID, field, sim_testing_set, provenance_id, diaobject_id, test_file=test_file
    )

    if not os.path.exists(hfile):
        (
            galID,
            RA,
            Dec,
            filters,
            magnitudes,
            magerrs,
            LSST_data,
            subaru_data,
            redshift,
            AGN_info,
        ) = pull_sims(
            ID, field, sim_testing_set, provenance_id, diaobject_id, test_file=test_file
        )

        # if the redshift comes back with a value, it means we have a known redshift and need to change the redshift_prior field
        if not (redshift is None):
            redshift_prior = None

        run_frankenblast_all.run(
            galID, field, RA, Dec, magnitudes, magerrs, LSST_data, subaru_data, redshift
        )

    if sim_testing_set:
        true_theta = test_theta[ID]
    else:
        # NEED TO SET UP PULLING THE OU INPUTS
        true_theta = None
        print("skipping")
        # raise Exception("Not supposed to be here yet")

    output_data = plot_sed.plot_frankenblast_sed(
        ID, root, hfile, sim_testing_set, true_theta=true_theta
    )

    # Save off the metadata for this specific SN+galaxy+photometry+fit+version+etc.
    # NEED TO SET UP FRAMEWORK HERE FOR ASDF/PARQUET SAVE
    if filetype == "parquet":
        IO.save_meta_parquet(
            provenance_id,
            diaobject_id,
            LSST_data,
            subaru_data,
            redshift_prior,
            sfr_model,
            gal_age_model,
            AGN_info,
            filters,
            magnitudes,
            magerrs,
            root,
        )
        IO.save_data_parquet(output_data, root)
    elif filetype == "asdf":
        IO.save_all_asdf(
            output_data,
            provenance_id,
            diaobject_id,
            LSST_data,
            subaru_data,
            redshift_prior,
            sfr_model,
            gal_age_model,
            AGN_info,
            filters,
            magnitudes,
            magerrs,
            root,
        )
    elif filetype == "database":
        # yet another blah
        raise Exception(("No database yet"))
    else:
        raise Exception("Invalid file type for saving results")


if __name__ == "__main__":

    print("TIME TO GET SOME GALAXY PROPERTIES!!")

    # Testing with wide field stuff first
    field = "deep"
    sim_testing_set = False
    # Change this depending on where you are pulling the data from
    test_file = os.environ["SBIPP_PHOT_ROOT"] + "/sbi_phot_subaru_deep_testing_set.h5"
    filetype = "asdf"
    cpu_num = 6
    # root = "/Users/hfausey/frankenblast_data/results/lsst_deep/"

    # Error handling - Sometimes frankenblast cannot figure things out, this skips ones
    # where it is going to get stuck

    if not sim_testing_set:
        """STUFF BELOW IS FOR TESTING WITH OU DATA"""

        test_file = "./my_files/data/compiled_catalog.csv"
        data = pd.read_csv(test_file)
        galaxy_ids = data["galID"].tolist()
        test_theta = None

    elif sim_testing_set:
        """STUFF BELOW IS FOR TESTING WITH SIMULATED TESTING SET WITH FRANKENBLAST"""

        with h5py.File(test_file, "r") as f:
            test_phot = np.array(f["phot"])
            test_theta = np.array(f["theta"])

        n_galaxies = len(test_phot)
        galaxy_ids = range(n_galaxies)
        # print(galaxy_ids)

        print(f"Loaded {n_galaxies} test galaxies")

        NF = int(len(test_phot[0]) / 2)

    # FOR PROVENANCE ID AND DIAOBJECT ID
    # If we are running everything in batches, I assume a list of provenance ids
    # and diaobject ids is going to be passed before touching anything else, in
    # which case we will have those from the start here.

    # BUT database is not up yet, so we are just using None for those two things
    # for now

    provenance_id = None
    diaobject_id = None

    ##LSST AND SUBARU DATA will get determined when checking for external data later

    ##FOR REDSHIFT PRIOR
    # At least for now it is just a FastUniform prior from 0-3 (same as TopHat)
    redshift_prior = "FastUniform 0-3"

    # FOR SFH
    #  1 is tau
    #  2 is constant sfh
    #  3 is delayed-tau
    #  4 is non-parametric

    # For frankenblast, right now at least, sfh is non-parametric

    sfr_model = 4

    # FOR GAL AGE MODEL
    # Frankenblast is going to almost always be MWA

    gal_age_model = "MWA"

    # AGN INFO will depend
    # could potentially swap between using and not using agn parameters, but will
    # need to be pulling that from elsewhere

    # FILTERS, MAGS, MAGERRS depends on external data as well as pulling Roman galaxy
    # photometry from other databases, so that is dealt with later

    # FILEPATH is dependent on the individual identifiers and will be set later.

    for ID in galaxy_ids:

        fit_one_galaxy(
            ID,
            field,
            sim_testing_set,
            provenance_id,
            diaobject_id,
            redshift_prior,
            sfr_model,
            gal_age_model,
            filetype,
            test_file=test_file,
            test_theta=test_theta,
        )
