#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 13:46:18 2026

@author: hmfausey
"""

import numpy as np
import pandas as pd
import h5py
import os
from my_files import detector_effects

##These are global parameters that will be used to hold info about the data files
# being used
all_data = None
test_phot = None
NF = None


def pull_sims(
    galID, field, sim_testing_set, provenance_id, diaobject_id, test_file=None
):
    """Provenance id and diaobject id may become the identifiers used to search,
    which is why they are currently passed to this function even though they
    aren't used."""

    ##This is defining which bands are being used
    if field == "wide":
        roman_filters = np.array(["R062", "Z087", "Y106", "J129", "H158"])
    elif field == "deep":
        roman_filters = np.array(["Z087", "Y106", "J129", "H158", "F184"])

    LSST_filters = np.array(
        ["lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y"]
    )

    Subaru_filters = np.array(["hsc_g", "hsc_r2", "hsc_i2"])

    # If sim_testing_set is False, its currently pulling from OU data
    if not sim_testing_set:
        global all_data

        # If data table not loaded yet
        if all_data is None:
            if not (test_file is None):
                # Now it is not being reloaded every time
                data = pd.read_csv(test_file)
                all_data = data
            else:
                raise Exception("No test file specified. Cannot aquire data.")

        # Locate the galaxy row
        matches = all_data.loc[all_data["galID"] == galID]

        if len(matches) == 0:
            raise ValueError(f"galID {galID} not found")

        row = matches.iloc[0]

        # Initialize magnitude array
        roman_magnitudes = np.zeros(len(roman_filters))
        roman_magerrs = np.zeros(len(roman_filters))

        ##For now this is mostly a dummy parameter but I assume it will be needed
        # in the future
        RA = row["RA"]
        Dec = row["Dec"]

        # Add filt
        for i in range(len(roman_filters)):
            ##roman_filters is an array of filter names, so this is pulling in
            # the correct filters for whether it is the wide or deep field
            roman_magnitudes[i] = row[roman_filters[i]]

            if ~np.isnan(roman_magnitudes[i]):
                ##OU data (or at least what I had access to) did not have uncertainties
                # so they are being set here
                _, magerr_roman = detector_effects.get_noise_flux(
                    roman_filters[i], roman_magnitudes[i], field
                )
                roman_magerrs[i] = magerr_roman
                # roman_magerrs[i] = 0.01
            else:
                roman_magerrs[i] = np.nan

        # These are the final roman filters
        filters = roman_filters
        magnitudes = roman_magnitudes
        magerrs = roman_magerrs

        # These two values are in principle coming from external data, but since
        # i dont have any for now they are false and none

        """Check for LSST data here
        blah blah blah
        get some array of lsst filters"""

        # For now, just saying LSST magnitudes are all nan since we dont have that data
        LSST_magnitudes = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        LSST_magerrs = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

        if np.isnan(LSST_magnitudes).all():
            LSST_data = False
        else:
            LSST_data = True
            filters = np.concatenate((LSST_filters, filters))
            magnitudes = np.concatenate((LSST_magnitudes, magnitudes))
            magerrs = np.concatenate((LSST_magerrs, magerrs))

        """Same for Subaru"""
        # For now, just saying Subaru magnitudes are all nan since we dont have that data

        Subaru_magnitudes = np.array([np.nan, np.nan, np.nan])
        Subaru_magerrs = np.array([np.nan, np.nan, np.nan])

        if np.isnan(Subaru_magnitudes).all():
            Subaru_data = False
        else:
            Subaru_data = True
            filters = np.concatenate((Subaru_filters, filters))
            magnitudes = np.concatenate((Subaru_magnitudes, magnitudes))
            magerrs = np.concatenate((Subaru_magerrs, magerrs))

        """Check for external redshift here
        blah blah blah
        get a redshift or find there is none"""

        redshift = None

        """Check for AGN info here
        blah blah blah
        find that we need to worry about AGN or not"""

        AGN_info = True

    else:
        # If we ARE using simulated data
        # Pulling from simulated testing sets using frankenblast

        global test_phot
        # global test_theta
        global NF

        if test_phot is None:
            if not (test_file is None):
                with h5py.File(test_file, "r") as f:
                    test_phot = np.array(f["phot"])
                    ##NF is the number of filters. it is half the length of test_phot
                    # because it is holding both the filter vals AND the uncertainties
                    NF = int(len(test_phot[0]) / 2)
            else:
                raise Exception("No test file specified. Cannot aquire data.")

        # These are the final roman filters
        filters = roman_filters

        # Check if the test file is lsst or subaru data
        fname = os.path.basename(test_file).lower()

        ##Here I'm just determining whether its lsst or subaru based on the file
        # name, but here is where we would put something to check if lsst or subaru
        # data exists
        LSST_data = "lsst" in fname
        Subaru_data = "subaru" in fname

        if LSST_data:
            filters = np.concatenate((LSST_filters, filters))
        if Subaru_data:
            filters = np.concatenate((Subaru_filters, filters))

        full = test_phot[galID]
        magnitudes = full[:NF]
        magerrs = full[NF:]

        ##Im just gonna go ahead and set this stuff with dummy parameters.
        # see comments above.
        redshift = None
        RA = 0.0
        Dec = 0.0

        AGN_info = True

    return (
        galID,
        RA,
        Dec,
        filters,
        magnitudes,
        magerrs,
        LSST_data,
        Subaru_data,
        redshift,
        AGN_info,
    )
