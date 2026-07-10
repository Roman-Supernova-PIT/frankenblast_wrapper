#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 12:40:09 2026

@author: hmfausey

IO functions for asdf, parquet, eventually just snappl hopefully
"""

import pandas as pd
import asdf


def save_meta_parquet(
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
):
    ##Saving metadata and regualr data seperately. This is all of the metadata
    # in a dictionary
    meta_data = {
        "provenance_id": provenance_id,  # Provenance of this spectrum
        "diaobject_id": diaobject_id,  # Object this is a spectrum for
        "LSST_data": LSST_data,  # Whether LSST data was included in the fit
        "Subaru_data": subaru_data,  # Extra field - whether subaru data included in the fit
        "redshift_prior": redshift_prior,  # Indicates whether the redshift is fixed, and if not, what redshift prior is used
        "SFR_model": sfr_model,  # Which star formation history is used
        "gal_age_model": gal_age_model,  # Which galaxy age metric is used (tage - regular age, MWA - mass weighted age)
        "AGN_info": AGN_info,  # Whether 'logAGNtau' and 'fAGN' are included in the fit
        "filters": filters,  # Array of strings indicating the filters used for the fit
        "mags": magnitudes,  # Array of floats, the associated magnitudes of each filter listed in 'filters'
        "mag_errs": magerrs,  # Array of floats, the associated errors of each filter listed in 'filters'
        "filepath": root,  # If we want to save a path of file with potentially more information
    }

    # Save off the metadata for this specific fit
    meta_df = pd.DataFrame([meta_data])

    meta_df.to_parquet(root + "metadata.parquet", engine="pyarrow", index=False)


def save_data_parquet(data, root):
    ##Since the regualr data was already in a dictionary, its just getting passed
    # here to be saved

    data_df = pd.DataFrame([data])

    data_df.to_parquet(root + "data.parquet", engine="pyarrow", index=False)


def save_all_asdf(
    data,
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
):
    ##If asdf is selected, the meta and data is all in the same file in two sub-
    # dictionaries. Data, as before, was already in one, so it just gets passed
    # as a parameter, added to the full asdf data, and saved off with the newly
    # organized metadata

    meta_data = {
        "provenance_id": provenance_id,  # Provenance of this spectrum
        "diaobject_id": diaobject_id,  # Object this is a spectrum for
        "LSST_data": LSST_data,  # Whether LSST data was included in the fit
        "Subaru_data": subaru_data,  # Extra field - whether subaru data included in the fit
        "redshift_prior": redshift_prior,  # Indicates whether the redshift is fixed, and if not, what redshift prior is used
        "SFR_model": sfr_model,  # Which star formation history is used
        "gal_age_model": gal_age_model,  # Which galaxy age metric is used (tage - regular age, MWA - mass weighted age)
        "AGN_info": AGN_info,  # Whether 'logAGNtau' and 'fAGN' are included in the fit
        "filters": filters,  # Array of strings indicating the filters used for the fit
        "mags": magnitudes,  # Array of floats, the associated magnitudes of each filter listed in 'filters'
        "mag_errs": magerrs,  # Array of floats, the associated errors of each filter listed in 'filters'
        "filepath": root,  # If we want to save a path of file with potentially more information
    }

    all_data = {
        "meta": meta_data,
        "data": data,
    }

    with asdf.AsdfFile(all_data) as af:
        af.write_to(root + "all_data.asdf")
