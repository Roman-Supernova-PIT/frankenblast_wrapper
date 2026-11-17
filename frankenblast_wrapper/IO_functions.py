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
    diaobject_id,
    new_provenance_id,
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
    """Saves metadata to a parquet file.

    Save only metadata. metadata and regular data are saved seperately.

    Parameters
    ----------
    diaobject_id : UUID
        The identifier associated with the supernova
    provenance_id : UUID
        Identifier for host galaxy photometry
    LSST_data : bool
        Whether LSST data is included
    subaru_data : bool
        Whether Subaru data is included
    redshift_prior : string
        Which redshift prior was used for the fit (None if fixed)
    sfr_model : int
        Which prospector star formation history model is assumed for the fit
        (Most likely 4 for delayed tau model)
    gal_age_model : string
        Whether galaxy age or mass-weighted age is used for the fit
    AGN_info : bool
        Whether AGN contamination is considered for the fit
    filter : array_like
        Which filter names were used for the fit
    magnitudes : array_like
        All host photometry
    magerrs : array_like
        All host photometry uncertainties (1-sigma)
    root : string
        File path where metadata should be saved

    Returns
    -------
    out : None
        Saves data and ends
    """

    meta_data = {
        "diaobject_id": diaobject_id,  # Object
        "provenance_id": new_provenance_id,  # Provenance
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

    ##Save off the metadata for this specific fit
    meta_df = pd.DataFrame([meta_data])

    meta_df.to_parquet(root + "metadata.parquet", engine="pyarrow", index=False)

    return None


def save_data_parquet(data, root):
    """Saves fit parameters results to a parquet file.

    Since the regular data is already in a dictionary, it iss just getting
    passed here to be saved

    Parameters
    ----------
    data : pandas dataframe
        Contains parameter estimates from the fit
    root : string
        File path where metadata should be saved

    Returns
    -------
    out : None
        Saves data and ends
    """

    data_df = pd.DataFrame([data])

    data_df.to_parquet(root + "data.parquet", engine="pyarrow", index=False)


def save_all_asdf(
    data,
    diaobject_id,
    provenance_id,
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
    """Saves metadata and data to an asdf file.

    Save both the metadata and regular data saved in the same asdf file.

    Parameters
    ----------
    data : pandas dataframe
        Contains parameter estimates from the fit
    diaobject_id : UUID
        The identifier associated with the supernova
    provenance_id : UUID
        Identifier for host galaxy photometry
    LSST_data : bool
        Whether LSST data is included
    subaru_data : bool
        Whether Subaru data is included
    redshift_prior : string
        Which redshift prior was used for the fit (None if fixed)
    sfr_model : int
        Which prospector star formation history model is assumed for the fit
        (Most likely 4 for delayed tau model)
    gal_age_model : string
        Whether galaxy age or mass-weighted age is used for the fit
    AGN_info : bool
        Whether AGN contamination is considered for the fit
    filter : array_like
        Which filter names were used for the fit
    magnitudes : array_like
        All host photometry
    magerrs : array_like
        All host photometry uncertainties (1-sigma)
    root : string
        File path where metadata should be saved

    Returns
    -------
    out : None
        Saves data and ends
    """

    ##If asdf is selected, the meta and data is all in the same file in two sub-
    # dictionaries. Data, as before, was already in one, so it just gets passed
    # as a parameter, added to the full asdf data, and saved off with the newly
    # organized metadata

    meta_data = {
        "diaobject_id": diaobject_id,  # Object
        "provenance_id": provenance_id,  # Provenance
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
