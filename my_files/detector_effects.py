#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 10:29:14 2026

@author: hmfausey
"""

import numpy as np

dim_pix = 4096  # number of pixels in one direction


filters = np.array(["R062", "Z087", "Y106", "J129", "H158", "F184"])
C_therm = np.array([0.003, 0.003, 0.003, 0.003, 0.048, 0.155])  # e/s/pix
C_zodi = np.array([0.315, 0.316, 0.357, 0.359, 0.339, 0.194])  # e/s/pix
ZP_e = np.array([26.619, 26.303, 26.356, 26.354, 26.377, 25.913])  # e/s
T_read = 3.04  # s
# flux_per_count in erg/cm2/s/Hz - only for WFI01 for now
flux_per_count = np.array(
    [
        8.507310180127394e-31,
        1.1735535182435983e-30,
        1.0812203456695343e-30,
        1.0770580837770768e-30,
        1.0739789902225378e-30,
        1.652567319829496e-30,
    ]
)

# From Rdox
""" NOTE:For all, used wide field but added random value for F184. Look into SNR for 
    more robust exposure time/depth
"""
exposure_times = {
    "wide": np.array([60, 85, 95, 152, 294, np.nan]),
    "deep": np.array([np.nan, 193, 294, 307, 420, 1636]),
    "all": np.array([60, 85, 95, 152, 294, 750]),
}


def get_noise_flux(used_filters, filter_vals, field):
    """For Roman"""
    """This one converts back to flux before returning instead of leaving it in counts"""
    """Using specifically erg/cm2/s/Hz, so dividing by exposure time before returning to get it back into per seconds at the end"""

    inds = np.where(np.isin(filters, used_filters))[0]
    exposure = exposure_times[field][inds]

    Vsky = (
        (C_therm[inds] + C_zodi[inds]) * exposure * np.ones(len(inds))
    )  # now in e/pix in an array per filter
    Rt = exposure / T_read
    Vread = 25 + (3072 / Rt) * np.ones(len(inds))  # e/pix in an array per filter

    counts = filter_vals / (flux_per_count[inds])

    Vsource = counts * exposure  # in e/pix in an array per filter
    # variance is number of counts, std dev is square root

    """need to update for host!!"""
    """NOTE: if this is being used for the frankenblast wrapper, we are already 
    just dealing with the host, so no need to worry about adding an additional source"""

    counts_host = 0 * np.ones(
        len(inds)
    )  # For now, will update according to GRB host galaxy info
    Vhost = counts_host * exposure  # in e/pix

    V_tot = Vsky + Vread + Vsource + Vhost

    F_source = (Vsource * flux_per_count[inds]) / exposure
    Ferr = (np.sqrt(V_tot) * flux_per_count[inds]) / exposure

    # swap to std.dev. instead of variance
    return F_source, Ferr


def get_noise(used_filters, filter_vals, field):
    """For Roman"""
    inds = np.where(np.isin(filters, used_filters))[0]
    exposure = exposure_times[field][inds]

    Vsky = (
        (C_therm[inds] + C_zodi[inds]) * exposure * np.ones(len(used_filters))
    )  # now in e/pix in an array per filter
    Rt = exposure / T_read
    Vread = 25 + (3072 / Rt) * np.ones(
        len(used_filters)
    )  # e/pix in an array per filter

    counts = filter_vals / (flux_per_count[inds])

    Vsource = counts * exposure  # in e/pix in an array per filter
    # variance is number of counts, std dev is square root

    """need to update for host!!"""
    """NOTE: if this is being used for the frankenblast wrapper, we are already 
    just dealing with the host, so no need to worry about adding an additional source"""

    counts_host = 0 * np.ones(
        len(used_filters)
    )  # For now, will update according to GRB host galaxy info
    Vhost = counts_host * exposure  # in e/pix

    V_tot = Vsky + Vread + Vsource + Vhost

    return Vsource, np.sqrt(V_tot)


def get_ZPimg(used_filters, field):

    inds = np.where(np.isin(filters, used_filters))[0]
    exposure = exposure_times[field][inds]

    ZPimg = ZP_e[inds] + 2.5 * np.log10(exposure)

    return ZPimg


def convert_counts_to_mag(counts, zero_point):

    nonzeropoint_mag = -2.5 * np.log10(counts)

    zeropointed_mag = nonzeropoint_mag + zero_point

    return zeropointed_mag


def convert_mag_to_flux_general(mAB):
    # Convert AB magnitude to flux in Jy
    flux = 3631 * 10 ** (-0.4 * mAB)
    flux = flux * 10 ** (-23)  # to get into erg/cm2/s/Hz
    return flux
