#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:11:20 2026

@author: hmfausey
"""

import os
from matplotlib import pyplot as plt

# import corner
from prospect.io import read_results as reader
from prospect.models import priors
from astropy.cosmology import WMAP9 as cosmo
from prospect.models.sedmodel import PolySpecModel

import numpy as np

from prospect.plotting.corner import allcorner
import fsps

os.environ["SPS_HOME"] = "/Users/hfausey/fsps/fsps"
os.environ["SBIPP_ROOT"] = "/Users/hfausey/frankenblast_data/data/SBI"
os.environ["SBIPP_PHOT_ROOT"] = "/Users/hfausey/frankenblast_data/data/sbipp_phot"
os.environ["SBIPP_TRAINING_ROOT"] = (
    "/Users/hfausey/frankenblast_data/data/sbi_training_sets"
)
os.environ["SED_OUTPUT_ROOT"] = "/Users/hfausey/frankenblast_data/data/sed_output"


def plot_frankenblast_sed(galID, hfile_root, hfile, sim_data, true_theta=None):
    ##Note, this is all for the complex model, but we will likely switch to a
    # simpler model and this will have to be changed

    # Reload data
    out, out_obs, out_model = reader.results_from(hfile)

    chain = out["chain"]

    model = build_model()

    mwa_chain = np.zeros(chain.shape[0])
    sfr_chain = np.zeros(chain.shape[0])

    for i, th in enumerate(chain):
        model.set_parameters(th)

        agebins = model.params["agebins"]
        masses = model.params["mass"]

        t0 = 10 ** agebins[:, 0]
        t1 = 10 ** agebins[:, 1]
        t_mid = 0.5 * (t0 + t1)

        mwa = np.sum(masses * t_mid) / np.sum(masses)

        mwa_chain[i] = mwa / 1e9  # Gyr

        dt = 10 ** agebins[0, 1] - 10 ** agebins[0, 0]
        sfr = masses[0] / dt

        sfr_chain[i] = sfr  # Msun / yr

    # wanted_params = np.array(["logmass", "zred"])
    wanted_params = np.array(out["theta_labels"])
    extra_params = np.array(["age", "SFR"])

    sub_chain_list = []
    sub_labels = []

    # Go through your wanted parameters
    for p in wanted_params:
        if p in out["theta_labels"]:
            idx = out["theta_labels"].index(p)
            sub_chain_list.append(out["chain"][:, idx])
            sub_labels.append(p)
        elif p in out.get("extras", {}):
            # pull from extras
            sub_chain_list.append(out["extras"][p])
            sub_labels.append(p)
        else:
            raise ValueError(f"Parameter {p} not found in chain or extras")

    sub_chain_list.append(mwa_chain)
    sub_labels.append("age")

    sub_chain_list.append(sfr_chain)
    sub_labels.append("SFR")

    sub_chain = np.column_stack(sub_chain_list)

    param_index_map = {
        "zred": 0,
        "logmass": 1,
        "logzsol": 2,
        "logsfr_ratios_1": 3,
        "logsfr_ratios_2": 4,
        "logsfr_ratios_3": 5,
        "logsfr_ratios_4": 6,
        "logsfr_ratios_5": 7,
        "logsfr_ratios_6": 8,
        "dust2": 9,
        "dust_index": 10,
        "dust1_fraction": 11,
        "log_fagn": 12,
        "log_agn_tau": 13,
        "duste_qpah": 14,
        "duste_umin": 15,
        "log_duste_gamma": 16,
        "age": 17,
        "SFR": 18,
    }

    if sim_data:
        ##If it is data simulated with the frankenblast training set model, we
        # know all of the inputs
        truths = []

        if not true_theta.all() == None:

            for p in wanted_params:
                if p == "logsfr_ratios":
                    for i in param_index_map["logsfr_ratios"]:
                        truths.append(true_theta[i])
                else:
                    truths.append(true_theta[param_index_map[p]])

            model.set_parameters(true_theta)

            true_agebins = model.params["agebins"]
            true_masses = model.params["mass"]

            t0 = 10 ** true_agebins[:, 0]
            t1 = 10 ** true_agebins[:, 1]
            t_mid = 0.5 * (t0 + t1)

            MWA_true = np.sum(true_masses * t_mid) / np.sum(true_masses) / 1e9  # Gyr

            t_recent = 10 ** true_agebins[0, 1] - 10 ** true_agebins[0, 0]
            sfr_true = true_masses[0] / t_recent

            truths.append(MWA_true)

            truths.append(sfr_true)

            truths = np.array(truths)
            # Save off truths
            np.savetxt(hfile_root + "truth_vals.txt", (truths), delimiter=" ")

            # Create plots for sfh
            dt = 10 ** true_agebins[:, 1] - 10 ** true_agebins[:, 0]  # years

            sfh = true_masses / dt

            t_left = 10 ** true_agebins[:, 0]
            t_right = 10 ** true_agebins[:, 1]

            t_left /= 1e9
            t_right /= 1e9

            sample_index = np.random.choice(
                len(chain), size=min(1000, len(chain)), replace=False
            )

            sfh_samples = []

            for i in sample_index:

                model.set_parameters(chain[i])

                masses = model.params["mass"]
                agebins = model.params["agebins"]

                dt = 10 ** agebins[:, 1] - 10 ** agebins[:, 0]

                sfh_samples.append(masses / dt)

            sfh_samples = np.array(sfh_samples)

            sfh_p16 = np.percentile(sfh_samples, 16, axis=0)
            sfh_p50 = np.percentile(sfh_samples, 50, axis=0)
            sfh_p84 = np.percentile(sfh_samples, 84, axis=0)

            edges = np.concatenate(
                [[10 ** agebins[0, 0] / 1e9], 10 ** agebins[:, 1] / 1e9]
            )

            plt.figure(figsize=(8, 5))

            # Truth
            plt.step(edges[:-1], sfh, where="post", color="black", lw=2, label="True")

            # Median posterior
            plt.step(
                edges[:-1], sfh_p50, where="post", color="c", lw=2, label="Recovered"
            )

            # Posterior envelope
            plt.fill_between(
                edges[:-1], sfh_p16, sfh_p84, step="post", color="c", alpha=0.3
            )

            plt.yscale("log")

            plt.xlabel("Lookback Time (Gyr)")
            plt.ylabel(r"SFR [$M_\odot\,yr^{-1}$]")

            plt.legend()

            plt.savefig(hfile_root + "sfr_plot.pdf")

            plt.show()
            plt.close()

        ##Make corner plot
        ndim = len(sub_labels)

        cfig, axes = plt.subplots(ndim, ndim, figsize=(2.5 * ndim, 2.5 * ndim))
        axes = allcorner(
            sub_chain.T, sub_labels, axes, color="royalblue", show_titles=True
        )

        for i in range(ndim):
            axes[i, i].axvline(truths[i], color="red", linestyle="--", linewidth=2)

        cfig.savefig(hfile_root + f"/{galID}_corner_full.pdf")
        cfig.show()
        plt.close(cfig)

    # Make a smaller corner plot with only what we care about
    special_params = np.array(["logmass", "zred", "logzsol", "dust2", "age", "SFR"])

    spec_chain_list = []
    spec_labels = []
    param_estimates = []

    # Get parameters and 1-sigma uncertainty (by percentile currently)
    for p in special_params:
        if p in out["theta_labels"]:

            idx = out["theta_labels"].index(p)

            if p == "dust2":
                spec_chain_list.append(out["chain"][:, idx] * 1.086)
                chain = out["chain"][:, idx] * 1.086
                spec_labels.append("Av")
            else:
                spec_chain_list.append(out["chain"][:, idx])
                chain = out["chain"][:, idx]
                spec_labels.append(p)

            median = np.median(chain)
            low, high = np.percentile(chain, [16, 84])
            param_estimates.append([median, high - median, median - low])

        else:
            if p == "age":
                spec_chain_list.append(mwa_chain)
                chain = mwa_chain
                spec_labels.append("age")
            elif p == "SFR":
                spec_chain_list.append(sfr_chain)
                chain = sfr_chain
                spec_labels.append("SFR")

            median = np.median(chain)
            low, high = np.percentile(chain, [16, 84])
            param_estimates.append([median, high - median, median - low])

    # Set up data dictionary
    output_data = {
        "logM": None,  # log stellar mass in solar masses (log[M/M_dot])
        "logM_err_high": None,  # upper 1-sigma error on log mass
        "logM_err_low": None,  # lower 1-sigma error on log mass
        "redshift": None,  # either a fixed redshift value, or fit redshift associated with 'redshift_prior'
        "redshift_err_high": None,  # upper 1-sigma error on redshift
        "redshift_err_low": None,  # lower 1-sigma error on log metallicity
        "logZ": None,  # log metallicity in solar metallicity (log[Z/Z_dot])
        "logZ_err_high": None,  # upper 1-sigma error on log metallicity
        "logZ_err_low": None,  # lower 1-sigma error on log metallicity
        "Av": None,  # dust attenuation in mags (for V band)
        "Av_err_high": None,  # upper 1-sigma error on A_v
        "Av_err_low": None,  # lower 1-sigma error on A_v
        "age": None,
        "age_err_high": None,
        "age_err_low": None,
        "SFR": None,
        "SFR_err_high": None,
        "SFR_err_low": None,
    }

    main_keys = np.array(["logM", "redshift", "logZ", "Av", "age", "SFR"])
    associated_names = np.array(["logmass", "zred", "logzsol", "Av", "age", "SFR"])

    for i in range(len(associated_names)):
        ##Put results in output_data dictionary
        (
            output_data[main_keys[i]],
            output_data[main_keys[i] + "_err_high"],
            output_data[main_keys[i] + "_err_low"],
        ) = param_estimates[i]

    spec_chain = np.column_stack(spec_chain_list)

    # Another corner plot! This time with way less parameters
    ndim = len(special_params)
    cfig, axes = plt.subplots(ndim, ndim, figsize=(10, 9))
    axes = allcorner(
        spec_chain.T, spec_labels, axes, color="royalblue", show_titles=True
    )

    cfig.savefig(hfile_root + f"/{galID}_corner_special.pdf")
    cfig.show()
    plt.close(cfig)

    return output_data


def build_model(obs=None, **extras):
    """prospector-alpha"""
    fit_order = [
        "zred",
        "logmass",
        "logzsol",
        "logsfr_ratios",
        "dust2",
        "dust_index",
        "dust1_fraction",
        "log_fagn",
        "log_agn_tau",
        "duste_qpah",
        "duste_umin",
        "log_duste_gamma",
    ]

    # -------------
    # MODEL_PARAMS
    model_params = {}

    # --- BASIC PARAMETERS ---
    model_params["zred"] = {
        "N": 1,
        "isfree": True,
        "init": 0.5,
        "prior": priors.FastUniform(
            a=0.0, b=3.0 + 1e-3
        ),  # priors.FastTruncatedNormal(a=0.0, b=1.5, mu=0.163, sig=0.052)
    }  # ANYA: based on YSE DR1 mu and sigma

    model_params["logmass"] = {
        "N": 1,
        "isfree": True,
        "init": 8.0,
        "units": "Msun",
        "prior": priors.FastUniform(a=6.0, b=12.0),
    }

    model_params["logzsol"] = {
        "N": 1,
        "isfree": True,
        "init": -0.5,
        "units": r"$\log (Z/Z_\odot)$",
        "prior": priors.FastUniform(a=-1.98, b=0.19),
    }

    model_params["imf_type"] = {
        "N": 1,
        "isfree": False,
        "init": 1,  # 1 = chabrier
        "units": None,
        "prior": None,
    }
    model_params["add_igm_absorption"] = {"N": 1, "isfree": False, "init": True}
    model_params["add_agb_dust_model"] = {"N": 1, "isfree": False, "init": True}
    model_params["pmetals"] = {"N": 1, "isfree": False, "init": -99}

    # --- SFH ---
    nbins_sfh = 7
    model_params["sfh"] = {"N": 1, "isfree": False, "init": 3}
    model_params["logsfr_ratios"] = {
        "N": 6,
        "isfree": True,
        "init": 0.0,
        "prior": priors.FastTruncatedEvenStudentTFreeDeg2(
            hw=np.ones(6) * 5.0, sig=np.ones(6) * 0.3
        ),
    }

    # add redshift scaling to agebins, such that
    # t_max = t_univ
    def zred_to_agebins(zred=None, **extras):
        amin = 7.1295
        nbins_sfh = 7
        tuniv = cosmo.age(zred)[0].value * 1e9
        tbinmax = tuniv * 0.9
        if zred <= 3.0:
            agelims = (
                [0.0, 7.47712]
                + np.linspace(8.0, np.log10(tbinmax), nbins_sfh - 2).tolist()
                + [np.log10(tuniv)]
            )
        else:
            agelims = np.linspace(amin, np.log10(tbinmax), nbins_sfh).tolist() + [
                np.log10(tuniv)
            ]
            agelims[0] = 0

        agebins = np.array([agelims[:-1], agelims[1:]])
        return agebins.T

    def logsfr_ratios_to_masses(
        logmass=None, logsfr_ratios=None, agebins=None, **extras
    ):
        """This converts from an array of log_10(SFR_j / SFR_{j+1}) and a value of
        log10(\Sum_i M_i) to values of M_i.  j=0 is the most recent bin in lookback
        time.
        """
        nbins = agebins.shape[0]
        sratios = 10 ** np.clip(logsfr_ratios, -100, 100)
        dt = 10 ** agebins[:, 1] - 10 ** agebins[:, 0]
        coeffs = np.array(
            [
                (1.0 / np.prod(sratios[:i]))
                * (np.prod(dt[1 : i + 1]) / np.prod(dt[:i]))
                for i in range(nbins)
            ]
        )
        m1 = (10**logmass) / coeffs.sum()

        return m1 * coeffs

    model_params["mass"] = {
        "N": 7,
        "isfree": False,
        "init": 1e6,
        "units": r"M$_\odot$",
        "depends_on": logsfr_ratios_to_masses,
    }

    model_params["agebins"] = {
        "N": 7,
        "isfree": False,
        "init": zred_to_agebins(np.atleast_1d(0.5)),
        "prior": None,
        "depends_on": zred_to_agebins,
    }

    # --- Dust Absorption ---
    model_params["dust_type"] = {
        "N": 1,
        "isfree": False,
        "init": 4,
        "units": "FSPS index",
    }
    model_params["dust1_fraction"] = {
        "N": 1,
        "isfree": True,
        "init": 1.0,
        "prior": priors.FastTruncatedNormal(a=0.0, b=2.0, mu=1.0, sig=0.3),
    }

    model_params["dust2"] = {
        "N": 1,
        "isfree": True,
        "init": 0.0,
        "units": "",
        "prior": priors.FastTruncatedNormal(a=0.0, b=4.0, mu=0.3, sig=1.0),
    }

    def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
        return dust1_fraction * dust2

    model_params["dust1"] = {
        "N": 1,
        "isfree": False,
        "depends_on": to_dust1,
        "init": 0.0,
        "units": "optical depth towards young stars",
        "prior": None,
    }
    model_params["dust_index"] = {
        "N": 1,
        "isfree": True,
        "init": 0.7,
        "units": "",
        "prior": priors.FastUniform(a=-1.0, b=0.2),
    }

    # --- Nebular Emission ---
    model_params["nebemlineinspec"] = {"N": 1, "isfree": False, "init": False}
    # model_params["add_neb_emission"] = {"N": 1, "isfree": False, "init": True}
    # model_params["add_neb_continuum"] = {"N": 1, "isfree": False, "init": True}
    model_params["gas_logz"] = {
        "N": 1,
        "isfree": False,
        "init": 0.0,
        "units": r"log Z/Z_\odot",
        "prior": priors.FastUniform(a=-2.0, b=0.5),
    }
    model_params["gas_logu"] = {
        "N": 1,
        "isfree": False,
        "init": -1.0,
        "units": r"Q_H/N_H",
        "prior": priors.FastUniform(a=-4, b=-1),
    }

    # --- AGN dust ---
    model_params["add_agn_dust"] = {"N": 1, "isfree": False, "init": True}

    model_params["log_fagn"] = {
        "N": 1,
        "isfree": True,
        "init": -7.0e-5,
        "prior": priors.FastUniform(a=-5.0, b=-4.9),
    }

    def to_fagn(log_fagn=None, **extras):
        return 10**log_fagn

    model_params["fagn"] = {"N": 1, "isfree": False, "init": 0, "depends_on": to_fagn}

    model_params["log_agn_tau"] = {
        "N": 1,
        "isfree": True,
        "init": np.log10(20.0),
        "prior": priors.FastUniform(a=np.log10(15.0), b=np.log10(15.1)),
    }

    def to_agn_tau(log_agn_tau=None, **extras):
        return 10**log_agn_tau

    model_params["agn_tau"] = {
        "N": 1,
        "isfree": False,
        "init": 0,
        "depends_on": to_agn_tau,
    }

    # --- Dust Emission ---
    model_params["duste_qpah"] = {
        "N": 1,
        "isfree": True,
        "init": 2.0,
        "prior": priors.FastTruncatedNormal(a=0.9, b=1.1, mu=2.0, sig=2.0),
    }

    model_params["duste_umin"] = {
        "N": 1,
        "isfree": True,
        "init": 1.0,
        "prior": priors.FastTruncatedNormal(a=0.9, b=1.1, mu=1.0, sig=10.0),
    }

    model_params["log_duste_gamma"] = {
        "N": 1,
        "isfree": True,
        "init": -2.0,
        "prior": priors.FastTruncatedNormal(a=-2.1, b=-1.9, mu=-2.0, sig=1.0),
    }

    def to_duste_gamma(log_duste_gamma=None, **extras):
        return 10**log_duste_gamma

    model_params["duste_gamma"] = {
        "N": 1,
        "isfree": False,
        "init": 0,
        "depends_on": to_duste_gamma,
    }

    # ---- Units ----
    model_params["peraa"] = {"N": 1, "isfree": False, "init": False}

    model_params["mass_units"] = {"N": 1, "isfree": False, "init": "mformed"}

    tparams = {}
    for i in fit_order:
        tparams[i] = model_params[i]
    for i in list(model_params.keys()):
        if i not in fit_order:
            tparams[i] = model_params[i]
    model_params = tparams

    return PolySpecModel(model_params)


def compute_mass_weighted_age(model):
    agebins = model.params["agebins"]  # shape (N, 2), log10(years)
    masses = model.params["mass"]  # formed mass per bin

    # Convert log age to linear years
    t0 = 10 ** agebins[:, 0]
    t1 = 10 ** agebins[:, 1]

    # Bin midpoint (you could also use more sophisticated weighting)
    t_mid = 0.5 * (t0 + t1)

    # Mass-weighted age (years)
    t_mw = np.sum(masses * t_mid) / np.sum(masses)

    return t_mw / 1e9  # convert to Gyr
