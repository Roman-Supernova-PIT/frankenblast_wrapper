#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 10:44:58 2026

@author: hmfausey
"""

import numpy as np
from prospect.io import read_results as reader
from frankenblast.frankenblast_model_simple import build_model
import arviz as avz
#import plotting
import pickle
import torch


def get_error(galID, hfile, root, model_name, w = 1):
    """Determines 1-, 2-, and 3-sigma uncertainties for parameters

    Loads in fit info and posterior chains to determine median parameter 
    estimate and the 1, 2, and 3 sigma uncertainties. Organizes outputs and puts
    them in a dictionary.

    Parameters
    ----------
    galID : int
        The galaxy identifier/id number
    hfile : sting
        Filename for .h5 file containing posteriors and fit information
    root : string
        File path to .h5 file
    model_name : string
        Which frankenblast model was used for the fit (important for loading in
        the .h5 file correctly)
    w : float
        Scaling parameter for correcting uncertainties, where necessary 
        (default 1)

    Returns
    -------
    out : output_data
        Dictionary of parameter estimates from the fit
    """
    
    ##Reload data
    out, out_obs, out_model = reader.results_from(hfile)

    full_chain = out["chain"]

    #model = build_model()
    
    sfr_chain = np.zeros(full_chain.shape[0])
    logsSFR_chain = np.zeros(full_chain.shape[0])
    
    ##Import the posterior chains of all parameters
    for i, th in enumerate(full_chain):
        
        if len(th) == 7:
            ##If we only have 7 free parameters but no fixed redshift value,
             #something has gone wrong.
            if out_obs["redshift"] is None:
                raise ValueError(
                    "7-parameter chain found, but no fixed redshift is available."
                )
            #Otherwise, but redshift into the model
            th_model = np.insert(th, 0, out_obs["redshift"])
         
        ##8 free parameters when redshift is unknown
        elif len(th) == 8:
            th_model = th
         
        else:
            raise ValueError(
                f"Unexpected posterior chain length {len(th)}. "
                "Expected 7 (fixed-z) or 8 (free-z)."
            )
         
        #model.set_parameters(th_model)
         
        '''
        tau = model.params["tau"][0]
        tage = model.params["tage"][0]
        mass = model.params["mass"][0]
        
        t = np.linspace(0, tage, 5000)
        
        sfh = t * np.exp(-t / tau)
        
        norm = np.trapz(sfh, t)
        
        sfh *= mass / norm
        
        sfr_chain[i] = sfh[-1]/(1e9) #To get into Mdot per year
        
        logsSFR_chain[i] = np.log10(sfr_chain[i]/mass)
        '''
        
    ##Index map for saved frankenblast columns
    
    if "zfixed" in model_name:
        param_index_map = {
            "logmass": 1,
            "logzsol": 2,
            "tau": 3,
            "tage": 4,
            "dust2": 5,
            "dust_index": 6,
            "dust1_fraction": 7,
        }
    else:
        param_index_map = {
            "zred": 0,
            "logmass": 1,
            "logzsol": 2,
            "tau": 3,
            "tage": 4,
            "dust2": 5,
            "dust_index": 6,
            "dust1_fraction": 7,
        }
    
    
    #Dictionary for saving results
    
    output_data = {
        "logM": None,  # log stellar mass in solar masses (log[M/M_dot])
        "logM_err_high": None,  # upper 1-sigma error on log mass
        "logM_err_low": None,  # lower 1-sigma error on log mass
        "logM_err_high_2sig": None,  # upper 2-sigma error on log mass
        "logM_err_low_2sig": None,  # lower 2-sigma error on log mass
        "logM_err_high_3sig": None,  # upper 3-sigma error on log mass
        "logM_err_low_3sig": None,  # lower 3-sigma error on log mass
        "redshift": None,  # either a fixed redshift value, or fit redshift associated with 'redshift_prior'
        "redshift_err_high": None,  # upper 1-sigma error on redshift
        "redshift_err_low": None,  # lower 1-sigma error on redshift
        "redshift_err_high_2sig": None,  # upper 2-sigma error on redshift
        "redshift_err_low_2sig": None,  # lower 2-sigma error on redshift
        "redshift_err_high_3sig": None,  # upper 3-sigma error on redshift
        "redshift_err_low_3sig": None,  # lower 3-sigma error on redshift
        "logZ": None,  # log metallicity in solar metallicity (log[Z/Z_dot])
        "logZ_err_high": None,  # upper 1-sigma error on log metallicity
        "logZ_err_low": None,  # lower 1-sigma error on log metallicity
        "logZ_err_high_2sig": None,  # upper 2-sigma error on log metallicity
        "logZ_err_low_2sig": None,  # lower 2-sigma error on log metallicity
        "logZ_err_high_3sig": None,  # upper 3-sigma error on log metallicity
        "logZ_err_low_3sig": None,  # lower 3-sigma error on log metallicity
        "Av": None,  # dust attenuation in mags (for V band)
        "Av_err_high": None,  # upper 1-sigma error on A_v
        "Av_err_low": None,  # lower 1-sigma error on A_v
        "Av_err_high_2sig": None,  # upper 2-sigma error on A_v
        "Av_err_low_2sig": None,  # lower 2-sigma error on A_v
        "Av_err_high_3sig": None,  # upper 3-sigma error on A_v
        "Av_err_low_3sig": None,  # lower 3-sigma error on A_v
        "tau": None, #e-folding time
        "tau_err_high": None, #upper 1-sigma error on tau
        "tau_err_low": None, #lower 1-sigma error on tau
        "tau_err_high_2sig": None, #upper 2-sigma error on tau
        "tau_err_low_2sig": None, #lower 2-sigma error on tau
        "tau_err_high_3sig": None, #upper 3-sigma error on tau
        "tau_err_low_3sig": None, #lower 3-sigma error on tau
        "age": None, #galaxy age
        "age_err_high": None, #upper 1-sigma error on age
        "age_err_low": None, #lower 1-sigma error on age
        "age_err_high_2sig": None, #upper 2-sigma error on age
        "age_err_low_2sig": None, #lower 2-sigma error on age
        "age_err_high_3sig": None, #upper 3-sigma error on age
        "age_err_low_3sig": None, #lower 3-sigma error on age
        "SFR": None, #star formation rate (recent)
        "SFR_err_high": None, #upper 1-sigma error on SFR
        "SFR_err_low": None, #lower 1-sigma error on SFR
        "SFR_err_high_2sig": None, #upper 2-sigma error on SFR
        "SFR_err_low_2sig": None, #lower 2-sigma error on SFR
        "SFR_err_high_3sig": None, #upper 3-sigma error on SFR
        "SFR_err_low_3sig": None, #lower 3-sigma error on SFR
        "logsSFR": None, #log specific star formation rate (mdot/Gyr)
        "logsSFR_err_high": None, #upper 1-sigma error on log(sSFR)
        "logsSFR_err_low": None, #lower 1-sigma error on log(sSFR)
        "logsSFR_err_high_2sig": None, #upper 2-sigma error on log(sSFR)
        "logsSFR_err_low_2sig": None, #lower 2-sigma error on log(sSFR)
        "logsSFR_err_high_3sig": None, #upper 3-sigma error on log(sSFR)
        "logsSFR_err_low_3sig": None, #lower 3-sigma error on log(sSFR)
    }
     
    
    #Save relevant chains and get error, but for things we actually want 
     #eg Av instead of dust2
    special_params = np.array(["logmass", "zred", "logzsol", "dust2", "tau", "tage", "SFR", "logsSFR"])
    spec_chain_list = []
    spec_labels = []
    param_estimates = []
    

    # Get parameters and 1-sigma uncertainty (by percentile currently)
    for p in special_params:
        
        if p in out["theta_labels"]:

            idx = out["theta_labels"].index(p)
            
            #Special cases
            if p == "dust2":
                #Converting from dust2 to Av
                spec_chain_list.append(out["chain"][:, idx] * 1.086)
                chain = out["chain"][:, idx] * 1.086
                spec_labels.append("Av")
            #Otherwise just add to proper column    
            else:
                spec_chain_list.append(out["chain"][:, idx])
                chain = out["chain"][:, idx]
                spec_labels.append(p)
        #Other special cases
        elif p == "SFR":
            spec_chain_list.append(sfr_chain)
            chain = sfr_chain
            spec_labels.append("SFR")
        elif p == "logsSFR":
            spec_chain_list.append(logsSFR_chain)
            chain = logsSFR_chain
            spec_labels.append("logsSFR")
        
        #Only adding error floor to mass for now
        median = np.median(chain)
        
        ##Highest density interval is probably the best for SBI/posterior
         #sampling, especially when it is not necessarity gaussian
 
        ##For temp scaling
        low, high = avz.hdi(chain, 0.683)
        #THIS IS WHERE TO PUT THE NEW FUNCTIONS WHEN CALIBRATING UNCERTAINTIES 
        err_plus, err_minus = (high-median), (median-low) 
        
        low2, high2 = avz.hdi(chain, 0.955)
        err_plus2, err_minus2 = (high2-median), (median-low2) 
        
        low3, high3 = avz.hdi(chain, 0.997)
        err_plus3, err_minus3 = (high3-median), (median-low3) 

        param_estimates.append([median, err_plus, err_minus, err_plus2, err_minus2, err_plus3, err_minus3])
        
        associated_keys = np.array(["logM", "redshift", "logZ", "tau", "Av", "age", "SFR", "logsSFR"])
        
        print(len(special_params))
        print(len(param_estimates))
    
    for i in range(len(special_params)):
        ##Put results in output_data dictionary
        (
            output_data[associated_keys[i]],
            output_data[associated_keys[i] + "_err_high"],
            output_data[associated_keys[i] + "_err_low"],
            output_data[associated_keys[i] + "_err_high_2sig"],
            output_data[associated_keys[i] + "_err_low_2sig"],
            output_data[associated_keys[i] + "_err_high_3sig"],
            output_data[associated_keys[i] + "_err_low_3sig"],
        ) = param_estimates[i]
            
        #plotting.corner_plot(galID, spec_chain_list, spec_labels, None, root)
        
    return output_data
    

    
    
    
    
    
               
