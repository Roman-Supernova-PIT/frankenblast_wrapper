#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 10:44:58 2026

@author: hmfausey
"""

import numpy as np
from prospect.io import read_results as reader
from frankenblast_model_simple import build_model
import arviz as avz
import plotting


def get_error(galID, hfile, hfile_root, sim_data, true_theta):
    # Get the associted uncertainties based on the psterior chains
    
    # Reload data
    out, out_obs, out_model = reader.results_from(hfile)

    chain = out["chain"]

    model = build_model()
    
    sfr_chain = np.zeros(chain.shape[0])
    logsSFR_chain = np.zeros(chain.shape[0])
    
    #Import the posterior chains of all parameters
    for i, th in enumerate(chain):
        model.set_parameters(th)
    
        tau = model.params["tau"][0]
        tage = model.params["tage"][0]
        mass = model.params["mass"][0]
        
        t = np.linspace(0, tage, 5000)
        
        sfh = t * np.exp(-t / tau)
        
        norm = np.trapz(sfh, t)
        
        sfh *= mass / norm
        
        sfr_chain[i] = sfh[-1]/(1e9) #To get into Mdot per year
        
        logsSFR_chain[i] = np.log10(sfr_chain[i]/mass)
        
    #Index map - to identify saved frankenblast columns
    
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
    
    
    #If we are using either simulated data from frankenblast, or OU data, then
     #we have truth values. If we aren't then there are no truth values and the
     #true_theta array should be none.
     
    if ~np.isnan(true_theta.all()):
        #We only have all of the truth values for siumlated data. For OU data we
         #have mass, redshift, and sSFR
        if sim_data:
            ##If it is data simulated with the frankenblast training set model, we
            # know all of the inputs
            truths = []
    
            if not true_theta.all() == None:
                sim_saved_params = np.array(out["theta_labels"])
                for p in sim_saved_params:
    
                    truths.append(true_theta[param_index_map[p]])
                
                model.set_parameters(true_theta)
                
                tau = true_theta[param_index_map["tau"]]
                tage = true_theta[param_index_map["tage"]]
                mass = 10**(true_theta[param_index_map["logmass"]])
                
                t = np.linspace(0, tage, 5000)
                
                sfh = t * np.exp(-t / tau)
                
                norm = np.trapz(sfh, t)
                
                sfh *= mass / norm
                                
                SFR_true = sfh/(1e9) #To get into Mdot per year
                
                logsSFR_true = np.log10(SFR_true/mass)
                
                truths.append(SFR_true)
    
                truths.append(logsSFR_true)
    
                truths = np.array(truths)
                
                np.savetxt(hfile_root + "truth_vals.txt", (truths), delimiter=" ")
        
        else:
            #If from OU, truths are log(M), redshift, and log(sSFR)[m-dot/Gyr] in that order
            #Want to put logsSFR in [m-dot] per year to match outputs
            true_theta[-1] = true_theta[-1] - 9 #From log(mdot/Gyr) to log(mdot/yr)
            np.savetxt(hfile_root + "truth_vals.txt", (true_theta), delimiter=" ")
            
        
    #Dictionary for saving results
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
        "tau": None, #e-folding time
        "tau_err_high": None, #upper 1-sigma error on tau
        "tau_err_low": None, #lower 1-sigma error on tau
        "age": None, #galaxy age
        "age_err_high": None, #upper 1-sigma error on age
        "age_err_low": None, #lower 1-sigma error on age
        "SFR": None, #star formation rate (recent)
        "SFR_err_high": None, #upper 1-sigma error on SFR
        "SFR_err_low": None, #lower 1-sigma error on SFR
        "logsSFR": None, #log specific star formation rate (mdot/Gyr)
        "logsSFR_err_high": None, #upper 1-sigma error on log(sSFR)
        "logsSFR_err_low": None, #lower 1-sigma error on log(sSFR)
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
        #Highest density interval is probably the best for SBI/posterior
         #sampling, especially when it is not necessarity gaussian
        low, high = avz.hdi(chain, 0.68)
            
        #THIS IS WHERE TO PUT THE NEW FUNCTIONS WHEN CALIBRATING UNCERTAINTIES 
        err_plus, err_minus = (high-median), (median-low) 


        param_estimates.append([median, err_plus, err_minus])
    
    #The name of the keys in the save file (in same order as the special params array)
    associated_keys = np.array(["logM", "redshift", "logZ", "tau", "Av", "age", "SFR", "logsSFR"])
    
    for i in range(len(special_params)):
        ##Put results in output_data dictionary
        (
            output_data[associated_keys[i]],
            output_data[associated_keys[i] + "_err_high"],
            output_data[associated_keys[i] + "_err_low"],
        ) = param_estimates[i]
    
    #Do some extra plotting here (SEDs and corner plots)
    plotting.corner_plot(galID, spec_chain_list, spec_labels, true_theta, hfile_root)
    
    return output_data


def error_floor(median, high, low):
    #Using and error floor. This is currently specific to only mass
    
    #0.3 dex for galaxy mass is typical but will explore further
    error_floor = 0.3 #dex
    err_measured_high, err_measured_low = (high-median), (median-low) 
    err_plus = np.sqrt(err_measured_high**2 + error_floor**2)
    err_minus = np.sqrt(err_measured_low**2 + error_floor**2)
    
    return err_plus, err_minus
    
        
            
