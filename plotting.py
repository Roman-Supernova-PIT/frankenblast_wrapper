#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 12:09:25 2026

@author: hmfausey
"""

import numpy as np

from matplotlib import pyplot as plt

from prospect.plotting.corner import allcorner

def corner_plot(galID, spec_chain_list, spec_labels, true_theta, hfile_root):
    #Make the corner plot
    spec_chain = np.column_stack(spec_chain_list)

    # Another corner plot! This time with way less parameters
    ndim = len(spec_labels)
    
    cfig, axes = plt.subplots(ndim, ndim, figsize=(10, 9))
    axes = allcorner(
        spec_chain.T, spec_labels, axes, color="royalblue", show_titles=True
    )
    if len(true_theta) == ndim:
        #This only works if we know true values for all parameters
        for i in range(ndim):
            axes[i, i].axvline(true_theta[i], color="red", linestyle="--", linewidth=2)

    cfig.savefig(hfile_root + f"/{galID}_corner_special.pdf")
    plt.show(cfig)
    plt.close(cfig)