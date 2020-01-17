#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.0.0"

"""
HISTORY:
    - 2020-01-17: created by Daniel Asmus


NOTES:
    -

TO-DO:
    -
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoMinorLocator
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.top'] = True
from scipy import stats

def make_KSscan(xquant, yquant, outname, border=3, xlim=None, dlim=None,
                plim=None, xlabel=None, title=None):
    """
    Make a scan of the 1D Kolmogorov-Smirnov two-sample test for different cuts
    in the X-quantity to test for significant difference between the
    y-distirbutions on both sides of the moving x threshold to cut the y
    population. The result is returned as an output plot file

    Parameters
    ----------
    xquant : TYPE array
        DESCRIPTION. x distribution of 2D values (cut axis)
    yquant : TYPE array
        DESCRIPTION. y distribution of 2D values (testing dimension)
    outname : TYPE string
        DESCRIPTION. Name of the output plot file
    border : TYPE integer, optional
        DESCRIPTION. The default is 3. Minimum size along the x axis for the
        start and end points of the scan
    xlim : TYPE float, optional
        DESCRIPTION. The default is None. x-limit for the plot
    dlim : TYPE float, optional
        DESCRIPTION. The default is None. d-limit for the plot
    plim : TYPE float, optional
        DESCRIPTION. The default is None. p-limit for the plot
    xlabel : TYPE string, optional
        DESCRIPTION. The default is None. name of the x quantity
    title : TYPE string, optional
        DESCRIPTION. The default is None. title for the plot

    Returns
    -------
    None.

    """

    ids = np.argsort(xquant)
    x = xquant[ids]
    y = yquant[ids]

    ksd = []
    ksp = []
    ksv = []
    no = len(x)

    for i in np.arange(border,no-border):
        s1 = y[:i]
        s2 = y[i:]
        d, p  = stats.ks_2samp(s1, s2)

        ksd.append(d)
        ksp.append(np.log10(p))
        ksv.append(x[i])

    ksv = np.array(ksv)
    ksp = np.array(ksp)
    ksd = np.array(ksd)

    fig, ax1 = plt.subplots()
    fig.subplots_adjust(right=0.87)

    ax2 = ax1.twinx()

    tkw = dict(size=4, width=1.5)

    pmajorLocator = MultipleLocator(1)
    dmajorLocator = MultipleLocator(0.1)
    dminorLocator = AutoMinorLocator(10)
    pminorLocator = AutoMinorLocator(10)
    xminorLocator = AutoMinorLocator(10)

    ax1.plot(ksv,ksd, color='blue')
    ax1.set_ylabel('KS distance')
    ax1.set_xlabel(xlabel)
    ax1.yaxis.label.set_color('blue')
    ax1.tick_params(axis='y', colors='blue', **tkw)

    if title:
        ax1.set_title(title)

    if dlim:
        ax1.set_ylim(dlim)
    else:
        ax1.set_ylim(0,1)

    if xlim:
        ax1.set_xlim(xlim)
    else:
        ax1.set_xlim(np.nanmin(ksv), np.nanmax(ksv))

    ax1.xaxis.set_minor_locator(xminorLocator)
    ax1.yaxis.set_major_locator(dmajorLocator)
    ax1.yaxis.set_minor_locator(dminorLocator)

    ax2.plot(ksv,ksp, color='green')
    ax2.set_ylabel('KS log P')
    ax2.yaxis.label.set_color('green')
    ax2.tick_params(axis='y', colors='green', **tkw)

#    ax2.xaxis.set_minor_locator(pminorLocator)
    ax2.yaxis.set_major_locator(pmajorLocator)
    ax2.yaxis.set_minor_locator(pminorLocator)

    if plim:
        ax2.set_ylim(plim)

    plt.savefig(outname,bbox_inches='tight')
    plt.close(fig)

