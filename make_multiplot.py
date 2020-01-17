#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.0.0"

"""
HISTORY:
    - 2020-01-15: created by Daniel Asmus


NOTES:
    -

TO-DO:
    -
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from scipy import stats

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.top'] = True

from .bces import bces as _bces


def make_multiplot(xqs, yqs, outname, xlabels=None, ylabels=None, alpha=0.25,
                   subsets=None, colour='blue', histlog=False, symsize=10,
                   violin=False, normviolin=True, vpoints=100, vmethod='scott'):
    """
    Do a multiplot "everything versus everything" for the quantities provided
    for the x and y axis (xqs and yqs) including some automatic basic
    correlation analysis and trend fitting

    Parameters
    ----------
    xqs : TYPE List of arrays
        DESCRIPTION.  List of arrays to be plotted on the x axis
    yqs : TYPE
        DESCRIPTION. List of arrays to be plotted on the y axis
    outname : TYPE List of arrays
        DESCRIPTION.
    xlabels : TYPE List of strings, optional
        DESCRIPTION. The default is None. Names of the x quantities
    ylabels : TYPE List of strings, optional
        DESCRIPTION. The default is None. Names of the y quantities
    alpha : TYPE float, optional
        DESCRIPTION. The default is 0.25. Transparency value for the symbols
    subsets : TYPE dictionary, optional
        DESCRIPTION. The default is None. Define the properties of subset to be
        highlighted in different colour and so on. See below for details
    colour : TYPE, optional
        DESCRIPTION. The default is 'blue'. Main color for the whole population
    histlog : TYPE bool, optional
        DESCRIPTION. The default is False. Flag whether to plot y axis of histograms in log
        space
    symsize : TYPE float, optional
        DESCRIPTION. The default is 10. Main symbol size for the whole
        population
    violin : TYPE bool, optional
        DESCRIPTION. The default is False. Flag whether to make violin instead
        of scatter plots
    normviolin : TYPE bool, optional
        DESCRIPTION. The default is True. flag whether to normalise the width
        of the violines to the plot width
    vpoints : TYPE int, optional
        DESCRIPTION. The default is 100. sampling density for the violins
    vmethod : TYPE string, optional
        DESCRIPTION. The default is 'scott'. method of how to calculate the
        violins

    Returns
    -------
    None.

    How to define subset dictionaries
    # --- subsamples
    ida = np.where(nh > 0)[0]
    id1 = np.where(nh <= 22)[0]
    thres = -1
    id2l = np.where((nh > 22) & (nh < 24) & (edd < thres))[0]
    id2h = np.where((nh > 22) & (nh < 24) & (edd >= thres))[0]

    subsets = dict()
    subsets['ids'] = [ida,id1, id2l, id2h]
    subsets['colour'] = ['black','grey','red','blue']
    subsets['alpha'] = [0, 0.25, 0.25, 0.25, 0.25]
    subsets['label'] = ['all','unobscured', 'CThin', 'CThin&high-$\lambda$']


    """

    nx = len(xqs)+1  # add one for the histograms
    ny = len(yqs)+1

    fig = plt.figure(1, figsize=(1+3*nx,1+3*ny))
    # Now, create the gridspec structure, as required

    # --- define ratios for the plots including the smaller histograms
    width_ratios = np.full(nx, 1.0)
    width_ratios[-1] = 0.5
    height_ratios = np.full(ny, 1.0)
    height_ratios[0] = 0.5

    gs = gridspec.GridSpec(ny, nx, width_ratios=width_ratios,
                           height_ratios=height_ratios)
    gs.update(left=0.1, right=0.95, bottom=0.1, top=0.93, wspace=0.03, hspace=0.03)

    # --- if no subset provided, treat the whole sample as subset (to keep the
    #     code below simple)
    if subsets is None:
        subsets = dict()
        subsets['ids'] = [range(len(xqs[0]))]
        subsets['colour'] = [colour]
        subsets['symsize'] = [symsize]
        subsets['alpha'] = [alpha]
        subsets['label'] = ['all']

    # --- axes lists for modifications at the end
    axes = []
    haxes = []
    haxes2 = []
    xhmax = []
    yhmax = []

    # --- main loop going over all plots in the grid
    for x in range(nx-1):
        for y in range(ny-1):

            # print(x,y)
            ax = plt.subplot(gs[y+1, x])
            axes.append(ax)

            # --- histograms for the y quantities (at the end of the rows)
            if x == nx-2:
                hax = plt.subplot(gs[y+1, x+1])
                haxes.append(hax)

            # --- histograms for the x quantities (as top row)
            if y == 0:
                hax2 = plt.subplot(gs[0, x])
                haxes2.append(hax2)

            # --- take care of masked arrays
            if isinstance(xqs[x],np.ma.MaskedArray):
                xqs[x][xqs[x].mask] = float('NaN')

            if isinstance(yqs[y],np.ma.MaskedArray):
                yqs[y][yqs[y].mask] = float('NaN')

            xs = np.array(xqs[x])
            ys = np.array(yqs[y])

            # --- how many valid data points?
            idgood = np.where(np.isfinite(xs) & np.isfinite(ys))[0]
            #ngood = len(idgood)

            xsg = xs[idgood]
            ysg = ys[idgood]

            # --- define the plotting range parameters according to the whole
            #     (good) sample
            xmin = np.nanmin(xs)
            xmax = np.nanmax(xs)
            xmid = 0.5*(xmax+xmin)
            xran = np.abs(xmax - xmin)

            ymin = np.nanmin(ys)
            ymax = np.nanmax(ys)
            ymid = 0.5*(ymax+ymin)
            yran = np.abs(ymax - ymin)

            # --- create histograms to get the optimal binning for the whole
            #     sample.
            xhist, xbins = np.histogram(xsg, bins='fd')
            yhist, ybins = np.histogram(ysg, bins='fd')

            # --- monitor the maximum number in a bin over all histograms to
            #     match their plotting range (in y) later
            xhmax.append(np.max(xhist))
            yhmax.append(np.max(yhist))

            # --- loop over all subplots
            ns = len(subsets['ids'])

            # --- violin plots
            if violin:

                # --- get the unique x positions (these are expected to be a
                #     small number, i.e. the data is grouped into only a few
                #     x values as for model data for example)
                xvpos = np.unique(xsg)
                nvx = len(xvpos)

                # --- add entries to the subset dictionary to contain all the
                #     required plotting information for each violin in each
                #     subplot for each subset and each x value. This is
                #     necessary in order to be able to normalise the area of
                #     each violing to be proportional to the number of objects
                #     contained by that violin. This is not the case by default
                #     where the width of each violing would be the same
                #     independent of how many objects contribute to it
                subsets['xvpos'] = [None] * ns
                subsets['yv'] = [None] * ns
                subsets['vwidth'] = [None] * ns
                subsets['nov'] = [None] * ns
                subsets['vnorm'] = [[]] * ns

                # --- maximum width of a violin based on the average spacing
                #     of the x values
                vwidth = 1.0*(np.max(xvpos) - np.min(xvpos))/(nvx-1)
#                print('vwidth:', vwidth)

                # --- go over each subset to pre-calculate the violin values
                for i in range(ns):

                    yv = [[]] * nvx
                    # --- determine all good data points to be plotted belonging
                    #     to this subset
                    idgg = np.array([z for z in subsets['ids'][i] if z in idgood])
                    xsg = np.array(xs[idgg])
                    ysg = np.array(ys[idgg])

                    # --- loop over all x positions, i.e. all violins of that
                    #     subset to fill in the corresponding y values
                    for j in range(nvx):
                        idv = np.where(xsg == xvpos[j])[0]
                        if len(idv) == 0:
                            continue
                        else:
                            yv[j] = ysg[idv]

                    # --- get the indices of all violins that actually contain
                    #     data
                    idvg = [j for j,_ in enumerate(yv) if len(yv[j]) > 0]

                    if len(idvg) == 0:
                            continue
                    else:

                        subsets['xvpos'][i] = [xvpos[j] for j in idvg]
                        subsets['yv'][i] = [yv[j] for j in idvg]
                        subsets['nov'][i] = [len(yv[j]) for j in idvg]

                        # --- compute the approx area of the violin
#                        for j in range(len(idvg)):
#                            vq1 = np.percentile(subsets['yv'][i][j],25)
#                            vq3 = np.percentile(subsets['yv'][i][j],75)
#                            va = subsets['nov'][i][j]/np.abs(vq3 - vq1)
#                            print('i,j,vq1,vq3, nov, va', i,j,vq1,vq3, subsets['nov'][i][j], va)
#                            subsets['vnorm'][i].append(va)
                        #print(x,y,i,subsets['nov'][i])
                        subsets['vnorm'][i] = [len(yv[j])/np.abs(np.percentile(yv[j],25) - np.percentile(yv[j],75)) for j in idvg]

                # --- compute a normalised violin width if desired
                if normviolin:

#                    normv = np.max([np.nanmax(subsets['nov'][i]) for i in range(ns)])

                    normv = np.max([np.nanmax(subsets['vnorm'][i]) for i in range(ns)])
#                    print('normv', normv)
                    for i in range(ns):
                        nv = len(subsets['xvpos'][i])
#                        subsets['vwidth'][i] = [vwidth*subsets['nov'][i][j]/normv for j in range(nv)]
                        subsets['vwidth'][i] = [vwidth*subsets['vnorm'][i][j]/normv for j in range(nv)]
#                        print("i, subsets['vnorm'][i]", i, subsets['vnorm'][i])
#                        print("i, subsets['vwidth'][i]", i, subsets['vwidth'][i])
                else:
                    for i in range(ns):
                        nv = len(subsets['xvpos'][i])
                        subsets['vwidth'][i] = [vwidth for j in range(nv)]


            # --- now the plotting of the subsets
            for i in range(ns):

                # --- find the valid data points for the subset
                #idgg = set(idgood).intersection(subsets['ids'][i])
                idgg = np.array([z for z in subsets['ids'][i] if z in idgood])
                xsg = np.array(xs[idgg])
                ysg = np.array(ys[idgg])

                ngoods = len(idgg)
                gstr = "$N=$"+str(ngoods)

                xmed = np.median(xsg)
                xstd = np.std(xsg)
                xstr = ("$<X> = "+ "{:4.2f}".format(xmed) + "\pm" +
                        "{:4.2f}".format(xstd) + "$")

                ymed = np.median(ysg)
                ystd = np.std(ysg)
                ystr = ("$<Y> = "+ "{:4.2f}".format(ymed) + "\pm" +
                        "{:4.2f}".format(ystd) + "$")

                scol = subsets['colour'][i]
                slab = subsets['label'][i]
                salpha = subsets['alpha'][i]
                symsize = subsets['symsize'][i]

                # --- plot the data points here
                if violin:
                    nv = len(subsets['xvpos'][i])
                    # --- for the violins, plot every violin independently
                    #    in order to scale their width according to the number
                    #    of objects inside
                    for j in range(nv):
#                        print("i,j,subsets['vwidth'][i][j]", i, j, subsets['vwidth'][i][j])
                        vparts = ax.violinplot(subsets['yv'][i][j],
                                               [subsets['xvpos'][i][j]],
                                               widths=subsets['vwidth'][i][j],
                                               showmedians=True,
                                               points=vpoints,
                                               bw_method=vmethod)

                        # --- the color for the violins have to be set by
                        #     accessing its parts from corresponding dictionary
                        for pc in vparts['bodies']:
                            pc.set_facecolor(scol)
                            pc.set_edgecolor(scol)
                            pc.set_alpha(salpha)

                        vparts['cmedians'].set_color(scol)
                        vparts['cbars'].set_color(scol)
                        vparts['cmins'].set_color(scol)
                        vparts['cmaxes'].set_color(scol)

                else:
                    ax.scatter(xsg, ysg, alpha=salpha,
                               color=scol, s=symsize,
                               label=slab, zorder=i)

                # --- Spearman Correlation rank
#                spear=stats.spearmanr(xsg, ysg)
#
#                spearstr = ("$\\rho_\mathrm{S}=$" + "{:4.2f}".format(spear[0])
#                            + "\n$\log\,p=$" +  "{:4.1f}".format(np.log10(spear[1])))

                # --- Kendall is better!
                kendall=stats.kendalltau(xsg, ysg)

                kendallstr = ("$\\tau_\mathrm{K}=$" + "{:4.2f}".format(kendall[0])
                            + "\n$\log\,p=$" +  "{:4.1f}".format(np.log10(kendall[1])))


                # --- linear regression to the data points

        #        # --- the SCIPY orthogonal regression does not work well
        #        # --- first get an initial guess from simple OLS:
        #        slope, intercept, r_value, p_value, std_err = linregress(xs,ys)
        #
        #        # --- then do orthogonal regression:
        #        model = odr.Model(lin_funct)
        #        data = odr.Data(xs, ys)
        #        init = odr.ODR(data, model, beta0=[slope, intercept])
        #        odrout = init.run()
                a,b,aerr,berr,covab = _bces(xsg-xmid,0*xsg,ysg-ymid,0*ysg,0*xsg)
                # ---
                # a[0] = y|x
                # a[1] = x|y
                # a[2] = bisector (do not use)
                # a[3] = orthogonal

                # --- plot the linear (orthogonal) fit
                xf = np.array([xmin,xmax])
                ax.plot(xf, ymid+b[3]+a[3]*(xf-xmid), color=scol, zorder=-32, ls='--', alpha=0.5)
                fitstr = "$y=$" + "{:4.2f}".format(a[3]) + "$x + $"+ "{:4.2f}".format(b[3])

                labstr = (slab + '\n' +
                          gstr + '\n' +
                          kendallstr + '\n' +
                          fitstr
                          )

                # --- decide the position for the label for the subset
                #     4 subsets is the current maximum
                if i == 0:
                    labx = 0.05
                    laby = 0.95
                    labva = 'top'
                    labha = 'left'
                elif i == 1:
                    labx = 0.95
                    laby = 0.95
                    labva = 'top'
                    labha = 'right'
                elif i == 2:
                    labx = 0.95
                    laby = 0.05
                    labva = 'bottom'
                    labha = 'right'
                else:
                    labx = 0.05
                    laby = 0.05
                    labva = 'bottom'
                    labha = 'left'

                # --- put the label for the subset with some parameters
                ax.text(labx, laby, labstr, transform=ax.transAxes,
                                 verticalalignment=labva,
                                 horizontalalignment=labha, fontsize=5,
                                 color=scol)


                # --- plotting of histogram if in the right position
                if x == nx-2:
                    # --- histograms for the y quantities (at the end of each
                    #     row)
                    # --- first plot the transparent area of the histo
                    hax.hist(ysg, facecolor=scol,
                                                bins=ybins, edgecolor=scol,
                                                histtype='stepfilled',
                                                alpha=salpha,
                                                orientation='horizontal')

                    # --- then plot the line of the histo
                    hax.hist(ysg, facecolor=scol, bins=ybins, edgecolor=scol,
                             linewidth=1, histtype='step',
                             orientation='horizontal')

                    # --- overplot the median line
                    hax.plot([0,np.max(yhmax)], [ymed,ymed], color=scol,
                              zorder=32, ls='--', alpha=0.5)

                    # --- do the Shapiro-Wilk test to see if the distribution
                    #     is consistent with a Normal distribution
                    if len(ysg) >= 3:
                        shap = stats.shapiro(ysg)
                    else:
                        shap = [0, 0]
                    shap = stats.shapiro(ysg)
                    shapstr = "$p_\mathrm{S} = " + "{:4.2f}".format(shap[1]) + "$"

                    # --- label for subset including some values
                    labstr = (slab + '\n' +
                              ystr + '\n' +
                              shapstr)

                    hax.text(labx, laby, labstr, transform=hax.transAxes,
                                 verticalalignment=labva,
                                 horizontalalignment=labha, fontsize=5,
                                 color=scol)

                # --- histogram for the x quantities in the top row
                if y == 0:
                    hax2.hist(xsg, facecolor=scol,
                                                  bins=xbins, edgecolor=scol,
                                                  histtype='stepfilled',
                                                  alpha=salpha)

                    hax2.hist(xsg, facecolor=scol, bins=xbins, edgecolor=scol,
                              linewidth=1, histtype='step')


                    hax2.plot([xmed,xmed], [0,np.max(xhmax)], color=scol,
                              zorder=32, ls='--', alpha=0.5)

                    if len(xsg) >= 3:
                        shap = stats.shapiro(xsg)
                    else:
                        shap = [0, 0]
                    shapstr = "$p_\mathrm{S} = " + "{:4.2f}".format(shap[1]) + "$"

                    labstr = (slab + '\n' +
                              xstr + '\n' +
                              shapstr)

                    hax2.text(labx, laby, labstr, transform=hax2.transAxes,
                                 verticalalignment=labva,
                                 horizontalalignment=labha, fontsize=5,
                                 color=scol)

            # --- give 10% edge to the plotting range to have no points on the
            #     axis
            xbord = 0.1*xran
            ybord = 0.1*yran
            ax.set_xlim(xmin-xbord, xmax+xbord)
            ax.set_ylim(ymin-ybord, ymax+ybord)

            # --- take care of the axis labels
            if x == 0:
                ax.set_ylabel('$'+ylabels[y]+'$')
            else:
               ax.get_yaxis().set_ticklabels([])

            if y == ny-2:
                ax.set_xlabel('$'+xlabels[x]+'$')
            else:
                ax.get_xaxis().set_ticklabels([])

            # --- axis labels for the histograms
            if x == nx-2:
                hax.set_ylim(ymin-ybord, ymax+ybord)
                hax.tick_params(labelleft='off',labelright='on')
                hax.set_ylabel('$'+ylabels[y]+'$')
                hax.yaxis.set_label_position('right')

            if y == 0:
                hax2.set_xlim(xmin-xbord, xmax+xbord)
                #print(x,y,xmin, xmax, xbord)
                #hax2.xaxis.tick_top()
                hax2.tick_params(labelbottom='off',labeltop='on')
                hax2.set_xlabel('$'+xlabels[x]+'$')
                hax2.xaxis.set_label_position('top')
                #pos = hax2.get_position()
#                pos2 = [pos.x0, pos.y0 + 0.1 * pos1.height ,  pos1.width*0.8, pos1.height*0.9]
#    ax.set_position(pos2) # set a new position

                #print(x,y,pos)
                #print(x,y,pos.x0, pos.y0, pos.height, pos.width)
                #hax2.tick_params(labeltop=True)
                #hax2.get_xaxis().set_ticklabels([])

    # --- adjust the y-range for the histograms so that the y-axis label
    #     applies to all
    xhmaxtot = np.max(xhmax)
    for i in range(len(haxes2)):
        if histlog:
            haxes2[i].set_yscale('log')
            haxes2[i].set_ylim(0.5,xhmaxtot)
        else:
            haxes2[i].set_ylim(0,xhmaxtot)
        if i > 0:
            haxes2[i].get_yaxis().set_ticklabels([])


    yhmaxtot = np.max(yhmax)
    for i in range(len(haxes)):
        if histlog:
            haxes[i].set_xscale('log')
            haxes[i].set_xlim(0.5,yhmaxtot)
        else:
            haxes[i].set_xlim(0,yhmaxtot)
        if i < len(haxes)-1:
            haxes[i].get_xaxis().set_ticklabels([])


    plt.savefig(outname,bbox_inches='tight')
    plt.clf()
    plt.close(fig)

