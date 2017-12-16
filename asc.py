#! /usr/bin/env python 
#
#   Copyright (C) Eugene Duff 2017 University of Oxford
#
# asc.py 
#
# Author: Eugene Duff <eugene.duff@gmail.com>
#
#   Part of FSL - FMRIB's Software Library
#   http://www.fmrib.ox.ac.uk/fsl
#   fsl@fmrib.ox.ac.uk
#   
#   Developed at FMRIB (Oxford Centre for Functional Magnetic Resonance
#   Imaging of the Brain), Department of Clinical Neurology, Oxford
#   University, Oxford, UK
#   
#   
#   LICENCE
#   
#   FMRIB Software Library, Release 5.0 (c) 2012, The University of
#   Oxford (the "Software")
#   
#   The Software remains the property of the University of Oxford ("the
#   University").
#   
#   The Software is distributed "AS IS" under this Licence solely for
#   non-commercial use in the hope that it will be useful, but in order
#   that the University as a charitable foundation protects its assets for
#   the benefit of its educational and research purposes, the University
#   makes clear that no condition is made or to be implied, nor is any
#   warranty given or to be implied, as to the accuracy of the Software,
#   or that it will be suitable for any particular purpose or for use
#   under any specific conditions. Furthermore, the University disclaims
#   all responsibility for the use which is made of the Software. It
#   further disclaims any liability for the outcomes arising from using
#   the Software.
#   
#   The Licensee agrees to indemnify the University and hold the
#   University harmless from and against any and all claims, damages and
#   liabilities asserted by third parties (including claims for
#   negligence) which arise directly or indirectly from the use of the
#   Software or the sale of any products based on the Software.
#   
#   No part of the Software may be reproduced, modified, transmitted or
#   transferred in any form or by any means, electronic or mechanical,
#   without the express permission of the University. The permission of
#   the University is not required if the said reproduction, modification,
#   transmission or transference is done without financial return, the
#   conditions of this Licence are imposed upon the receiver of the
#   product, and all original and amended source code is included in any
#   transmitted product. You may be held legally responsible for any
#   copyright infringement that is caused or encouraged by your failure to
#   abide by these terms and conditions.
#   
#   You are not permitted under this Licence to use this Software
#   commercially. Use for which any financial return is received shall be
#   defined as commercial use, and includes (1) integration of all or part
#   of the source code or the Software into a product for sale or license
#   by or on behalf of Licensee to third parties or (2) use of the
#   Software or any derivative of it for research with the final aim of
#   developing software products for sale or license to a third party or
#   (3) use of the Software or any derivative of it for research with the
#   final aim of developing non-software products for sale or license to a
#   third party, or (4) use of the Software to provide any service to an
#   external organisation for which payment is received. If you are
#   interested in using the Software commercially, please contact Isis
#   Innovation Limited ("Isis"), the technology transfer company of the
#   University, to negotiate a licence. Contact details are:
#   innovation@isis.ox.ac.uk quoting reference DE/9564.
""" Additive Signal Change Analysis

This module provides tools for estimating and plotting Additive Signal Change (ASC). Functions are provided for both netmat based analyses, and image based seed analyses (using nibabel). 

Provides a procedural interface primarily for interactively. 

Submodules include: 
    :mod:`asc_funcs`
        provides helper functions
Todo:
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning,module='sklearn') 
warnings.filterwarnings('ignore', module='matplotlib')
warnings.filterwarnings('ignore', module='numpy')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os,argparse,glob,logging,pickle,datetime
import scipy.stats as stats
import numpy as np
import asc_funcs
from asc_funcs import flattenall as fa
from colorline import colorline
from operator import mul
from fractions import Fraction

from matplotlib.pylab import find
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.backends.backend_pdf import PdfPages

# optional
try:
    import mne
    from mne.viz import plot_connectivity_circle
except:
    print("MNE not detected, no visualisation")
FinalFig2.pdf

np.seterr(divide='ignore', invalid='ignore')

#####################################
##  main

def _main(input_dir='.',design=None,inds1=None,inds2=None,subj_order=False,pcorrs=False,min_corr_diff=0,out_base='asc',prefix='dr_stage1',errdist_perms=0,exclude_conns=True,data_pre='',pctl=5,neg_norm=False,nosubj=False,rel=True):
    """  Perform additive signal analysis. """ 
    # initiate figure
    fig = plt.figure(figsize=(20.27, 11.69))

    # logging
    logdir = os.path.join(input_dir, 'logs')

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    logging.basicConfig(filename=os.path.join(logdir, 'asc.log'), level=logging.DEBUG)
    logging.info('ASC analysis: %s', datetime.datetime.now())
    logging.info('Workdir:\t\t%s', input_dir)

    AB_con=plot_conn(dir=input_dir,design=design,inds1=inds1,inds2=inds2,fig=fig,errdist_perms=errdist_perms,prefix=prefix,exclude_conns=bool(exclude_conns),data_pre=data_pre,savefig=out_base+'.png',pctl=float(pctl),min_corr_diff=min_corr_diff,pcorrs=pcorrs,rel=rel,neg_norm=neg_norm,nosubj=nosubj,nofig=False)
    
    outdata=open(out_base+'.npz','wb')
    pickle.dump(AB_con,outdata)
    outdata.close()

    return()

################################################################################
# calculate and plot connectivity matrices from a dual-regression directory

def plot_conn(dir=".",inds1=None,inds2=None,fig=None,errdist_perms=0,prefix='dr_stage1',exclude_conns=True,data_pre='',design=None, savefig='',pctl=5,min_corr_diff=0,pcorrs=False,neg_norm=True,nosubj=False,subj_order=True,nofig=False,rel=True,fdr_alpha=0.2):
    """  Perform Additive Signal Change analysis and plot results.
    Key arguments
    Optional:
        dir -- directory (default: current dir)
        inds1 -- indexes of dual regression files corresponding to group 1
        inds2 -- indexes of dual regression files corresponding to group 2
        design -- FEAT style design file specifying samples according to +1, -1
        errdist_perms -- number of permutations for estimating the expected distribution
    """

    # load indexes
    if isinstance(inds1,str):
        inds1 = np.loadtxt(inds1).astype(int)
        inds2 = np.loadtxt(inds2).astype(int)

    # load FSL design file
    if isinstance(design,str):
        design=np.loadtxt(design,skiprows=5)

        inds1 = np.where(design[:,0]==1)[0]
        inds2 = np.where(design[:,0]==-1)[0]

    # if necessary, read dual_regression directory to determine indices
    if inds1 is None:

        dr_files=np.sort(glob.glob(prefix+'_subject?????.txt'))  
        npts = len(dr_files) 

        if subj_order:
            inds1 = np.arange(npts/2)
            inds2 = np.arange(npts/2) + npts/2
        else:
            inds1 = np.arange(0,npts,2)
            inds2 = np.arange(1,npts,2)

    # read data for two states
    A = asc_funcs.dr_loader(dir,subj_inds=inds1,prefix=prefix,nosubj=nosubj)
    B = asc_funcs.dr_loader(dir,subj_inds=inds2,prefix=prefix,nosubj=nosubj)
    # generate contrast
    AB_con = asc_funcs.FC_con(A,B)

    # calculate limits on contrast
    [AB_con, inds_plots] = plot_conn_stats(AB_con,fig,errdist_perms=errdist_perms,exclude_conns=exclude_conns,savefig=savefig,pctl=pctl,min_corr_diff=min_corr_diff,neg_norm=True,refresh=True,nofig=False,rel=rel,fdr_alpha=fdr_alpha)

    return(AB_con)

################################################################################
# plot_conn_stats 


def plot_conn_stats(AB_con,fig,flatten=True,errdist_perms=0,pctl=5,min_corr_diff=0,pcorrs=False,neg_norm=True,fdr_alpha=0.2,exclude_conns=True,savefig=None,ccstatsmat=None,inds_cc=None,vvstatsmat=None,refresh=False,nofig=False,rel=True):
    """ generates correlation and ASC stats for a pair of states. """

    # generate basic correlation and variance stats 
    if ccstatsmat is None:
        inds_cc = find(mne.stats.fdr_correction(fa(AB_con.get_corr_stats(pcorrs=pcorrs,rel=rel)[1]),alpha=fdr_alpha)[0])
        ccstatsmat=-fa(AB_con.get_corr_stats(pcorrs=pcorrs,rel=rel)[0])
        vvstatsmat=-AB_con.get_std_stats(pcorrs=pcorrs,rel=rel)

    # generate ASC limits 
    lims=AB_con.get_ASC_lims(pcorrs=pcorrs,errdist_perms=errdist_perms,refresh=refresh,pctl=pctl)

    # gen correlation matrices for plotting
    Acorrs_mat=np.mean(AB_con.A.get_corrs(pcorrs=pcorrs),0,keepdims=True)
    Acorrs=fa(Acorrs_mat)   # flattening
    Bcorrs_mat=np.mean(AB_con.B.get_corrs(pcorrs=pcorrs),0,keepdims=True)
    Bcorrs=fa(Bcorrs_mat)  # flattening

    # min_corr_diff is the minimum change in correlation considered interesting (significant but tiny effects may not be interesting)
    if min_corr_diff != 0:
        inds_corr_diff = find(abs(Acorrs-Bcorrs)>min_corr_diff)
        inds_cc = np.intersect1d(inds_corr_diff,inds_cc)

    ##################################
    # set plot colours and other specs

    plot_colors=[(0.2,0.6,1),(0.62,0.82,0.98),(0.40,0.95,0.46),(0.6,0.95,0.6),(0.15,0.87,0.87),(0.8,0.8,0.8)]

    cdict1 = {'red':   ((0.0, 0.0, 0.0),
                       (0.75, 0.5, 0.5),
                       (1.0, 1.0, 1.0)),
             'green': ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0)),
             'blue':  ((0.0, 0.0, 1.0),
                       (0.25, 0.5, 0.5),
                       (1.0, 0, 0))}

    cdict2 = {'red':   ((0.0, 0.0, 0.0),
                       (0.5, 0.0, 0.1),
                       (1.0, 1.0, 1.0)),
             'green': ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0)),
             'blue':  ((0.0, 0.0, 1.0),
                       (0.5, 0.1, 0.0),
                       (1.0, 0.0, 0.0))}

    blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)
    plt.register_cmap(cmap=blue_red1)
    cmap2=matplotlib.colors.ListedColormap(name='Test',colors=plot_colors)
    plt.register_cmap(cmap=cmap2)

    fontsize=9 

    current_palette = [(0.2980392156862745, 0.4470588235294118, 0.6901960784313725), 
            (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
            (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
            (0.5058823529411764, 0.4470588235294118, 0.6980392156862745),
            (0.8, 0.7254901960784313, 0.4549019607843137),
            (0.39215686274509803, 0.7098039215686275, 0.803921568627451)]
    sb_cols=current_palette + current_palette + current_palette 

    # colours for variance stats

    vcols=[]
    # scaling stats for plotting colours for variance change
    vv_norm=vvstatsmat/6
    vv_norm[np.isnan(vv_norm)]=0
    n_nodes = AB_con.A.get_covs().shape[1]

    for a in np.arange(n_nodes):
        vcols.append(blue_red1((vv_norm[a]+1)/2))

    vmax=3
    vmin=-3

    # node info
    node_angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    height=np.ones(n_nodes)*4
    dist_mat = node_angles[None, :] - node_angles[:, None]
    dist_mat[np.diag_indices(n_nodes)] = 1e9
    node_width = np.min(np.abs(dist_mat))
    node_edgecolor='black'
    group_cols=[]

    # colours for ROIs
    ROI_info = AB_con.A.ROI_info
    for a in ROI_info['ROI_RSNs']:     
        val=1-0.35*a/max(ROI_info['ROI_RSNs'])
        group_cols.append((val*0.8,val*0.9,val))

    plots = ['uncorrelated','common','additive','other']
    titles= {'uncorrelated':"Addition of uncorrelated signal", 'common':"Addition of common signal",'additive':"Mixed additive signals",'other':"Changes not explained \n by additive signal changes"}

    #################################
    # plot data
     
    indices=np.triu_indices(n_nodes,1)
    notin=inds_cc.copy()
    inds_plots={}

    for plot in plots[:3]:
        if errdist_perms>0:
            inds_plots[plot]=np.intersect1d(inds_cc,find(fa(lims[plot]['pctls'])))
            notin = np.setdiff1d(notin,find(fa(lims[plot]['pctls'])))
            minstr='min_pctls'
            maxstr='max_pctls'
        else:
            inds_plots[plot]=np.intersect1d(inds_cc,find(fa(lims[plot]['pctls_noerr'])))
            notin = np.setdiff1d(notin,find(fa(lims[plot]['pctls_noerr'])))
            minstr='min'
            maxstr='max'

    inds_plots['other'] = notin
    if exclude_conns:
        inds_plots['common']=np.setdiff1d(inds_plots['common'],inds_plots['uncorrelated'])
        inds_plots['additive']=np.setdiff1d(inds_plots['additive'],inds_plots['common'])
        inds_plots['additive']=np.setdiff1d(inds_plots['additive'],inds_plots['uncorrelated'])

    plotccstats= ccstatsmat.astype(float)

    # flip color of changes to negative corrs (neg_norm option)
    if neg_norm==True:
       plotccstats= plotccstats*np.sign(Acorrs)

    cnt=-1

    #################################
    # produce the four plots for the four ASC classes 

    if fig != None:
        fig.clf()

        for plot in plots:

            cnt+=1

            ################################################
            # mne plot function
            # TODO: test for mne / nofig 

            pp=plot_connectivity_circle(plotccstats.flatten()[inds_plots[plot]],ROI_info['ROI_names'][0:n_nodes],(indices[0][inds_plots[plot]],indices[1][inds_plots[plot]]),fig=fig,colormap='BlueRed1',vmin=vmin,vmax=vmax,node_colors=vcols,subplot=241+cnt,title=titles[plot],interactive=True,fontsize_names=fontsize,facecolor='w',colorbar=False,node_edgecolor=node_edgecolor,textcolor='black',padding=3,node_linewidth=0.5) 

            # titles
            ax=plt.gca()
            ax.set_title(titles[plot],color='black') 
            
            #  color node faces
            bars = pp[1].bar(node_angles, height*2.2, width=node_width, bottom=10.4, \
                            edgecolor='0.9', lw=2, facecolor='.9', \
                            align='center',linewidth=1)

            for bar, color in zip(bars, group_cols):
                bar.set_facecolor(color)
                bar.set_edgecolor(color)

            # plot correlation info below circle plots
            if plot=='other': 
                plotrange='additive'
            else:
                plotrange=plot

            #  sorting plotting only those indices requested 
            sort_array=np.zeros((len(inds_plots[plot]),),dtype=('f4,f4'))
            sort_array['f0']=fa(lims[plotrange][minstr])[0,inds_plots[plot]]  # 
            sort_array['f1']=fa(lims[plotrange][maxstr])[0,inds_plots[plot]]  # 
            ii=np.argsort(sort_array,order=['f0','f1'])

            # plot conn info
            if len(ii)>0: 
                # width of nodes       
                width=np.max((20,len(ii)+10))
                # ii_ext
                ii_ext=np.r_[ii[0],ii,ii[-1]]

                # fbwx: midpoints for under plots
                fbwx=np.arange(len(ii_ext))+(width - len(ii_ext))/2.
                fbwx[0]=fbwx[0]+0.5  #=np.r_[fbwx[0]-0.5, fbwx,fbwx[-1]+0.5]
                fbwx[-1]=fbwx[-1]-0.5  

                #  axis settings
                ax=plt.subplot(245+cnt,axisbg='white')
                ax.set_ylim([-1.,1])
                ax.set_yticks([-1,0,1])
                ax.set_yticks([-0.75,-.25,0,0.25,.5,.75,1],minor=True)
                ax.yaxis.grid(color=[0.7,.95,.95],linestyle='-',linewidth=.5,which='minor')
                ax.yaxis.grid(color=[0.65,.85,.85],linestyle='-',linewidth=2,which='major')

                # first plot bands for ASC / uncorr / common  (fill between)
                if len(fbwx)==1:            
                    # if only one element  
                    plt.fill_between(np.r_[fbwx-0.5,fbwx+0.5],np.r_[fa(lims['additive'][minstr])[0,inds_plots[plot]][ii_ext],fa(lims['additive'][minstr])[0,inds_plots[plot]][ii_ext]],np.r_[fa(lims['additive'][maxstr])[0,inds_plots[plot]][ii_ext],fa(lims['additive'][maxstr])[0,inds_plots[plot]][ii_ext]] ,color='Grey',alpha=0.4)
                    plt.fill_between(np.r_[fbwx-0.5,fbwx+0.5],np.r_[fa(lims['common'][minstr])[0,inds_plots[plot]][ii_ext],fa(lims['common'][minstr])[0,inds_plots[plot]][ii_ext]],np.r_[fa(lims['common'][maxstr])[0,inds_plots[plot]][ii_ext],fa(lims['common'][maxstr])[0,inds_plots[plot]][ii_ext]] ,color='Blue',alpha=0.4)

                    plt.fill_between(np.r_[fbwx-0.5,fbwx+0.5],np.r_[fa(lims['uncorrelated'][minstr])[0,inds_plots[plot]][ii_ext],fa(lims['uncorrelated'][minstr])[0,inds_plots[plot]][ii_ext]],np.r_[fa(lims['uncorrelated'][maxstr])[0,inds_plots[plot]][ii_ext],fa(lims['uncorrelated'][maxstr])[0,inds_plots[plot]][ii_ext]] ,color='Green',alpha=0.6)
                else:
                    # if multple elements  
                    plt.fill_between(fbwx,fa(lims['additive'][minstr])[0,inds_plots[plot]][ii_ext],fa(lims['additive'][maxstr])[0,inds_plots[plot]][ii_ext],color=[0.67,0.76,0.85])
                    plt.fill_between(fbwx,fa(lims['common'][minstr])[0,inds_plots[plot]][ii_ext],fa(lims['common'][maxstr])[0,inds_plots[plot]][ii_ext],color='Blue',alpha=0.4)
                    plt.fill_between(fbwx,fa(lims['uncorrelated'][minstr])[0,inds_plots[plot]][ii_ext],fa(lims['uncorrelated'][maxstr])[0,inds_plots[plot]][ii_ext],color='Green',alpha=0.6)
                
                if neg_norm == True:
                    iipospos=np.in1d(ii,find(abs(Acorrs[0,inds_plots[plot]])>abs(Bcorrs[0,inds_plots[plot]])))
                    iinegpos=np.in1d(ii,find(abs(Acorrs[0,inds_plots[plot]])<abs(Bcorrs[0,inds_plots[plot]])))
                else:
                    iipospos=np.in1d(ii,find(Acorrs[0,inds_plots[plot]]>Bcorrs[0,inds_plots[plot]]))
                    iinegpos=np.in1d(ii,find(Acorrs[0,inds_plots[plot]]<Bcorrs[0,inds_plots[plot]]))

                iipos=ii[iipospos]
                iineg=ii[iinegpos]

                xes = np.arange(len(ii))+(width - len(ii))/2.
                
                # now plot correlation in A and B conditions
                plt.plot(np.array([xes,xes])[:,find(iipospos)],[Acorrs[0,inds_plots[plot][iipos]],Bcorrs[0,inds_plots[plot][iipos]]],color=[0,0,1],alpha=1,linewidth=1.5,zorder=1)
                plt.plot(np.array([xes,xes])[:,find(iinegpos)],[Acorrs[0,inds_plots[plot][iineg]],Bcorrs[0,inds_plots[plot][iineg]]],color=[1,0,0],alpha=1,linewidth=1.5,zorder=1)

                plt.fill_between(fbwx,fa(lims['uncorrelated'][minstr])[0,inds_plots[plot]][ii_ext],fa(lims['uncorrelated'][maxstr])[0,inds_plots[plot]][ii_ext],color='Green',alpha=0.6)
                line3 = plt.Rectangle((0, 0), 0, 0,color=current_palette[0])
                ax.add_patch(line3)
                ax.set_xticks([])

                # plot points
                line2=plt.scatter((xes)[find(iipospos)],Bcorrs[0,inds_plots[plot][iipos]].T,color='blue',zorder=2)
                line2=plt.scatter((xes)[find(iinegpos)],Bcorrs[0,inds_plots[plot][iineg]].T,color='red',zorder=2)
                line2=plt.scatter((xes)[find(iipospos)],Acorrs[0,inds_plots[plot][iipos]].T,color='white',zorder=2)
                line2=plt.scatter((xes)[find(iinegpos)],Acorrs[0,inds_plots[plot][iineg]].T,color='white',zorder=2)

                # plot line between, colouring line two according to pos or neg change
                cmap=ListedColormap([(0.2980392156862745, 0.4470588235294118, 0.6901960784313725), (0.3333333333333333, 0.6588235294117647, 0.40784313725490196), (0.7686274509803922, 0.3058823529411765, 0.3215686274509804)])
                norm= BoundaryNorm([-2,0,1,2],cmap.N)
                z=np.zeros(xes.shape[0]+1,)

                # plot network membership above
                colorline(fbwx[:-1], z+1.05,ROI_info['ROI_RSNs'][indices[0][np.r_[inds_plots[plot],inds_plots[plot][-1]]]]-1.5,cmap=cmap,norm=norm,linewidth=5)
                colorline(fbwx[:-1], z+1.1,ROI_info['ROI_RSNs'][indices[1][np.r_[inds_plots[plot],inds_plots[plot][-1]]]]-1.5,cmap=cmap,norm=norm,linewidth=5)
                plt.show()

        if savefig!=None:
            if nofig==23: 
                pp = PdfPages(fname)
                fig.tight_layout(h_pad=1, pad=4)
                pp.savefig(fig=fig)
                pp.close()  
            else:
                fig.savefig(savefig)

    # add stats to AB_con struct
    AB_con.lims['covs']['inds_plots']=inds_plots 
    AB_con.lims['covs']['cc_stats']=ccstatsmat
    AB_con.lims['covs']['vv_stats']=vvstatsmat

    return(AB_con,inds_plots)

################################################################################
##  command line funcs

if __name__=="__main__":
    """ Main commnd """

    DESC = "FMRI Additive Signal Analysis"
        
    PARSER = argparse.ArgumentParser(description=DESC, argument_default=argparse.SUPPRESS)

    requiredArgs = PARSER.add_argument_group('required arguments')
    requiredArgs.add_argument('-i', '--input_dir', help='dual_regression dir', required=True)

    optionalArgs = PARSER.add_argument_group('optional arguments')
    optionalArgs.add_argument('-d', '--design', help='design file', required=False)
    optionalArgs.add_argument('--inds1', help='index file 1', required=False)
    optionalArgs.add_argument('--inds2', help='index file 2', required=False)

    optionalArgs.add_argument('-o', '--out_base', help='output base file name [asc]', required=False)
    optionalArgs.add_argument('--pcorrs', help='use partial correlation', required=False,type=bool)
    optionalArgs.add_argument('--errdist_perms', help='permutations for monte carlo', required=False,type=int)
    optionalArgs.add_argument('--min_corr_diff', help='minimum correlation change', required=False,type=float)
    optionalArgs.add_argument('--prefix', help='dr prefix', required=False)
    optionalArgs.add_argument('--pctl', help='percentile of connections to show (FDR)', required=False,type=float)
    optionalArgs.add_argument('--subj_order', help='subject_order', required=False,type=bool)
    optionalArgs.add_argument('--exclude_conns', help='exclude connections', required=False, type=bool)
    
    ARGS = PARSER.parse_args()

    _main(**vars(ARGS))


