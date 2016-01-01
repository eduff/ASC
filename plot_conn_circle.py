import matplotlib

from matplotlib.colors import LinearSegmentedColormap
import scipy.stats as stats
import max_corr_funcs as cf
import mne, ml_funcs
from ml_funcs import flattenall as fa
import seaborn as sns
import numpy as np
import scipy, os.path
import matplotlib.pyplot as plt
from matplotlib.pylab import find
import itertools
import colorline
from colorline import colorline
from matplotlib.colors import ListedColormap, BoundaryNorm
from operator import mul
from fractions import Fraction
from mne.viz import plot_connectivity_circle
rtoz=ml_funcs.rtoz

def plot_conn(dir,inds1,inds2,fig,flatten=True):
    
    data_pre='' # out_' 
    prefix='dr_stage1'
    filt=0
    img=0


    plotcolors=[(0.2,0.6,1),(0.62,0.82,0.98),(0.40,0.95,0.46),(0.6,0.95,0.6),(0.15,0.87,0.87),(0.8,0.8,0.8)]

    la=np.logical_and

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
    cmap2=matplotlib.colors.ListedColormap(name='Test',colors=plotcolors)
    plt.register_cmap(cmap=cmap2)

    current_palette = sns.color_palette()
    sb_cols=current_palette + current_palette + current_palette 
    group_cols=[]

    if os.path.isfile(data_pre+'ROI_order.txt'):
        ROI_order=np.loadtxt(data_pre+'ROI_order.txt').astype(int)-1

    if os.path.isfile(data_pre+'ROI_RSNs.txt'):
        ROI_RSNs=(np.loadtxt(data_pre+'ROI_RSNs.txt').astype(int))
    else:
        ROI_RSNs=np.np.arange(len(ROI_order))

    if os.path.isfile('goodnodes.txt'):
        goodnodes=np.loadtxt('goodnodes.txt').astype(int)
        ROI_RSNs=ROI_RSNs[np.in1d(ROI_order,goodnodes)]
        ROI_order=ROI_order[np.in1d(ROI_order,goodnodes)]
    else:
        ROI_RSNs=ROI_RSNs[ROI_order]

    if os.path.isfile('ROI_RSN_include.txt'):
        ROI_RSN_include=(np.loadtxt('ROI_RSN_include.txt').astype(int))
        ROI_order = ROI_order[np.in1d(ROI_RSNs,ROI_RSN_include)]
        ROI_RSNs = ROI_RSNs[np.in1d(ROI_RSNs,ROI_RSN_include)]

    if os.path.isfile('ROI_names.txt'):
        ROI_names=np.array(open('ROI_names.txt').read().splitlines())
    else:
        ROI_names=[ str(a) for a in np.arange(len(ROI_RSNs)) ]



    ROI_names=ROI_names[ROI_order]
    n_nodes=len(ROI_names)

    A_orig=cf.dr_loader('.',subj_ids=inds1,ROIs=ROI_order)
    B_orig=cf.dr_loader('.',subj_ids=inds2,ROIs=ROI_order)

    #if flatten():
    
    ccstatsmat=-fa(stats.ttest_rel(rtoz(A_orig.get_corrs()),rtoz(B_orig.get_corrs()))[0])
    inds_cc=find(mne.stats.fdr_correction(scipy.stats.norm.sf(abs(ccstatsmat)),alpha=0.05)[0])

    vvstatsmat=-(stats.ttest_rel(A_orig.get_stds(),B_orig.get_stds())[0])
    inds_vv=find(mne.stats.fdr_correction(scipy.stats.norm.sf(abs(vvstatsmat)),alpha=0.05)[0])

    A=cf.flatten_tcs(A_orig)
    B=cf.flatten_tcs(B_orig)

    Acorrs=fa(A.get_corrs())
    Bcorrs=fa(B.get_corrs())

    vv_norm=vvstatsmat/3
    vcols=[]
    for a in np.arange(n_nodes):
        vcols.append(blue_red1((vv_norm[a]+1)/2))
    #vvs_s=squeeze(vvs_s)
    #vvst_s=squeeze(vvst_s)

    indices=np.triu_indices(n_nodes,1)

    node_angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    height=np.ones(n_nodes)*3
    dist_mat = node_angles[None, :] - node_angles[:, None]
    dist_mat[np.diag_indices(n_nodes)] = 1e9
    node_width = np.min(np.abs(dist_mat))
    node_edgecolor='black'
    
    fig.clf()

    # calculate variance
    #for a in ndindex(varmat.shape[0:2]):
    #    varvarmat[a][ix_(np.arange(varmat.shape[-1]),np.arange(varmat.shape[-1]))]=tile(varmat[a],(rois,1))
    #sz=varmat.shape[-1]

    lims=cf.corr_lims_all(A,B,errdist=True,errdist_perms=50,pctl_shared=20,pctl=20,dof=700)
    plots = ['unshared','shared','combined','other']
    titles= {'unshared':"Change in unshared signal", 'shared':"Change in shared signal",'combined':"Change in both shared and unshared signals",'other':"Change in synchronisation"}
    notin=inds_cc
    inds_plots={}
    vmax=3
    vmin=-3


    for plot in plots[:3]:
        inds_plots[plot]=np.intersect1d(inds_cc,find(fa(lims[plot]['pctls'])))
        notin = np.setdiff1d(notin,find(fa(lims[plot]['pctls'])))
    inds_plots['other'] = notin

    inds_plots['combined']=np.setdiff1d(inds_plots['combined'],inds_plots['shared'])
    inds_plots['combined']=np.setdiff1d(inds_plots['combined'],inds_plots['unshared'])

    cnt=-1
    fontsize=7 

    for plot in plots:
        cnt+=1
        pp=plot_connectivity_circle(ccstatsmat.astype(float).flatten()[inds_plots[plot]],ROI_names[0:n_nodes],(indices[0][inds_plots[plot]],indices[1][inds_plots[plot]]),fig=fig,colormap='BlueRed1',vmin=vmin,vmax=vmax,node_colors=vcols,subplot=241+cnt,title=titles[plot],interactive=True,fontsize_names=fontsize,facecolor='w',colorbar=False,node_edgecolor=node_edgecolor,textcolor='black') 

        ax=plt.gca()
        ax.set_title(titles[plot],color='black')

        bars = pp[1].bar(node_angles, height, width=node_width, bottom=15.5, \
                        edgecolor=node_edgecolor, lw=2, facecolor='.9', \
                        align='center',linewidth=0)

        for a in ROI_RSNs:                          
            group_cols.append(sb_cols[a-1])

        for bar, color in zip(bars, group_cols):
            bar.set_facecolor(color)
            bar.set_edgecolor(color)


        ax.text(.8,18,'DMN',size=10,color='black')
        ax.text(2.8,21,'Motor',size=10,color='black')
        ax.text(5,18,'Visual',size=10,color='black')

        sort_array=np.zeros((len(inds_plots[plot]),),dtype=('f4,f4'))

        if plot=='other':
            plotrange='combined'
        else:
            plotrange=plot

        sort_array['f0']=fa(lims[plotrange]['pctls_min'])[0,inds_plots[plot]]
        sort_array['f1']=fa(lims[plotrange]['pctls_max'])[0,inds_plots[plot]]
        ii=np.argsort(sort_array,order=['f0','f1'])
    
        ax=plt.subplot(245+cnt,axisbg='white')

        if len(ii)>0: 
            #if a==0:
            #    ii2=ii[in1d(ii,find(ccmat1>ccmat2))]
            #elif a==1:
            #    ii2=ii[in1d(ii,find(ccmat1<ccmat2))]
            #else:
            #    ii2=ii[in1d(ii,setdiff1d(arange(len(changes)),inds_orig))]
            width=np.max((20,len(ii)+10))

            fbwx=np.arange(len(ii))+(width - len(ii))/2.
            fbwx[0]=fbwx[0]-0.5
            fbwx[-1]=fbwx[-1]+.5

            #ax.set_yticks([0,1])
            ax.set_yticks([0,1])
            ax.set_yticks([-.25,0,0.25,.5,.75,1],minor=True)
            ax.yaxis.grid(color=[0.7,.95,.95],linestyle='-',linewidth=.5,which='minor')
            ax.yaxis.grid(color=[0.65,.85,.85],linestyle='-',linewidth=2,which='major')

            if len(fbwx)==1:            
                plt.fill_between(r_[fbwx-0.5,fbwx+0.5],r_[fa(lims['combined']['pctls_min'])[0,inds_plots[plot]][ii],fa(lims['combined']['pctls_min'])[0,inds_plots[plot]][ii]],r_[fa(lims['combined']['pctls_max'])[0,inds_plots[plot]][ii],fa(lims['combined']['pctls_max'])[0,inds_plots[plot]][ii]] ,alpha=0.4)
                plt.fill_between(r_[fbwx-0.5,fbwx+0.5],r_[fa(lims['unshared']['pctls_min'])[0,inds_plots[plot]][ii],fa(lims['unshared']['pctls_min'])[0,inds_plots[plot]][ii]],r_[fa(lims['unshared']['pctls_max'])[0,inds_plots[plot]][ii],fa(lims['unshared']['pctls_max'])[0,inds_plots[plot]][ii]] ,color='Green',alpha=0.4)
                plt.fill_between(r_[fbwx-0.5,fbwx+0.5],r_[fa(lims['shared']['pctls_min'])[0,inds_plots[plot]][ii],fa(lims['shared']['pctls_min'])[0,inds_plots[plot]][ii]],r_[fa(lims['shared']['pctls_max'])[0,inds_plots[plot]][ii],fa(lims['shared']['pctls_max'])[0,inds_plots[plot]][ii]] ,color='Blue',alpha=0.4)
            else:
                plt.fill_between(fbwx,fa(lims['combined']['pctls_min'])[0,inds_plots[plot]][ii],fa(lims['combined']['pctls_max'])[0,inds_plots[plot]][ii],alpha=0.4)
                plt.fill_between(fbwx,fa(lims['unshared']['pctls_min'])[0,inds_plots[plot]][ii],fa(lims['unshared']['pctls_max'])[0,inds_plots[plot]][ii],color='Green',alpha=0.4)
                plt.fill_between(fbwx,fa(lims['shared']['pctls_min'])[0,inds_plots[plot]][ii],fa(lims['shared']['pctls_max'])[0,inds_plots[plot]][ii],color='Blue',alpha=0.4)

            iipospos=np.in1d(ii,find(Acorrs[0,inds_plots[plot]]>Bcorrs[0,inds_plots[plot]]))
            iipos=ii[iipospos]

            iinegpos=np.in1d(ii,find(Acorrs[0,inds_plots[plot]]<Bcorrs[0,inds_plots[plot]]))
            iineg=ii[iinegpos]

            xes = np.arange(len(ii))+(width - len(ii))/2.

            plt.plot(np.array([xes,xes])[:,find(iipospos)],[Acorrs[0,inds_plots[plot][iipos]],Bcorrs[0,inds_plots[plot][iipos]]],color=[0,0,1],alpha=1,linewidth=1.5,zorder=1)
            plt.plot(np.array([xes,xes])[:,find(iinegpos)],[Acorrs[0,inds_plots[plot][iineg]],Bcorrs[0,inds_plots[plot][iineg]]],color=[1,0,0],alpha=1,linewidth=1.5,zorder=1)

            plt.fill_between(fbwx,fa(lims['unshared']['pctls_min'])[0,inds_plots[plot]][ii],fa(lims['unshared']['pctls_max'])[0,inds_plots[plot]][ii],color='Green',alpha=0.4)
            line3 = plt.Rectangle((0, 0), 0, 0,color=current_palette[0])
            ax.add_patch(line3)
            line2=plt.scatter((xes)[find(iipospos)],Bcorrs[0,inds_plots[plot][iipos]].T,color='blue',zorder=2)
            line2=plt.scatter((xes)[find(iinegpos)],Bcorrs[0,inds_plots[plot][iineg]].T,color='red',zorder=2)
            line1=plt.scatter(xes,Acorrs[0,inds_plots[plot][ii]].T,color='white',zorder=2)
            # color line two according to pos or neg change
            cmap=ListedColormap([(0.2980392156862745, 0.4470588235294118, 0.6901960784313725), (0.3333333333333333, 0.6588235294117647, 0.40784313725490196), (0.7686274509803922, 0.3058823529411765, 0.3215686274509804)])
            norm= BoundaryNorm([-2,0,1,2],cmap.N)
            z=np.zeros(xes.shape[0]+1,)

            colorline(np.r_[xes-1,xes[-1]], z+1.05,ROI_RSNs[indices[0][np.r_[inds_plots[plot],inds_plots[plot][-1]]]]-1.5,cmap=cmap,norm=norm,linewidth=5)
            colorline(np.r_[xes-1,xes[-1]], z+1.1,ROI_RSNs[indices[1][np.r_[inds_plots[plot],inds_plots[plot][-1]]]]-1.5,cmap=cmap,norm=norm,linewidth=5)
            plt.show()

def plot_connectivity_circle_DR(ccstatsma, ROIs, inds, fig, ):

    vmin = -3
    vmax = 3

    inds=np.intersect1d(inds_orig,find(changes==b))
   
