import scipy.stats as stats
import mne
from mne.viz import plot_connectivity_circle
import max_corr_funcs as cf
import ml_funcs as ml
from ml_funcs import flattenall as fa
import numpy as np
import scipy, os.path
import itertools
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pylab import find
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap, BoundaryNorm
from colorline import colorline
from operator import mul
from fractions import Fraction
import seaborn as sns

# calculate and plot connectivity matrices from an ICA dir
def plot_conn(dir,inds1,inds2,fig,errdist_perms=0,prefix='dr_stage1',exclude_conns=True,data_pre='',savefig='',pctl=5,min_corr_diff=0,pcorrs=False,neg_norm=True,nosubj=False):

    A = cf.dr_loader(dir,subj_inds=inds1,prefix=prefix,nosubj=nosubj)
    B = cf.dr_loader(dir,subj_inds=inds2,prefix=prefix,nosubj=nosubj)

    AB_con = cf.FC_con(A,B)

    [AB_con, inds_plots] = plot_conn_stats(AB_con,fig,errdist_perms=errdist_perms,exclude_conns=exclude_conns,savefig=savefig,pctl=pctl,min_corr_diff=min_corr_diff,neg_norm=True,refresh=True)

    return(AB_con)

def plot_conn_stats(AB_con,fig,flatten=True,errdist_perms=0,pctl=5,min_corr_diff=0,pcorrs=False,neg_norm=True,fdr_alpha=0.4,exclude_conns=True,savefig='',ccstatsmat=None,inds_cc=None,vvstatsmat=None,refresh=False):
    

    # gen basic stats 
    if ccstatsmat is None:
        inds_cc = find(mne.stats.fdr_correction(fa(AB_con.get_corr_stats(pcorrs=pcorrs)[1]),alpha=fdr_alpha)[0])
        ccstatsmat=-fa(AB_con.get_corr_stats(pcorrs=pcorrs)[0])
        vvstatsmat=-AB_con.get_std_stats()

    ROI_info = AB_con.A.ROI_info

    vv_norm=vvstatsmat/6
    vv_norm[np.isnan(vv_norm)]=0
    n_nodes = AB_con.A.get_covs().shape[1]

    lims=AB_con.get_lims(pcorrs=pcorrs,errdist_perms=errdist_perms,refresh=refresh,pctl=pctl)

    # Acorrs=np.mean(fa(AB_con.A.get_corrs(pcorrs=pcorrs)),0)
    # Bcorrs=np.mean(fa(AB_con.B.get_corrs(pcorrs=pcorrs)),0)

    Acorrs_mat=np.mean(AB_con.A.get_corrs(pcorrs=pcorrs),0,keepdims=True)
    Acorrs=fa(Acorrs_mat)
    Bcorrs_mat=np.mean(AB_con.B.get_corrs(pcorrs=pcorrs),0,keepdims=True)
    Bcorrs=fa(Bcorrs_mat)

    if min_corr_diff != 0:
        inds_corr_diff = find(abs(Acorrs-Bcorrs)>min_corr_diff)
        inds_cc = np.intersect1d(inds_corr_diff,inds_cc)

    # set plot colours

    fig.clf()
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

    current_palette = sns.color_palette()
    sb_cols=current_palette + current_palette + current_palette 

    # variance stats

    vcols=[]
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

    for a in ROI_info['ROI_RSNs']:     
        val=1-0.5*a/max(ROI_info['ROI_RSNs'])
        group_cols.append((val,val,val))

    # incl_zeros = lims['covs']['incl_zeros']
    #inds_cc=find(mne.stats.fdr_correction(2*(0.5-abs(0.5-fa(incl_zeros))),alpha=0.02)[0])

    plots = ['unshared','common','combined','other']
    titles= {'unshared':"Addition of uncorrelated signal", 'common':"Addition of common signal",'combined':"Addition of mixed signals",'other':"Changes not explained \n by simple signal additions"}

    indices=np.triu_indices(n_nodes,1)
    notin=inds_cc.copy()
    inds_plots={}

    for plot in plots[:3]:
        inds_plots[plot]=np.intersect1d(inds_cc,find(fa(lims[plot]['pctls'])))
        notin = np.setdiff1d(notin,find(fa(lims[plot]['pctls'])))

    inds_plots['other'] = notin

    if exclude_conns:
        # inds_plots['common']=np.setdiff1d(inds_plots['common'],inds_plots['unshared'])
        inds_plots['combined']=np.setdiff1d(inds_plots['combined'],inds_plots['common'])
        inds_plots['combined']=np.setdiff1d(inds_plots['combined'],inds_plots['unshared'])
    
    # produce the four plots 
    cnt=-1
    plotccstats= ccstatsmat.astype(float)

    for plot in plots:

        cnt+=1

        # flip color of negative corrs
        if neg_norm==True:
           plotccstats= plotccstats*np.sign(Acorrs)
        
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

        sort_array=np.zeros((len(inds_plots[plot]),),dtype=('f4,f4'))
        
        # plot corr info

        if plot=='other': 
            plotrange='combined'
        else:
            plotrange=plot

        sort_array['f0']=fa(lims[plotrange]['min_pctls'])[0,inds_plots[plot]]
        sort_array['f1']=fa(lims[plotrange]['max_pctls'])[0,inds_plots[plot]]
        ii=np.argsort(sort_array,order=['f0','f1'])

        if len(ii)>0: 
            width=np.max((20,len(ii)+10))
            ii_ext=np.r_[ii[0],ii,ii[-1]]
            fbwx=np.arange(len(ii_ext))+(width - len(ii_ext))/2.
            fbwx[0]=fbwx[0]+0.5  #=np.r_[fbwx[0]-0.5, fbwx,fbwx[-1]+0.5]
            fbwx[-1]=fbwx[-1]-0.5  

            ax=plt.subplot(245+cnt,axisbg='white')

            ax.set_ylim([-1.,1])
            ax.set_yticks([-1,0,1])
            ax.set_yticks([-0.75,-.25,0,0.25,.5,.75,1],minor=True)
            ax.yaxis.grid(color=[0.7,.95,.95],linestyle='-',linewidth=.5,which='minor')
            ax.yaxis.grid(color=[0.65,.85,.85],linestyle='-',linewidth=2,which='major')

            if len(fbwx)==1:            
                plt.fill_between(np.r_[fbwx-0.5,fbwx+0.5],np.r_[fa(lims['combined']['min_pctls'])[0,inds_plots[plot]][ii_ext],fa(lims['combined']['min_pctls'])[0,inds_plots[plot]][ii_ext]],np.r_[fa(lims['combined']['max_pctls'])[0,inds_plots[plot]][ii_ext],fa(lims['combined']['max_pctls'])[0,inds_plots[plot]][ii_ext]] ,alpha=0.4)
                plt.fill_between(np.r_[fbwx-0.5,fbwx+0.5],np.r_[fa(lims['common']['min_pctls'])[0,inds_plots[plot]][ii_ext],fa(lims['common']['min_pctls'])[0,inds_plots[plot]][ii_ext]],np.r_[fa(lims['common']['max_pctls'])[0,inds_plots[plot]][ii_ext],fa(lims['common']['max_pctls'])[0,inds_plots[plot]][ii_ext]] ,color='Blue',alpha=0.4)

                plt.fill_between(np.r_[fbwx-0.5,fbwx+0.5],np.r_[fa(lims['unshared']['min_pctls'])[0,inds_plots[plot]][ii_ext],fa(lims['unshared']['min_pctls'])[0,inds_plots[plot]][ii_ext]],np.r_[fa(lims['unshared']['max_pctls'])[0,inds_plots[plot]][ii_ext],fa(lims['unshared']['max_pctls'])[0,inds_plots[plot]][ii_ext]] ,color='Green',alpha=0.6)
            else:
                plt.fill_between(fbwx,fa(lims['combined']['min_pctls'])[0,inds_plots[plot]][ii_ext],fa(lims['combined']['max_pctls'])[0,inds_plots[plot]][ii_ext],alpha=0.4)
                plt.fill_between(fbwx,fa(lims['common']['min_pctls'])[0,inds_plots[plot]][ii_ext],fa(lims['common']['max_pctls'])[0,inds_plots[plot]][ii_ext],color='Blue',alpha=0.4)
                plt.fill_between(fbwx,fa(lims['unshared']['min_pctls'])[0,inds_plots[plot]][ii_ext],fa(lims['unshared']['max_pctls'])[0,inds_plots[plot]][ii_ext],color='Green',alpha=0.6)

            if neg_norm == True:
                iipospos=np.in1d(ii,find(abs(Acorrs[0,inds_plots[plot]])>abs(Bcorrs[0,inds_plots[plot]])))
                iinegpos=np.in1d(ii,find(abs(Acorrs[0,inds_plots[plot]])<abs(Bcorrs[0,inds_plots[plot]])))
            else:
                iipospos=np.in1d(ii,find(Acorrs[0,inds_plots[plot]]>Bcorrs[0,inds_plots[plot]]))
                iinegpos=np.in1d(ii,find(Acorrs[0,inds_plots[plot]]<Bcorrs[0,inds_plots[plot]]))

            iipos=ii[iipospos]
            iineg=ii[iinegpos]

            xes = np.arange(len(ii))+(width - len(ii))/2.

            plt.plot(np.array([xes,xes])[:,find(iipospos)],[Acorrs[0,inds_plots[plot][iipos]],Bcorrs[0,inds_plots[plot][iipos]]],color=[0,0,1],alpha=1,linewidth=1.5,zorder=1)
            plt.plot(np.array([xes,xes])[:,find(iinegpos)],[Acorrs[0,inds_plots[plot][iineg]],Bcorrs[0,inds_plots[plot][iineg]]],color=[1,0,0],alpha=1,linewidth=1.5,zorder=1)

            plt.fill_between(fbwx,fa(lims['unshared']['min_pctls'])[0,inds_plots[plot]][ii_ext],fa(lims['unshared']['max_pctls'])[0,inds_plots[plot]][ii_ext],color='Green',alpha=0.6)
            line3 = plt.Rectangle((0, 0), 0, 0,color=current_palette[0])
            ax.add_patch(line3)
            ax.set_xticks([])

            #
            line2=plt.scatter((xes)[find(iipospos)],Bcorrs[0,inds_plots[plot][iipos]].T,color='blue',zorder=2)
            line2=plt.scatter((xes)[find(iinegpos)],Bcorrs[0,inds_plots[plot][iineg]].T,color='red',zorder=2)
            line2=plt.scatter((xes)[find(iipospos)],Acorrs[0,inds_plots[plot][iipos]].T,color='white',zorder=2)
            line2=plt.scatter((xes)[find(iinegpos)],Acorrs[0,inds_plots[plot][iineg]].T,color='white',zorder=2)
            # color line two according to pos or neg change
            cmap=ListedColormap([(0.2980392156862745, 0.4470588235294118, 0.6901960784313725), (0.3333333333333333, 0.6588235294117647, 0.40784313725490196), (0.7686274509803922, 0.3058823529411765, 0.3215686274509804)])
            norm= BoundaryNorm([-2,0,1,2],cmap.N)
            z=np.zeros(xes.shape[0]+1,)

            # plot network membership
            colorline(fbwx[:-1], z+1.05,ROI_info['ROI_RSNs'][indices[0][np.r_[inds_plots[plot],inds_plots[plot][-1]]]]-1.5,cmap=cmap,norm=norm,linewidth=5)
            colorline(fbwx[:-1], z+1.1,ROI_info['ROI_RSNs'][indices[1][np.r_[inds_plots[plot],inds_plots[plot][-1]]]]-1.5,cmap=cmap,norm=norm,linewidth=5)
            plt.show()

    if savefig!='':
        fig.savefig(savefig)

    AB_con.lims['covs']['inds_plots']=inds_plots 
    AB_con.lims['covs']['cc_stats']=ccstatsmat
    AB_con.lims['covs']['vv_stats']=vvstatsmat
    return(AB_con,inds_plots)
