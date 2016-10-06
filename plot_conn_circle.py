import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import scipy.stats as stats
import max_corr_funcs as cf
import mne
import ml_funcs as ml
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
rtoz=ml.rtoz

# plot connectivity matrices from an ICA dir
def plot_conn(dir,inds1,inds2,fig,flatten=True,errdist_perms=0,prefix='dr_stage1',exclude=True,filt=0,data_pre='',savefig='',pctl=5):
     
    # basic settings

    # colours
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

    # loading data

    if os.path.isfile(data_pre+'ROI_order.txt'):
        ROI_order=np.loadtxt(data_pre+'ROI_order.txt').astype(int)-1
    else:
        ROI_order = []

    if os.path.isfile(data_pre+'ROI_RSNs.txt'):
        ROI_RSNs=(np.loadtxt(data_pre+'ROI_RSNs.txt').astype(int))
    else: 
        ROI_RSNs = []

    if os.path.isfile('goodnodes.txt'):
        goodnodes=np.loadtxt('goodnodes.txt').astype(int)
        ROI_RSNs=ROI_RSNs[np.in1d(ROI_order,goodnodes)]
        ROI_order=ROI_order[np.in1d(ROI_order,goodnodes)]
    elif ROI_RSNs != []:
        ROI_RSNs=ROI_RSNs[ROI_order]

    if os.path.isfile('ROI_RSN_include.txt'):
        ROI_RSN_include=(np.loadtxt('ROI_RSN_include.txt').astype(int))
        if ROI_order == []:
            ROI_order = arange(len(ROI_RSNs))
        ROI_order = ROI_order[np.in1d(ROI_RSNs,ROI_RSN_include)]
        ROI_RSNs = ROI_RSNs[np.in1d(ROI_RSNs,ROI_RSN_include)]

    if plt.is_numlike(inds1):
        A_orig=cf.dr_loader('.',subj_ids=inds1,ROIs=ROI_order,prefix=prefix)
        B_orig=cf.dr_loader('.',subj_ids=inds2,ROIs=ROI_order,prefix=prefix)
    else:
        A_orig=inds1
        B_orig=inds2

    if os.path.isfile('ROI_names.txt'):
        ROI_names=np.array(open('ROI_names.txt').read().splitlines())
    else:
        ROI_names=np.array([ str(a) for a in np.arange(A_orig.get_covs().shape[-1]) ])

    n_nodes=len(ROI_names)

    if ROI_order == []:
        ROI_order = np.arange(n_nodes)

    if ROI_RSNs == []:
        ROI_RSNs = np.ones(n_nodes).astype(int)

    ROI_names=ROI_names[ROI_order]
    n_nodes=len(ROI_names)

    # stats generation
 
    ccstats=stats.ttest_rel(rtoz(A_orig.get_corrs()),rtoz(B_orig.get_corrs()))
    ccstatsmat=-fa(ccstats[0])
    ccstatsmatp=fa(ccstats[1])
    inds_cc=find(mne.stats.fdr_correction(ccstatsmatp,alpha=0.05)[0])

    # inds_cc=find(mne.stats.fdr_correction(scipy.stats.norm.sf(abs(ccstatsmatp)),alpha=0.2)[0])
    # get std (remove stats due to rounding in simulations)
    A_orig_stds = np.round(A_orig.get_stds(),6)
    B_orig_stds = np.round(B_orig.get_stds(),6)
    
    vvstatsmat=-(stats.ttest_rel(A_orig_stds,B_orig_stds)[0])
    vvpctg = (B_orig_stds - A_orig_stds)/(A_orig_stds)
    inds_vv=find(mne.stats.fdr_correction(scipy.stats.norm.sf(abs(vvstatsmat)),alpha=0.2)[0])

    if A_orig.tcs == []: 
        A=cf.FC(np.mean(A_orig.get_covs(),0),cov_flag=True, dof=A_orig.dof)
        B=cf.FC(np.mean(B_orig.get_covs(),0),cov_flag=True , dof=B_orig.dof)
    else:
        A=cf.flatten_tcs(A_orig)
        B=cf.flatten_tcs(B_orig)

    Acorrs=fa(A.get_corrs())
    Bcorrs=fa(B.get_corrs())

    vv_norm=vvstatsmat/6
    #vv_norm= vvpctg * 254
    vv_norm[np.isnan(vv_norm)]=0
    vcols=[]
    for a in np.arange(n_nodes):
        vcols.append(blue_red1((vv_norm[a]+1)/2))
    #vvs_s=squeeze(vvs_s)
    #vvst_s=squeeze(vvst_s)

    indices=np.triu_indices(n_nodes,1)

    node_angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    height=np.ones(n_nodes)*4
    dist_mat = node_angles[None, :] - node_angles[:, None]
    dist_mat[np.diag_indices(n_nodes)] = 1e9
    node_width = np.min(np.abs(dist_mat))
    node_edgecolor='black'
   
    #return(A,B)
    fig.clf()

    # calculate variance
    #for a in ndindex(varmat.shape[0:2]):
    #    varvarmat[a][ix_(np.arange(varmat.shape[-1]),np.arange(varmat.shape[-1]))]=tile(varmat[a],(rois,1))
    #sz=varmat.shape[-1]
    lims=cf.corr_lims_all(A,B,errdist_perms=errdist_perms,pctl=pctl,show_pctls=True)
    # incl_zeros = stats.norm.ppf(lims['covs']['incl_zeros'])
    incl_zeros = lims['covs']['incl_zeros']

    # ccstatsmat = lims['covs'][ 

    #inds_cc=find(mne.stats.fdr_correction(2*(0.5-abs(0.5-fa(incl_zeros))),alpha=0.02)[0])
    plots = ['unshared','common','combined','other']
    titles= {'unshared':"Addition of uncorrelated signal", 'common':"Addition of common signal",'combined':"Addition of mixed signals",'other':"Changes not explained \n by simple signal additions"}
    notin=inds_cc
    inds_plots={}
    vmax=3
    vmin=-3
    for plot in plots[:3]:
        inds_plots[plot]=np.intersect1d(inds_cc,find(fa(lims[plot]['pctls'])))
        notin = np.setdiff1d(notin,find(fa(lims[plot]['pctls'])))
    inds_plots['other'] = notin

    if exclude:
        inds_plots['common']=np.setdiff1d(inds_plots['common'],inds_plots['unshared'])
        inds_plots['combined']=np.setdiff1d(inds_plots['combined'],inds_plots['common'])
        inds_plots['combined']=np.setdiff1d(inds_plots['combined'],inds_plots['unshared'])
    cnt=-1
    fontsize=9 

    # produce the four plots 

    for plot in plots:
        cnt+=1
        pp=plot_connectivity_circle(ccstatsmat.astype(float).flatten()[inds_plots[plot]],ROI_names[0:n_nodes],(indices[0][inds_plots[plot]],indices[1][inds_plots[plot]]),fig=fig,colormap='BlueRed1',vmin=vmin,vmax=vmax,node_colors=vcols,subplot=241+cnt,title=titles[plot],interactive=True,fontsize_names=fontsize,facecolor='w',colorbar=False,node_edgecolor=node_edgecolor,textcolor='black',padding=3,node_linewidth=0.5) 

        ax=plt.gca()
        ax.set_title(titles[plot],color='black') 
        
        bars = pp[1].bar(node_angles, height*2.2, width=node_width, bottom=10.4, \
                        edgecolor='0.9', lw=2, facecolor='.9', \
                        align='center',linewidth=1)
       
        #for aa in np.arange(len(sb_cols)):
        #    sb_cols[aa]=(min(sb_cols[0][0]*(1.9),1),min(sb_cols[0][1]*(1.9),1),min(sb_cols[0][2]*(1.9),1))

        for a in ROI_RSNs:     
            val=1-0.5*a/max(ROI_RSNs)
            group_cols.append((val,val,val))

        for bar, color in zip(bars, group_cols):
            bar.set_facecolor(color)
            bar.set_edgecolor(color)

        #ax.text(.8,18,'DMN',size=10,color='black')
        #ax.text(2.8,21,'Motor',size=10,color='black')
        #ax.text(5,18,'Visual',size=10,color='black')
   
        sort_array=np.zeros((len(inds_plots[plot]),),dtype=('f4,f4'))

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

            #if a==0:
            #    ii2=ii[in1d(ii,find(ccmat1>ccmat2))]
            #elif a==1:
            #    ii2=ii[in1d(ii,find(ccmat1<ccmat2))]
            #else:
            #    ii2=ii[in1d(ii,setdiff1d(arange(len(changes)),inds_orig))]

            # pad out inds_plots
            # pad out inds_plots
            #inds_plots_pad = {}
            #for tt in inds_plots.keys():
            #    np.r_[inds_plots[tt][0],inds_plots['other'],inds_plots[tt][-1]]

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

            iipospos=np.in1d(ii,find(Acorrs[0,inds_plots[plot]]>Bcorrs[0,inds_plots[plot]]))
            iipos=ii[iipospos]

            iinegpos=np.in1d(ii,find(Acorrs[0,inds_plots[plot]]<Bcorrs[0,inds_plots[plot]]))
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
            colorline(fbwx[:-1], z+1.05,ROI_RSNs[indices[0][np.r_[inds_plots[plot],inds_plots[plot][-1]]]]-1.5,cmap=cmap,norm=norm,linewidth=5)
            colorline(fbwx[:-1], z+1.1,ROI_RSNs[indices[1][np.r_[inds_plots[plot],inds_plots[plot][-1]]]]-1.5,cmap=cmap,norm=norm,linewidth=5)
            plt.show()

    if savefig!='':
        fig.savefig(savefig)

    return(lims,A,B,inds_plots)


def plot_connectivity_circle_DR(ccstatsmat, ROIs, inds, fig, ):

    vmin = -3
    vmax = 3

    inds=np.intersect1d(inds_orig,find(changes==b))
   

def plot_connectivity_circle_thr(con, node_names, thr=-1, indices=None, n_lines=None,
                             node_angles=None, node_width=None,
                             node_colors=None, facecolor='black',
                             textcolor='white', node_edgecolor='black',
                             linewidth=1.5, colormap='hot', vmin=None,
                             vmax=None, colorbar=True, title=None,
                             colorbar_size=0.2, colorbar_pos=(-0.3, 0.1),
                             fontsize_title=12, fontsize_names=8,
                             fontsize_colorbar=8, padding=6.,
                             fig=None, subplot=111, interactive=True):

    if thr>0:
        triu=np.triu_indices(10,k=1)
        conu=con[triu]
        inds1=np.tile(np.arange(10),(10,1))
        inds2=inds1.T

        inds1_t = inds1[triu]
        inds2_t = inds2[triu]

        plot_connectivity_circle(conu[conu>thr], node_names, indices=(inds1_t[conu>thr],inds2_t[conu>thr]), n_lines=n_lines,
                             node_angles=node_angles, node_width=node_width,
                             node_colors=node_colors, facecolor=facecolor,
                             textcolor=textcolor, node_edgecolor=node_edgecolor,
                             linewidth=linewidth, colormap=colormap, vmin=vmin,
                             vmax=vmax, colorbar=colorbar, title=title,
                             colorbar_size=colorbar_size, colorbar_pos=colorbar_pos,
                             fontsize_title=fontsize_title, fontsize_names=fontsize_names,
                             fontsize_colorbar=fontsize_colorbar, padding=padding,
                             fig=fig, subplot=subplot, interactive=interactive) 
    

def gen_lims_perms(errdist_perms=100,nreps=100,a=0.57741163399813,b=0.7948717948717948,c=0.69230769230769229,pctl=5):

    A_real=cf.dr_loader('.',subj_ids=np.arange(15),dof='EstEff',prefix='test')
    Af_real= cf.flatten_tcs(A_real)

    tcs = Af_real.tcs[0,:3,:]
    shp = tcs[:2,:].shape

    corrmat1=np.array([[1,a,b],[a,1,c],[b,c,1]])
    tmp=ml.gen_sim_data(tcs,covmat=corrmat1,nreps=nreps)
    Af=cf.FC(tmp[:,:2,:])

    tmp1=ml.gen_sim_data(tcs,covmat=corrmat1,nreps=nreps)
    zz1=np.zeros((nreps,shp[0],shp[1]))
    zz1[:,1,:]=tmp1[:,2,:]
    Bf=cf.FC(tmp1[:,:2,:]+0.6*zz1)

    lims1=cf.corr_lims_all(Af,Bf,errdist_perms=errdist_perms,show_pctls=True,pctl=pctl)

    return(lims1)




