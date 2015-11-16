from matplotlib.colors import LinearSegmentedColormap
import mne

import scripts2
#import ml_funcs
import seaborn as sns

import scipy, glob,re, os.path
import scipy.stats as stats
from matplotlib.pylab import find
import itertools
#from ml_funcs import corr_calc
import colorline
from colorline import colorline
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.collections import LineCollection

#t21=ml_funcs.transp21
#rtoz=ml_funcs.rtoz

from operator import mul
from fractions import Fraction

from mne.viz import plot_connectivity_circle

#data_pre='out_'

if os.path.isfile(data_pre+'ROI_order.txt'):
    ROI_order=loadtxt(data_pre+'ROI_order.txt').astype(int)-1
#else:
#    ROI_order=arange(len(ROI_names))

if os.path.isfile(data_pre+'ROI_RSNs.txt'):
    ROI_RSNs=(loadtxt(data_pre+'ROI_RSNs.txt').astype(int))
else:
    ROI_RSNs=arange(len(ROI_order))

if os.path.isfile('goodnodes.txt'):
    goodnodes=loadtxt('goodnodes.txt').astype(int)
    ROI_RSNs=ROI_RSNs[in1d(ROI_order,goodnodes)]
    ROI_order=ROI_order[in1d(ROI_order,goodnodes)]
else:
    ROI_RSNs=ROI_RSNs[ROI_order]

if os.path.isfile('ROI_RSN_include.txt'):
    ROI_RSN_include=(loadtxt('ROI_RSN_include.txt').astype(int))
    ROI_order = ROI_order[in1d(ROI_RSNs,ROI_RSN_include)]
    ROI_RSNs = ROI_RSNs[in1d(ROI_RSNs,ROI_RSN_include)]

if os.path.isfile('ROI_names.txt'):
    ROI_names=array(open('ROI_names.txt').read().splitlines())

ROI_names=ROI_names[ROI_order]
n_nodes=len(ROI_names)

cond_names=open('../cond_names.txt').read().splitlines()
#RSN_names=array(open('../RSN_names.txt').read().splitlines())

def nCk(n,k): 
    return int( reduce(mul, (Fraction(n-i, i+1) for i in range(k)), 1) )

plotcolors=[(0.2,0.6,1),(0.62,0.82,0.98),(0.40,0.95,0.46),(0.6,0.95,0.6),(0.15,0.87,0.87),(0.8,0.8,0.8)]

la=logical_and

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
register_cmap(cmap=blue_red1)
cmap2=matplotlib.colors.ListedColormap(name='Test',colors=plotcolors)
register_cmap(cmap=cmap2)

indices=triu_indices(n_nodes,1)

contrs_all=[]
conds=5
outsize=nCk(conds,2)

for a in itertools.combinations(np.arange(conds),2):
    contrs_all.append(a)

node_angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
height=ones(n_nodes)*3

dist_mat = node_angles[None, :] - node_angles[:, None]
dist_mat[np.diag_indices(n_nodes)] = 1e9
node_width = np.min(np.abs(dist_mat))
node_edgecolor='black'

#sb_cols=[(0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
# (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
# (0.7686274509803922, 0.3058823529411765, 0.3215686274509804)]

current_palette = sns.color_palette()
sb_cols=current_palette + current_palette + current_palette 
group_cols=[]

for a in ROI_RSNs:                          
    group_cols.append(sb_cols[a-1])

#for aaa in [2]:
for aaa in arange(1):
#for aaa in arange(outsize):

    #contrs=[contrs_all[aaa][::-1]]
    contrs=[contrs_all[aaa][::-1]]

    fig=plt.figure(1)
    #prefix='wa'
    execfile('/home/fs0/eduff/code/ampconn/scripts.py')
    fig=plt.figure(aaa+figadd)
    data=ccstatsmat[0,:]
    inds=find(mne.stats.fdr_correction(scipy.stats.norm.sf(abs(data)))[0])
    # stouffers-z for some change in variance
    fig.clf()

    vv_norm=tmp/3

    vcols=[]
    for a in arange(n_nodes):
        vcols.append(blue_red1((vv_norm[a]+1)/2))
    vvs_s=squeeze(vvs_s)
    vvst_s=squeeze(vvst_s)

    ccols=[]
    changes=zeros((ccmat2.shape[1],)).astype(int)+48
       
    snr=squeeze(out[2][0][0]==0)
    #sync= la(-snr,(abs(minimum(ccmat2[0,:],ccmat1[0,:])>0.08)))
    #sync= la(-snr,abs(ccstatsmat)>1.8)
    sync=~snr

    vchange_stat=(abs(vvs_s)+abs(vvst_s))/2**.5
    vthr=1.64
    #snr=vchangestat>vthr
    #snr=lo(abs(vvs_s)>vthr,abs(vvst_s)>vthr)
    changes[find(sync)]=3
    #ppns[a,5]=sum(changes==5)

    # changes in signal
    edge=vthr
    #changes[la(snr,minimum(vvs_s,vvst_s)>-edge)]=int(0)
    changes[la(snr,abs(maximum(vvs_s,vvst_s))>abs(minimum(vvs_s,vvst_s)) )]=int(0)
    #changes[la(snr,maximum(abs(vvs_s),abs(vvst_s))>0)]=int(0)
    # ppns[a,0]=sum(changes==0)

    # changes in noise
    #changes[la(snr,maximum(vvs_s,vvst_s)<edge)]=int(2)
    changes[la(snr,abs(maximum(vvs_s,vvst_s))<abs(minimum(vvs_s,vvst_s)) )]=int(1)
    # ppns[a,2]=sum(changes==
    # 1 change in signal
    #changes[la(snr,la(maximum(vvs_s,vvst_s)>,abs(minimum(vvs_s,vvst_s))<edge))]=int(1)
    # ppns[a,1]=sum(changes==1)

    # 1 change in noise 
    #changes[la(snr,la(minimum(vvs_s,vvst_s)<-edge,abs(maximum(vvs_s,vvst_s))<edge))]=int(3)
    # ppns[a,3]=sum(changes==3)

    # changes in opposite dirs
    changes[la(snr,la(maximum(vvs_s,vvst_s)>vthr,(minimum(vvs_s,vvst_s))<-vthr))]=int(2)
    # ppns[a,4]=sum(changes==4)

    #for a in arange(528):
    #    ccols.append(blue_red1(plotcolors[changes[a]]))

    inds_orig=inds.copy()
    #plot_connectivity_circle(data[inds],ROI_names,(indices[0][inds],indices[1][inds]),fig=fig,colormap='BlueRed1',vmin=-6,vmax=6,node_colors=vcols)   

    titles=["Change in shared signal","Change in unshared signal","Change in both shared and unshared signals","Change in synchronisation"]
    for b in arange(4):

        inds=intersect1d(inds_orig,find(changes==b))

        fontsize=7
        pp=plot_connectivity_circle(ccstatsmat[0,inds].astype(float),ROI_names[0:n_nodes],(indices[0][inds],indices[1][inds]),fig=fig,colormap='BlueRed1',vmin=-3,vmax=3,node_colors=vcols,subplot=241+b,title=titles[b],interactive=True,fontsize_names=fontsize,facecolor='w',colorbar=False,node_edgecolor=node_edgecolor,textcolor='black') 

        ax=gca()
        ax.set_title(titles[b],color='black')

        bars = pp[1].bar(node_angles, height, width=node_width, bottom=15.5, \
                        edgecolor=node_edgecolor, lw=2, facecolor='.9', \
                        align='center',linewidth=0)

        for bar, color in zip(bars, group_cols):
            bar.set_facecolor(color)
            bar.set_edgecolor(color)

        ax.text(.8,18,'DMN',size=10,color='black')
        ax.text(2.8,21,'Motor',size=10,color='black')
        ax.text(5,18,'Visual',size=10,color='black')

        sort_array=zeros((len(inds),),dtype=('f4,f4'))
        sort_array['f0']=out[0][1][0,0,inds]
        sort_array['f1']=out[1][1][0,0,inds]
        ii=argsort(sort_array,order=['f0','f1'])

        #fig=figure(figadd+100+aaa)
        #fig.clf()
              #ax=fig.add_subplot(1,1,1,axisbg='white')

        ax=subplot(245+b,axisbg='white')
        if len(ii)>0: 
            #if a==0:
            #    ii2=ii[in1d(ii,find(ccmat1>ccmat2))]
            #elif a==1:
            #    ii2=ii[in1d(ii,find(ccmat1<ccmat2))]
            #else:
            #    ii2=ii[in1d(ii,setdiff1d(arange(len(changes)),inds_orig))]
            width=max((20,len(ii)+10))

            fbwx=arange(len(ii))+(width - len(ii))/2.
            fbwx[0]=fbwx[0]-0.5
            fbwx[-1]=fbwx[-1]+.5

            #ax.set_yticks([0,1])
            ax.set_yticks([0,1])
            ax.set_yticks([-.25,0,0.25,.5,.75,1],minor=True)
            ax.yaxis.grid(color=[0.7,.95,.95],linestyle='-',linewidth=.5,which='minor')
            ax.yaxis.grid(color=[0.65,.85,.85],linestyle='-',linewidth=2,which='major')

             
            if len(fbwx)==1:            
                fill_between(r_[fbwx-0.5,fbwx+0.5],r_[out[0][1][0][0,inds[ii]].flatten(),out[1][1][0][0,inds[ii]].flatten()],r_[out[1][0][0][0,inds[ii]].flatten(),out[1][1][0][0,inds[ii]].flatten()],alpha=0.4)
            else:
                fill_between(fbwx,out[0][1][0][0,inds[ii]].flatten(),out[1][1][0][0,inds[ii]].flatten(),alpha=0.4)
            iipospos=in1d(ii,find(ccmat1[0,inds]>ccmat2[0,inds]))
            iipos=ii[iipospos]

            iinegpos=in1d(ii,find(ccmat1[0,inds]<ccmat2[0,inds]))
            iineg=ii[iinegpos]

            xes = arange(len(ii))+(width - len(ii))/2.


            #plot(array([arange(len(ii))+(width - len(ii))/2.,arange(len(ii))+(width - len(ii))/2.]),[ccmat1[0,inds[ii]],ccmat2[0,inds[ii]]],color=current_palette[1],alpha=0.5)
            #scatter(array([arange(len(ii))+(width - len(ii))/2.,arange(len(ii))+(width - len(ii))/2.])[:,find(iipospos)],[ccmat1[0,inds[iipos]],ccmat2[0,inds[iipos]]],color=[1,0,0],alpha=1,linewidth=1.5)
            plot(array([xes,xes])[:,find(iipospos)],[ccmat1[0,inds[iipos]],ccmat2[0,inds[iipos]]],color=[1,0,0],alpha=1,linewidth=1.5,zorder=1)
            #scatter(array([arange(len(ii))+(width - len(ii))/2.,arange(len(ii))+(width - len(ii))/2.])[:,find(iinegpos)],[ccmat1[0,inds[iineg]],ccmat2[0,inds[iineg]]],color=[0,0,1],alpha=1,linewidth=1.5)
            plot(array([xes,xes])[:,find(iinegpos)],[ccmat1[0,inds[iineg]],ccmat2[0,inds[iineg]]],color=[0,0,1],alpha=1,linewidth=1.5,zorder=1)

            line3 = plt.Rectangle((0, 0), 0, 0,color=current_palette[0])
            ax.add_patch(line3)
            line2=plt.scatter((xes)[find(iipospos)],ccmat1[0,inds[iipos]].T,color='red',zorder=2)
            line2=plt.scatter((xes)[find(iinegpos)],ccmat1[0,inds[iineg]].T,color='blue',zorder=2)
            line1=plt.scatter(xes,ccmat2[0,inds[ii]].T,color='white',zorder=2)
            # color line two according to pos or neg change
            cmap=ListedColormap([(0.2980392156862745, 0.4470588235294118, 0.6901960784313725), (0.3333333333333333, 0.6588235294117647, 0.40784313725490196), (0.7686274509803922, 0.3058823529411765, 0.3215686274509804)])
            norm= BoundaryNorm([-2,0,1,2],cmap.N)
            z=zeros(xes.shape[0]+1,)
            #colorline(xes, z+0.3,ROI_RSNs[indices[0][inds]]-1.5,cmap=cmap,norm=norm,linewidth=5)
            #colorline(xes, z-0.4,ROI_RSNs[indices[1][inds]]-1.5,cmap=cmap,norm=norm,linewidth=5)

            colorline(r_[xes,xes[-1]+1], z+1.05,ROI_RSNs[indices[0][r_[inds,inds[-1]]]]-1.5,cmap=cmap,norm=norm,linewidth=5)
            colorline(r_[xes,xes[-1]+1], z+1.1,ROI_RSNs[indices[1][r_[inds,inds[-1]]]]-1.5,cmap=cmap,norm=norm,linewidth=5)
            plt.show()

            # plot line color
            #tcs1 = ccmat1[0,inds[ii]].T
            #tcs2 = ccmat2[0,inds[ii]].T
            #xax=arange(len(ii))+(width - len(ii))/2.
            #xax_i=arange((len(ii)-.999)*10)/10. + (width - len(ii))/2.
            #tcs1_i = interp(xax_i,xax,tcs1)
            #tcs2_i = interp(xax_i,xax,tcs2)
                
            #points=array([xax_i,tcs1_i]).T.reshape(-1,1,2)
            #segments = np.concatenate([points[:-1], points[1:]], axis=1)
            #lc = LineCollection(segments, cmap=cmap, norm=norm)
            #lc.set_array(array(tcs1_i<tcs2_i).astype(int))
            #lc.set_linewidth(2)
            #ax.add_collection(lc)

            #line2=plt.plot(,ccmat1[0,inds[ii]].T,color='blue',alpha=0.95,linewidth=2)

            #fill_between((arange(len(ii))+(width - len(ii))/2.)[find(iipospos)],ccmat1[0,inds[iipos]],ccmat2[0,inds[iipos]],color='red',alpha=0.5)
            #fill_between((arange(len(ii))+(width - len(ii))/2.)[find(iinegpos)],ccmat1[0,inds[iineg]],ccmat2[0,inds[iineg]],color='red',alpha=0.5)

            ax.set_xlim([0,width])
            ax.set_ylim([-0.5,1.15])
            #ax.set_
            ax.xaxis.grid(False)
 

            #titles=["Change in shared signal","Change in unshared signal","Change in both shared and unshared signals","Change in synchronisation"]

            #lgnd=ax.legend((line1[0],line3,line2[0]),('Rest correlation',cond_names[contrs[0][0]] + " correlation that could be \n explained by change in signal levels.","Observed "+cond_names[contrs[0][0]] + ' correlation'),loc=4,frameon=True,markerscale=2) # bbox_to_anchor=(0.53,.97),f

            #frame=lgnd.get_frame()
            #frame.set_facecolor([0.85,0.95,0.95])
            if b != 0:
                ax.set_yticklabels([])
            ax.set_xticklabels([])

            if filt:
                fig.savefig(cond_names[contrs[0][0]]+'-'+ cond_names[contrs[0][1]]+'_'+str(lf)+ '_' + '.pdf')
            else:
                fig.savefig(cond_names[contrs[0][0]]+'-'+ cond_names[contrs[0][1]]+ '.pdf')


