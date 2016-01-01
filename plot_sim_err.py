from numpy import *
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import max_corr_funcs as cf
import matplotlib.pyplot as mp

FC=cf.FC
img=0
repeat_fig=0
calcflag=1
# set img, calcflag,
ccs=arange(-.999,1,0.01)

vvs_all = r_[arange(-0.25,0,0.025)+1,arange(.2,0,-00.025)+1]**.5
#vvs_all = array([0.95])
if img==0:
    vvs=vvs_all[:10]
else:
    vvs=vvs_all[10:]

stds=vvs*100

figs=[8,9]

#fig=figure(figs[img]) 

if not(repeat_fig):
    #fig.clf()
    plot_colour='green'
else:
    plot_colour='blue'

#ax=gca()
ooA=ones((2,2))
ooB=ones((2,2))
errdist_perms=2500
dof=1000
out_hist_unshared=zeros((len(ccs),len(vvs_all),len(ccs)))
out_hist_lim_l=zeros((len(ccs),len(vvs_all),len(ccs)))
out_hist_lim_u=zeros((len(ccs),len(vvs_all),len(ccs)))
out_hist_common_l=zeros((len(ccs),len(vvs_all),len(ccs)))
out_hist_common_u=zeros((len(ccs),len(vvs_all),len(ccs)))

if calcflag == 1:
    out=zeros((len(ccs),len(vvs_all),2,errdist_perms))
    out_common=zeros((len(ccs),len(vvs_all),2))
    out_unshared=zeros((len(ccs),len(vvs_all)))

    # for a in arange(8):
    for cc in arange(len(ccs)):
        ooA[[0,1],[1,0]]=ccs[cc]
        ooA[[0,1],[0,1]]=2

        tmpA=FC(ooA,cov_flag=True)
        tmpB=FC(ooB,cov_flag=True)
        #display(cc)
        for vv in arange(len(vvs_all)):

            ooB[[0,1],[0,1]]=vvs_all[vv]
            ooB[[0,1],[1,0]]=ccs[cc]*vvs_all[vv]
            #tmp=max_corr_funcs_diff.corr_lims_2(ccs[cc],.99,1.0,1.0*vvs_all[vv],1,1)
            tmp=cf.corr_lims(1.0,vvs_all[vv],1.0,vvs_all[vv],ccs[cc])
            tmp=cf.corr_lims(1.0,vvs_all[vv],1.0,1,ccs[cc])
            #tmp=scripts2.range(ccs[cc],.99,1.0,1.0*vvs_all[vv],1,1)
            #out[cc,vv,:]=[tmp[0][0],tmp[0][1]]
            out_unshared=cf.corr_lims_unshared_mat(tmpA,tmpB,errdist=True,errdist_perms=errdist_perms,pctl=5,dof=dof)
            out_hist_unshared[cc,vv,:]=mp.hist(out_unshared[2][:,0,1,0],normed=True,range=[0,1],bins=r_[ccs,[1]])[0]

            out_common=cf.corr_lims_common_mat(tmpA,tmpB,errdist=True,errdist_perms=errdist_perms,pctl=5,dof=dof)
            out_hist_common_l[cc,vv,:]=mp.hist(out_common[2][0][:,0,1,0],normed=True,range=[0,1],bins=r_[ccs,[1]])[0]
            out_hist_common_u[cc,vv,:]=mp.hist(out_common[2][1][:,0,1,0],normed=True,range=[0,1],bins=r_[ccs,[1]])[0]

            out_lims=cf.corr_lims_mat(tmpA,tmpB,errdist=True,errdist_perms=errdist_perms,pctl=5,dof=dof)

            out_hist_lim_l[cc,vv,:]=mp.hist(out_lims[2][0][:,0,1,0],normed=True,range=[0,1],bins=r_[ccs,[1]])[0]
            out_hist_lim_u[cc,vv,:]=mp.hist(out_lims[2][1][:,0,1,0],normed=True,range=[0,1],bins=r_[ccs,[1]])[0]

            #out_common[cc,vv,:]=corr_lims_common(1,vvs_all[vv],std_y,vvs_all[vv],ccs[cc])
            ##out_common[cc,vv,:]=calc_rho_xyb(1,vvs_all[vv],std_y,1,ccs[cc])
            #out_unshared[cc,vv]=corr_lims_unshared(std_x,vvs_all[vv],std_y,vvs_all[vv],ccs[cc])
            ##out_unshared[cc,vv]=calc_rho_unshared(std_x,vvs_all[vv],std_y,1,ccs[cc])
        savez('out_hists2.npz',out_hist_unshared,out_hist_lim_l,out_hist_lim_u,out_hist_common_l,out_hist_common_u)
        
# blue pallette
for aaa in arange(8):
    #fig.clf()
    ax=gca()
    pllt=sns.dark_palette("#5178C7",16)

    cdict = {'red': ((0,0.151,0.151),(1,0.4,0.4)),'green':((0,0.2,0.2333),(1,0.6,0.6)),'blue': ((0,0.32, 0.32),(1,0.9,0.9))}

    blue_blue = LinearSegmentedColormap('BlueBlue', cdict)
    register_cmap(cmap=blue_blue)
    cmap=get_cmap('BlueBlue')

    cdict2 = {'red': ((0,0.4,0.4),(1,0.151,0.151)),'green':((0,0.6,0.6),(1,0.2,0.2)),'blue': ((0,0.9, 0.9),(1,0.32,0.32))}
    blue_blue_r = LinearSegmentedColormap('BlueBlue_r', cdict2)
    register_cmap(cmap=blue_blue_r)
    cmap=get_cmap('BlueBlue_r')

# green pallette

    pllt=sns.dark_palette("#5178C7", 16)

    cdict = {'red': ((0,0.151,0.151),(1,0.4,0.4)),'green': ((0,0.32, 0.32),(1,0.9,0.9)) ,'blue': ((0,0.2,0.2333),(1,0.6,0.6))}

    green_green = LinearSegmentedColormap('GreenGreen', cdict)
    register_cmap(cmap=green_green)
    cmap=get_cmap('GreenGreen')

    cdict2 = {'red': ((0,0.4,0.4),(1,0.151,0.151)),'green':((0,0.9, 0.9),(1,0.32,0.32)),'blue': ((0,0.6,0.6),(1,0.2,0.2))}

    green_green_r = LinearSegmentedColormap('GreenGreen_r', cdict2)
    register_cmap(cmap=green_green_r)
    cmap=get_cmap('GreenGreen_r')

    if img==0:
        if plot_colour=='blue':
            cmap=blue_blue_r
        else:
            cmap=green_green_r
    else:
        if plot_colour=='blue':
            cmap=blue_blue
        else:
            cmap=green_green

    for a in arange(aaa,len(vvs),19):
        #fill_between(ccs,out[:,a+img*10,0],out[:,a+img*10,1],color=cmap(256-a*256/len(vvs)),label=str(stds[a]))
        fill_between(ccs,ccs,out[:,a+img*10,1],color=cmap(256-a*256/len(vvs)),label=str(stds[a]))
        # plt.plot([],[],color=pllt[-(a+1),:],label=str(stds[a]),linewidth=15)

    cmap=cm.gray

    for a in arange(aaa,len(vvs),19):
        fill_between(ccs,out[:,a+img*10,0],ccs,color=cmap(256-a*128/len(vvs)),label=str(stds[a]))
        # plt.plot([],[],color=pllt[-(a+1),:],label=str(stds[a]),linewidth=15)

    for a in arange(aaa,len(vvs),19):

        cmap=blue_blue
        fill_between(ccs,out_common[:,a+img*10,0],out_common[:,a+img*10,1],color=cmap(256-a*256/len(vvs)),label=str(stds[a]))

        cmap=green_green
        fill_between(ccs,out_unshared[:,a+img*10],ccs,color=cmap(256-a*256/len(vvs)),label=str(stds[a]))

        cmap=blue_blue
        plot(ccs,out_unshared[:,a+img*10],color=cmap(256-a*256/len(vvs)))


    X=array([0.0001,0.0001])
    Y=array([0.0001,0.0001])


#cc=ax.pcolor(X,Y,array([[vv,vvs[-1]],[vvs[0],vvs[-1]]]),cmap=cmap)      
    vvs=r_[vvs,1]

    cmap=green_green

    if not(repeat_fig):
      cc=ax.pcolor(array([[.2]]),cmap=cmap,visible=False,vmin=min(vvs**2),vmax=max(vvs**2))      
      cbar=colorbar(cc,shrink=0.5,ticks=vvs**2,fraction=0.1)

    plot(ccs,ccs,color=[0.9,0.9,0.9],label='Test',linewidth=5)

    ax.set_xlabel('Correlation in condition 1')
    ax.set_ylabel('Correlation in condition 2')
    cbar.set_label('Proportional change in variance \n in region one in second condition.')

    if img==0:
        ax.set_title('Changes in correlation that can be explained \n by decreases in signal amplitude of one region of .'+str(vvs[aaa]**2)+'% in one region')
        
        # fig.savefig('Decreases_sim.pdf')
        #fig.savefig('Decreases_both'+str(aaa)+'.pdf')

    else:
        ax.set_title('Changes in correlation that can be explained \n by an increase in signal amplitude of '+str(vvs[aaa]**2)+'% in both regions.')

        #fig.savefig('Increases_both'+str(aaa)+'.pdf')


