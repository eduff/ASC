
from numpy import *
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import max_corr_funcs as cf
import matplotlib.pyplot as plt
import os,numbers
#import argparse


def plot_sim_err(ccs,out=[],errdist_perms=0,calcflag=1,fig=[],plotType='combined',plotExt='',vInds=[]):

    if type(ccs) == str:
        ccs=float(ccs)

    if type(errdist_perms) == str:
        errdist_perms=float(errdist_perms)

    if isinstance(ccs,numbers.Number):
        ccs=[ccs]

    FC=cf.FC
    img=0
    repeat_fig=0
    if 'calcflag' not in locals():
        calcflag=1

    if 'outname' not in locals():
        outname='test'

    if 'show_pctls' not in locals():
        show_pctls = True

    if 'errdist_perms' not in locals():
        errdist_perms=0

    vvs_all_1 = r_[arange(-0.5,0,0.05)+1,(arange(.5,0,-00.05)[::-1])+1]
    # vvs_all_2 = ones(vvs_all_1.shape)
    vvs_all_2 = r_[arange(-0.5,0,0.05)+1,(arange(.5,0,-00.05)[::-1])+1]

    if vInds == []:
        vInds= arange(len(vvs_all_1)-1,0,-1)
    elif isinstance(vInds,numbers.Number):
        vInds = [vInds]

    stds_1=vvs_all_1**.5
    stds_2=vvs_all_2**.5

    figs=[8,9]

    if not(repeat_fig):
        plot_colour='green'
    else:
        plot_colour='blue'

    ooA=ones((2,2))
    ooB=ones((2,2))

    dof=1000

    if calcflag == 1:
        out={}

        if errdist_perms > 0:
            type_list=['min','max','min_pctls','max_pctls','pctls_raw','min_pctls_raw','max_pctls_raw']
        else:
            type_list=['min','max']

        for us in ['common','unshared','combined']:
            out[us] = {}
            for mm in type_list:
                if mm[-3:] == 'raw':
                    out[us][mm]=zeros((errdist_perms,len(ccs),len(vvs_all_1)))
                else:
                    out[us][mm]=zeros((len(ccs),len(vvs_all_1)))

        # for a in arange(8):
        for cc in arange(len(ccs)):
            print(str(cc))
            ooA[[0,1],[1,0]]=ccs[cc]
            ooA[[0,1],[0,1]]=1

            tmpA=FC(ooA,cov_flag=True,dof=dof)

            #display(cc)
            for vv in arange(len(vvs_all_1)):

                ooB[0,0]=vvs_all_1[vv]
                ooB[1,1]=vvs_all_2[vv]
                ooB[[0,1],[1,0]]=ccs[cc]*(vvs_all_1[vv]*vvs_all_2[vv])**0.5

                tmpB=FC(ooB,cov_flag=True,dof=dof)
                lims = cf.corr_lims_all(tmpA,tmpB,errdist_perms=errdist_perms,pctl=5,dof=dof) # ,show_pctls=show_pctls)
                # lims[cc,vv] = cf.corr_lims_all(tmpA,tmpB,errdist=True,errdist_perms=errdist_perms,pctl=5,dof=dof)

                for us in ['common','unshared','combined']:

                    for mm in type_list:
                        if mm in lims[us].keys():
                            if mm[-3:] == 'raw':
                                out[us][mm][:,cc,vv]=lims[us][mm][:,0,1]
                            else:
                                out[us][mm][cc,vv]=lims[us][mm][0,1]
                        
                ## out_unshared[cc,vv]=calc_rho_unshared(std_x,vvs_all[vv],std_y,1,ccs[cc])

                print('vv:'+str(vv))
        if len(ccs)==1:          
            outname = str(ccs[0]*100)

        savez('out_'+outname+'.npz',out)
# blue pallette
    #return() 
    if fig == []:
        return()

    fig.clf()

#loop over variances

    ax=plt.gca()

# blue pallette

    pllt=sns.dark_palette("#5178C7",16)

    cdict = {'red': ((0,0.151,0.151),(1,0.4,0.4)),'green':((0,0.2,0.2333),(1,0.6,0.6)),'blue': ((0,0.32, 0.32),(1,0.9,0.9))}

    blue_blue = LinearSegmentedColormap('BlueBlue', cdict)
    plt.register_cmap(cmap=blue_blue)
    cmap=plt.get_cmap('BlueBlue')

    cdict2 = {'red': ((0,0.4,0.4),(1,0.151,0.151)),'green':((0,0.6,0.6),(1,0.2,0.2)),'blue': ((0,0.9, 0.9),(1,0.32,0.32))}
    blue_blue_r = LinearSegmentedColormap('BlueBlue_r', cdict2)
    plt.register_cmap(cmap=blue_blue_r)
    cmap=plt.get_cmap('BlueBlue_r')

# green pallette

    cdict = {'red': ((0,0.151,0.151),(1,0.4,0.4)),'green': ((0,0.32, 0.32),(1,0.9,0.9)) ,'blue': ((0,0.2,0.2333),(1,0.6,0.6))}

    green_green = LinearSegmentedColormap('GreenGreen', cdict)
    plt.register_cmap(cmap=green_green)
    cmap=plt.get_cmap('GreenGreen')

    cdict2 = {'red': ((0,0.4,0.4),(1,0.151,0.151)),'green':((0,0.9, 0.9),(1,0.32,0.32)),'blue': ((0,0.6,0.6),(1,0.2,0.2))}

    green_green_r = LinearSegmentedColormap('GreenGreen_r', cdict2)
    plt.register_cmap(cmap=green_green_r)
    cmap=plt.get_cmap('GreenGreen_r')

# red pallette

    cdict = {'red': ((0,0.2,0.2),(0.5,0.9,0.9),(1,0.9,0.9)),'green': ((0,0.2, 0.2),(.5,0.9,0.9),(1,0.2,0.2)) ,'blue': ((0,0.9,0.9),(0.5,0.9,0.9),(1,0.2,0.2))}

    red_red = LinearSegmentedColormap('RedRed', cdict)
    plt.register_cmap(cmap=red_red)
    cmap=plt.get_cmap('RedRed')

    cdict2= {'red': ((0,0.9,0.9),(0.5,0.2,0.2),(0.5,0.9,0.9),(1,0.9,0.9)),'green': ((0,0.9, 0.9),(.5,0.2,0.2),(.5,0.2,0.2),(1,0.9,0.9)) ,'blue': ((0,0.9,0.9),(0.5,0.9,0.9),(0.5,0.2,0.2),(1,0.9,0.9))}

    red_red_r = LinearSegmentedColormap('RedRed_r', cdict2)
    plt.register_cmap(cmap=red_red_r)
    cmap2=plt.get_cmap('RedRed_r')

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

    cmap=red_red
    vvs = vvs_all_1
    if plotExt != '':
        plotExt = '_' + plotExt
    for vInd in vInds:
        if plotExt[-3:]=='raw':
            pctls =  array([1,5,10,20,30,40])
            for pctl in pctls:
                plt.fill_between(ccs,percentile(out[plotType]['min'+plotExt][:,:,vInd],pctl,0),percentile(out[plotType]['max'+plotExt][:,:,vInd],100-pctl,0),color=cmap(128+sign(vInd-10)*pctl*128/40),label=str(stds_1[vInd]))
                #if vInd>4:
                #    plt.fill_between(ccs,out['combined']['min'][:,vInd],out['combined']['max'][:,vInd],color=cmap((vInd+1)*256/len(vvs_all)),label=str(stds_1[vInd]))
        else:
            plt.fill_between(ccs,out[plotType]['min'+plotExt][:,vInd]-0.01,out[plotType]['max'+plotExt][:,vInd]+0.01,color=cmap(int(vInd*256/len(vvs_all_1))),label=str(stds_1[vInd]))
            # ut[
            #X=array([0.0001,0.0001])
            #Y=array([0.0001,0.0001])
            #cc=ax.pcolor(X,Y,array([[vv,vvs[-1]],[vvs[0],vvs[-1]]]),cmap=cmap)      
    if not(repeat_fig):
      ax=plt.gca()
      ax.set_ylim([-1,1])
      ax.set_xlim([-1,1])

    if plotExt[-3:]=='raw':

      cmap=red_red_r
      cc=ax.pcolor(array([[.3]]),cmap=cmap,visible=False,vmin=min(vvs),vmax=max(vvs))      
      cbar=plt.colorbar(cc,shrink=0.5,fraction=0.1)
      cbar.set_ticks(array([min(vvs),1,max(vvs)]))

      pctls_1 = str(-100+abs(array(pctls)).min()/2.0)
      #pctls_2 = str(-100+abs(array(pctls)).max()/2.0)
      pctls_3 = str(0)
      #pctls_4 = str(100-abs(array(pctls)).max()/2.0)
      pctls_5 = str(100-abs(array(pctls)).min()/2.0)

      cbar.set_ticklabels( [pctls_1,pctls_3, pctls_5])
      cbar.set_label('Percentile of max/min range of change')
    else:
      cc=ax.pcolor(array([[.3]]),cmap=cmap,visible=False,vmin=min(vvs),vmax=max(vvs))      
      cbar=plt.colorbar(cc,shrink=0.5,ticks=vvs,fraction=0.1)
      cbar.set_label('Proportional change in variance \n in both regions in second condition.')

    plt.plot(ccs,ccs,color=[0.9,0.9,0.9],label='Test',linewidth=5)

    ax.set_xlabel('Correlation in condition 1')
    ax.set_ylabel('Correlation in condition 2')

    #if imgtitle==0:
    ax.set_title('Changes in correlation that can be explained \n by decreases in signal amplitude of one region of .'+str(vvs_all_1[vInds[0]])+'% in one region')
    
    # fig.savefig('Decreases_sim.pdf')
    fig.savefig('Decreases_both'+str(vInds[0])+'.pdf')
    return(out)

    #else:
    #    ax.set_title('Changes in correlation that can be explained \n by an increase in signal amplitude of '+str(vvs_all[vInds[0]])+'% in both regions.')

    #    fig.savefig('Increases_both'+str(aaa)+'.pdf')


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




#
#if __name__=="__main__":
#
#    parser = argparse.ArgumentParser(description="Calc corrs")
#    parser.add_argument("-e", "--errdist_perms", default=0, help="errdist_perms")
#    requiredNamed = parser.add_argument_group('required arguments')
#    requiredNamed.add_argument('-c', '--cc', help='correlation', required=True)
#    args = parser.parse_args()
#
#    plot_sim_err(args.cc,errdist_perms=args.errdist_perms)
#
