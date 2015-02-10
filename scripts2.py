from numpy import *
import numpy as np
import matplotlib.pylab as pl
import glob,os
import nibabel as nb

def range(p1,p2,stdx1,stdx2,stdy1,stdy2,outname=[]):
    # Inputs (can be matrices or filenames):
    #
    # p1 - first corr
    # p2 - second corr
    # stdx1 - first std of region x
    # stdx2 - second std of region x
    # stdy1 - first std of region y
    # stdy2 - second std of region y
    # outputname - output name (when files input
    #
    # Outputs:
    # out[x][0] outputs for 1>2 
    # out[x][1] outputs for 2>1
    #
    # out[0][x] min values for p2
    # out[1][x] max values for p2
    # out[2][x] coupling 
    # out[3][x] coupling shared 
    # out[4][x] coupling unshared

    # load images

    if type(p1)==str:
        p1_img=nb.load(p1)
        p1_name=p1
        p1=p1_img.get_data()

        p2_img=nb.load(p2)
        p2_name=p2
        p2=p2_img.get_data()
    
        if outname == []:
            outname=p2_name.replace('.nii.gz','')  

    # arrayify if float test

    if ~isinstance(p1,np.ndarray):
        p1=array([p1])
        p2=array([p2])
        stdx1=array([stdx1])
        stdx2=array([stdx2])

    if type(stdx1)==str:
        stdx1_img=nb.load(stdx1)
        stdx1_name=stdx1
        stdx1=stdx1_img.get_data()

        stdx2_img=nb.load(stdx2)
        stdx2_name=stdx2
        stdx2=stdx2_img.get_data()

    if type(stdy1)==str:

        if os.path.exists(stdy1):

            filename,Ext = os.path.splitext(stdy1)

            if (Ext=='.txt'):
                stdy1=loadtxt(stdy1)
                stdy2=loadtxt(stdy2)

                if len(stdy1)!=1:
                    stdy1=std(stdy1)
                    stdy2=std(stdy2)
        else:
            stdy1=float(stdy1)
            stdy2=float(stdy2)

    if type(stdy1)!=array:
        stdy1=zeros(stdx1.shape).astype(stdx1.dtype)+stdy1
        stdy2=zeros(stdx1.shape).astype(stdx1.dtype)+stdy2

    # create index arrays
    # remove 0-variance elements - create mask

    ii= np.unravel_index(pl.find((stdx1!=0) & (stdx2!=0) & (stdy1!=0) & (stdy2!=0)),stdx1.shape)
    p1m=p1[ii]
    p2m=p2[ii]

    stdx1m=stdx1[ii]
    stdx2m=stdx2[ii]

    stdy1m=stdy1[ii]
    stdy2m=stdy2[ii]

    # set up coupling class arrays
    coupling=zeros((2,)+p1.shape)
    coupling_shared=zeros((2,)+p1.shape)
    coupling_unshared=zeros((2,)+p1.shape)
    coupling_dir=zeros((2,)+p1.shape)

    # set up min/max p arrays
    min_p=zeros((2,)+p1.shape)
    max_p=zeros((2,)+p1.shape)

    #min_p=zeros(r_[2,array(stdx1m.shape)]).astype(stdx1m.dtype)
    #max_p=zeros(r_[2,array(stdx1m.shape)]).astype(stdx1m.dtype)
    class_p=zeros(r_[2,array(stdx1m.shape)]).astype(stdx1m.dtype)

    # aross a>b and b>a 

    for bb in [0,1]:

        ps = array([p1m,p2m])
        sign_ps=sign(ps[bb,:])
        ps=ps*array([sign_ps,sign_ps])

        if bb == 0:
            ssx=stdx1m/stdx2m
            ssy=stdy1m/stdy2m
        else:
            ssx=stdx2m/stdx1m
            ssy=stdy2m/stdy1m
        
        # identify different types of changes

        xdown = pl.find(ssx>1)
        ydown = pl.find(ssy>1)
        bothup = ((ssx>1) & (ssy>1))
        bothdown = ((ssx<1) & (ssy<1))

        diff = pl.find(((ssx>1) & (ssy<1)) | ((ssx<1) & (ssy>1))) 

        # generate parameter types minwxu

        minwxu=ps[bb,:].copy()
        maxwxu=ones(p1m.shape)
        minwxc=ps[bb,:].copy()
        maxwxc=ones(p1m.shape)

        # now go through scenarios filling out elements 

        maxwxu[xdown]=minimum(1,1/ssx[xdown])
        minwxu[ydown]=ssy[ydown]*ps[bb,ydown]

        maxwxc[ydown]=minimum(maxwxc[ydown],ps[bb,ydown]*1/sqrt(1-1/ssy[ydown]**2))
        minwxc[xdown]=sign(minwxc[xdown])*maximum(abs(minwxc[xdown]),sqrt(1-1/ssx[xdown]**2))

        uymin=ones(stdx1m.shape)
        uymax=ones(stdx1m.shape)
        oones = ones(stdx1m.shape) 

        # ydown: c=1 wx=low

        uymin[ydown]=calc_uy(ssy[ydown],minwxc[ydown],ps[bb,ydown])
        uymax[ydown]=calc_uy(ssy[ydown],maxwxc[ydown],ps[bb,ydown])

        uxmin=ones(stdx1m.shape)
        uxmax=ones(stdx1m.shape)
        uxmax.flat[xdown]=calc_ux(ssx,maxwxc)
        uxmin.flat[xdown]=calc_ux(ssx,minwxc)

        ## Ranges
        # u=1  
        # go through max,min wx,u

        cxwhu1= calc_c(ssx,maxwxc)
        cywhu1= calc_c(ssy,ps[bb,:]/maxwxc) 

        cxwlu1= calc_c(ssx,minwxc)
        cywlu1= calc_c(ssy,ps[bb,:]/minwxc)

        p2ul = ps[bb,:]*ssx*ssy*cxwlu1*cywlu1
        p2uh = ps[bb,:]*ssx*ssy*cxwhu1*cywhu1

        # cx=1

        #cxwhc1= calc_c(ssx,maxwxc,uxmax)
        #cywhc1= calc_c(ssy,p1/maxwxc,uymax)

        #cxwlc1= calc_c(ssx,minwxc,uxmin)
        #cywlc1= calc_c(ssy,p1/minwxu,uymin)

        p2cl = ps[bb,:]*ssx*ssy #*cxwlc1*cywlc1
        #p2ch = p1*ssx*ssy*cxwhc1*cywhc1

        min_p_tmp = nanmin(c_[p2cl,p2ul,p2uh],1)
        min_p_tmp[min_p_tmp<-1]=-1
        max_p_tmp = nanmax(c_[p2cl,p2ul,p2uh],1)
        max_p_tmp[max_p_tmp>1]=1

        extremes=array([min_p_tmp,max_p_tmp])

        # classify corr changes
        absmin_p_tmp = extremes[abs(extremes).argmin(0),arange(len(ii[0]))] 
        absmax_p_tmp = extremes[abs(extremes).argmax(0),arange(len(ii[0]))] 

        # calculate std of correlation
        #coupling_tmp = (ps[1-bb,:]<min_p_tmp*(1-0.15*sign(min_p_tmp))) | (ps[1-bb,:]>max_p_tmp*(1+0.15*sign(min_p_tmp)))
        coupling_tmp = (ps[1-bb,:]<min_p_tmp) | (ps[1-bb,:]>max_p_tmp) 
        # coupling_tmp = (ps[1-bb,:]<min_p_tmp-1.3*abs(min_p_tmp)) | (ps[1-bb,:]>max_p_tmp+1.3*abs(max_p_tmp)) 
        coupling[(bb,)+ii]  = coupling_tmp 
        coupling_dir[(bb,)+ii]  = ( abs(ps[1-bb,:])<(abs(extremes).min(0)) ) & coupling_tmp
        coupling_dir[(bb,)+ii]  = (( abs(ps[1-bb,:])>(abs(extremes).max(0)) ) ^ coupling_tmp)
        
        # switch negative vals around
        tmp=min_p_tmp.copy()
        min_p_tmp=max_p_tmp
        #max_p_tmp[neg_ps[bb,]]=-tmp[neg_ps[bb,]]

        min_p[(bb,)+ii] = min_p_tmp
        max_p[(bb,)+ii] = max_p_tmp

        # increase corr: increased signal
        increase_corr=ps[bb,:]>ps[abs(1-bb),:]

        # increase corr: decrease signal
        coupling_shared[(bb,)+ii] = (increase_corr & bothup) | (~increase_corr & bothup)
        coupling_unshared[(bb,)+ii] = (increase_corr & bothup) | (~increase_corr & bothdown)
        
        if outname != []:
            p1[:,:,:]=coupling[bb,:,:,:]
            p1_img.to_filename(outname + '_coupling_' + str(bb)+'.nii.gz')

            p1[:,:,:]=coupling_dir[bb,:,:,:]
            p1_img.to_filename(outname + '_coupling_dir_' + str(bb)+'.nii.gz')

            p1[:,:,:]=min_p[bb,:,:,:]
            p1_img.to_filename(outname + '_min_' + str(bb)+'.nii.gz')


            p1[:,:,:]=max_p[bb,:,:,:]
            p1_img.to_filename(outname + '_max_' + str(bb)+'.nii.gz')
    
    return(min_p,max_p,coupling,coupling_shared,coupling_unshared)

def  calc_ux(ssx,wa):
    ux=ones(ssx.shape)
    ux=sqrt((1/ssx**2 - wa**2)/(1-wa**2))
    return(ux)

def calc_uy(ssy,wa,p1):
    uy=ones(ssy.shape)
    uz=zeros(ssy.shape)
    uy=sqrt(maximum(uz,(1/ssy**2-(p1**2/wa**2))*wa**2/(wa**2-p1**2)))
    return(uy)

def calc_c(ssx,wa,ux=[]):

    uz=zeros(ssx.shape)
    if ux == []:
        ux=ones(wa.shape)
    cc=sqrt(maximum(uz,(1/(ssx**2) - (ux**2)*(1 - wa**2)) * (1/wa**2)))  
    return(cc)

def loadDR(directory):
    filelist=sort(glob.glob(directory + '/dr_stage1_subject?????.txt'))

    matlist=[]

    for a in arange(len(filelist)):
        matlist.append(loadtxt(filelist[a]))

    return matlist

def plot_corr_example():

    ccs=arange(0.001,1,0.001)
    vvs_all=arange(-0.2,2.025,0.025)+1
    stds=vvs_all*100


    out=zeros((len(ccs),len(vvs_all),2))
    els = [[0,8],[9,17]]
    figs=[2,3]


    for img in [0,1]:
        fig=plt.figure(figs[img]) 
        fig.clf()
        vvs=vvs_all[els[img][0]:(els[img][0]+8)]

        for a in arange(8):
            for cc in arange(len(ccs)):
                for vv in arange(len(vvs)):
                    tmp=range(ccs[cc],.99,1.0,1.0*vvs[vv],1,1)
                    out[cc,vv,:]=[tmp[0][0],tmp[1][0]]

        cdict = {'red': ((0,0.133333,0.13333),(1,0.31,0.31)),'green':((0,0.133333,0.13333),(1,0.47,0.47)),'blue': ((0,0.133333,0.13333),(1,0.78,0.78))}

        cmap=get_cmap('BlueBlue')
        blue_blue = LinearSegmentedColormap('BlueBlue', cdict)
        register_cmap(cmap=blue_blue)
                
        for a in arange(len(vvs)):
            fill_between(ccs,out[:,a,0],out[:,a,1],color=pllt[-(a+1),:],label=str(stds[a]))
            plt.plot([],[],color=pllt[-(a+1),:],label=str(stds[a]),linewidth=15)

        X=array([0.0001,0.0001])
        Y=array([0.0001,0.0001])

        cc=ax.pcolor(X,Y,array([[vvs[0],vvs[0]],[vvs[1],vvs[1]]]),cmap=cmap)      
        colorbar()


