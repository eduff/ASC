#!/usr/bin/env python
#
# asc_funcs.py 
#
# Author: Eugene Duff <eugene.duff@gmail.com>
#
#
#############################################################################
# general connectivity matrix class, with methods for calculating correlation etc.

# from numpy import *
import numpy as np
import matplotlib.cm
import matplotlib.pylab as pl
import glob,os, numbers
import scipy.stats as stats
import scipy.sparse
import spectrum
import scipy.linalg as linalg
from itertools import combinations, chain
from scipy.misc import comb
from scipy.interpolate import interp1d
from scipy import sparse
from scipy.sparse import coo_matrix
from scipy.signal import welch
from multiprocessing import Process, Queue, current_process, freeze_support  # for parallel processing of seed-based connectivity

# optional modules
try:
    import nibabel as nb
except:
    print("nibabel not detected, no image loading available")

class FC:
    """ Functional Connectivity Class.

    
    Stores and provides FC data of a number of subjects, including raw time series (optional), covariance, 
    partial correlation etc.
    """  
    ### todo doc
    def __init__(self,tcs,cov_flag=False,dof=None,ROI_info={},mask=None):
        """  Initialise the FC object. """
        #  reading covariance matrix
        if cov_flag==True:
            self.tcs=None

            covs=tcs
            
            if dof is None:
                raise ValueError("Need to specify dof if providing cov.")
            else:
                self.dof=dof
  
        else:
            # reading timecourses
            if tcs.ndim==2:
                tcs=(np.atleast_3d(tcs).transpose(2,0,1))
            
            self.tcs = tcs
            if dof is None:
                self.dof=tcs.shape[-1]-1
            elif dof == 'EstEff':
                AR=np.zeros((tcs.shape[0],tcs.shape[1],15))
                ps=np.zeros((tcs.shape[0],tcs.shape[1],15))
                for subj in np.arange(tcs.shape[0]): 
                    for ROI in np.arange(tcs.shape[1]):
                        AR[subj,ROI,:]=spectrum.aryule(pl.demean(tcs[subj,ROI,:]),15)[0]
                        ps[subj,ROI,:]=spectrum.correlation.CORRELATION(pl.demean(tcs[subj,ROI,:]),maxlags=14,norm='coeff')
                ps = np.mean(np.mean(ps,0),0)
                AR2 = np.mean(np.mean(AR,0),0)
                dof_nom=tcs.shape[-1]-1
                self.dof = int(dof_nom / (1-np.dot(ps[:15].T,AR2))/(1 - np.dot(np.ones(len(AR2)).T,AR2)))
            else:
                self.dof=dof

            covs = get_covs(tcs)

        if ROI_info=={}:
            shp=covs.shape[-1]
            ROI_info['ROI_names']=np.arange(shp).astype(str)
            ROI_info['ROI_RSNs']=np.ones(shp,).astype(int)

        if mask:
            self.mask=mask

        self.ROI_info = ROI_info

        self.covs = covs


    #####  FC functions #####

    def get_stds(self,pcorrs=False):
        """ calculate std_devs. """

        if not( 'stds' in self.__dict__):
            sz = self.get_covs(pcorrs=pcorrs).shape
            if len(sz)==3:
                self.stds=np.diagonal(self.get_covs(pcorrs=pcorrs),axis1=len(sz)-2,axis2=len(sz)-1)**(0.5) 
            else:
                self.stds=self.get_covs(pcorrs=pcorrs).diagonal()**0.5
        return(self.stds)

    def get_stds_m(self,pcorrs=False):
        """ calculate std_devs (return matrix form). """
        stds_m = self.get_stds(pcorrs=pcorrs)
        return(np.tile(stds_m,(1,self.covs.shape[1])).reshape(stds_m.shape[0],stds_m.shape[1],stds_m.shape[1]))

    def get_stds_m_t(self,pcorrs=False):
        """ calculate std_devs (return transpose matrix form). """

        return transpose(self.get_stds_m(pcorrs=pcorrs),(0,2,1))

    def get_corrs(self,pcorrs=False):
        """ get correlation (partial). """

        if pcorrs:
            return(self._get_pcorrs())
        else:
            if not( 'corrs' in self.__dict__):
                # initialise as normal/sparse matrix
                els=self.get_covs().nonzero()
                self.corrs=self.get_covs()*0
                dims=np.arange(len(els))
                dims1 = tuple(np.setdiff1d(dims,dims[-1]))
                dims2 = tuple(np.setdiff1d(dims,dims[-2]))

                els1=tuple(els[index] for index in dims1)
                els2=tuple(els[index] for index in dims2)

                self.corrs[els] = np.array(self.get_covs()[els])/(self.get_stds()[els1]*self.get_stds()[els2])
            return(self.corrs)
    # get partial correlation 
    def _get_pcorrs(self):

        if not( 'pcorrs' in self.__dict__):
            
            self.pcorrs=self.get_corrs()*0
            for a in range(len(self.pcorrs)):
                self.pcorrs[a,:,:]=corr2pcorr(self.corrs[a,:,:])
               
        return self.pcorrs 

    def get_covs(self,pcorrs=False):
        if pcorrs:
            return self._get_pcovs()
        else:
            return self.covs

    def _get_pcovs(self):
        if not( 'pcorrs' in self.__dict__):
            pcorrs=self._get_pcorrs()

        multiplier = (self.get_stds_m()*transpose(self.get_stds_m(),(0,2,1)))
        
        return(pcorrs*multiplier)


class FC_con:
    """ Class for contrasts between two states, A and B, providing functions to calculate stats, ASC analysis limits etc. """
    def __init__(self,A,B):     
        self.A=A
        self.B=B

    def get_corr_stats(self,pcorrs=False,rel=True): 
        """  get basic correlation statistics between two states. """
        if pcorrs:
            out = self.get_pcorr_stats(self,rel=rel)
        else:
            if not( 'corr_stats' in self.__dict__):
                if rel == True:
                    self.corr_stats = stats.ttest_rel(rtoz(self.A.get_corrs(pcorrs=False)),rtoz(self.B.get_corrs(pcorrs=False)))
                else:
                    self.corr_stats = stats.ttest_ind(rtoz(self.A.get_corrs(pcorrs=False)),rtoz(self.B.get_corrs(pcorrs=False)))
            out = self.corr_stats
        
        return out

    def get_pcorr_stats(self,rel=True): 
        """  get basic partial correlation statistics between two states. """
        if not( 'pcorr_stats' in self.__dict__):
            if rel == True:
                self.pcorr_stats = stats.ttest_rel(rtoz(self.A.get_corrs(pcorrs=True)),rtoz(self.B.get_corrs(pcorrs=True)))
            else:
                self.pcorr_stats = stats.ttest_ind(rtoz(self.A.get_corrs(pcorrs=True)),rtoz(self.B.get_corrs(pcorrs=True)))
            out = self.pcorr_stats

        return self.pcorr_stats

    def get_std_stats(self,pcorrs=False,rel=True): 
        """ get basic std statistics between two states. """
        if not( 'std_stats' in self.__dict__):
            if rel == True:
                self.std_stats = stats.ttest_rel(self.A.get_stds(pcorrs=pcorrs),self.B.get_stds(pcorrs=pcorrs))[0]
            else:
                self.std_stats = stats.ttest_ind(self.A.get_stds(pcorrs=pcorrs),self.B.get_stds(pcorrs=pcorrs))[0]
        return self.std_stats

    def get_ASC_lims(self,pcorrs=False,pctl=5,errdist_perms=50,refresh=False,sim_sampling=40):
        """  get ASC limits. """
        
        if not( 'lims' in self.__dict__) or refresh:

            # take mean of all subjects if required 
            if self.A.tcs is None and np.ndim(self.A.get_covs())==3: 
                A=FC(np.mean(self.A.get_covs(pcorrs=pcorrs),0),cov_flag=True, dof=self.A.dof,ROI_info=self.A.ROI_info)
                B=FC(np.mean(self.B.get_covs(pcorrs=pcorrs),0),cov_flag=True, dof=self.B.dof,ROI_info=self.B.ROI_info)
            elif self.A.tcs is None:
                A=self.A
                B=self.B
            else:
                A=flatten_tcs(self.A)
                B=flatten_tcs(self.B)

            # calculate ASC limits
            self.lims=ASC_lims_all(A,B,errdist_perms=errdist_perms,pctl=pctl,pcorrs=pcorrs,sim_sampling=sim_sampling)
        return(self.lims)

    def get_plot_inds(self,inds_cc=None,exclude_conns=True):
        """  get ASC limits. """
                              
        self.lims = gen_plot_inds(self.lims,inds_cc=inds_cc,exclude_conns=exclude_conns)

def gen_plot_inds(lims,inds_cc=None,exclude_conns=True):
    """ determine which connections should be plot in each ASC class. """ 
    
    plots = list(lims.keys())+ ['other']
    plots.remove('covs') 
    
    # check sparseness for image-based (seed region analysis)
    if scipy.sparse.issparse(lims[plots[0]]['pctls']):
        voxels =  lims[plots[0]]['pctls'][0,1:].toarray()
        if not inds_cc:
            inds_cc=np.arange(voxels.shape[-1]).astype(int)

        inds_plots={}
        notin = inds_cc.copy()

        for plot in plots[:3]:
            inds_plots[plot]=np.intersect1d(inds_cc,pl.find(lims[plot]['pctls'][0,1:].toarray()))
            notin = np.setdiff1d(notin,inds_plots[plot])

        inds_plots['other'] = notin
        
        # exclude from additive if in common or uncorrelated
        if exclude_conns:
            inds_plots['common']=np.setdiff1d(inds_plots['common'],inds_plots['uncorrelated'])
            inds_plots['additive']=np.setdiff1d(inds_plots['additive'],inds_plots['common'])
            inds_plots['additive']=np.setdiff1d(inds_plots['additive'],inds_plots['uncorrelated'])
    else:
        # check if correlation matrix is provided
        if not inds_cc:
            inds_cc=np.triu_indices(n_nodes,1)
        indices=np.triu_indices(n_nodes,1)
        notin=inds_cc
        inds_plots={}

        for plot in plots[:3]:
            inds_plots[plot]=np.intersect1d(inds_cc,pl.find(fa(lims[plot]['pctls'])))
            notin = np.setdiff1d(notin,pl.find(fa(lims[plot]['pctls'])))

        inds_plots['other'] = notin

        # exclude from additive if in common or uncorrelated
        if exclude_conns:
            inds_plots['common']=np.setdiff1d(inds_plots['common'],inds_plots['uncorrelated'])
            inds_plots['additive']=np.setdiff1d(inds_plots['additive'],inds_plots['common'])
            inds_plots['additive']=np.setdiff1d(inds_plots['additive'],inds_plots['uncorrelated'])

    #self.inds_plots=inds_plots
    lims['covs']['inds_plots']=inds_plots

    return(lims)

def ASC_lims_all(A,B,pcorrs=False,errdist_perms=0,dof=None,pctl=10,ch_type='All',sim_sampling=40,mask=None,keepRaw=True):
    """ core ASC limits routine. 

    Takes two states, A and B and convert to  analyses differences. """
    if not dof:
        dof=A.dof

    if ch_type == 'All':
        ch_type = ['covs','uncorrelated','common','additive']
    elif type(ch_type)==str:                                                                
        ch_type=[ch_type]

    # masking for image-based analysis
    if mask is None:
        shp = A.get_covs().shape
        if scipy.sparse.issparse(A.covs):
            mask = A.get_covs().nonzero()
            #todo fix masking
        else:
            mask =  np.triu_indices(shp[-1],k=1) 

            if len(shp) == 3:
                first=np.repeat(np.arange(shp[0]),len(mask[0]))
                second=np.tile(mask[0],shp[0])
                third=np.tile(mask[1],shp[0])
                mask=tuple((first,second,third))

    # set up basic variables

    # flattened covariance and corr matrices
    covsA=np.array(A.get_covs(pcorrs=pcorrs)[mask])
    covsB=np.array(B.get_covs(pcorrs=pcorrs)[mask])

    Acorrs=np.array(A.get_corrs(pcorrs=pcorrs)[mask]).flatten()
    Bcorrs=np.array(B.get_corrs(pcorrs=pcorrs)[mask]).flatten()   
    shp=Bcorrs.shape

    # masks for stds
    dims=np.arange(len(mask))
    els1 = tuple(np.setdiff1d(dims,dims[-1]))
    els2 = tuple(np.setdiff1d(dims,dims[-2]))
    
    mask1=tuple(mask[index] for index in els1)
    mask2=tuple(mask[index] for index in els2)

    Astdm=A.get_stds()[mask1]
    Astdmt=A.get_stds()[mask2]
    Bstdm=B.get_stds()[mask1]
    Bstdmt=B.get_stds()[mask2]

    pctls=None
    lims_struct={}

    # variance diffs
    diffX=(Bstdm**2-Astdm**2)
    diffY=(Bstdmt**2-Astdmt**2)

    # calculate limits for dfferent types of ASC class 
    for a in ch_type:
        
        lims_struct[a]={}

        if a=='uncorrelated':
            # uncorrelated signal produces straight forward limits
            lims_struct[a]={}
            uncorrelated = covsA / (Bstdm*Bstdmt)

            # large reductions in variance can be greater than is possible for uncorrelated signal, given initial correlation. 
            # uncorrelated[uncorrelated>1]=1 
            # uncorrelated[uncorrelated<-1]=-1
            zz =   A.get_covs()*0
            lims_struct[a]['min'] = zz.copy()
            lims_struct[a]['min'][mask] = uncorrelated
            lims_struct[a]['max'] = zz.copy()
            lims_struct[a]['max'][mask] = uncorrelated

            # check whether B (exactly) equal to uncorrelated
            lims_struct['uncorrelated']['pctls_noerr']=zz.copy()
            lims_struct['uncorrelated']['pctls_noerr'][mask]=(Bcorrs> lims_struct['uncorrelated']['min'][mask]) != (Bcorrs> lims_struct['uncorrelated']['max'][mask])

        elif a== 'common':
            # to identify limits for common, we sample across possible common signals

            # 3d inds = np.tile(np.arange(sim_sampling),(shp[0],shp[1],shp[1],1))
            # 3d Common=np.zeros((shp[0],shp[1],shp[2],sim_sampling,sim_sampling))
            # 3d Common=np.zeros((shp,sim_sampling,sim_sampling))

            shp=Acorrs.shape
            inds = np.tile(np.arange(sim_sampling),(shp[-1],1))

            # Common stores the limits for each sampled common signal / default to nan
            Common_max=np.zeros(shp)*np.nan
            Common_min=np.zeros(shp)*np.nan

            # sampling range (for correlation of new signal with X_A and Y_A)
            smpling=(np.arange(sim_sampling)/(sim_sampling-1.0))
            smpling[-1]=0.99999 # make final value slightl less than 1

            # orthogonal components form (see supp materials)
            a11=Astdm   # std of node 1
            a21=Astdmt*Acorrs  # std of node 2 projected to node 1
            a22=np.sqrt(Astdmt**2-a21**2)     # remaining std of node 2
            
            # set up data matrices
            Common=np.zeros(shp+(sim_sampling*2,))  # final common signal correlation

            # correlation max and min bounds  remove?
            #cc1=np.zeros(shp+(sim_sampling*2,))  # lower correlation
            #cc2=np.zeros(shp+(sim_sampling*2,))  # upper correlation bound

            # 
            kp=np.zeros(shp+(sim_sampling*2,))
            km=np.zeros(shp+(sim_sampling*2,))

            # 
            a31a=np.zeros(shp+(sim_sampling*2,)) # std of node 2 projected to node 1
            a31b=np.zeros(shp+(sim_sampling*2,))
            a32a=np.zeros(shp+(sim_sampling*2,))
           
            # sign of change in variance
            signX=np.sign(Bstdm-Astdm)
            signY=np.sign(Bstdmt-Astdmt)

            cnt=0
                
            #  loop over different correlations of X_A with N (new signal), which may be +ve or -ve

            for bbb in [-1,1]:
                #  loop over different correlations of X_A with N, cal
                for aaa in (np.arange(len(smpling))/len(smpling))[::(-bbb)]:

                    #match sign to that of original correlation 
                    corrA=signX*aaa

                   # calculate a_31 from quadratic 
                    aa=1
                    bb=2*a11*corrA**2
                    cc=-(corrA**2)*( diffX )
                    a31 = (-bb + np.sqrt(bb**2 - 4*aa*cc))/(2*aa)
                    #a31a[:,cnt] = (-bb + np.sqrt(bb**2 - 4*aa*cc))/(2*aa)
                    #a31=a31a[:,cnt]

                    # now calculate a_32 from quadratic
                    aa = ( Bstdm**2-Astdm**2 ) -2*a11*a31
                    a32a[:,cnt]=bbb*np.sqrt(abs(aa - a31**2 )) 
                    a32=a32a[:,cnt]
                    
                    # calculate k, positive and negative  (Case 1, Common Signal, Supporting Info)
                    bb = 2*(a21*a31+a22*a32)
                    cc = -( diffY )
                    kp[:,cnt] = (-bb + np.sqrt(bb**2 - 4*aa*cc))/(2*aa)
                    km[:,cnt] = (-bb - np.sqrt(bb**2 - 4*aa*cc))/(2*aa)
                    ks=np.c_[kp[:,cnt],km[:,cnt]] 

                    # exclude wrong sign
                    ks[(ks[:,0]*np.sign(Acorrs)*np.sign(diffX)*np.sign(diffY))<0,0]=np.inf
                    ks[(ks[:,1]*np.sign(Acorrs)*np.sign(diffX)*np.sign(diffY))<0,1]=np.inf

                    # find the minimum k
                    k=ks[np.arange(shp[0]),np.argmin(abs(ks),axis=1)]
                    
                    # now calculate correlations  remove?
                    #cc1[:,cnt]=(a11*a31)/((a11)*np.sqrt(a31**2+a32**2))
                    #cc2[:,cnt]=(a21*k*a31 + a22*k*a32)/(np.sqrt((a21**2+a22**2))*np.sqrt((k*a31)**2+(k*a32)**2))

                    # now calculate covariances                     
                    cov_max_comm = a11*a21 + a11*a31*k + (a21*a31) + a22*a32 + k*(a31**2  + a32**2 )
                    
                    # calculate correlations
                    Common[:,cnt]=cov_max_comm/(( Bstdm)*( Bstdmt)) 
    
                    # check that changes are within observed variance limits
                    Common[((a31)**2 + (a32)**2) >abs(diffX),cnt]=np.nan
                    Common[((a31*k)**2 + (a32*k)**2) >abs(diffY),cnt]=np.nan

                    # if diffX negative, can't remove more than initial variance  
                    Common[(2*(a11*a31))<(diffX-abs(diffX)),cnt]=np.nan

                    # exclude sign flips
                    Common[np.sign(Acorrs)*np.sign(k)!=(np.sign(diffY)*np.sign(diffX)),cnt]=np.nan

                    Common[k==np.inf,cnt]=np.nan
                    cnt+=1

            # Common >=0  
            Common_min = np.fmin(Common_min,np.nanmin(Common,1))
            lims_struct[a]['min'] = A.get_covs()*0 
            lims_struct[a]['min'][mask]=Common_min

            Common_max = np.fmax(Common_max,np.nanmax(Common,1))
            lims_struct[a]['max'] = A.get_covs()*0 
            lims_struct[a]['max'][mask]=Common_max

            # common results 
            lims_struct['common']['pctls_noerr']=A.get_covs()*0
            lims_struct['common']['pctls_noerr'][mask]=(Bcorrs> lims_struct['common']['min'][mask]) != (Bcorrs> lims_struct['common']['max'][mask])

        elif a == 'additive':

            # calculate bounds for Additive Signal addition 
            # using formulation from Supporting Info., ASC paper
 
            # set up data
            posX=(diffX>=0)
            posY=(diffY>=0)

            # known variables
            a11=Astdm
            a21=Astdmt*Acorrs
            a22=np.sqrt(Astdmt**2-a21**2)

            # unknown variables
            a31=np.zeros(a22.shape)
            a32=np.zeros(a22.shape)
            a41=np.zeros(a22.shape)
            a42p=np.zeros(a22.shape)
            a42n=np.zeros(a22.shape)
                
            # unknown values for max increases in signal (pos.corr) (See Supporting info)
            a32[posX]=np.sqrt(abs(diffX[posX]-a31[posX]**2))
            a31[posX]=0
            a41[posY]=np.sqrt(abs(a22[posY]**2/(a22[posY]**2+a21[posY]**2)))*np.sqrt(abs(diffY[posY]))
            a42p[posY]=-a41[posY]*(a21[posY]/a22[posY])
            a42n[posY]=a41[posY]*(a21[posY]/a22[posY])

            # unknown values for max decreases
            a31[~posX]=diffX[~posX]/a11[~posX]
            a32[~posX]=np.sqrt(-a31[~posX]**2-diffX[~posX])
            a41[~posY]=np.sqrt(abs(a22[~posY]**2/(a22[~posY]**2+a21[~posY]**2)))*np.sqrt(abs(diffY[~posY]))
            a42p[~posY]= (-a21[~posY]*a41[~posY] + diffY[~posY])/a22[~posY] 
            a42n[~posY]= (a21[~posY]*a41[~posY] + diffY[~posY])/a22[~posY] 
    
            # varous possible values for min/max lims options e.g. a32,a41 pos/neg
            lims1=a11*a21+a11*a41 + a22*a32 + a32*a42p
            corr1=lims1/(Bstdm*Bstdmt) 
            lims2=a11*a21-a11*a41 - a22*a32 - a32*a42n
            corr2=lims2/(Bstdm*Bstdmt) 
            lims3=a11*a21+a11*a41 - a22*a32 - a32*a42p
            corr3=lims3/(Bstdm*Bstdmt) 
            lims4=a11*a21-a11*a41 + a22*a32 + a32*a42n
            corr4=lims4/(Bstdm*Bstdmt) 

            # find min/max across these options
            corrmin=np.minimum.reduce([corr1,corr2,corr3,corr4])
            corrmax=np.maximum.reduce([corr1,corr2,corr3,corr4])

            # todo mask 

            # check correlation between x_A and x_B,y_B to determine if changes have overshot corr=1/-1.  If so max_corr = +-1 (see Supp Info)
            cc_xA_yB=a11*(a21+a41)/(Astdm*Bstdmt)
            cc_xA_xB=(a11**2+a11*a31)/(Astdm*Bstdm)

            corr1s=((cc_xA_yB)>(cc_xA_xB))&(Acorrs>0)
            corrm1s=((cc_xA_yB)<-(cc_xA_xB))&(Acorrs<0) 

            corrmin[corrm1s]=-1
            corrmax[corr1s]=1

            # fill output using mask
            lims_struct[a]['min'] = A.get_covs()*0 
            lims_struct[a]['max'] = A.get_covs()*0 

            lims_struct[a]['min'][mask]=corrmin
            lims_struct[a]['max'][mask]=corrmax

            lims_struct['additive']['pctls_noerr']=A.get_covs()*0
            lims_struct['additive']['pctls_noerr'][mask]=(Bcorrs> lims_struct['additive']['min'][mask]) != (Bcorrs> lims_struct['additive']['max'][mask])

    # Run above with randomised Monte Carlo inputs for modelling of uncertainty (see paper, Supp. Info 1.1.4)
    if errdist_perms > 0:

        # shape of permuted data distribution
        shp_dist = tuple((errdist_perms,shp[0])) 

        # set up data
        # simulated distributon of underlying covariances
        A_sim_dist=np.zeros(shp_dist)
        B_sim_dist=np.zeros(shp_dist)

        lims_struct['covs']['corrs_raw_A'] = np.zeros(shp_dist)
        lims_struct['covs']['corrs_raw_B'] = np.zeros(shp_dist)
        lims_struct['covs']['covs_raw_A'] = np.zeros(shp_dist) 
        lims_struct['covs']['covs_raw_B'] = np.zeros(shp_dist)

        # distributions of min and max across all 

        lims_struct['uncorrelated']['pctls_raw'] = np.zeros(shp_dist) 

        lims_struct['common']['min_pctls_raw'] = np.zeros(shp_dist)
        lims_struct['common']['max_pctls_raw'] = np.zeros(shp_dist)
        lims_struct['additive']['min_pctls_raw'] = np.zeros(shp_dist)
        lims_struct['additive']['max_pctls_raw'] = np.zeros(shp_dist)

        # generate prior cov matrices
        sims_gen_A=wishart_gen(A)
        sims_gen_B=wishart_gen(B)

        # generate simulated data
        A_sims=sims_gen_A.get_sims(errdist_perms)
        B_sims=sims_gen_B.get_sims(errdist_perms)

        for perm_i in np.arange(errdist_perms):
            # print(perm_i.astype(str))
            A_sim=FC(A_sims[perm_i],cov_flag=True,dof=A.dof)
            B_sim=FC(B_sims[perm_i],cov_flag=True,dof=B.dof)
            out = ASC_lims_all(A_sim,B_sim,errdist_perms=0,pcorrs=pcorrs,dof=dof,ch_type=ch_type,sim_sampling=sim_sampling)

            # uncorrelated
            if np.ndim(A.covs)==3:
                mask_sim=(mask[1],mask[2])
            else:
                mask_sim = mask

            if 'uncorrelated' in ch_type:
                lims_struct['uncorrelated']['pctls_raw'][perm_i,:] = out['uncorrelated']['min'][mask_sim]

                # shared
            if 'covs' in ch_type:
                lims_struct['covs']['corrs_raw_A'][perm_i,:]=A_sim.get_corrs(pcorrs=pcorrs)[mask_sim]
                lims_struct['covs']['corrs_raw_B'][perm_i,:]=B_sim.get_corrs(pcorrs=pcorrs)[mask_sim]
                lims_struct['covs']['corrs_raw_A'][perm_i,:]=A_sim.get_covs(pcorrs=pcorrs)[mask_sim]
                lims_struct['covs']['corrs_raw_B'][perm_i,:]=B_sim.get_covs(pcorrs=pcorrs)[mask_sim]

            if 'common' in ch_type:
                lims_struct['common']['min_pctls_raw'][perm_i,:]= out['common']['min'][mask_sim]
                lims_struct['common']['max_pctls_raw'][perm_i,:]= out['common']['max'][mask_sim]

            # additive
            if 'additive' in ch_type:
                lims_struct['additive']['min_pctls_raw'][perm_i,:]= out['additive']['min'][mask_sim]
                lims_struct['additive']['max_pctls_raw'][perm_i,:]= out['additive']['max'][mask_sim]

        lims_struct=calc_percentiles(A,B,lims_struct,pctl,ch_type=['covs','uncorrelated','common','additive'],pcorrs=False)

        if keepRaw==False:
            for pp in ch_type:
                if pp=='uncorrelated':
                    del lims_struct[pp]['pctls_raw']
                elif pp=='covs':
                    del lims_struct['covs']['corrs_raw_B']
                    del lims_struct['covs']['corrs_raw_A']

                    del lims_struct['covs']['covs_raw_B']
                    del lims_struct['covs']['covs_raw_A']
                else:
                    del lims_struct[pp]['min_pctls_raw']
                    del lims_struct[pp]['max_pctls_raw']

    return lims_struct

def calc_percentiles(A,B,lims_struct,pctl,mask=None,ch_type=['covs','uncorrelated','common','additive'],pcorrs=False):
    """ calculate percentiles for  . """
    if mask is None:
        shp = A.get_covs().shape
        if scipy.sparse.issparse(A.covs):
            mask = A.get_covs().nonzero()
            #todo fix masking
        else:
            mask =  np.triu_indices(shp[-1],k=1) 

            if len(shp) == 3:
                first=np.repeat(np.arange(shp[0]),len(mask[0]))
                second=np.tile(mask[0],shp[0])
                third=np.tile(mask[1],shp[0])
                mask=tuple((first,second,third))

    Acorrs=np.array(A.get_corrs(pcorrs=pcorrs)[mask]).flatten()
    Bcorrs=np.array(B.get_corrs(pcorrs=pcorrs)[mask]).flatten()   
    shp=Bcorrs.shape

    if 'covs' in ch_type:

        raw = lims_struct['covs']['corrs_raw_A'] - lims_struct['covs']['corrs_raw_B'] 
        
        lims_struct['covs']['incl_zeros'] = percentileofscore(raw,0,0)
        
    if 'uncorrelated' in ch_type:
        uncorrelated_lims_err = lims_struct['uncorrelated']['pctls_raw'] 

        pctl_max = np.nanpercentile(uncorrelated_lims_err,100-pctl,0)
        pctl_min = np.nanpercentile(uncorrelated_lims_err,pctl,0)

        lims_struct['uncorrelated']['min_pctls'] = A.get_covs()*0
        lims_struct['uncorrelated']['min_pctls'][mask] = pctl_min
        lims_struct['uncorrelated']['max_pctls'] = A.get_covs()*0
        lims_struct['uncorrelated']['max_pctls'][mask] = pctl_max
        lims_struct['uncorrelated']['pctls']=A.get_covs()*0
        lims_struct['uncorrelated']['pctls'][mask]=(Bcorrs > pctl_min) != (Bcorrs> pctl_max)

    # common
    if 'common' in ch_type:

        corr_min_common_err = lims_struct['common']['min_pctls_raw'] 
        corr_max_common_err = lims_struct['common']['max_pctls_raw']

        pctl_out_max = np.nanpercentile(corr_max_common_err,100-pctl, 0)

        lims_struct['common']['max_pctls'] = A.get_covs()*0
        lims_struct['common']['max_pctls'][mask] = pctl_out_max

        pctl_out_min = np.nanpercentile(corr_min_common_err,pctl, 0)

        lims_struct['common']['min_pctls'] = A.get_covs()*0
        lims_struct['common']['min_pctls'][mask] = pctl_out_min

        lims_struct['common']['pctls'] = A.get_covs()*0
        lims_struct['common']['pctls'][mask] = (Bcorrs> pctl_out_max) != (Bcorrs> pctl_out_min)


    # additive 
    if 'additive' in ch_type:

        corr_min_additive_err =  lims_struct['additive']['min_pctls_raw'] 
        corr_max_additive_err = lims_struct['additive']['max_pctls_raw'] 

        pctl_out_min = np.nanpercentile(corr_min_additive_err,pctl,0)
        lims_struct['additive']['min_pctls'] =  A.get_covs()*0
        lims_struct['additive']['min_pctls'][mask] =  pctl_out_min 
        
        pctl_out_max = np.nanpercentile(corr_max_additive_err,100-pctl,0)
        lims_struct['additive']['max_pctls'] =  A.get_covs()*0
        lims_struct['additive']['max_pctls'][mask] =  pctl_out_max 

        lims_struct['additive']['pctls'] =  A.get_covs()*0
        lims_struct['additive']['pctls'][mask] = (Bcorrs> pctl_out_max) != (Bcorrs> pctl_out_min)

    return(lims_struct)

def abs_sort(x):
    """ sort according to absolute value. """
    if len(x.shape)==1:
        return x[np.argsort(abs(x),0)]
    else:
        out  = np.array(x)[np.argsort(abs(x),0),np.arange(x.shape[1])]
        return out

def corr2pcorr(cc):
    """ convert correlation to partial corr. """
    pinvA=linalg.pinv(cc)
    iis=np.tile(np.atleast_3d(pinvA.diagonal()).T,pinvA.shape[1])
    dd=np.diag(pinvA)

    output=-pinvA/np.sqrt(iis*iis.T)
    output[np.where(np.eye(cc.shape[1]))]=1

    return(output)

class wishart_gen:
    """ class to generate simulated samples for a covariance matrix. """
    def __init__(self,A):
        self.A=A

    def get_sims(self,errdist_perms,recalc=False):
        """ generate simulated wishart samples """
        A=self.A
        
        if not( 'covs_sim' in self.__dict__) or recalc==True:

            covs=A.get_covs()
            dof=A.dof
            # if sparse cov matrix 
            if scipy.sparse.issparse(A.get_covs()):
                
                corrs=A.get_corrs()
                cvs=covs.diagonal()[1:]

                # generate random samples from wishart
                (vars1,vars2,covmat1) = wishart_2(covs[0,0],covs.diagonal()[1:],corrs[0,1:].toarray(),dof,size=errdist_perms)
                covs_sim=[]

                for a in np.arange(vars2.shape[0]):
                    covmat=A.get_covs()*0
                    covmat[0,0]=vars1[a,0]
                    diag_inds=np.diag_indices(covs.shape[-1])
                    covmat[diag_inds[0][1:],diag_inds[1][1:]]=vars2[a,:]
                    covmat[0,1:]=covmat1[a,:]
                    covs_sim.append(covmat)
            else:
                # simulate underlying "true" covariance
                whA=stats.wishart(dof,np.squeeze(covs)[:,:])
                covs_sim=whA.rvs(10*errdist_perms)/(dof)
                covs_sim=list(covs_sim)

                # create  simulated observed covariances from underlying true cov.
                ppA=np.zeros((1,10*errdist_perms))
                whA=[]

                for yb in np.arange(10*errdist_perms):
                    whA.append(stats.wishart(dof,covs_sim[yb]))
                    ppA[0,yb]=whA[-1].pdf(np.squeeze(A.get_covs())*dof)
                   
                # generate sample distribution of covariances
                ppA=ppA/sum(ppA)
                ppA_cul=(np.dot(ppA,np.triu(np.ones(len(ppA.T)))).T)  ## memory issues
                
                rand_els = stats.uniform(0,1).rvs(errdist_perms) 
                els=np.sort(np.searchsorted(ppA_cul.flatten(),rand_els)) 
                covs_sim=[]
                for xb in np.arange(errdist_perms):
                    covs_sim.append(whA[els[xb]].rvs()/dof)
            self.covs_sim=covs_sim

        return(self.covs_sim)

def wishart_2(vars1,vars2,rho,dof,size=1):
    """ Generate wishart from 2 variances and correlation matrix. """
    
    rho=np.atleast_3d(rho).T
    vars1=np.atleast_3d(vars1).T
    vars2=np.atleast_3d(vars2).T

    chis1 = np.atleast_3d(stats.chi2.rvs(dof,size=size))
    chis2 = np.atleast_3d(stats.chi2.rvs(dof-1,size=size))
    norms = np.atleast_3d(stats.norm.rvs(0,1,size=size))

    sqrt_rho = np.sqrt(1-rho**2)

    vars1_sim = vars1*chis1/dof
    covs_sim = np.sqrt(vars1*vars2)*(rho*chis1 + sqrt_rho *(chis1**0.5)*norms)/dof
    vars2_sim = vars2 * ( chis2*(sqrt_rho**2) + (sqrt_rho*norms + rho*(chis1**.5))**2 )/dof
    
    return(vars1_sim.T,vars2_sim.T,covs_sim.T)

def wishart_pdf(cov,samples,dof):
    """ wishart pdf given cov and dof. """
    if scipy.sparse.issparse(cov):
        var1=cov[0,0]
        var2=cov.diagonal()[1:]
        covs1=covs[0,:]
        
        for a in len(var2):
            whA.append(stats.wishart(dof,covsA_sim[b,:,:]))
        ppA[0,yb]=whA[-1].pdf(A.get_covs()[xa,:,:]*dof)

    return(cov) 

def var_gamma_pdf(x,std1,std2,rho,dof):
    """ marginal distribution of covariance. """
    ors = (1-rho**2)
    gamma_d = stats.gamma(dof/2.0)
    first = abs(x)**((dof-1)/2)  / ( gamma_d * np.sqrt(2^(dof-1)*pi*ors*(std1*std2)**n+1) )
    second = special((dof-1)/2.0,abs(x)/(std1*std2*ors))
    third = e**( (rho*x) / std1*std2*ors)

    return(first*second*third)

def pcorr2corr(pcorr):
    """ calculate approx correlation from pcorr. """
    ipcorr=linalg.pinv(pcorr)
    iis=np.tile(np.atleast_3d(ipcorr.diagonal()).T,pcorr.shape[1])
    dd=diag(ipcorr)

    tmp=-ipcorr*np.sqrt(iis*iis.T)
    tmp[where(np.eye(cc.shape[1]))]=dd

    return(tmp)

def rprior(n=1,r=3,M=np.eye(2)):
    """ calculate r prior: n,r,M. """
    out=np.zeros((n,len(M),len(M)))
    Minv=np.la.pinv(M)
    for a in np.arange(n):
        out[a,:,:]= scipy.linalg.cho_solve(scipy.linalg.cho_factor( stats.wishart(r,Minv).rvs()),np.eye(len(Minv)))
    return(out)

def worker(input, output):
    """ worker for pprocessing """
    for func, args in iter(input.get, 'STOP'):
        result = calculate(func, args)
        output.put(result)

def calculate(func, args):
    """ calculator for pprocessing """
    result = func(*args)
    return results

def func_star(a_b):
        """ Convert `f([1,2])` to `f(1,2)` call. """ 
        niceness=os.nice(0)
        os.nice(5-niceness)
        return ASC_lims_all_pool(*a_b)

def runwithvv2(c):
    """ parallelise variance calculation """
    vv1s=range(len(vvs_all))
    args=itertools.izip(itertools.repeat(ccs),itertools.repeat(ccval),itertools.repeat(vvs_all),vv1s,itertools.repeat(vv1s),itertools.repeat(100),itertools.repeat(100))
    pool=Pool()
    out=pool.imap(func_star,args)
    pool.close()
    pool.join()
    return(out,pool)

def flatten_tcs(A,dof='EstEff'):
    """ Concat timecourses, demeaning, estimating dof """
    tcs_dm=pl.demean(A.tcs,2)
    shp=tcs_dm.shape
    out=FC((np.reshape(tcs_dm.swapaxes(0,1),[shp[1],-1])),ROI_info=A.ROI_info)
    # subtract dofs from demeaning 
    out.dof=out.dof-shp[0]
    tcs = out.tcs

    if dof is None:
        self.dof=tcs.shape[-1]-1
    elif dof == 'EstEff':
        AR=np.zeros((tcs.shape[0],tcs.shape[1],15))
        ps=np.zeros((tcs.shape[0],tcs.shape[1],15))
        for subj in np.arange(tcs.shape[0]): 
            for ROI in np.arange(tcs.shape[1]):
                AR[subj,ROI,:]=spectrum.aryule(pl.demean(tcs[subj,ROI,:]),15)[0]
                ps[subj,ROI,:]=spectrum.correlation.CORRELATION(pl.demean(tcs[subj,ROI,:]),maxlags=14,norm='coeff')
        ps = np.mean(np.mean(ps,0),0)
        AR = np.mean(np.mean(AR,0),0)
        dof_nom=tcs.shape[-1]-1
        dof = int(dof_nom / (1-np.dot(ps[:15].T,AR))/(1 - np.dot(np.ones(len(AR)).T,AR)))

    out.dof=dof

    return(out)

def seed_loader(filenames,seed_mask_file,mask_file,subj_inds=None,dof=None):
    """ load data for seed analysis """
    files=[]
    
    seed=nb.load(seed_mask_file)
    seed_mask = seed.get_data()
    seed_points = where(seed_mask)

    mask=nb.load(mask_file)
    mask_data = mask.get_data()
    mask_points = where(mask_data)

    covmat=coo_matrix((len(mask_points[0])+1,len(mask_points[0])+1)).tocsr()

    if type(filenames)==str:
        filenames=[filenames]

    dof_cnt=0

    for file in filenames:
        print(file)

        newfile = nb.load(file)
        files.append(newfile)

        # load 
        data=newfile.get_data()
        seed_data = data[seed_points[0],seed_points[1],seed_points[2],:]
        seed_mean = pl.demean(np.mean(seed_data,0))
        data_mask = data[mask_points[0],mask_points[1],mask_points[2],:]
        
        vars = var(data_mask,1)
        covs = sum(seed_mean*pl.demean(data_mask,1),1)/len(seed_mean)
        # corrs = sum(seed_mean*pl.demean(data_mask,1),1)/(np.std(seed_mean)*np.std(data_mask,1)*len(seed_mean))

        # data: corner, top row, first colomn, diag
        rows=np.r_[0,np.zeros(vars.shape),np.arange(len(vars))+1,np.arange(len(vars))+1]
        cols=np.r_[0,np.arange(len(vars))+1,np.zeros(vars.shape),np.arange(len(vars))+1]
        covmat_data=np.r_[var(seed_mean),covs,covs,vars]
        covmat = covmat + coo_matrix((covmat_data,(rows,cols)),shape=(len(vars)+1,len(vars)+1)).tocsr()
        dof_cnt = dof_cnt+seed_data.shape[-1]-1
         
        # FC_cov = FC(covmat)

        newfile.uncache()

    if not dof:
        dof=dof_cnt

    covmat=covmat/len(filenames)

    out = FC(covmat,cov_flag=True,dof=dof,mask=mask)

    return out

def seed_saver(filename,AB_con,mask=None,save_minmax=False,save_corr=False):
    """ Save seed analysis results """
    # todo test for nb
    
    lims=AB_con.get_ASC_lims()
    plots = list(lims.keys())
    plots.remove('covs') 

    if ( not 'mask' in AB_con.A.__dict__.keys() ):
        if ( not mask ):
            raise ValueError("No mask in contrast object, please provide.")
    else:
        mask = AB_con.A.mask

    if type(mask)==str:
        mask=nb.load(mask)

    mask_data = mask.get_data()
    mask_points = where(mask_data)

    img=set_new_data(mask,mask_data*0)

    for pp in plots: 
        data=img.get_data()*0
        data[where(mask.get_data())] = AB_con.lims[pp]['pctls'][0,1:].toarray()
        img = set_new_data(img,data)
        nb.save(img,filename + '_' + pp + '.nii.gz')
        if save_minmax:
            data=img.get_data()*0
            data[where(mask.get_data())] = AB_con.lims[pp]['min'][0,1:].toarray()
            img = set_new_data(img,data)
            nb.save(img,filename + '_min_' + pp + '.nii.gz')
            if pp != 'uncorrelated':
                data=img.get_data()*0
                data[where(mask.get_data())] = AB_con.lims[pp]['max'][0,1:].toarray()
                img = set_new_data(img,data)
                nb.save(img,filename + '_max_' + pp + '.nii.gz')

    if save_corr:
        data=img.get_data()*0
        data[where(mask.get_data())] = AB_con.A.get_corrs()[0,1:].toarray()
        img = set_new_data(img,data)
        nb.save(img,filename + '_Acorr' + '.nii.gz')
            
        data=img.get_data()*0
        data[where(mask.get_data())] = AB_con.B.get_corrs()[0,1:].toarray()
        img = set_new_data(img,data)
        nb.save(img,filename + '_Bcorr' + '.nii.gz')
 
    return()

class FC_seed:
    """ shell file for FC seed file """
    def __init__(self,filenames,seed_mask,mask,subj_inds):

        for file in filenames:
            
            FC.filelist.append(nb.load(file))


def dr_loader(dir,prefix='dr_stage1',subj_inds=None,ROI_order=None,subjorder=True,dof='EstEff',nosubj=False,read_roi_info=True):
    """ load dual_regression data time courses. 
    
    todo: expand doc
    todo: work with variable durations
    """
    if nosubj:
        dr_files=np.sort(glob.glob(prefix+'*.txt')) 
    else:
        dr_files=np.sort(glob.glob(prefix+'_subject?????.txt')) 
     
    if subj_inds is None:
        subj_inds=np.arange(len(dr_files))
    
    dr_files=dr_files[subj_inds]

    subjs=len(dr_files)
    maskflag=False

    data=np.atleast_3d(np.loadtxt(dr_files[0]).T)

    ROI_info={}

    # load various order specifications from dr_directory
    if read_roi_info:
        if ROI_order is None:
            if os.path.isfile('ROI_order.txt'):
                ROI_order=np.loadtxt('ROI_order.txt').astype(int)-1
            else:
                ROI_order = np.arange((data.shape[0]))

        ROI_info['ROI_order']=ROI_order
     
        if os.path.isfile('ROI_RSNs.txt'):
            ROI_info['ROI_RSNs']=(np.loadtxt('ROI_RSNs.txt').astype(int))
        else: 
            ROI_info['ROI_RSNs'] = np.zeros((data.shape[0],))+1
        
        if os.path.isfile('goodnodes.txt'):
            goodnodes=np.loadtxt('goodnodes.txt').astype(int)
        else:
            goodnodes=ROI_order

        ROI_info['goodnodes']=goodnodes
        ROI_info['ROI_order']=ROI_info['ROI_order'][np.in1d(ROI_order,goodnodes)]
        ROI_info['ROI_RSNs']=ROI_info['ROI_RSNs'][ROI_info['ROI_order']]

        if os.path.isfile('ROI_RSN_include.txt'):
            ROI_info['ROI_RSN_include']=(np.loadtxt('ROI_RSN_include.txt').astype(int))
            ROI_info['ROI_order'] = ROI_info['ROI_order'][np.in1d(ROI_info['ROI_RSNs'],ROI_info['ROI_RSN_include'])]
            ROI_info['ROI_RSNs'] = ROI_info['ROI_RSNs'][np.in1d(ROI_info['ROI_RSNs'],ROI_info['ROI_RSN_include'])]

        if os.path.isfile('ROI_names.txt'):
            with open('ROI_names.txt', 'r') as f:
                ROI_info['ROI_names']=np.array(f.read().splitlines())
        else:
            ROI_info['ROI_names']=np.array([ str(a) for a in np.arange(data.shape[0]) ])

        ROI_info['ROI_names']=ROI_info['ROI_names'][ROI_info['ROI_order']]

        ROI_order=ROI_info['ROI_order']

    for cnt in np.arange(subjs):

            tmpdata=np.atleast_3d(np.loadtxt(dr_files[cnt]).T)
            if cnt==0:
                shp=tmpdata.shape
                if ROI_order is None:
                   ROI_order = np.arange(shp[1])
                   if not ROI_info == {}:
                       ROI_info['ROI_order'] = ROI_order

                datamat=np.zeros((subjs,shp[0],shp[1]))
                mask=np.zeros((subjs,shp[0],shp[1]))

            if tmpdata.shape[1] < shp[1]:
                mask[cnt,:,tmpdata.shape[1]:]=1
                maskflag=True

            datamat[cnt,:,:shp[1]]=tmpdata.squeeze()

    if maskflag:
        datamat=ma.masked_array(datamat,mask)
    
    if not ROI_order is None:
        datamat=datamat[:,ROI_order,:]

    if ( not subjorder ):
        datamat=swapaxes(datamat,0,1)

    A=FC(datamat,dof=dof,ROI_info=ROI_info)

    return A

def dr_saver(A,dir,prefix='dr_stage1',goodnodes=None,aug=0):
    """ save tcs into dual_regression format """
    tcs =  A.tcs
    dof = A.dof

    if goodnodes is None:
        goodnodes = range(tcs.shape[1])

    for subj in np.arange(tcs.shape[0]):
        numb = str(subj+aug)
        np.savetxt(dir+'/'+prefix+'_subject'+numb.zfill(5)+'.txt', np.atleast_3d(tcs[subj,goodnodes,:].T))

def percentileofscore(data,score,axis):
    """ calculate a percentile for given data. """

    data_sort = np.sort(data,axis)
    results = np.argmax((data_sort > score),axis) / np.float(data.shape[axis])

    return(results)


def get_covs(A,B=None):
    """ get covariances from FC. """
    if B is None:
        covs=np.zeros((A.shape[0],A.shape[1],A.shape[1]))

        for a in np.arange(A.shape[0]):
            covs[a,:,:]=np.cov(A[a,:,:])
    else:
        covs = sum(pl.demean(A,2)*pl.demean(B,2),2)/(A.shape[2])

    return covs

def get_corrs(A,B=None,pcorrs=False):
    """ get correlations from FC. """
    
    if B is None:

        covs=get_covs(A)
        stds=A.get_stds()
        stds_m =np.tile(stds,(1,covs.shape[1])).reshape(stds.shape[0],stds.shape[1],stds.shape[1]) 
        stds_m_t = transpose(stds_m,(0,2,1))
        
        corrs=covs/(stds_m*stds_m_t)

    else:
        corrs = sum(pl.demean(A,2)*pl.demean(B,2),2)/(A.shape[2]*np.std(A,2)*np.std(B,2))

    return corrs

def make_psd_tcs(tcs,PSD=False,nreps=1,norm=False,tpts=None):
    """ generate timecourses with given cross PSD """
    if tpts is None:
        tpts = tcs.shape[-1]

    if nreps>1:
        if len(tcs.shape)==3:
            tcs=tcs[0,:,:]
        tcs = np.tile(tcs,(nreps,)+tuple(np.ones(len(tcs.shape))))

    if len(tcs.shape)==1:
        tcs=np.atleast_2d(tcs)
    
    ltcs = tcs.shape[-1]
    rand_tcs=np.zeros(tcs.shape)
    if PSD == False:
        Px=welch(tcs,return_onesided=True,nperseg=tcs.shape[-1]/4.0)

    # sqrt
    # interpolate Px
    interp =  interp1d(Px[0],Px[1])
    tmp=np.arange(tpts)*Px[0][-1]/tpts
    
    iPx=(tmp,interp(tmp))

    Ax = np.sqrt(iPx[1])
    rand_tcs=np.zeros(tcs.shape)

    rnds = np.random.random(Ax.shape)
    rnds_fft = np.fft.fft(rnds/np.std(rnds,-1,keepdims=True))

    Zx = Ax*rnds_fft 
    
    rand_tcs=np.real(np.fft.ifft(Zx))

    if norm:
        rand_tcs = (rand_tcs -np.mean(rand_tcs,-1,keepdims=True) )/ np.std(rand_tcs,-1,keepdims=True)
    else:
        rand_tcs + np.mean(tcs,-1,keepdims=True)

    return np.real(rand_tcs)

def gen_sim_data(tcs,covmat=np.array([]),nreps=-1):
    """ generate simulated data matching PSD and covariance matrix. """
    if tcs.ndim <3:
        tcs=np.rollaxis(np.atleast_3d(tcs),2,0)

    if len(covmat)!=0:
        if tcs.shape[1]==1:
            tcs = np.tile(tcs,(covmat.shape[0],1))

    if covmat.ndim==2:
        covmat=np.tile(covmat,(tcs.shape[0],1,1))
        #covmat = np.atleast_3d(covmat).transpose((2,0,1))

    if tcs.shape[0]==1 & len(tcs.shape)==2:
        gen_tcs =  make_psd_tcs(tcs) 
    else:
        if len(covmat)==0:
            if tcs.shape[-2]>=1:
                covmat=get_covs(tcs)

        rand_tcs=make_psd_tcs(tcs,norm=True,nreps=nreps)

        if nreps>1:
            L=np.linalg.cholesky(covmat)

            gen_tcs=np.zeros(rand_tcs.shape)
            for a in np.arange(nreps):
                gen_tcs[a,:,:]=np.dot(L,rand_tcs[a,:,:]) + np.mean(tcs,(tcs.ndim)-1,keepdims=True)
        else:
            gen_tcs=np.zeros(rand_tcs.shape)
            
            for a in np.arange(covmat.shape[0]):
                L=np.linalg.cholesky(covmat[a,:,:])
                gen_tcs[a,:,:]=np.dot(L,rand_tcs[a,:,:]) + np.mean(tcs[a,:,:],1,keepdims=True)
    return gen_tcs

def inst_cov(tcs):
    ntcs=tcs.shape[0]
    inds = np.triu_indices(ntcs)

    out = np.zeros((ntcs,ntcs,tcs.shape[-1]))

    out[inds[0],inds[1],:] = tcs[inds[0],:] * tcs[inds[1],:]
    out[inds[1],inds[0],:] = tcs[inds[0],:] * tcs[inds[1],:]

    return(out)


def rightShift1(tup, n):
    """ shift tuple elements right. """
    try:
        n = len(tup) - ( n % len(tup))
    except ZeroDivisionError:
        return tuple()
    return tup[n:] + tup[0:n]

def is_numeric(x):
    """ check if input is numeric. """
    try:
        a = 5+x
        return(True)

    except TypeError:

        return(False)

def ASC_lims_all_sim(ccs,vvs,pcorrs=False,errdist=False,errdist_perms=100,dof=None,pctl=10,ch_type='All',sim_sampling=40):
    """ simulate new data todo """
    out=None
    for a in range(len(ccs)):

        ooA=np.ones((2,2))
        ooA[[0,1],[1,0]]=ccs[a]
        ooA[[0,1],[0,1]]=1
        tmpA=FC(ooA,cov_flag=True,dof=600)

        ooB=np.ones((2,2))
        ooB[[0,1],[1,0]]=ccs[a]
        ooB[0,0]=vvs[a,0]
        ooB[1,1]=vvs[a,1]
        tmpB=FC(ooB,cov_flag=True,dof=600)

        out.append(ASC_lims_all(tmpA,tmpB,pcorrs=pcorrs,errdist=errdist,errdist_perms=errdist_perms,dof=dof,pctl=pctl,ch_type=ch_type,sim_sampling=sim_sampling))
    
    return(out)

def ASC_lims_all_pool(ccs,vvs,pcorrs=False,errdist=False,errdist_perms=100,dof=None,pctl=5,ch_type='All',sim_sampling=40,show_pctls=True):
    """ pool ASC for parallel processing. """
    out=None
    for a in range(len(ccs)):

        ooA=np.ones((2,2))
        ooA[[0,1],[1,0]]=ccs[a]
        ooA[[0,1],[0,1]]=1
        tmpA=FC(ooA,cov_flag=True,dof=600)

        ooB=np.ones((2,2))
        ooB[[0,1],[1,0]]=ccs[a]
        ooB[0,0]=vvs[a,0]
        ooB[1,1]=vvs[a,1]
        tmpB=FC(ooB,cov_flag=True,dof=600)

        out.append(ASC_lims_all(tmpA,tmpB,pcorrs=pcorrs,errdist=errdist,errdist_perms=errdist_perms,dof=dof,pctl=pctl,ch_type=ch_type,sim_sampling=sim_sampling,show_pctls=show_pctls))
    
    return(out)

def set_new_data(image, new_data):
    """
    From an image and a numpy array it creates a new image with
    the same header of the image and the numpy array as its data.
    :param image: nibabel image
    :param new_data: numpy array 
    :return: nibabel image
    """
    # see if nifty1
    if image.header['sizeof_hdr'] == 348:
        new_image = nb.Nifti1Image(new_data, image.affine, header=image.header)
    # see if nifty2
    elif image.header['sizeof_hdr'] == 540:
        new_image = nb.Nifti12mage(new_data, image.affine, header=image.header)

def rtoz(x):
    """ calculate z from r. """
    els=x.nonzero()
    out = x*0
    out[els]=(0.5*(np.log(1+x[els]) - np.log(1-x[els])))

    return out


def flattenall(x,nd=2):
    """ flatten first two dims of matrices.  """
    shp=x.shape
    if (len(shp)==2) & (shp[0]==shp[1]):
        nd=1

    firstdim=int(np.product(shp[:-2]))  
    tmp=np.reshape(x,(firstdim,shp[-2],shp[-1]))
    uts=int(0.5*((shp[-1]**2)-shp[-1]))
    outmat=np.zeros((firstdim,uts))

    for aa in np.arange(firstdim):
        if nd==1:
            outmat[(uts*aa):(uts*(aa+1))]=triu_all(tmp[aa,:,:])
        elif len(x.shape)>2:
            outmat[aa,:]= triu_all(tmp[aa,:,:])
        else:
            #  if already flattened
            outmat=x
    return outmat

def triu_all(x):
    """ upper triangular of first two dims. """
    return x.flatten()[pl.find(np.triu(np.ones(x.shape),1))]
