from numpy import *
import os
import numpy as np
import ml_funcs as ml
import numpy.linalg as la
import matplotlib.pylab as pl
import matplotlib.cm as cm
import glob,os, numbers
import nibabel as nb
import scipy.stats
import scipy.sparse
import spectrum
from itertools import combinations, chain
from scipy.misc import comb
from scipy.interpolate import interp1d
from scipy.sparse import coo_matrix

from scipy.signal import welch
from multiprocessing import Process, Queue, current_process, freeze_support

def corr_lims(std_xa,std_xb,std_ya,std_yb,rho_a,rho_b=array([0])):

           cx=cmaxmin(std_xa,std_xb,std_ya,rho_a)[0]
           cx_neg=c_neg_maxmin(std_xa,std_xb,std_ya,rho_a)[0]

           cy=cmaxmin(std_ya,std_yb,std_xa,rho_a)[0]
           cy_neg=c_neg_maxmin(std_ya,std_yb,std_xa,rho_a)[0]

           corr_maxa=( std_xa*std_ya*rho_a*(1+cx*cy) + cx*std_ya**2 + cy * std_xa**2) / (std_xb*std_yb)
           corr_maxa_neg=( std_xa*std_ya*rho_a*(1+cx_neg*cy_neg) + cx_neg*std_ya**2 + cy_neg * std_xa**2) / (std_xb*std_yb)
           corr_min=( std_xa*std_ya*rho_a*(1+cx*cy) + cx*std_ya**2 + cy * std_xa**2) / (std_xb*std_yb)
           return([corr_maxa_neg, corr_maxa],(rho_b> corr_maxa_neg) != (rho_b> corr_maxa))

## def corr_lims_mat(A,B,pcorrs=False):
def corr_lims_mat(A,B,pcorrs=False,errdist=False,errdist_perms=1000,dof=[],pctl=5):

    if dof==[]:
        dof=A.dof
    
    covsA=A.get_covs(pcorrs=pcorrs)
    covsB=B.get_covs(pcorrs=pcorrs)
    
    Astdm=A.get_stds_m(pcorrs=pcorrs)
    Astdmt=A.get_stds_m_t(pcorrs=pcorrs)
    Bstdm=B.get_stds_m(pcorrs=pcorrs)
    Bstdmt=B.get_stds_m_t(pcorrs=pcorrs)
    Acorrs=A.get_corrs(pcorrs=pcorrs)
    Bcorrs=B.get_corrs(pcorrs=pcorrs)

    cx=cmaxmin_m(Astdm,Bstdm,Astdmt,Acorrs)[0]
    cy=cmaxmin_m(Astdmt,Bstdmt,Astdm,Acorrs)[0]

    cx_neg=c_neg_maxmin_m(Astdm,Bstdm,Astdmt,Acorrs)[0]
    cy_neg=c_neg_maxmin_m(Astdmt,Bstdmt,Astdm,Acorrs)[0]

    #corr_maxa=( Astdm*Astdmt*A.corrs*(1+cx*cy) + cx*Astdmt**2 + cy * Astdm**2) / (Bstdm * Bstdmt)
    corr_maxa=( Astdm*Astdmt*A.corrs*(1+cx_neg*cy_neg) + cx_neg*Astdmt**2 + cy_neg * Astdm**2) / (Bstdm * Bstdmt)
    corr_mina=( Astdm*Astdmt*A.corrs*(1+cx*cy) + cx*Astdmt**2 + cy * Astdm**2) / (Bstdm * Bstdmt)

    # (Astds*Astds*Acorrs + wx*Astds*rho_xb + wy*Astds*rho_yb + wx*wy )/(Bstds*Bstds)

    A_sim=[]
    B_sim=[]
    corr_maxa_err=[]
    corr_mina_err=[]
    pctls=[]

    if errdist:

        shp=covsA.shape

        corr_maxa_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        corr_mina_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))

        for a in arange(shp[0]):
            # inv wishart distribution for covariance 
            
            whA=scipy.stats.invwishart(dof,covsA[a,:,:]*(dof-1))
            whB=scipy.stats.invwishart(dof,covsB[a,:,:]*(dof-1))
            
            covsA_sim=whA.rvs(errdist_perms)/(dof)
            covsB_sim=whB.rvs(errdist_perms)/(dof)
            for b in arange(errdist_perms):

                whA=scipy.stats.wishart(dof,covsA_sim[a,:,:])
                whB=scipy.stats.wishart(dof,covsB_sim[a,:,:])

                A_sim=FC(whA.rvs()/dof,cov_flag=True,dofs=dof)
                B_sim=FC(whB.rvs()/dof,cov_flag=True,dofs=dof)

                tmp = corr_lims_mat(A_sim,B_sim,dof=dof) 
                corr_maxa_err[b,a,:,:]=tmp[0]
                corr_mina_err[b,a,:,:]=tmp[1]
        corr_mina_err[abs(corr_mina_err)>1]=sign(corr_mina_err[abs(corr_mina_err)>1]) 
        pctl_out = [nanpercentile(corr_mina_err,pctl,0),nanpercentile(corr_mina_err,100-pctl, 0)]

        corr_maxa_err[abs(corr_maxa_err)>1]=sign(corr_maxa_err[abs(corr_maxa_err)>1]) 
        pctl_out_neg = [nanpercentile(corr_maxa_err,pctl,0),nanpercentile(corr_maxa_err,100-pctl, 0)]
        
        pctls = (Bcorrs> minimum(pctl_out_neg[0] , pctl_out_neg[1])) != (Bcorrs> maximum(pctl_out[0] ,  pctl_out[1]))
        return([corr_mina, corr_maxa],pctls,[corr_mina_err, corr_maxa_err])
    else:
        return(corr_mina, corr_maxa)

    #return([corr_maxa_neg, corr_maxa],[corr_maxa_neg_err, corr_mina_err],[pctl_out_neg,pctl_out])
    #return([corr_maxa_neg, corr_maxa],[corr_maxa_neg_err, corr_mina_err],[pctl_out_neg,pctl_out])

    #return([corr_maxa_neg, corr_maxa], pctls,[corr_maxa_neg_err, corr_mina_err])

def cmaxmin(std_xa,std_xb,std_ya,rho_a):
           tmp=nan_to_num(sqrt((std_xa*rho_a)**2 - (std_xa**2-std_xb**2)))
           #return(array([(-std_xa*rho_a-abs(tmp))/std_ya,(-std_xa*rho_a+abs(tmp))/std_ya]))

           #tmp=nan_to_num(sqrt((std_xa*rho_a)**2 - (std_xa**2-std_xb**2)))
           return(abs_sort(array([(std_xa*rho_a+tmp)/std_ya,(std_xa*rho_a-tmp)/std_ya])))

def cmaxmin_m(std_xa,std_xb,std_ya,rho_a):

           tmp=nan_to_num(sqrt((std_xa*rho_a)**2 - (std_xa**2-std_xb**2)))

           data=array([(std_xa*rho_a+tmp)/std_ya ,(std_xa*rho_a-tmp)/std_ya])

           inds=abs(data[0,:])>abs(data[1,:])
        
           out=data[0,:,:,:]
           out[inds]=data[1,inds]

           return(out)

def c_neg_maxmin_m(std_xa,std_xb,std_ya,rho_a):

           tmp=nan_to_num(sqrt((std_xa*rho_a)**2 - (std_xa**2-std_xb**2)))

           data=array([-(std_xa*rho_a+tmp)/std_ya ,-(std_xa*rho_a-tmp)/std_ya])

           inds=abs(data[0,:])>abs(data[1,:])
        
           out=data[0,:,:,:]
           out[inds]=data[1,inds]
           return(out)

def c_neg_maxmin(std_xa,std_xb,std_ya,rho_a):
           tmp=nan_to_num(sqrt((std_xa*rho_a)**2 - (std_xa**2-std_xb**2)))
           # return(array([(std_xa*rho_a-abs(tmp)/std_ya,(std_xa*rho_a+abs(tmp))/std_ya)]))

           return(abs_sort(array([-(std_xa*rho_a+tmp)/std_ya,-(std_xa*rho_a-tmp)/std_ya])))

# general connectivity matrix class
class FC:
    def __init__(self,tcs,cov_flag=False,dof=[],ROI_info=[]):
    
        if cov_flag==True:
            self.tcs=[]
            #if tcs.ndim==2:
            #    tcs=(atleast_3d(tcs).transpose(2,0,1))
            covs=tcs
            if dof == []:
                raise ValueError("Need to specify dof if providing cov.")
            else:
                self.dof=dof
  
        else:
            if tcs.ndim==2:
                tcs=(atleast_3d(tcs).transpose(2,0,1))
            
            self.tcs = tcs
            if dof == []:
                self.dof=tcs.shape[-1]-1
            elif dof == 'EstEff':
                AR=zeros((tcs.shape[0],tcs.shape[1],15))
                ps=zeros((tcs.shape[0],tcs.shape[1],15))
                for subj in arange(tcs.shape[0]): 
                    for ROI in arange(tcs.shape[1]):
                        AR[subj,ROI,:]=spectrum.aryule(pl.demean(tcs[subj,ROI,:]),15)[0]
                        ps[subj,ROI,:]=spectrum.correlation.CORRELATION(pl.demean(tcs[subj,ROI,:]),maxlags=14,norm='coeff')
                ps = np.mean(mean(ps,0),0)
                AR2 = np.mean(mean(AR,0),0)
                dof_nom=tcs.shape[-1]-1
                self.dof = int(dof_nom / (1-np.dot(ps[:15].T,AR2))/(1 - np.dot(ones(len(AR2)).T,AR2)))

            else:
                self.dof=dof

            covs = get_covs(tcs)

        self.ROI_info = ROI_info

        self.covs = covs

    def get_stds(self,pcorrs=False):

        if not( 'stds' in self.__dict__):
            sz = self.get_covs(pcorrs=pcorrs).shape
            self.stds=diagonal(self.get_covs(pcorrs=pcorrs),axis1=len(sz)-2,axis2=len(sz)-1)**(0.5) 
        return(self.stds)

    def get_stds_m(self,pcorrs=False):

        stds_m = self.get_stds(pcorrs=pcorrs)
        return(tile(stds_m,(1,self.covs.shape[1])).reshape(stds_m.shape[0],stds_m.shape[1],stds_m.shape[1]))

    def get_stds_m_t(self,pcorrs=False):

        return transpose(self.get_stds_m(pcorrs=pcorrs),(0,2,1))

    def get_corrs(self,pcorrs=False):

        if pcorrs:
            return(self.get_pcorrs())
        else:
            if not( 'corrs' in self.__dict__):
                self.corrs = self.get_covs()/(self.get_stds_m()*self.get_stds_m_t())
            return(self.corrs)
    
    def get_pcorrs(self):

        if not( 'pcorrs' in self.__dict__):
            if not( 'corrs' in self.__dict__):
                self.corrs = (self.get_covs())/(self.get_stds_m()*self.get_stds_m_t())
            
            self.pcorrs=zeros(self.corrs.shape)
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
            pcorrs=self.get_pcorrs()

        multiplier = (self.get_stds_m()*transpose(self.get_stds_m(),(0,2,1)))
        
        return(pcorrs*multiplier)


class FC_con:
    def __init__(self,A,B):     
        self.A=A
        self.B=B

    def get_common(self,errdist=False):
        if not( 'common' in self.__dict__):
            self.common=corr_lims_common_mat(self.A,self.B,errdist=errdist)
        
        return(self.common)

    def get_unshared(self,errdist=False):
        if not( 'unshared' in self.__dict__):
            self.unshared=corr_lims_unshared_mat(self.A,self.B,errdist=errdist)

        return(self.unshared)

    def get_lims(self,errdist=False):
        if not( 'lims' in self.__dict__):
            self.lims=corr_lims_all(self.A,self.B,errdist=errdist)

        return(self.lims)

    def unshared_prob(self):
        if not( 'unshared' in self.__dict__):
            self.unshared=corr_lims_unshared_mat(self.A,self.B,errdist=errdist)

    def get_corr_stats(self,pcorrs=False): 
        if pcorrs:
            out = self.get_pcorr_stats(self)
        else:
            if not( 'corr_stats' in self.__dict__):
                self.corr_stats = scipy.stats.ttest_rel(ml.rtoz(self.A.get_corrs(pcorrs=False)),ml.rtoz(self.B.get_corrs(pcorrs=False)))
            out = self.corr_stats
        
        return out

    def get_pcorr_stats(self): 
        if not( 'pcorr_stats' in self.__dict__):
               self.pcorr_stats=scipy.stats.ttest_rel(ml.rtoz(self.A.get_corrs(pcorrs=pcorrs)),ml.rtoz(self.B.get_corrs(pcorrs=pcorrs)))

        return self.pcorr_stats

    
    def get_std_stats(self,pcorrs=False): 
        if not( 'std_stats' in self.__dict__):
            self.std_stats = scipy.stats.ttest_rel(self.A.get_stds(pcorrs=pcorrs),self.B.get_stds(pcorrs=pcorrs))[0]

        return self.std_stats

    def get_lims(self,pcorrs=False,pctl=0.05,errdist_perms=50,refresh=False):
        
        if not( 'lims' in self.__dict__) or refresh:
            if self.A.tcs == []: 
                A=FC(np.mean(self.A.get_covs(pcorrs=pcorrs),0),cov_flag=True, dof=self.A.dof,ROI_info=self.A.ROI_info)
                B=FC(np.mean(self.B.get_covs(pcorrs=pcorrs),0),cov_flag=True, dof=self.B.dof,ROI_info=self.B.ROI_info)
            else:
                A=flatten_tcs(self.A)
                B=flatten_tcs(self.B)
            self.lims=corr_lims_all(A,B,errdist_perms=errdist_perms,pctl=pctl,pcorrs=pcorrs)
        return(self.lims)


class FC2:
    def __init__(self,tcs,cov_flag=False,dof=[],ROI_info=[]):
    
        if cov_flag==True:
            self.tcs=[]
            if tcs.ndim==2:
                tcs=(atleast_3d(tcs).transpose(2,0,1))
            covs=tcs
            if dof == []:
                raise ValueError("Need to specify dof if providing cov.")
            else:
                self.dof=dof
  
        else:
            if tcs.ndim==2:
                tcs=(atleast_3d(tcs).transpose(2,0,1))
            
            self.tcs = tcs
            if dof == []:
                self.dof=tcs.shape[-1]-1
            elif dof == 'EstEff':
                AR=zeros((tcs.shape[0],tcs.shape[1],15))
                ps=zeros((tcs.shape[0],tcs.shape[1],15))
                for subj in arange(tcs.shape[0]): 
                    for ROI in arange(tcs.shape[1]):
                        AR[subj,ROI,:]=spectrum.aryule(pl.demean(tcs[subj,ROI,:]),15)[0]
                        ps[subj,ROI,:]=spectrum.correlation.CORRELATION(pl.demean(tcs[subj,ROI,:]),maxlags=14,norm='coeff')
                ps = np.mean(mean(ps,0),0)
                AR2 = np.mean(mean(AR,0),0)
                dof_nom=tcs.shape[-1]-1
                self.dof = int(dof_nom / (1-np.dot(ps[:15].T,AR2))/(1 - np.dot(ones(len(AR2)).T,AR2)))

            else:
                self.dof=dof

            covs = get_covs(tcs)

        self.ROI_info = ROI_info

        self.covs = covs

    def get_stds(self,pcorrs=False):

        if not( 'stds' in self.__dict__):
            self.stds=diagonal(self.get_covs(pcorrs=pcorrs),axis1=1,axis2=2)**(0.5) 
        return(self.stds)

    def get_stds_m(self,pcorrs=False):

        stds_m = self.get_stds(pcorrs=pcorrs)
        return(tile(stds_m,(1,self.covs.shape[1])).reshape(stds_m.shape[0],stds_m.shape[1],stds_m.shape[1]))

    def get_stds_m_t(self,pcorrs=False):

        return transpose(self.get_stds_m(pcorrs=pcorrs),(0,2,1))

    def get_corrs(self,pcorrs=False):

        if pcorrs:
            return(self.get_pcorrs())
        else:
            if not( 'corrs' in self.__dict__):
                self.corrs = self.get_covs()/(self.get_stds_m()*self.get_stds_m_t())
            return(self.corrs)
    
    def get_pcorrs(self):

        if not( 'pcorrs' in self.__dict__):
            if not( 'corrs' in self.__dict__):
                self.corrs = (self.get_covs())/(self.get_stds_m()*self.get_stds_m_t())
            
            self.pcorrs=zeros(self.corrs.shape)
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
            pcorrs=self.get_pcorrs()

        multiplier = (self.get_stds_m()*transpose(self.get_stds_m(),(0,2,1)))
        
        return(pcorrs*multiplier)




# calculate noise related change 
def corr_lims_unshared(std_x,std_xb,std_y,std_yb,rho_xy):

    corr_unshared= array((std_x*std_y*rho_xy) / (std_xb*std_yb))
    
    corr_unshared[abs(corr_unshared) > 1]=sign(corr_unshared[abs(corr_unshared) > 1])

    return corr_unshared

def corr_lims_unshared_mat(A,B,pcorrs=False,errdist=False,errdist_perms=1000,dof=[],pctl=5):

    if dof==[]:
        dof=A.dof

    covsA=A.get_covs()
    covsB=B.get_covs()
    Bcorrs=B.get_corrs()   
    Bstds=B.get_stds_m()
    Bstds_t=B.get_stds_m_t()

    unshared = covsA / (Bstds*Bstds_t)
    unshared[unshared>1]=1 
    unshared[abs(unshared)>1]=sign(unshared[abs(unshared)>1]) 
    # unshared = calc_rho_unshared(A,B)
    # corr_unshared[abs(corr_unshared) > 1]=sign(corr_unshared[abs(corr_unshared) > 1])

    A_sim=[]
    B_sim=[]
    
    shp=covsA.shape

    covsA_sim=[]

    if errdist:
        unshared_lims_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        ppA=[]

        corr_maxa_neg_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        corr_maxa_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))

        for a in arange(shp[0]):
           
             # generate init samples
            whA=scipy.stats.wishart(dof,covsA[a,:,:])
            whB=scipy.stats.wishart(dof,covsB[a,:,:])
            covsA_sim=whA.rvs(errdist_perms)/(dof)
            covsB_sim=whB.rvs(errdist_perms)/(dof)

            # inv wishart distribution for covariance 
            # whA=scipy.stats.invwishart(dof,covsA[a,:,:]*(dof))
            # whB=scipy.stats.invwishart(dof,covsB[a,:,:]*(dof)) 
            # covsA_sim=whA.rvs(errdist_perms)
            # covsB_sim=whB.rvs(errdist_perms)

            # generate all samples 
            
            for b in arange(errdist_perms):

                whA=scipy.stats.wishart(dof,covsA_sim[b,:,:])
                whB=scipy.stats.wishart(dof,covsB_sim[b,:,:])
                A_sim=FC(whA.rvs()/dof,cov_flag=True,dof=dof)
                B_sim=FC(whB.rvs()/dof,cov_flag=True,dof=dof)

                unshared_lims_err[b,a,:,:] = corr_lims_unshared_mat(A_sim,B_sim,dof=dof)

        unshared_lims_err[abs(unshared_lims_err)>1]=sign(unshared_lims_err[abs(unshared_lims_err)>1]) 
        pctl_out = [nanpercentile(unshared_lims_err,pctl,0),nanpercentile(unshared_lims_err,100-pctl,0)]
        pctls=(Bcorrs> pctl_out[0]) != (Bcorrs> pctl_out[1])
        return(unshared, pctls ,unshared_lims_err,covsA_sim)
    else:
        return(unshared)

def corr_lims_all(A,B,pcorrs=False,errdist_perms=0,dof=[],pctl=10,chType='All',sim_sampling=40):

    if dof==[]:
        dof=A.dof

    if chType == 'All':
        chType = ['covs','unshared','common','combined']
    elif type(chType)==str:
        chType=[chType]

    covsA=A.get_covs(pcorrs=pcorrs)
    covsB=B.get_covs(pcorrs=pcorrs)

    Acorrs=A.get_corrs(pcorrs=pcorrs)   
    Astds=A.get_stds_m(pcorrs=pcorrs)

    Bcorrs=B.get_corrs(pcorrs=pcorrs)   
    Bstds=B.get_stds_m(pcorrs=pcorrs)
    
    shp=A.covs.shape
    Astdm=A.get_stds_m(pcorrs=pcorrs)
    Astdmt=A.get_stds_m_t(pcorrs=pcorrs)
    Bstdm=B.get_stds_m(pcorrs=pcorrs)
    Bstdmt=B.get_stds_m_t(pcorrs=pcorrs)

    A_sim=[]
    B_sim=[]
    #corr_maxa_err=[]
    #corr_mina_err=[]
    pctls=[]
    
    lims_struct={}

    for a in chType:
        
        lims_struct[a]={}

        if a=='unshared':

            lims_struct[a]={}
            unshared = covsA / (Bstdm*Bstdmt)
            unshared[unshared>1]=1 
            unshared[abs(unshared)>1]=sign(unshared[abs(unshared)>1])
        
            lims_struct[a]['min'] = unshared
            lims_struct[a]['max'] = unshared
            sdf

        elif a== 'common':

            inds = tile(arange(sim_sampling),(shp[0],shp[1],shp[1],1))

            #corr2=zeros((shp[0],shp[1],shp[2],sim_sampling,sim_sampling))
            corr2Common=zeros((shp[0],shp[1],shp[2],sim_sampling,sim_sampling))
            corr2Common[:]=nan
            aa=(arange(sim_sampling)/(sim_sampling-1.0))

            for aaa in arange(len(aa)):
                rho_xb=aa[aaa]
                Acorrs_abs = abs(Acorrs)
                (rho_yb_l,rho_yb_u)=calc_pbv(Acorrs,rho_xb)
                bb= (arange(sim_sampling)/(sim_sampling-1.0)) #/(rho_yb_u-rho_yb_l) + rho_yb_l
                # loop over range of possible Y corrs
                tmp = calc_weight(Astdm,Bstdm,rho_xb)
                inds_0=[(abs(tmp[0])<abs(tmp[1]))]
                inds_1=[(abs(tmp[0])>=abs(tmp[1]))]
                wx=zeros(tmp[0].shape)
                wx[:]=nan
                wx[inds_0]=tmp[0][inds_0]
                wx[inds_1]=tmp[1][inds_1]
                
                for bbb in arange(len(bb)):

                    rho_yb=sign(Acorrs_abs)*bb[bbb]
                    tmp = calc_weight(Astdmt,Bstdmt,rho_yb)
                    inds_0=[(abs(tmp[0])<abs(tmp[1]))]
                    inds_1=[(abs(tmp[0])>=abs(tmp[1]))]
                    wy=zeros(tmp[0].shape)
                    wy[:]=nan
                    wy[inds_0]=tmp[0][inds_0]
                    wy[inds_1]=tmp[1][inds_1]
                    corr2Common[:,:,:,aaa,bbb] = (Astdm*Astdmt*Acorrs_abs + wx*Astdm*rho_xb + wy*Astdmt*rho_yb + wx*wy )/(Bstdm*Bstdmt)
                    # prevent negative weights
                    corr2Common[:,:,:,aaa,bbb][(sign(Bstdm-Astdm)*wx)<0]=nan
                    corr2Common[:,:,:,aaa,bbb][(sign(Bstdmt-Astdmt)*wy)<0]=nan
                    # prevent negative inital shared components
                    #Aorr2Common[:,:,:,aaa,:][sign(covsA)*==-1]=nan
                    #corr2Common[:,:,:,aaa,:][transpose(sign(covsA)*sign(covsB),(0,2,1))==-1]=nan
                inds_u=np.maximum(0,floor((rho_yb_u)*sim_sampling).astype(int))
                inds_u=tile(inds_u,(sim_sampling,1,1,1)).transpose(1,2,3,0)
                # remove els out correlation range
                corr2Common[:,:,:,aaa,:][inds_u<inds]=nan
                # remove els with negative weights
                inds_l=np.maximum(0,floor((rho_yb_l)*sim_sampling).astype(int))
                #inds_l=np.maximum(len(aa)/2,floor((rho_yb_l+1)/2.0*sim_sampling).astype(int))
                inds_l=tile(inds_l,(sim_sampling,1,1,1)).transpose(1,2,3,0)
                corr2Common[:,:,:,aaa,:][inds_l>inds]=nan
            Acorr_tile = transpose(tile(Acorrs,(len(aa),len(bb),1,1,1)),[2,3,4,0,1])
            #corr2[corr2==0]=nan
            #corr2[sign(corr2)!=sign(Acorrs)
            #corr2[corr2<-1]=-1
            #corr2[corr2>1]=1
            #corr2Common[:,:,:,pl.find(sign(Acorrs)!=sign(aa)),:]=nan
            #corr2Common[:,:,:,:,pl.find(sign(Acorrs)!=sign(bb))]=nan
            # remove corr sign flips

            corr2Common=corr2Common*sign(Acorr_tile)
            corr2Common[sign(Acorr_tile)!=sign(corr2Common)]=nan
            corr2Common[corr2Common==0]=nan
            corr2Common[corr2Common<-1]=nan
            corr2Common[corr2Common>1]=nan
            #Acorr_tile[corr2Common>=0]=-1
            #corr2Common[corr2Common<Acorr_tile]=nan 


            #tmp[sign(tmp)!=sign(Acorrs)]=0
            lims_struct[a]['min']=nanmin(nanmin(corr2Common,4),3) 
            #tmp[sign(tmp)!=sign(Acorrs)]=0
            lims_struct[a]['max']=nanmax(nanmax(corr2Common,4),3) 
            lims_struct[a]['corr2Common']=nanmax(nanmax(corr2Common,4),3) 
            #lims_struct[a]['min']corrminCommon=nanmin(nanmin(corr2Common,4),3) 
            #lims_struct[a]['max']corrmaxCommon=nanmax(nanmax(corr2Common,4),3)

        elif a == 'combined':

            cx=cmaxmin_m(Astdm,Bstdm,Astdmt,Acorrs)[0]
            cy=cmaxmin_m(Astdmt,Bstdmt,Astdm,Acorrs)[0]

            cx_neg=c_neg_maxmin_m(Astdm,Bstdm,Astdmt,Acorrs)[0]
            cy_neg=c_neg_maxmin_m(Astdmt,Bstdmt,Astdm,Acorrs)[0]

            limsa=( Astdm*Astdmt*A.corrs*(1+cx_neg*cy_neg) + cx_neg*Astdmt**2 + cy_neg * Astdm**2) / (Bstdm * Bstdmt)
            limsb=( Astdm*Astdmt*A.corrs*(1+cx*cy) + cx*Astdmt**2 + cy * Astdm**2) / (Bstdm * Bstdmt)

            lims_struct[a]['min']=np.minimum(limsa,limsb)
            lims_struct[a]['max']=np.maximum(limsa,limsb)
            lims_struct[a]['max'][lims_struct[a]['max']>1]=1
            lims_struct[a]['min'][lims_struct[a]['min']<-1]=-1
    
    if errdist_perms > 0:

        unshared_lims_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        A_sim_dist=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        B_sim_dist=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        corr_err_A=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        corr_err_B=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        covs_err_A=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        covs_err_B=zeros((errdist_perms,shp[0],shp[1],shp[1]))

        # corr_mina_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        corr_min_common_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        corr_max_common_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        corr_min_Combined_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        corr_max_Combined_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))
     
        for xa in arange(shp[0]):
            # inv wishart distribution for covariance 
            
            #whA=scipy.stats.invwishart(dof,covsA[a,:,:]*(dof-1))
            #whB=scipy.stats.invwishart(dof,covsB[a,:,:]*(dof-1))


            # generate initial cov matrices
            
            whA=scipy.stats.wishart(dof,A.get_covs()[xa,:,:])
            whB=scipy.stats.wishart(dof,B.get_covs()[xa,:,:])

            covsA_sim=whA.rvs(100*errdist_perms)/(dof)
            covsB_sim=whB.rvs(100*errdist_perms)/(dof)

            ppA=zeros((1,100*errdist_perms))
            ppB=zeros((1,100*errdist_perms))
            
            whA=[]
            whB=[]

            for yb in arange(100*errdist_perms):
                whA.append(scipy.stats.wishart(dof,covsA_sim[yb,:,:]))
                ppA[0,yb]=whA[-1].pdf(covsA[xa,:,:]*dof)
                #ppA[0,yb]=1
                whB.append(scipy.stats.wishart(dof,covsB_sim[yb,:,:]))
                ppB[0,yb]=whB[-1].pdf(covsB[xa,:,:]*dof)
                #ppB[0,yb]=1
               
            # generate sample
            ppA=ppA/sum(ppA)
            ppB=ppB/sum(ppB)
            ppA_cul=(dot(ppA,triu(ones(len(ppA.T)))).T) 
            ppB_cul=(dot(ppB,triu(ones(len(ppB.T)))).T) 
            rand_els = scipy.stats.uniform(0,1).rvs(errdist_perms) 
            els=sort(searchsorted(ppA_cul.flatten(),rand_els)) 

            # return ppA,ppA_cul,whA

            for xb in arange(errdist_perms):

                A_sim=FC(whA[els[xb]].rvs()/dof,cov_flag=True,dof=A.dof)
                B_sim=FC(whB[els[xb]].rvs()/dof,cov_flag=True,dof=B.dof)
                out = corr_lims_all(A_sim,B_sim,errdist_perms=0,pcorrs=pcorrs,dof=dof,chType=chType,sim_sampling=sim_sampling)

                # unshared
                if 'unshared' in chType:
                    unshared_lims_err[xb,xa,:,:] = out['unshared']['min']

                    # shared
                if 'covs' in chType:
                    corr_err_B[xb,xa,:,:]=B_sim.get_corrs(pcorrs=pcorrs)
                    corr_err_A[xb,xa,:,:]=A_sim.get_corrs(pcorrs=pcorrs)
                    covs_err_B[xb,xa,:,:]=B_sim.get_covs(pcorrs=pcorrs)
                    covs_err_A[xb,xa,:,:]=A_sim.get_covs(pcorrs=pcorrs)

                if 'common' in chType:
                    corr_min_common_err[xb,xa,:,:]= out['common']['min']
                    corr_max_common_err[xb,xa,:,:]= out['common']['max']

                # combined
                if 'combined' in chType:
                    corr_min_Combined_err[xb,xa,:,:]= out['combined']['min']
                    corr_max_Combined_err[xb,xa,:,:]= out['combined']['max']

                #tmp = corr_lims_mat(A_sim,B_sim,dof=dof) 
                #corr_maxa_err[xb,xa,:,:]=tmp[0].squeeze()
                #corr_mina_err[xb,xa,:,:]=tmp[1].squeeze()

        if 'covs' in chType:
            #lims_struct['cov']
            
            lims_struct['covs']['corrs_raw_A'] = corr_err_A
            lims_struct['covs']['corrs_raw_B'] = corr_err_B
            lims_struct['covs']['covs_raw_A'] = covs_err_A
            lims_struct['covs']['covs_raw_B'] = covs_err_B

            raw = corr_err_A-corr_err_B
            
            lims_struct['covs']['incl_zeros'] = percentileofscore(raw,0,0)
            #  lims_struct['correlation']['pctls']=(Bcorrs> pctl_min) != (Bcorrs> pctl_max)
            
        if 'unshared' in chType:
            unshared_lims_err[abs(unshared_lims_err)>1]=sign(unshared_lims_err[abs(unshared_lims_err)>1]) 
            #lims_struct['unshared']['err']=unshared_lims_err

            pctl_max = nanpercentile(unshared_lims_err,100-pctl,0)
            pctl_min = nanpercentile(unshared_lims_err,pctl,0)

            lims_struct['unshared']['pctls_raw'] = unshared_lims_err

            lims_struct['unshared']['min_pctls'] = pctl_min
            lims_struct['unshared']['max_pctls'] = pctl_max
            lims_struct['unshared']['pctls']=(Bcorrs> pctl_min) != (Bcorrs> pctl_max)


        # common
        if 'common' in chType:
            corr_max_common_err[abs(corr_max_common_err)>1]=sign(corr_max_common_err[abs(corr_max_common_err)>1]) 
            pctl_out_max = nanpercentile(corr_max_common_err,100-pctl, 0)
            lims_struct['common']['max_pctls'] = pctl_out_max

            corr_min_common_err[abs(corr_min_common_err)>1]=sign(corr_min_common_err[abs(corr_min_common_err)>1]) 

            pctl_out_min = nanpercentile(corr_min_common_err,pctl, 0)
            lims_struct['common']['min_pctls'] = pctl_out_min
            lims_struct['common']['ccmax'] = corr_max_common_err
            lims_struct['common']['ccmin'] = corr_min_common_err

            lims_struct['common']['pctls'] = (Bcorrs> pctl_out_max) != (Bcorrs> pctl_out_min)

            lims_struct['common']['min_pctls_raw'] = corr_min_common_err
            lims_struct['common']['max_pctls_raw'] = corr_max_common_err

        # combined 

        if 'combined' in chType:
            #corr_mina_err[abs(corr_min_Combined_err)>1]=sign(corr_min_Combined_err[abs(corr_min_Combined_err)>1]) 
            #lims_struct['combined']['min_err'] = corr_min_Combined_err

            #pctl_out = [percentile(corr_min_Combined_err,pctl,0),percentile(corr_min_Combined_err,100-pctl, 0)]
            pctl_out_min = nanpercentile(corr_min_Combined_err,pctl,0)
            lims_struct['combined']['min_pctls'] =  pctl_out_min 
            
            #corr_maxa_err[abs(corr_max_Combined_err)>1]=sign(corr_max_Combined_err[abs(corr_max_Combined_err)>1]) 
            #pctl_out_neg = [percentile(corr_maxa_err,pctl,0),percentile(corr_maxa_err,100-pctl, 0)]
            pctl_out_max = nanpercentile(corr_max_Combined_err,100-pctl,0)
            lims_struct['combined']['max_pctls'] =  pctl_out_max 

            #lims_struct['combined']['pctls'] = (Bcorrs> minimum(pctl_out_max[0] , pctl_out_max[1])) != (Bcorrs> maximum(pctl_out_min[0] ,  pctl_out_min[1]))
            lims_struct['combined']['pctls'] = (Bcorrs> pctl_out_max) != (Bcorrs> pctl_out_min)

            lims_struct['combined']['min_pctls_raw'] = corr_min_Combined_err
            lims_struct['combined']['max_pctls_raw'] = corr_max_Combined_err

    return lims_struct

# calculate amount of signal with correlation rho_xb to initial signal produces variance change from std_x to std_xb
def calc_weight(std_x,std_xb,rho_xb):
    tmp = sqrt((std_x*rho_xb)**2 + std_xb**2 -std_x**2)
    return(-std_x*rho_xb + tmp,-std_x*rho_xb - tmp)

# calculate max,min correlation of rho_bc given rho_ab, rho_ac
def calc_pbv(rho_xy,rho_xb):
    min_cc =  (rho_xy*rho_xb) - sqrt(1 - rho_xy**2)*sqrt(1 - rho_xb**2)
    max_cc =  (rho_xy*rho_xb) + sqrt(1 - rho_xy**2)*sqrt(1 - rho_xb**2)
    #tmp = sqrt((rho_xy*rho_xb)**2 + 1 - rho_xy**2 -rho_xb**2)
    return(min_cc,max_cc)


def cmaxmin_m(std_xa,std_xb,std_ya,rho_a):

           tmp=nan_to_num(sqrt((std_xa*rho_a)**2 - (std_xa**2-std_xb**2)))

           data=array([(std_xa*rho_a+tmp)/std_ya ,(std_xa*rho_a-tmp)/std_ya])

           inds=abs(data[0,:])>abs(data[1,:])
        
           out=data[0,:,:,:]
           out[inds]=data[1,inds]

           return(out)





# calculate rho_xyb given variances and init correlation rho_xy
def calc_rho_xyb(std_x,std_xb,std_y,std_yb,rho_xy,rho_xb):
    rho_yb = calc_pbv(rho_xy,rho_xb)[0]
    wx = calc_weight(std_x,std_xb,rho_xb)[0] 
    wy = calc_weight(std_y,std_yb,rho_yb)[0] 
    rho_xyb = (std_x*std_y*rho_xy + wx*std_x*rho_xb + wy*std_y*rho_yb + wx*wy )/(std_xb*std_yb)
    return(rho_xyb)

def abs_sort(x):
    if len(x.shape)==1:
        return x[argsort(abs(x),0)]
    else:
        out  = array(x)[argsort(abs(x),0),arange(x.shape[1])]
        return out

# calc common max range of change in signal
def  corr_lims_common(std_x,std_xb,std_y,std_yb,rho_xy,sim_sampling=40):
    corr2=zeros((sim_sampling,sim_sampling))
    aa=(arange(-1,1.000,2.0/sim_sampling))

    for a in arange(sim_sampling/2.0)+sim_sampling/2.0:
        rho_xb=aa[a]
        (rho_yb_l,rho_yb_u)=calc_pbv(rho_xy,rho_xb)
        for rho_yb in arange(max([floor(rho_yb_l*sim_sampling/2.0)*2.0/sim_sampling,0]),rho_yb_u-0.0001,2.0/sim_sampling):
            b=rho_yb*sim_sampling/2.0+sim_sampling/2.0
            wy = calc_weight(std_y,std_yb,rho_yb)[0]
            wx = calc_weight(std_x,std_xb,rho_xb)[0]
            corr2[a,b] = (std_x*std_y*rho_xy + wx*std_x*rho_xb + wy*std_y*rho_yb + wx*wy )/(std_xb*std_yb)
    tmp = arange(max([floor(rho_yb_l*sim_sampling/2.0)*2.0/sim_sampling,0]),rho_yb_u-0.0001,2.0/sim_sampling)
    #return(corr2,tmp)
    return(nanmin(corr2[corr2!=0]),nanmax(corr2[corr2!=0]))

def corr_lims_common_mat(A,B,pcorrs=False,errdist=False,errdist_perms=300,dof=None,pctl=5,sim_sampling=101):

    if dof==None:
        dof=A.dof
    
    # calculate limits if change in A,B due to same source
    
    shp=A.covs.shape
    corr2=zeros((shp[0],shp[1],shp[2],sim_sampling,sim_sampling))
    corr2Common=zeros((shp[0],shp[1],shp[2],sim_sampling,sim_sampling))
    aa=(arange(sim_sampling)/(sim_sampling-1.0))

    covsA=A.get_covs(pcorrs=pcorrs)
    covsB=B.get_covs(pcorrs=pcorrs)

    Acorrs=A.get_corrs(pcorrs=pcorrs)[0,0,1]
    Astds=A.get_stds_m(pcorrs=pcorrs)
    Astds_t=A.get_stds_m_t(pcorrs=pcorrs)

    Bcorrs=B.get_corrs(pcorrs=pcorrs)[0,0,1]
    Bstds=B.get_stds_m(pcorrs=pcorrs)
    Bstds_t=B.get_stds_m_t(pcorrs=pcorrs)

    inds = tile(arange(sim_sampling),(shp[0],shp[1],shp[1],1))
        
    # loop over range of rho_xb common 
    for aaa in arange(len(aa)):
        rho_xb=aa[aaa]
        (rho_yb_l,rho_yb_u)=calc_pbv(Acorrs,rho_xb)
        bb= (arange(sim_sampling)/(sim_sampling-1.0)) #/(rho_yb_u-rho_yb_l) + rho_yb_l
        tmp=[] 
        # loop over range of possible Y corrs 

        for bbb in arange(len(bb)):
            rho_yb=bb[bbb]
            tmp = calc_weight(Astds,Bstds,sign(Acorrs)*rho_xb)
            inds_0=[(abs(tmp[0])<abs(tmp[1]))]
            inds_1=[(abs(tmp[0])>=abs(tmp[1]))]
            wx=zeros(tmp[0].shape)
            wx[inds_0]=tmp[0][inds_0]
            wx[inds_1]=tmp[1][inds_1]
            
            wy = calc_weight(Astds,Bstds,rho_yb)[0]
            tmp = calc_weight(Astds,Bstds,sign(Acorrs)*rho_xb)
            inds_0=[(abs(tmp[0])<abs(tmp[1]))]
            inds_1=[(abs(tmp[0])>=abs(tmp[1]))]
            wy=zeros(tmp[0].shape)
            wy[inds_0]=tmp[0][inds_0]
            wy[inds_1]=tmp[1][inds_1]
            corr2[:,:,:,aaa,bbb] = (Astds*Astds*Acorrs + wx*Astds*rho_xb + wy*Astds*rho_yb + wx*wy )/(Bstds*Bstds)

        inds_u=(floor(rho_yb_u*sim_sampling)).astype(int)
        inds_u=tile(inds_u,(sim_sampling,1,1,1)).transpose(1,2,3,0)

        corr2Common=corr2.copy() 
        corr2Common[:,:,:,aaa,:][inds_u<inds]=nan
        
        inds_l=(floor(rho_yb_l*sim_sampling)).astype(int)
        inds_l=tile(inds_l,(sim_sampling,1,1,1)).transpose(1,2,3,0)
        corr2Common[:,:,:,aaa,:][inds_l>inds]=nan
    
    corr2[corr2==0]=nan
    corr2[corr2<-1]=-1
    corr2[corr2>1]=1

    corr2Common[corr2Common==0]=nan
    corr2Common[corr2Common<-1]=-1
    corr2Common[corr2Common>1]=1


    corrmincommon=nanmin(nanmin(corr2,4),3) 
    corrmaxcommon=nanmax(nanmax(corr2,4),3)

    corrminCommon=nanmin(nanmin(corr2Common,4),3) 
    corrmaxCommon=nanmax(nanmax(corr2Common,4),3)

    A_sim=[]
    B_sim=[]
 
    pctlscommon=[]
    corr_min_common_err=[]
    corr_max_common_err=[]

    pctlsCommon=[]
    corr_min_Common_err=[]
    corr_max_Common_err=[]

    if errdist:
        shp=covsA.shape

        corr_min_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        corr_max_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))

        corr_min_common_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        corr_max_common_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        
        corr_min_Common_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        corr_max_Common_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))

        for a in arange(shp[0]):
            # generate init samples
            whA=scipy.stats.wishart(dof,covsA[a,:,:])
            whB=scipy.stats.wishart(dof,covsB[a,:,:])
            covsA_sim=whA.rvs(errdist_perms)/(dof)
            covsB_sim=whB.rvs(errdist_perms)/(dof)

            ppA=zeros((1,errdist_perms))
            ppB=zeros((1,errdist_perms))
            
            whA=[]
            whB=[]
            # generate all montecarlo samples 
  
            for b in arange(errdist_perms):

                whA.append(scipy.stats.wishart(dof,covsA_sim[b,:,:]))
                ppA[0,b]=whA[-1].pdf(covsA[a,:,:]*dof)
                whB.append(scipy.stats.wishart(dof,covsB_sim[b,:,:]))
                ppB[0,b]=whB[-1].pdf(covsB[a,:,:]*dof)
           
            ppA=ppA/sum(ppA)
            ppB=ppB/sum(ppB)

            # select var
            
            ppA_cul=(dot(ppA,triu(ones(len(ppA.T)))).T) 
            ppB_cul=(dot(ppB,triu(ones(len(ppB.T)))).T) 
            rand_els = scipy.stats.uniform(0,1).rvs(errdist_perms) 
            els=sort(searchsorted(ppA_cul.flatten(),rand_els)) 
            for b in arange(errdist_perms):

                A_sim=FC(whA[els[b]].rvs()/dof,cov_flag=True,dof=A.dof)
                B_sim=FC(whB[els[b]].rvs()/dof,cov_flag=True,dof=B.dof)

                tmp = corr_lims_common_mat(A_sim,B_sim,dof=dof,sim_sampling=40.0) 
                corr_min_common_err[b,a,:,:]=tmp[0][0]
                corr_max_common_err[b,a,:,:]=tmp[0][1]
                corr_min_Common_err[b,a,:,:]=tmp[1][0]
                corr_max_Common_err[b,a,:,:]=tmp[1][1]

        corr_max_common_err[abs(corr_max_common_err)>1]=sign(corr_max_common_err[abs(corr_max_common_err)>1]) 
        pctl_out_max = [nanpercentile(corr_max_common_err,pctl,0),nanpercentile(corr_max_common_err,100-pctl, 0)]

        corr_min_common_err[abs(corr_min_common_err)>1]=sign(corr_min_common_err[abs(corr_min_common_err)>1]) 
        pctl_out_min = [nanpercentile(corr_min_common_err,pctl,0),nanpercentile(corr_min_common_err,100-pctl, 0)]
        #pctls_common = (Bcorrs> minimum(pctl_out_min[0] , pctl_out_min[1])) != (Bcorrs> maximum(pctl_out[0] ,  pctl_out[1]))


        pctl_out_max = [nanpercentile(corr_max_Common_err,pctl,0),nanpercentile(corr_max_Common_err,100-pctl, 0)]

        pctl_out_min = [nanpercentile(corr_min_Common_err,pctl,0),nanpercentile(corr_min_Common_err,100-pctl, 0)]
        #pctls_common = (Bcorrs> minimum(pctl_out_min[0] , pctl_out_min[1])) != (Bcorrs> maximum(pctl_out[0] ,  pctl_out[1]))

        return([[corrmincommon,corrmaxcommon],pctls_common,[corr_min_common_err, corr_max_common_err]],[[corrminCommon,corrmaxCommon],pctls_Common,[corr_min_Common_err, corr_max_Common_err]])
    else:
        return([[corrmincommon,corrmaxcommon],[corrminCommon,corrmaxCommon]])

def plot_class(A,B,thresh=1.96):
   
    corr_lims,corr_lims_TF = corr_lims_mat(A,B)
    common,common_TF = corr_lims_common_mat(A,B)
    unshared,unshared_r, unshared_TF = corr_lims_unshared_mat(A,B)

    corrsA=A.get_corrs()
    corrsB=B.get_corrs()

    #ttests=stats.ttest_rel(ccsB,ccsA)[0]
    
    return corr_unshared

def corr2pcorr(cc):

    pinvA=linalg.pinv(cc)
    iis=tile(atleast_2d(pinvA.diagonal()).T,pinvA.shape[1])
    dd=diag(pinvA)

    tmp=-pinvA/sqrt(iis*iis.T)
    tmp[where(eye(cc.shape[1]))]=1

    return(tmp)

def pcorr2corr(pcorr):
    ipcorr=linalg.pinv(pcorr)
    iis=tile(atleast_2d(ipcorr.diagonal()).T,pcorr.shape[1])
    dd=diag(ipcorr)

    tmp=-ipcorr*sqrt(iis*iis.T)
    tmp[where(eye(cc.shape[1]))]=dd

    return(tmp)

def rprior(n=1,r=3,M=eye(2)):
    out=zeros((n,len(M),len(M)))
    Minv=la.pinv(M)
    for a in arange(n):
        out[a,:,:]= scipy.linalg.cho_solve(scipy.linalg.cho_factor( scipy.stats.wishart(r,Minv).rvs()),eye(len(Minv)))
    return(out)

def calc_hists(ccs,cc,vvs_all,vv1,vv2,errdist_perms,dof):
    # calculate histograms of unshared, common, and mixed effects.
    # out_hist_unshared=zeros((len(ccs),len(vvs_all),len(vvs_all),len(ccs)))
    ooA=ones((2,2))
    ooB=ones((2,2))
        
    ooA[[0,1],[1,0]]=ccs[cc]
    ooA[[0,1],[0,1]]=1
    tmpA=FC(ooA,cov_flag=True)

    if isinstance(vv2,numbers.Number):
        vv2=[vv2]
    elif vv2==[]:
        vvs=vvs_all
    out_hist_unshared=zeros((len(vv2),len(ccs)))
    out_hist_common_l=zeros((len(vv2),len(ccs)))
    out_hist_common_u=zeros((len(vv2),len(ccs)))
    out_hist_lim_l=zeros((len(vv2),len(ccs)))
    out_hist_lim_u=zeros((len(vv2),len(ccs)))
    #print('vv1:'+str(vv1))

    for vv2_cnt in range(len(vv2)):
        ooB[[0],[0]]=vvs_all[vv1]
        ooB[[1],[1]]=vvs_all[vv2[vv2_cnt]]
        ooB[[0,1],[1,0]]=ccs[cc]*(vvs_all[vv1]**.5)*(vvs_all[vv2[vv2_cnt]]**.5)

        tmpB=FC(ooB,cov_flag=True)
        
        out_unshared=corr_lims_unshared_mat(tmpA,tmpB,errdist=True,errdist_perms=errdist_perms,pctl=5,dof=dof)
        out_hist_unshared[vv2_cnt,:]=np.histogram(out_unshared[2][:,0,1,0],normed=True,range=[0,1],bins=r_[ccs,[1]])[0]
        out_common=corr_lims_common_mat(tmpA,tmpB,errdist=True,errdist_perms=errdist_perms,pctl=5,dof=dof)
        out_hist_common_l[vv2_cnt,:]=np.histogram(out_common[2][0][:,0,1,0],normed=True,range=[0,1],bins=r_[ccs,[1]])[0]
        out_hist_common_u[vv2_cnt,:]=np.histogram(out_common[2][1][:,0,1,0],normed=True,range=[0,1],bins=r_[ccs,[1]])[0]

        out_lims=corr_lims_mat(tmpA,tmpB,errdist=True,errdist_perms=errdist_perms,pctl=5,dof=dof)

        out_hist_lim_l[vv2_cnt,:]=np.histogram(out_lims[2][0][:,0,1,0],normed=True,range=[0,1],bins=r_[ccs,[1]])[0]
        out_hist_lim_u[vv2_cnt,:]=np.histogram(out_lims[2][1][:,0,1,0],normed=True,range=[0,1],bins=r_[ccs,[1]])[0]
        
    return(array(out_hist_unshared),array(out_hist_common_l),array(out_hist_common_u),array(out_hist_lim_l),array(out_hist_lim_u))


def worker(input, output):
    for func, args in iter(input.get, 'STOP'):
        result = calculate(func, args)
        output.put(result)

def calculate(func, args):
    result = func(*args)
    return results

def func_star(a_b):
        #Convert `f([1,2])` to `f(1,2)` call 
        niceness=os.nice(0)
        os.nice(5-niceness)
        return corr_lims_all_pool(*a_b)

def runwithvv2(c):
    vv1s=range(len(vvs_all))
    args=itertools.izip(itertools.repeat(ccs),itertools.repeat(ccval),itertools.repeat(vvs_all),vv1s,itertools.repeat(vv1s),itertools.repeat(100),itertools.repeat(100))
    pool=Pool()
    out=pool.imap(func_star,args)
    pool.close()
    pool.join()
    return(out,pool)

def plot_fb(ccs,low,high,vvs=None,cmap=cm.gray,colorbar=True):
   
    fig=pl.gcf()
    pl.plot(ccs,ccs,color=[0.55,0.55,0.55],label='Test',linewidth=5)

    for a in arange(len(low)):
        srted=np.sort(c_[low[a],high[a]])
        pl.fill_between(ccs,srted[:,0],srted[:,1],color=cmap(a*256/len(low)))

    ax=pl.gca()

    if vvs != None:
        cc=ax.pcolor(array([[.2]]),cmap=cmap,visible=False,vmin=min(vvs),vmax=max(vvs))      
        if colorbar:
            cbar=pl.colorbar(cc,shrink=0.5,ticks=vvs,fraction=0.1)
            cbar.set_label('Proportional change in variance \n in region one in second condition.')

    ax.set_ylim(-1,1)
    
    ax.set_xlabel('Correlation in condition 1')
    ax.set_ylabel('Correlation in condition 2')

    # ax.set_title('Change in correlation after adding/removing common signal')

def flatten_tcs(A,dof='EstEff'):
    tcs_dm=pl.demean(A.tcs,2)
    shp=tcs_dm.shape
    out=FC(reshape(tcs_dm.swapaxes(0,1),[shp[1],-1]),ROI_info=A.ROI_info)
    # subtract dofs from demeaning 
    out.dof=out.dof-shp[0]
    tcs = out.tcs

    if dof == []:
        self.dof=tcs.shape[-1]-1
    elif dof == 'EstEff':
        AR=zeros((tcs.shape[0],tcs.shape[1],15))
        ps=zeros((tcs.shape[0],tcs.shape[1],15))
        for subj in arange(tcs.shape[0]): 
            for ROI in arange(tcs.shape[1]):
                AR[subj,ROI,:]=spectrum.aryule(pl.demean(tcs[subj,ROI,:]),15)[0]
                ps[subj,ROI,:]=spectrum.correlation.CORRELATION(pl.demean(tcs[subj,ROI,:]),maxlags=14,norm='coeff')
        ps = np.mean(mean(ps,0),0)
        AR = np.mean(mean(AR,0),0)
        dof_nom=tcs.shape[-1]-1
        dof = int(dof_nom / (1-np.dot(ps[:15].T,AR))/(1 - np.dot(ones(len(AR)).T,AR)))

    out.dof=dof

    return(out)

def seed_loader(filenames,seed_mask_file,mask_file,subj_ids=[],dof=300):

    # load data
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

    for file in filenames:

        newfile = nb.load(file)
        files.append(newfile)

        # load 

        data=newfile.get_data()
        seed_data = data[seed_points[0],seed_points[1],seed_points[2],:]
        seed_mean = pl.demean(mean(seed_data,0))
        data_mask = data[mask_points[0],mask_points[1],mask_points[2],:]
        
        vars = var(data_mask,1)
        covs = sum(seed_mean*pl.demean(data_mask,1),1)/len(seed_mean)
        # corrs = sum(seed_mean*pl.demean(data_mask,1),1)/(std(seed_mean)*std(data_mask,1)*len(seed_mean))

        # data: corner, top row, first colomn, diag
        rows=np.r_[0,zeros(vars.shape),arange(len(vars))+1,arange(len(vars))+1]
        cols=np.r_[0,arange(len(vars))+1,zeros(vars.shape),arange(len(vars))+1]
        covmat_data=np.r_[var(seed_mean),covs,covs,vars]
        covmat = covmat + coo_matrix((covmat_data,(rows,cols)),shape=(len(vars)+1,len(vars)+1)).tocsr()

        # FC_cov = FC(covmat

        newfile.uncache()
    out = FC(covmat,cov_flag=True,dof=dof)
    return out
    
class FC_seed:
    def __init__(self,filenames,seed_mask,mask,subj_ids):

        for file in filenames:
            
            FC.filelist.append(nb.load(file))



def dr_loader(dir,prefix='dr_stage1',subj_ids=[],ROI_order=[],subjorder=True,dof='EstEff',nosubj=False,read_roi_info=True):
    
    if nosubj:
        dr_files=sort(glob.glob(prefix+'*.txt')) 
    else:
        dr_files=sort(glob.glob(prefix+'_subject?????.txt')) 
        
    if len(subj_ids)==0:
        subj_ids=arange(len(dr_files))
    
    dr_files=dr_files[subj_ids]

    subjs=len(dr_files)
    maskflag=False

    data=atleast_2d(loadtxt(dr_files[0]).T)

    ROI_info={}

    if read_roi_info:

        if ROI_order == []:
            if os.path.isfile('ROI_order.txt'):
                ROI_order=np.loadtxt('ROI_order.txt').astype(int)-1
        ROI_info['ROI_order']=ROI_order
     
        if os.path.isfile('ROI_RSNs.txt'):
            ROI_info['ROI_RSNs']=(np.loadtxt('ROI_RSNs.txt').astype(int))
        else: 
            ROI_info['ROI_RSNs'] = []
        
        if os.path.isfile('goodnodes.txt'):
            ROI_info['goodnodes']=np.loadtxt('goodnodes.txt').astype(int)
            ROI_info['ROI_RSNs']=ROI_info['ROI_RSNs'][np.in1d(ROI_order,goodnodes)]
            ROI_info['ROI_order']=ROI_info['ROI_order'][np.in1d(ROI_order,goodnodes)]
        elif ROI_info['ROI_RSNs'] != []:
            ROI_info['goodnodes]']=None
            ROI_info['ROI_RSNs']=ROI_info['ROI_RSNs'][ROI_info['ROI_order']]

        if ROI_info['ROI_order'] == []:
            ROI_info['ROI_order'] = arange(len(ROI_info['ROI_RSNs']))

        if os.path.isfile('ROI_RSN_include.txt'):
            ROI_info['ROI_RSN_include']=(np.loadtxt('ROI_RSN_include.txt').astype(int))


            ROI_info['ROI_order'] = ROI_info['ROI_order'][np.in1d(ROI_info['ROI_RSNs'],ROI_info['ROI_RSN_include'])]
            ROI_info['ROI_RSNs'] = ROI_info['ROI_RSNs'][np.in1d(ROI_info['ROI_RSNs'],ROI_info['ROI_RSN_include'])]

        if os.path.isfile('ROI_names.txt'):
            ROI_info['ROI_names']=np.array(open('ROI_names.txt').read().splitlines())
        else:
            ROI_info['ROI_names']=np.array([ str(a) for a in np.arange(A_orig.get_covs(pcorrs=pcorrs).shape[-1]) ])

        # n_nodes=len(ROI_info[ROI_names)

        ROI_info['ROI_names']=ROI_info['ROI_names'][ROI_info['ROI_order']]


    for cnt in arange(subjs):

            tmpdata=atleast_2d(loadtxt(dr_files[cnt]).T)
            if cnt==0:
                shp=tmpdata.shape
                if ROI_order is []:
                   ROI_order = np.arange(shp[1])
                   if not ROI_info == {}:
                       ROI_info['ROI_order'] = ROI_order

                datamat=zeros((subjs,shp[0],shp[1]))
                mask=zeros((subjs,shp[0],shp[1]))

            if tmpdata.shape[1] < shp[1]:
                mask[cnt,:,tmpdata.shape[1]:]=1
                maskflag=True

            datamat[cnt,:,:shp[1]]=tmpdata

    if maskflag:
        datamat=ma.masked_array(datamat,mask)
    
    if not ROI_order is []:
        datamat=datamat[:,ROI_order,:]

    if ( not subjorder ):
        datamat=swapaxes(datamat,0,1)

    A=FC(datamat,dof=dof,ROI_info=ROI_info)

    return A

def dr_saver(A,dir,prefix='dr_stage1',goodnodes=[],aug=0):
   tcs =  A.tcs
   dof = A.dof

   if goodnodes == []:
       goodnodes = range(tcs.shape[1])


   for subj in arange(tcs.shape[0]):
        numb = str(subj+aug)
        savetxt(dir+'/'+prefix+'_subject'+numb.zfill(5)+'.txt', atleast_2d(tcs[subj,goodnodes,:].T))

def percentileofscore(data,score,axis):

    data_sort = np.sort(data,axis)
    results = np.argmax((data_sort > score),axis) / np.float(data.shape[axis])

    return(results)


def get_covs(A,B=None):

    if B is None:

        covs=zeros((A.shape[0],A.shape[1],A.shape[1]))
        
        for a in arange(A.shape[0]):
            covs[a,:,:]=cov(A[a,:,:])
    else:
        covs = sum(pl.demean(A,2)*pl.demean(B,2),2)/(A.shape[2])

    return covs

def get_corrs(A,B=None,pcorrs=False):
    
    if B is None:

        covs=get_covs(A)
        stds=diagonal(covs,axis1=1,axis2=2)**(0.5) 
        stds_m =tile(stds,(1,covs.shape[1])).reshape(stds.shape[0],stds.shape[1],stds.shape[1]) 
        stds_m_t = transpose(stds_m,(0,2,1))
        
        corrs=covs/(stds_m*stds_m_t)

    else:
        corrs = sum(pl.demean(A,2)*pl.demean(B,2),2)/(A.shape[2]*std(A,2)*std(B,2))

    return corrs


def comb_index(n, k):
    count = comb(n, k, exact=True)
    index = np.fromiter(chain.from_iterable(combinations(range(n), k)), 
                        int, count=count*k)
    return index.reshape(-1, k)

def make_psd_tcs(tcs,PSD=False,nreps=1,norm=False,tpts=[]):
    
    if tpts == []:
        tpts = tcs.shape[-1]

    if nreps>1:
        if len(tcs.shape)==3:
            tcs=tcs[0,:,:]
        tcs = tile(tcs,(nreps,)+tuple(ones(len(tcs.shape))))

    if len(tcs.shape)==1:
        tcs=np.atleast_2d(tcs)
    
    ltcs = tcs.shape[-1]
    rand_tcs=zeros(tcs.shape)
    if PSD == False:
        Px=welch(tcs,return_onesided=True,nperseg=tcs.shape[-1]/4.0)

    # sqrt
    # interpolate Px
    interp =  interp1d(Px[0],Px[1])
    tmp=arange(tpts)*Px[0][-1]/tpts
    
    iPx=(tmp,interp(tmp))

    #tI=tile(arange(tpts)*tr/(2*tpts),(tcs.shape[0],tcs.shape[1],1))
    #interp(tI,tPx[0],Px[1])

    Ax = np.sqrt(iPx[1])
    rand_tcs=zeros(tcs.shape)

    rnds = np.random.random(Ax.shape)
    rnds_fft = np.fft.fft(rnds/std(rnds,-1,keepdims=True))

    # Zx=zeros(Ax.shape)
    Zx = Ax*rnds_fft 
    
    #Ph=np.random.randn(Ax.shape[0])*360
    #Zx = Ax*(e**(Ph*1j))
    #ZZx=zeros(tcs.shape).astype(complex)
    #ZZx[:,:,-(Zx.shape[2]+1):-1]=Zx
#    for aa in arange(tcs.shape[0]):
#        for bb in arange(tcs.shape[1]):
#            iPx[1][a,b] = interp(Px
#            llx=len(Px[0])
#            init=aa*llx
#            endd=min(aa*llx+llx,tcs.shape[2])
#            rand_tcs[:,:,aa*llx:(aa*llx+min(aa*llx+llx,tcs.shape[2]))]=np.fft.ifft(Zx)[:,:,:(endd-(aa*llx))]
    rand_tcs=np.fft.ifft(Zx)

    if norm:
        rand_tcs = (rand_tcs -mean(rand_tcs,-1,keepdims=True) )/ std(rand_tcs,-1,keepdims=True)
    else:
        rand_tcs + mean(tcs,-1,keepdims=True)

    return real(rand_tcs)

def gen_sim_data(tcs,covmat=array([]),nreps=-1):

    if tcs.ndim <3:
        tcs=rollaxis(atleast_3d(tcs),2,0)

    if len(covmat)!=0:
        if tcs.shape[1]==1:
            tcs = tile(tcs,(covmat.shape[0],1))

    if covmat.ndim==2:
        covmat=tile(covmat,(tcs.shape[0],1,1))
        #covmat = atleast_3d(covmat).transpose((2,0,1))

    if tcs.shape[0]==1 & len(tcs.shape)==2:
        gen_tcs =  make_psd_tcs(tcs) #+  mean(tcs,-1,keepdims=True)
    else:
        if len(covmat)==0:
            if tcs.shape[-2]>=1:
                covmat=get_covs(tcs)

        rand_tcs=make_psd_tcs(tcs,norm=True,nreps=nreps)

        if nreps>1:
            L=np.linalg.cholesky(covmat)

            gen_tcs=zeros(rand_tcs.shape)
            for a in arange(nreps):
                gen_tcs[a,:,:]=np.dot(L,rand_tcs[a,:,:]) + mean(tcs,1,keepdims=True)
        else:
            gen_tcs=zeros(rand_tcs.shape)
            
            for a in arange(covmat.shape[0]):

                L=np.linalg.cholesky(covmat[a,:,:])
                gen_tcs[a,:,:]=np.dot(L,rand_tcs[a,:,:]) + mean(tcs[a,:,:],1,keepdims=True)
    return gen_tcs

def rightShift1(tup, n):
    try:
        n = len(tup) - ( n % len(tup))
    except ZeroDivisionError:
        return tuple()
    return tup[n:] + tup[0:n]

def is_numeric(x):
    try:
        a = 5+x
        return(True)

    except TypeError:

        return(False)

def corr_lims_all_sim(ccs,vvs,pcorrs=False,errdist=False,errdist_perms=100,dof=[],pctl=5,chType='All',sim_sampling=40):

    out=[]
    for a in range(len(ccs)):

        ooA=ones((2,2))
        ooA[[0,1],[1,0]]=ccs[a]
        ooA[[0,1],[0,1]]=1
        tmpA=FC(ooA,cov_flag=True,dof=600)

        ooB=ones((2,2))
        ooB[[0,1],[1,0]]=ccs[a]
        ooB[0,0]=vvs[a,0]
        ooB[1,1]=vvs[a,1]
        tmpB=FC(ooB,cov_flag=True,dof=600)

        out.append(corr_lims_all(tmpA,tmpB,pcorrs=pcorrs,errdist=errdist,errdist_perms=errdist_perms,dof=dof,pctl=pctl,chType=chType,sim_sampling=sim_sampling))
    
    return(out)

def corr_lims_all_pool(ccs,vvs,pcorrs=False,errdist=False,errdist_perms=100,dof=[],pctl=5,chType='All',sim_sampling=40,show_pctls=True):

    out=[]
    for a in range(len(ccs)):

        ooA=ones((2,2))
        ooA[[0,1],[1,0]]=ccs[a]
        ooA[[0,1],[0,1]]=1
        tmpA=FC(ooA,cov_flag=True,dof=600)

        ooB=ones((2,2))
        ooB[[0,1],[1,0]]=ccs[a]
        ooB[0,0]=vvs[a,0]
        ooB[1,1]=vvs[a,1]
        tmpB=FC(ooB,cov_flag=True,dof=600)

        out.append(corr_lims_all(tmpA,tmpB,pcorrs=pcorrs,errdist=errdist,errdist_perms=errdist_perms,dof=dof,pctl=pctl,chType=chType,sim_sampling=sim_sampling,show_pctls=show_pctls))
    
    return(out)
 
