from numpy import *
import os
import numpy as np
import numpy.linalg as la
import matplotlib.pylab as pl
import matplotlib.cm as cm
import glob,os, numbers
import nibabel as nb
import scipy.stats as stats
import scipy.sparse
import spectrum
from itertools import combinations, chain
from scipy.misc import comb
from scipy.interpolate import interp1d
import collections
import scipy.sparse as sparse
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
def corr_lims_mat(A,B,pcorrs=False,errdist=False,errdist_perms=1000,dof=None,pctl=5):

    if dof is None:
        dof=A.dof
    
    covsA=A.get_covs(pcorrs=pcorrs)
    covsB=B.get_covs(pcorrs=pcorrs)
    
    Astdm=A.get_stds_m(pcorrs=pcorrs)
    Astdmt=A.get_stds_m_t(pcorrs=pcorrs)
    Bstdm=B.get_stds_m(pcorrs=pcorrs)
    Bstdmt=B.get_stds_m_t(pcorrs=pcorrs)
    Acorrs=A.get_corrs(pcorrs=pcorrs)
    Bcorrs=B.get_corrs(pcorrs=pcorrs)

    cx=cmaxmin(Astdm,Bstdm,Astdmt,Acorrs)[0]
    cy=cmaxmin(Astdmt,Bstdmt,Astdm,Acorrs)[0]

    cx_neg=c_neg_maxmin(Astdm,Bstdm,Astdmt,Acorrs)[0]
    cy_neg=c_neg_maxmin(Astdmt,Bstdmt,Astdm,Acorrs)[0]

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
            
            whA=stats.invwishart(dof,covsA[a,:,:]*(dof-1))
            whB=stats.invwishart(dof,covsB[a,:,:]*(dof-1))
            
            covsA_sim=whA.rvs(errdist_perms)/(dof)
            covsB_sim=whB.rvs(errdist_perms)/(dof)
            for b in arange(errdist_perms):

                whA=stats.wishart(dof,covsA_sim[a,:,:])
                whB=stats.wishart(dof,covsB_sim[a,:,:])

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

    tmp = nan_to_num(sqrt((std_xa*rho_a)**2 - (std_xa**2-std_xb**2)))
    #return(array([(-std_xa*rho_a-abs(tmp))/std_ya,(-std_xa*rho_a+abs(tmp))/std_ya]))
    #tmp=nan_to_num(sqrt((std_xa*rho_a)**2 - (std_xa**2-std_xb**2)))
    if len(std_xa.shape)==1: 
        out = abs_sort(array([(std_xa*rho_a+tmp)/std_ya,(std_xa*rho_a-tmp)/std_ya]))
    else:
        data=array([(std_xa*rho_a+tmp)/std_ya ,(std_xa*rho_a-tmp)/std_ya])

        inds=abs(data[0,:])>abs(data[1:wq,:])

        out=data[0,:,:,:]
        out[inds]=data[1,inds]

    return(out)

def c_neg_maxmin(std_xa,std_xb,std_ya,rho_a):

    tmp=nan_to_num(sqrt((std_xa*rho_a)**2 - (std_xa**2-std_xb**2)))

    if len(std_xa.shape)==1: 

        out=abs_sort(array([-(std_xa*rho_a+tmp)/std_ya,-(std_xa*rho_a-tmp)/std_ya]))
    else:
        data=array([-(std_xa*rho_a+tmp)/std_ya ,-(std_xa*rho_a-tmp)/std_ya])

        inds=abs(data[0,:])>abs(data[1,:])

        out=data[0,:]
        out[inds]=data[1,inds]

    return(out)

# return(array([(std_xa*rho_a-abs(tmp)/std_ya,(std_xa*rho_a+abs(tmp))/std_ya)]))
# general connectivity matrix class
class FC:
    def __init__(self,tcs,cov_flag=False,dof=None,ROI_info={},mask=None):
    
        if cov_flag==True:
            self.tcs=None
            #if tcs.ndim==2:
            #    tcs=(atleast_3d(tcs).transpose(2,0,1))
            covs=tcs
            if dof is None:
                raise ValueError("Need to specify dof if providing cov.")
            else:
                self.dof=dof
  
        else:
            if tcs.ndim==2:
                tcs=(atleast_3d(tcs).transpose(2,0,1))
            
            self.tcs = tcs
            if dof is None:
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

        if ROI_info=={}:
            shp=covs.shape[-1]
            ROI_info['ROI_names']=arange(shp).astype(str)
            ROI_info['ROI_RSNs']=ones(shp,).astype(int)

        if mask:
            self.mask=mask

        self.ROI_info = ROI_info

        self.covs = covs

    def get_stds(self,pcorrs=False):

        if not( 'stds' in self.__dict__):
            sz = self.get_covs(pcorrs=pcorrs).shape
            if len(sz)==3:
                self.stds=diagonal(self.get_covs(pcorrs=pcorrs),axis1=len(sz)-2,axis2=len(sz)-1)**(0.5) 
            else:
                self.stds=self.get_covs(pcorrs=pcorrs).diagonal()**0.5
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
                # initialise as normal/sparse matrix
                els=self.get_covs().nonzero()
                self.corrs=self.get_covs()*0
                dims=arange(len(els))
                dims1 = tuple(setdiff1d(dims,dims[-1]))
                dims2 = tuple(setdiff1d(dims,dims[-2]))

                els1=tuple(els[index] for index in dims1)
                els2=tuple(els[index] for index in dims2)

                self.corrs[els] = array(self.get_covs()[els])/(self.get_stds()[els1]*self.get_stds()[els2])
            return(self.corrs)
    
    def get_pcorrs(self):

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
            pcorrs=self.get_pcorrs()

        multiplier = (self.get_stds_m()*transpose(self.get_stds_m(),(0,2,1)))
        
        return(pcorrs*multiplier)


class FC_con:
    def __init__(self,A,B):     
        self.A=A
        self.B=B

    #def get_lims(self,errdist_perms=False):
    #    if not( 'lims' in self.__dict__):
    #        self.lims=corr_lims_all(self.A,self.B,errdist=errdist_perms)
    #    return(self.lims)

    def get_corr_stats(self,pcorrs=False): 
        if pcorrs:
            out = self.get_pcorr_stats(self)
        else:
            if not( 'corr_stats' in self.__dict__):
                self.corr_stats = stats.ttest_rel(rtoz(self.A.get_corrs(pcorrs=False)),rtoz(self.B.get_corrs(pcorrs=False)))
            out = self.corr_stats
        
        return out

    def get_pcorr_stats(self): 
        if not( 'pcorr_stats' in self.__dict__):
               self.pcorr_stats=stats.ttest_rel(rtoz(self.A.get_corrs(pcorrs=pcorrs)),rtoz(self.B.get_corrs(pcorrs=pcorrs)))

        return self.pcorr_stats

    def get_std_stats(self,pcorrs=False): 
        if not( 'std_stats' in self.__dict__):
            self.std_stats = stats.ttest_rel(self.A.get_stds(pcorrs=pcorrs),self.B.get_stds(pcorrs=pcorrs))[0]

        return self.std_stats

    def get_lims(self,pcorrs=False,pctl=0.05,errdist_perms=50,refresh=False,sim_sampling=40):
        
        if not( 'lims' in self.__dict__) or refresh:
            if self.A.tcs is None and np.ndim(self.A.get_covs())==3: 
                A=FC(np.mean(self.A.get_covs(pcorrs=pcorrs),0),cov_flag=True, dof=self.A.dof,ROI_info=self.A.ROI_info)
                B=FC(np.mean(self.B.get_covs(pcorrs=pcorrs),0),cov_flag=True, dof=self.B.dof,ROI_info=self.B.ROI_info)
            elif self.A.tcs is None:
                A=self.A
                B=self.B
            else:
                A=flatten_tcs(self.A)
                B=flatten_tcs(self.B)
            self.lims=corr_lims_all(A,B,errdist_perms=errdist_perms,pctl=pctl,pcorrs=pcorrs,sim_sampling=sim_sampling)
        return(self.lims)

    def get_plot_inds(self,inds_cc=None,exclude_conns=True):
                              
        self.lims = gen_plot_inds(self.lims,inds_cc=inds_cc,exclude_conns=exclude_conns)


def gen_plot_inds(lims,inds_cc=None,exclude_conns=True):
    
    plots = list(lims.keys())+ ['other']
    plots.remove('covs') 
    
    if scipy.sparse.issparse(lims[plots[0]]['pctls']):
        voxels =  lims[plots[0]]['pctls'][0,1:].toarray()
        if not inds_cc:
            inds_cc=arange(voxels.shape[-1]).astype(int)

        inds_plots={}
        notin = inds_cc.copy()

        for plot in plots[:3]:
            inds_plots[plot]=np.intersect1d(inds_cc,pl.find(lims[plot]['pctls'][0,1:].toarray()))
            notin = np.setdiff1d(notin,inds_plots[plot])

        inds_plots['other'] = notin

        if exclude_conns:
            inds_plots['common']=np.setdiff1d(inds_plots['common'],inds_plots['unshared'])
            inds_plots['combined']=np.setdiff1d(inds_plots['combined'],inds_plots['common'])
            inds_plots['combined']=np.setdiff1d(inds_plots['combined'],inds_plots['unshared'])
    else:
        if not inds_cc:
            inds_cc=np.triu_indices(n_nodes,1)
        indices=np.triu_indices(n_nodes,1)
        notin=inds_cc
        inds_plots={}

        for plot in plots[:3]:
            inds_plots[plot]=np.intersect1d(inds_cc,pl.find(fa(lims[plot]['pctls'])))
            notin = np.setdiff1d(notin,pl.find(fa(lims[plot]['pctls'])))

        inds_plots['other'] = notin

        if exclude_conns:
            inds_plots['common']=np.setdiff1d(inds_plots['common'],inds_plots['unshared'])
            inds_plots['combined']=np.setdiff1d(inds_plots['combined'],inds_plots['common'])
            inds_plots['combined']=np.setdiff1d(inds_plots['combined'],inds_plots['unshared'])
    #self.inds_plots=inds_plots
    lims['covs']['inds_plots']=inds_plots

    return(lims)
     

def corr_lims_all(A,B,pcorrs=False,errdist_perms=0,dof=None,pctl=10,ch_type='All',sim_sampling=40,mask=None):

    if not dof:
        dof=A.dof

    if ch_type == 'All':
        ch_type = ['covs','unshared','common','combined']
    elif type(ch_type)==str:
        ch_type=[ch_type]

    if mask is None:
        shp = A.get_covs().shape
        if scipy.sparse.issparse(A.covs):
            mask = A.get_covs().nonzero()
            #todo fix masking
        else:
            mask =  triu_indices(shp[-1],k=1) 

            if len(shp) == 3:
                first=repeat(arange(shp[0]),len(mask[0]))
                second=tile(mask[0],shp[0])
                third=tile(mask[1],shp[0])
                mask=tuple((first,second,third))

    covsA=array(A.get_covs(pcorrs=pcorrs)[mask])
    covsB=array(B.get_covs(pcorrs=pcorrs)[mask])

    Acorrs=array(A.get_corrs(pcorrs=pcorrs)[mask]).flatten()

    Bcorrs=array(B.get_corrs(pcorrs=pcorrs)[mask]).flatten()   
    shp=Bcorrs.shape

    dims=arange(len(mask))
    els1 = tuple(setdiff1d(dims,dims[-1]))
    els2 = tuple(setdiff1d(dims,dims[-2]))

    mask1=tuple(mask[index] for index in els1)
    mask2=tuple(mask[index] for index in els2)

    Astdm=A.get_stds()[mask1]
    Astdmt=A.get_stds()[mask2]
    Bstdm=B.get_stds()[mask1]
    Bstdmt=B.get_stds()[mask2]

    #Bstdm=B.get_stds_m(pcorrs=pcorrs)[mask]
    #Bstdmt=B.get_stds_m_t(pcorrs=pcorrs)[mask]

    #corr_maxa_err=None
    #corr_mina_err=None
    pctls=None
    
    lims_struct={}

    for a in ch_type:
        
        lims_struct[a]={}

        if a=='unshared':

            lims_struct[a]={}
            unshared = covsA / (Bstdm*Bstdmt)
            unshared[unshared>1]=1 
            unshared[unshared<-1]=-1
        
            lims_struct[a]['min'] = A.get_covs()*0  
            lims_struct[a]['min'][mask] = unshared

            lims_struct[a]['max'] = A.get_covs()*0 
            lims_struct[a]['max'][mask] = unshared

        elif a== 'common':

            # 3d inds = tile(arange(sim_sampling),(shp[0],shp[1],shp[1],1))
            inds = tile(arange(sim_sampling),(shp[-1],1))

            # 3d corr2Common=zeros((shp[0],shp[1],shp[2],sim_sampling,sim_sampling))
            # corr2Common=zeros((shp,sim_sampling,sim_sampling))

            corr2Common_max=zeros(shp)
            corr2Common_max[:]=nan
            corr2Common_min=zeros(shp)
            corr2Common_min[:]=nan

            aa=(arange(sim_sampling)/(sim_sampling-1.0))

            for aaa in arange(len(aa)):
                # calc range of Y corrs for common
                rho_xb=aa[aaa]
                #Acorrs_abs = abs(Acorrs)
                # calculate range of possible ys
                (rho_yb_l,rho_yb_u)=calc_pbv(Acorrs,rho_xb)

                # loop over range of possible Y corrs
                weights = calc_weight(Astdm,Bstdm,rho_xb)
                inds_0=[(abs(weights[0])<abs(weights[1]))]
                inds_1=[(abs(weights[0])>=abs(weights[1]))]

                wx=zeros(weights[0].shape)
                wx[:]=nan
                wx[inds_0]=weights[0][inds_0]
                wx[inds_1]=weights[1][inds_1]
                weightsx=weights 
                bb= (arange(sim_sampling)/(sim_sampling-1.0)) 
                corr2Common=zeros(shp+(sim_sampling,))
                corr2Common[:]=nan

                for bbb in arange(len(bb)):

                    rho_yb=bb[bbb]
                    weights = calc_weight(Astdmt,Bstdmt,rho_yb)
                    inds_0=[(abs(weights[0])<abs(weights[1]))]
                    inds_1=[(abs(weights[0])>=abs(weights[1]))]
                    wy=zeros(weights[0].shape)
                    wy[:]=nan
                    wy[inds_0]=weights[0][inds_0]
                    wy[inds_1]=weights[1][inds_1]
                    # corr2Common = varA+varadd + 2cov(Aadd) 
                    corr2Common[:,bbb] = (Astdm*Astdmt*Acorrs + wx*wy + wx*Astdmt*rho_xb + wy*Astdm*rho_yb )/(Bstdm*Bstdmt)

                    # prevent negative weights
                    corr2Common[:,bbb][sign(wx) != sign(wy)] = nan
                    corr2Common[:,bbb][logical_or(rho_yb_l > bb[bbb] , rho_yb_u < bb[bbb])] = nan
                    corr2Common[:,bbb][sign(corr2Common[:,bbb])!=sign(Acorrs)]=nan
                    corr2Common[:,bbb][sign(corr2Common[:,bbb])!=sign(Acorrs)]=nan
                    
                    # prevent negative inital shared components
                    corr2Common[:,bbb][sign(covsA)==-1]=nan
                    #corr2Common[:,bbb][transpose(sign(covsA)*sign(covsB),(0,2,1))==-1]=nan
                    corr2Common[:,bbb][sign(covsA)*sign(covsB)==-1]=nan
                corr2Common[corr2Common==0]=nan
                corr2Common[corr2Common<-1]=nan
                corr2Common[corr2Common>1]=nan
                corr2Common_bbb_max = nanmax(corr2Common,1)
                corr2Common_bbb_min = nanmin(corr2Common,1)

                corr2Common_max = fmax(corr2Common_max,corr2Common_bbb_max)
                corr2Common_min = fmin(corr2Common_min,corr2Common_bbb_min)

                #if corr2Common_min<0.4:
                #    sdf
                #a sdfsdf 
            #if Bstdm > 1.2:
            #    sdf
                #inds_u=np.maximum(0,floor((rho_yb_u)*sim_sampling).astype(int))
                #inds_u=tile(inds_u,(sim_sampling,1)).transpose(1,0)
                ## remove els out correlation range
                #corr2Common[:,aaa,:][inds_u<inds]=nan
                ## remove els with negative weights
                #inds_l=np.maximum(0,floor((rho_yb_l)*sim_sampling).astype(int))
                ## inds_l=np.maximum(len(aa)/2,floor((rho_yb_l+1)/2.0*sim_sampling).astype(int))
                #inds_l=tile(inds_l,(sim_sampling,1)).transpose(1,0)
                #corr2Common[:,aaa,:][inds_l>inds]=nan
            
            #Acorr_tile = transpose(tile(Acorrs,(len(aa),len(bb),1)),[2,0,1])
            #corr2[corr2==0]=nan
            #corr2[sign(corr2)!=sign(Acorrs)
            #corr2[corr2<-1]=-1
            #corr2[corr2>1]=1
            #corr2Common[:,:,:,pl.find(sign(Acorrs)!=sign(aa)),:]=nan
            #corr2Common[:,:,:,:,pl.find(sign(Acorrs)!=sign(bb))]=nan

            # remove corr sign flips
            #corr2Common=corr2Common*sign(Acorrs)
            #corr2Common[sign(Acorrs)!=sign(corr2Common)]=nan
            #corr2Common[corr2Common==0]=nan
            #corr2Common[corr2Common<-1]=nan
            #corr2Common[corr2Common>1]=nan
            #Acorr_tile[corr2Common>=0]=-1
            #corr2Common[corr2Common<Acorr_tile]=nan 
            #tmp[sign(tmp)!=sign(Acorrs)]=0
            lims_struct[a]['min'] = A.get_covs()*0 
            lims_struct[a]['min'][mask]=corr2Common_min
            #tmp[sign(tmp)!=sign(Acorrs)]=0

            lims_struct[a]['max'] = A.get_covs()*0 
            lims_struct[a]['max'][mask]=corr2Common_max
            #*lims_struct[a]['corr2Common']=
            ##lims_struct[a]['min']corrminCommon=nanmin(nanmin(corr2Common,4),3) 
            ##lims_struct[a]['max']corrmaxCommon=nanmax(nanmax(corr2Common,4),3)

        elif a == 'combined':

            cx=cmaxmin(Astdm,Bstdm,Astdmt,Acorrs)[0]
            cy=cmaxmin(Astdmt,Bstdmt,Astdm,Acorrs)[0]

            cx_neg=c_neg_maxmin(Astdm,Bstdm,Astdmt,Acorrs)[0]
            cy_neg=c_neg_maxmin(Astdmt,Bstdmt,Astdm,Acorrs)[0]

            limsa=( Astdm*Astdmt*Acorrs*(1+cx_neg*cy_neg) + cx_neg*Astdmt**2 + cy_neg * Astdm**2) / (Bstdm * Bstdmt)
            limsb=( Astdm*Astdmt*Acorrs*(1+cx*cy) + cx*Astdmt**2 + cy * Astdm**2) / (Bstdm * Bstdmt)

            lims_struct[a]['min'] = A.get_covs()*0 
            lims_struct[a]['max'] = A.get_covs()*0 

            lims_struct[a]['min'][mask]=np.minimum(limsa,limsb)
            lims_struct[a]['max'][mask]=np.maximum(limsa,limsb)
            lims_struct[a]['max'][lims_struct[a]['max']>1]=1
            lims_struct[a]['min'][lims_struct[a]['min']<-1]=-1
    
    if errdist_perms > 0:
        shp_dist = tuple((errdist_perms,shp[0])) 
        unshared_lims_err=zeros(shp_dist)

        A_sim_dist=zeros(shp_dist)
        B_sim_dist=zeros(shp_dist)
        corr_err_A=zeros(shp_dist)
        corr_err_B=zeros(shp_dist)
        covs_err_A=zeros(shp_dist)
        covs_err_B=zeros(shp_dist)

        corr_min_common_err=zeros(shp_dist)
        corr_max_common_err=zeros(shp_dist)
        corr_min_Combined_err=zeros(shp_dist)
        corr_max_Combined_err=zeros(shp_dist)
     
        for xa in arange(1):
            # inv wishart distribution for covariance 
            
            #whA=stats.invwishart(dof,covsA[a,:,:]*(dof-1))
            #whB=stats.invwishart(dof,covsB[a,:,:]*(dof-1))

            # generate initial cov matrices 
            #if scipy.sparse.issparse(A.covs):
            #    whA=stats.wishart(dof,A.
            #else:
            #whA=stats.wishart(dof,A.get_covs()[xa,:,:])
            #whB=stats.wishart(dof,B.get_covs()[xa,:,:])

            #covsA_sim=whA.rvs(100*errdist_perms)/(dof)
            #covsB_sim=whB.rvs(100*errdist_perms)/(dof)
            
            # generate shaped prior cov matrices

            sims_gen_A=wishart_gen(A)
            sims_gen_B=wishart_gen(B)

            ppA=zeros((1,100*errdist_perms))
            ppB=zeros((1,100*errdist_perms))
            A_sims=sims_gen_A.get_sims(errdist_perms)
            B_sims=sims_gen_B.get_sims(errdist_perms)

            for xb in arange(errdist_perms):
                print(xb.astype(str))
                A_sim=FC(A_sims[xb],cov_flag=True,dof=A.dof)
                B_sim=FC(B_sims[xb],cov_flag=True,dof=B.dof)

                out = corr_lims_all(A_sim,B_sim,errdist_perms=0,pcorrs=pcorrs,dof=dof,ch_type=ch_type,sim_sampling=sim_sampling)

                # unshared
                if ndim(A.covs)==3:
                    mask_sim=(mask[1],mask[2])
                else:
                    mask_sim = mask

                if 'unshared' in ch_type:
                    unshared_lims_err[xb,:] = out['unshared']['min'][mask_sim]

                    # shared
                if 'covs' in ch_type:
                    corr_err_B[xb,:]=B_sim.get_corrs(pcorrs=pcorrs)[mask_sim]
                    corr_err_A[xb,:]=A_sim.get_corrs(pcorrs=pcorrs)[mask_sim]
                    covs_err_B[xb,:]=B_sim.get_covs(pcorrs=pcorrs)[mask_sim]
                    covs_err_A[xb,:]=A_sim.get_covs(pcorrs=pcorrs)[mask_sim]

                if 'common' in ch_type:
                    corr_min_common_err[xb,:]= out['common']['min'][mask_sim]
                    corr_max_common_err[xb,:]= out['common']['max'][mask_sim]

                # combined
                if 'combined' in ch_type:
                    corr_min_Combined_err[xb,:]= out['combined']['min'][mask_sim]
                    corr_max_Combined_err[xb,:]= out['combined']['max'][mask_sim]

        if 'covs' in ch_type:
            #lims_struct['cov']
            
            lims_struct['covs']['corrs_raw_A'] = corr_err_A
            lims_struct['covs']['corrs_raw_B'] = corr_err_B
            lims_struct['covs']['covs_raw_A'] = covs_err_A
            lims_struct['covs']['covs_raw_B'] = covs_err_B

            raw = corr_err_A-corr_err_B
            
            lims_struct['covs']['incl_zeros'] = percentileofscore(raw,0,0)
            #  lims_struct['correlation']['pctls']=(Bcorrs> pctl_min) != (Bcorrs> pctl_max)
            
        if 'unshared' in ch_type:
            unshared_lims_err[abs(unshared_lims_err)>1]=sign(unshared_lims_err[abs(unshared_lims_err)>1]) 
            #lims_struct['unshared']['err']=unshared_lims_err

            pctl_max = nanpercentile(unshared_lims_err,100-pctl,0)
            pctl_min = nanpercentile(unshared_lims_err,pctl,0)
            lims_struct['unshared']['pctls_raw'] = unshared_lims_err

            lims_struct['unshared']['min_pctls'] = A.get_covs()*0
            lims_struct['unshared']['min_pctls'][mask] = pctl_min
            lims_struct['unshared']['max_pctls'] = A.get_covs()*0
            lims_struct['unshared']['max_pctls'][mask] = pctl_max
            lims_struct['unshared']['pctls']=A.get_covs()*0
            lims_struct['unshared']['pctls'][mask]=(Bcorrs> pctl_min) != (Bcorrs> pctl_max)
        # common
        if 'common' in ch_type:
            corr_max_common_err[abs(corr_max_common_err)>1]=sign(corr_max_common_err[abs(corr_max_common_err)>1]) 
            pctl_out_max = nanpercentile(corr_max_common_err,100-pctl, 0)

            lims_struct['common']['max_pctls'] = A.get_covs()*0
            lims_struct['common']['max_pctls'][mask] = pctl_out_max

            corr_min_common_err[abs(corr_min_common_err)>1]=sign(corr_min_common_err[abs(corr_min_common_err)>1]) 

            pctl_out_min = nanpercentile(corr_min_common_err,pctl, 0)
            lims_struct['common']['min_pctls'] = A.get_covs()*0
            lims_struct['common']['min_pctls'][mask] = pctl_out_min

            #lims_struct['common']['ccmax'] = corr_max_common_err
            #lims_struct['common']['ccmin'] = corr_min_common_err

            lims_struct['common']['pctls'] = A.get_covs()*0
            lims_struct['common']['pctls'][mask] = (Bcorrs> pctl_out_max) != (Bcorrs> pctl_out_min)

            lims_struct['common']['min_pctls_raw'] = corr_min_common_err
            lims_struct['common']['max_pctls_raw'] = corr_max_common_err

        # combined 

        if 'combined' in ch_type:
            #corr_mina_err[abs(corr_min_Combined_err)>1]=sign(corr_min_Combined_err[abs(corr_min_Combined_err)>1]) 
            #lims_struct['combined']['min_err'] = corr_min_Combined_err

            #pctl_out = [percentile(corr_min_Combined_err,pctl,0),percentile(corr_min_Combined_err,100-pctl, 0)]
            pctl_out_min = nanpercentile(corr_min_Combined_err,pctl,0)
            lims_struct['combined']['min_pctls'] =  A.get_covs()*0
            lims_struct['combined']['min_pctls'][mask] =  pctl_out_min 
            
            #corr_maxa_err[abs(corr_max_Combined_err)>1]=sign(corr_max_Combined_err[abs(corr_max_Combined_err)>1]) 
            #pctl_out_neg = [percentile(corr_maxa_err,pctl,0),percentile(corr_maxa_err,100-pctl, 0)]
            pctl_out_max = nanpercentile(corr_max_Combined_err,100-pctl,0)
            lims_struct['combined']['max_pctls'] =  A.get_covs()*0
            lims_struct['combined']['max_pctls'][mask] =  pctl_out_max 

            #lims_struct['combined']['pctls'] = (Bcorrs> minimum(pctl_out_max[0] , pctl_out_max[1])) != (Bcorrs> maximum(pctl_out_min[0] ,  pctl_out_min[1]))
            lims_struct['combined']['pctls'] =  A.get_covs()*0
            lims_struct['combined']['pctls'][mask] = (Bcorrs> pctl_out_max) != (Bcorrs> pctl_out_min)

            lims_struct['combined']['min_pctls_raw'] = corr_min_Combined_err
            lims_struct['combined']['max_pctls_raw'] = corr_max_Combined_err

    return lims_struct

def calc_percentiles(out_con,pctl,ch_type=['covs','unshared','common','combined'],pcorrs=False):
        A = out_con.A
        B = out_con.B
        mask=A.get_covs().nonzero()
        lims_struct=out_con.lims.copy()

        Acorrs=array(A.get_corrs(pcorrs=pcorrs)[mask]).flatten()
        Bcorrs=array(B.get_corrs(pcorrs=pcorrs)[mask]).flatten()   
        shp=Bcorrs.shape

        if 'covs' in ch_type:

            raw = lims_struct['covs']['corrs_raw_A'] - lims_struct['covs']['corrs_raw_B'] 
            
            lims_struct['covs']['incl_zeros'] = percentileofscore(raw,0,0)
            #  lims_struct['correlation']['pctls']=(Bcorrs> pctl_min) != (Bcorrs> pctl_max)
            
        if 'unshared' in ch_type:
            unshared_lims_err = lims_struct['unshared']['pctls_raw'] 

            pctl_max = nanpercentile(unshared_lims_err,100-pctl,0)
            pctl_min = nanpercentile(unshared_lims_err,pctl,0)

            lims_struct['unshared']['min_pctls'] = A.get_covs()*0
            lims_struct['unshared']['min_pctls'][mask] = pctl_min
            lims_struct['unshared']['max_pctls'] = A.get_covs()*0
            lims_struct['unshared']['max_pctls'][mask] = pctl_max
            lims_struct['unshared']['pctls']=A.get_covs()*0
            lims_struct['unshared']['pctls'][mask]=(Bcorrs > pctl_min) != (Bcorrs> pctl_max)

        # common
        if 'common' in ch_type:

            corr_min_common_err = lims_struct['common']['min_pctls_raw'] 
            corr_max_common_err = lims_struct['common']['max_pctls_raw']

            pctl_out_max = nanpercentile(corr_max_common_err,100-pctl, 0)

            lims_struct['common']['max_pctls'] = A.get_covs()*0
            lims_struct['common']['max_pctls'][mask] = pctl_out_max

            pctl_out_min = nanpercentile(corr_min_common_err,pctl, 0)

            lims_struct['common']['min_pctls'] = A.get_covs()*0
            lims_struct['common']['min_pctls'][mask] = pctl_out_min

            #lims_struct['common']['ccmax'] = corr_max_common_err
            #lims_struct['common']['ccmin'] = corr_min_common_err

            lims_struct['common']['pctls'] = A.get_covs()*0
            lims_struct['common']['pctls'][mask] = (Bcorrs> pctl_out_max) != (Bcorrs> pctl_out_min)

        # combined 

        if 'combined' in ch_type:

            corr_min_Combined_err =  lims_struct['combined']['min_pctls_raw'] 
            corr_max_Combined_err = lims_struct['combined']['max_pctls_raw'] 

            pctl_out_min = nanpercentile(corr_min_Combined_err,pctl,0)
            lims_struct['combined']['min_pctls'] =  A.get_covs()*0
            lims_struct['combined']['min_pctls'][mask] =  pctl_out_min 
            
            pctl_out_max = nanpercentile(corr_max_Combined_err,100-pctl,0)
            lims_struct['combined']['max_pctls'] =  A.get_covs()*0
            lims_struct['combined']['max_pctls'][mask] =  pctl_out_max 

            lims_struct['combined']['pctls'] =  A.get_covs()*0
            lims_struct['combined']['pctls'][mask] = (Bcorrs> pctl_out_max) != (Bcorrs> pctl_out_min)

        return(lims_struct)


# calculate amount of signal with correlation rho_xb to initial signal produces variance change from std_x to std_xb
def calc_weight(std_xa,std_xb,rho_xb):
    tmp = sqrt((std_xa*rho_xb)**2 + std_xb**2 -std_xa**2)
    return(-std_xa*rho_xb + tmp,-std_xa*rho_xb - tmp)

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
        
           out=data[0,:]
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

def corr2pcorr(cc):

    pinvA=linalg.pinv(cc)
    iis=tile(atleast_2d(pinvA.diagonal()).T,pinvA.shape[1])
    dd=diag(pinvA)

    tmp=-pinvA/sqrt(iis*iis.T)
    tmp[where(eye(cc.shape[1]))]=1

    return(tmp)

# generate simulated samples for an observed covariance
class wishart_gen:
    def __init__(self,A):
        self.A=A

    def get_sims(self,errdist_perms,recalc=False):
        A=self.A
        
        if not( 'covs_sim' in self.__dict__) or recalc==True:

            if scipy.sparse.issparse(A.get_covs()):
                covs=A.get_covs()
                corrs=A.get_corrs()
                vars1=covs[0,0]
                vars2=corrs[0,1:].toarray()
                cvs=covs.diagonal()[1:]
                # generate random samples from wishart
                (vars1,vars2,covmat1) = wishart_2(covs[0,0],covs.diagonal()[1:],corrs[0,1:].toarray(),A.dof,size=errdist_perms)
                covs_sim=[]

                for a in arange(vars2.shape[0]):
                    covmat=A.get_covs()*0
                    covmat[0,0]=vars1[a,0]
                    diag_inds=np.diag_indices(covs.shape[-1])
                    covmat[diag_inds[0][1:],diag_inds[1][1:]]=vars2[a,:]
                    covmat[0,1:]=covmat1[a,:]
                    covs_sim.append(covmat)
            else:
                covs=self.A.get_covs()
                dof=self.A.dof
                whA=stats.wishart(dof,squeeze(covs)[:,:])
                covs_sim=whA.rvs(errdist_perms)/(dof)
                covs_sim=list(covs_sim)

                ppA=zeros((1,100*errdist_perms))
                ppB=zeros((1,100*errdist_perms))
                
                whA=[]
                whB=[]

                # create  
                for yb in arange(errdist_perms):
                    whA.append(stats.wishart(dof,covs_sim[yb]))
                    ppA[0,yb]=whA[-1].pdf(squeeze(A.get_covs())*dof)
                    #ppA[0,yb]=1
                    #whB.append(stats.wishart(dof,covsB_sim[yb,:,:]))
                    #ppB[0,yb]=whB[-1].pdf(B.get_covs()[xa,:,:]*dof)
                    #ppB[0,yb]=1
                   
                # generate sample
                ppA=ppA/sum(ppA)
                #ppB=ppB/sum(ppB)
                ppA_cul=(dot(ppA,triu(ones(len(ppA.T)))).T) 
                ## memory issues
                #ppB_cul=(dot(ppB,triu(ones(len(ppB.T)))).T) 
                rand_els = stats.uniform(0,1).rvs(errdist_perms) 
                els=sort(searchsorted(ppA_cul.flatten(),rand_els)) 
                covs_sim=[]
                for xb in arange(errdist_perms):
                    covs_sim.append(whA[els[xb]].rvs()/dof)

            self.covs_sim=covs_sim

        return(self.covs_sim)

def wishart_2(vars1,vars2,rho,dof,size=1):
    
    rho=atleast_2d(rho).T
    vars1=atleast_2d(vars1).T
    vars2=atleast_2d(vars2).T

    chis1 = atleast_2d(stats.chi2.rvs(dof,size=size))
    chis2 = atleast_2d(stats.chi2.rvs(dof-1,size=size))
    norms = atleast_2d(stats.norm.rvs(0,1,size=size))

    sqrt_rho = np.sqrt(1-rho**2)

    vars1_sim = vars1*chis1/dof
    covs_sim = np.sqrt(vars1*vars2)*(rho*chis1 + sqrt_rho *(chis1**0.5)*norms)/dof
    vars2_sim = vars2 * ( chis2*(sqrt_rho**2) + (sqrt_rho*norms + rho*(chis1**.5))**2 )/dof
    
    #vars2_sim = vars2 * ( chis2*(sqrt_rho**2) + ( rho*(chis1**.5))**2 )

    return(vars1_sim.T,vars2_sim.T,covs_sim.T)

def wishart_pdf(cov,samples,dof):
    if scipy.sparse.issparse(cov):
        var1=cov[0,0]
        var2=cov.diagonal()[1:]
        covs1=covs[0,:]
        
        for a in len(var2):
            whA.append(stats.wishart(dof,covsA_sim[b,:,:]))
        ppA[0,yb]=whA[-1].pdf(A.get_covs()[xa,:,:]*dof)

    return(cov) 

def var_gamma_pdf(x,std1,std2,rho,dof):
    # marginal distribution of covariance (not used)
    ors = (1-rho**2)
    gamma_d = stats.gamma(dof/2.0)
    first = abs(x)**((dof-1)/2)  / ( gamma_d * sqrt(2^(dof-1)*pi*ors*(std1*std2)**n+1) )
    second = special((dof-1)/2.0,abs(x)/(std1*std2*ors))
    third = e**( (rho*x) / std1*std2*ors)

    return(first*second*third)



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
        out[a,:,:]= scipy.linalg.cho_solve(scipy.linalg.cho_factor( stats.wishart(r,Minv).rvs()),eye(len(Minv)))
    return(out)

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
    out=FC((reshape(tcs_dm.swapaxes(0,1),[shp[1],-1])),ROI_info=A.ROI_info)
    # subtract dofs from demeaning 
    out.dof=out.dof-shp[0]
    tcs = out.tcs

    if dof is None:
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

def seed_loader(filenames,seed_mask_file,mask_file,subj_inds=None,dof=None):

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

    dof_cnt=0

    for file in filenames:
        print(file)

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
        dof_cnt = dof_cnt+seed_data.shape[-1]-1
         
        # FC_cov = FC(covmat)

        newfile.uncache()

    if not dof:
        dof=dof_cnt

    covmat=covmat/len(filenames)

    out = FC(covmat,cov_flag=True,dof=dof,mask=mask)

    return out

def seed_saver(filename,AB_con,mask=None,save_minmax=False,save_corr=False):

    lims=AB_con.get_lims()
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
            if pp != 'unshared':
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
    def __init__(self,filenames,seed_mask,mask,subj_inds):

        for file in filenames:
            
            FC.filelist.append(nb.load(file))



def dr_loader(dir,prefix='dr_stage1',subj_inds=None,ROI_order=None,subjorder=True,dof='EstEff',nosubj=False,read_roi_info=True):
    
    if nosubj:
        dr_files=sort(glob.glob(prefix+'*.txt')) 
    else:
        dr_files=sort(glob.glob(prefix+'_subject?????.txt')) 
        
    if subj_inds is None:
        subj_inds=arange(len(dr_files))
    
    dr_files=dr_files[subj_inds]

    subjs=len(dr_files)
    maskflag=False

    data=atleast_2d(loadtxt(dr_files[0]).T)

    ROI_info={}

    if read_roi_info:

        if ROI_order is None:
            if os.path.isfile('ROI_order.txt'):
                ROI_order=np.loadtxt('ROI_order.txt').astype(int)-1
            else:
                ROI_order = arange((data.shape[0]))

        ROI_info['ROI_order']=ROI_order
     
        if os.path.isfile('ROI_RSNs.txt'):
            ROI_info['ROI_RSNs']=(np.loadtxt('ROI_RSNs.txt').astype(int))
        else: 
            ROI_info['ROI_RSNs'] = zeros((data.shape[0],))+1
        
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
            ROI_info['ROI_names']=np.array([ str(a) for a in np.arange(data.shape[-1]) ])

        # n_nodes=len(ROI_info[ROI_names)

        ROI_info['ROI_names']=ROI_info['ROI_names'][ROI_info['ROI_order']]


    for cnt in arange(subjs):

            tmpdata=atleast_2d(loadtxt(dr_files[cnt]).T)
            if cnt==0:
                shp=tmpdata.shape
                if ROI_order is None:
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
    
    if not ROI_order is None:
        datamat=datamat[:,ROI_order,:]

    if ( not subjorder ):
        datamat=swapaxes(datamat,0,1)

    A=FC(datamat,dof=dof,ROI_info=ROI_info)

    return A

def dr_saver(A,dir,prefix='dr_stage1',goodnodes=None,aug=0):
   tcs =  A.tcs
   dof = A.dof

   if goodnodes is None:
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
        stds=A.get_stds()
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

def make_psd_tcs(tcs,PSD=False,nreps=1,norm=False,tpts=None):
    
    if tpts is None:
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

def corr_lims_all_sim(ccs,vvs,pcorrs=False,errdist=False,errdist_perms=100,dof=None,pctl=5,ch_type='All',sim_sampling=40):

    out=None
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

        out.append(corr_lims_all(tmpA,tmpB,pcorrs=pcorrs,errdist=errdist,errdist_perms=errdist_perms,dof=dof,pctl=pctl,ch_type=ch_type,sim_sampling=sim_sampling))
    
    return(out)

def corr_lims_all_pool(ccs,vvs,pcorrs=False,errdist=False,errdist_perms=100,dof=None,pctl=5,ch_type='All',sim_sampling=40,show_pctls=True):

    out=None
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

        out.append(corr_lims_all(tmpA,tmpB,pcorrs=pcorrs,errdist=errdist,errdist_perms=errdist_perms,dof=dof,pctl=pctl,ch_type=ch_type,sim_sampling=sim_sampling,show_pctls=show_pctls))
    
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
    els=x.nonzero()
    out = x*0
    out[els]=(0.5*(np.log(1+x[els]) - np.log(1-x[els])))

    return out


def flattenall(x,nd=2):
   
    shp=x.shape
    if (len(shp)==2) & (shp[0]==shp[1]):
        nd=1

    firstdim=np.product(shp[:-2])
    tmp=np.reshape(x,(firstdim,shp[-2],shp[-1]))
    uts=0.5*((shp[-1]**2)-shp[-1])
    outmat=np.zeros((firstdim,uts))

    for aa in arange(firstdim):
        if nd==1:
            outmat[(uts*aa):(uts*(aa+1))]=triu_all(tmp[aa,:,:])
        elif len(x.shape)>2:
            outmat[aa,:]= triu_all(tmp[aa,:,:])
        else:
            #  if already flattened
            outmat=x
    return outmat



def triu_all(x):
    return x.flatten()[pl.find(triu(ones(x.shape),1))]
