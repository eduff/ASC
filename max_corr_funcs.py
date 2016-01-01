from numpy import *
import os
import numpy as np
import numpy.linalg as la
import matplotlib.pylab as pl
import matplotlib.cm as cm
import glob,os, numbers
import nibabel as nb
import scipy.stats
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

                A_sim=FC(whA.rvs(),cov_flag=True)
                B_sim=FC(whB.rvs(),cov_flag=True)

                tmp = corr_lims_mat(A_sim,B_sim,dof=dof) 
                corr_maxa_err[b,a,:,:]=tmp[0]
                corr_mina_err[b,a,:,:]=tmp[1]
        corr_mina_err[abs(corr_mina_err)>1]=sign(corr_mina_err[abs(corr_mina_err)>1]) 
        pctl_out = [percentile(corr_mina_err,pctl,0),percentile(corr_mina_err,100-pctl, 0)]

        corr_maxa_err[abs(corr_maxa_err)>1]=sign(corr_maxa_err[abs(corr_maxa_err)>1]) 
        pctl_out_neg = [percentile(corr_maxa_err,pctl,0),percentile(corr_maxa_err,100-pctl, 0)]
        
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

class FC:
    def __init__(self,tcs,cov_flag=False,dof=[]):
    
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
            else:
                self.dof=dof
                
            # corrs=zeros((tcs.shape[0],tcs.shape[1],tcs.shape[1]))
            covs=zeros((tcs.shape[0],tcs.shape[1],tcs.shape[1]))

            for a in arange(tcs.shape[0]):
                covs[a,:,:]=cov(tcs[a,:,:])

        self.covs = covs

    def get_stds(self,pcorrs=False):

        if ~( 'stds' in self.__dict__):
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
            if ~( 'corrs' in self.__dict__):
                self.corrs = self.get_covs()/(self.get_stds_m()*transpose(self.get_stds_m(),(0,2,1)))
            return(self.corrs)
    
    def get_pcorrs(self):

        if ~( 'pcorrs' in self.__dict__):
            if ~( 'corrs' in self.__dict__):
                self.corrs = (self.get_covs())/(self.get_stds_m()*transpose(self.get_stds_m(),(0,2,1)))
            
            self.pcorrs=zeros(self.corrs.shape)
            for a in range(len(self.pcorrs)):
                iis=tile(atleast_2d(pinvA.diagonal()).T,self.covs.shape[1])
                tmp=-pinvA/sqrt(iis*iis.T)
                tmp[where(eye(tmp.shape[0]))]=1
                self.pcorrs[a,:,:]=tmp
        return self.pcorrs
    
    def get_covs(self,pcorrs=False):
        if pcorrs:
            return self._get_pcovs()
        else:
            return self.covs

    def _get_pcovs(self):
        if ~( 'pcorrs' in self.__dict__):
            pcorrs=self.get_pcorrs()

        multiplier = (self.get_stds_m()*transpose(self.get_stds_m(),(0,2,1)))
        
        return(pcorrs*multiplier)


class FC_con:
    def __init__(self,A,B):     
        self.A=A
        self.B=B

    def get_shared(self,errdist=False):
        if ~( 'shared' in self.__dict__):
            self.shared=corr_lims_shared_mat(self.A,self.B,errdist=errdist)
        
        return(self.shared)

    def get_unshared(self,errdist=False):
        if ~( 'unshared' in self.__dict__):
            self.unshared=corr_lims_unshared_mat(self.A,self.B,errdist=errdist)

        return(self.unshared)

    def get_lims(self,errdist=False):
        if ~( 'lims' in self.__dict__):
            self.lims=corr_lims_mat(self.A,self.B,errdist=errdist)

        return(self.lims)

    def unshared_prob(self):
        if ~( 'unshared' in self.__dict__):
            self.unshared=corr_lims_unshared_mat(self.A,self.B,errdist=errdist)

        


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
                A_sim=FC(whA.rvs()/dof,cov_flag=True)
                B_sim=FC(whB.rvs()/dof,cov_flag=True)

                unshared_lims_err[b,a,:,:] = corr_lims_unshared_mat(A_sim,B_sim,dof=dof)

        unshared_lims_err[abs(unshared_lims_err)>1]=sign(unshared_lims_err[abs(unshared_lims_err)>1]) 
        pctl_out = [percentile(unshared_lims_err,pctl,0),percentile(unshared_lims_err,100-pctl,0)]
        pctls=(Bcorrs> pctl_out[0]) != (Bcorrs> pctl_out[1])
        return(unshared, pctls ,unshared_lims_err,covsA_sim)
    else:
        return(unshared)


def corr_lims_all(A,B,pcorrs=False,errdist=False,errdist_perms=1000,dof=[],pctl=5,pctl_shared=5,chType='All',sim_sampling=80):

    if dof==[]:
        dof=A.dof

    if chType == 'All':
        chType = ['unshared','shared','combined']
    elif type(chType)==str:
        chType=[chType]

    covsA=A.get_covs()
    covsB=B.get_covs()

    Acorrs=A.get_corrs()   
    Astds=A.get_stds_m()

    Bcorrs=B.get_corrs()   
    Bstds=B.get_stds_m()
    
    shp=A.covs.shape
    lims_struct={}

    Astdm=A.get_stds_m(pcorrs=pcorrs)
    Astdmt=A.get_stds_m_t(pcorrs=pcorrs)
    Bstdm=B.get_stds_m(pcorrs=pcorrs)
    Bstdmt=B.get_stds_m_t(pcorrs=pcorrs)

    A_sim=[]
    B_sim=[]
    corr_maxa_err=[]
    corr_mina_err=[]
    pctls=[]
    
    for a in chType:
        
        lims_struct[a]={}

        if a=='unshared':

            lims_struct[a]={}
            unshared = covsA / (Bstdm*Bstdmt)
            unshared[unshared>1]=1 
            unshared[abs(unshared)>1]=sign(unshared[abs(unshared)>1])
        
            lims_struct[a]['min'] = unshared
            lims_struct[a]['max'] = unshared

        elif a== 'shared':

            inds = tile(arange(sim_sampling),(shp[0],shp[1],shp[1],1))

            corr2=zeros((shp[0],shp[1],shp[2],sim_sampling,sim_sampling))
            corr2Common=zeros((shp[0],shp[1],shp[2],sim_sampling,sim_sampling))
            aa=(arange(sim_sampling)/(sim_sampling-1.0))

            for aaa in arange(len(aa)):
                rho_xb=aa[aaa]
                (rho_yb_l,rho_yb_u)=calc_pbv(Acorrs,rho_xb)
                bb= (arange(sim_sampling)/(sim_sampling-1.0)) #/(rho_yb_u-rho_yb_l) + rho_yb_l
                tmp=[] 
                # loop over range of possible Y corrs 
                for bbb in arange(len(bb)):
                    rho_yb=bb[bbb]
                    tmp = calc_weight(Astdm,Bstdm,rho_xb)
                    inds_0=[(abs(tmp[0])<abs(tmp[1]))]
                    inds_1=[(abs(tmp[0])>=abs(tmp[1]))]
                    wx=zeros(tmp[0].shape)
                    wx[inds_0]=tmp[0][inds_0]
                    wx[inds_1]=tmp[1][inds_1]
                    
                    tmp = calc_weight(Astdmt,Bstdmt,rho_yb)
                    inds_0=[(abs(tmp[0])<abs(tmp[1]))]
                    inds_1=[(abs(tmp[0])>=abs(tmp[1]))]
                    wy=zeros(tmp[0].shape)
                    wy[inds_0]=tmp[0][inds_0]
                    wy[inds_1]=tmp[1][inds_1]

                    corr2[:,:,:,aaa,bbb] = (Astdm*Astdmt*Acorrs + wx*Astdm*rho_xb + wy*Astdmt*rho_yb + wx*wy )/(Bstdm*Bstdmt)
                    tmp = corr2[:,:,:,aaa,bbb] 
                inds_u=(floor((rho_yb_u+1)*sim_sampling)).astype(int)
                inds_u=tile(inds_u,(sim_sampling,1,1,1)).transpose(1,2,3,0)
                
                corr2Common=corr2.copy() 
                
                corr2Common[:,:,:,aaa,:][inds_u<inds]=nan
                inds_l=(floor(rho_yb_l*sim_sampling)).astype(int)
                inds_l=tile(inds_l,(sim_sampling,1,1,1)).transpose(1,2,3,0)
                corr2Common[:,:,:,aaa,:][inds_l>inds]=nan

            Acorr_tile = transpose(tile(Acorrs,(len(aa),len(bb),1,1,1)),[2,3,4,0,1])
            corr2[corr2==0]=nan
            #corr2[sign(corr2)!=sign(Acorrs)
            corr2[corr2<-1]=-1
            corr2[corr2>1]=1
            #corr2Common[:,:,:,plt.find(sign(Acorrs)!=sign(aa)),:]=0
            #corr2Common[:,:,:,:,plt.find(sign(Acorrs)!=sign(bb))]=0
            corr2Common[corr2Common==0]=nan
            corr2Common[corr2Common<-1]=-1
            corr2Common[corr2Common>1]=1
            Acorr_tile[corr2Common>=0]=-1
            corr2Common[corr2Common<Acorr_tile]=nan 

            tmp=nanmin(nanmin(corr2Common,4),3) 
            #tmp[sign(tmp)!=sign(Acorrs)]=0
            lims_struct[a]['min']=tmp
            tmp=nanmax(nanmax(corr2Common,4),3) 
            #tmp[sign(tmp)!=sign(Acorrs)]=0
            lims_struct[a]['max']=tmp
            #lims_struct[a]['min']corrminCommon=nanmin(nanmin(corr2Common,4),3) 
            #lims_struct[a]['max']corrmaxCommon=nanmax(nanmax(corr2Common,4),3)

        elif a == 'combined':

            cx=cmaxmin_m(Astdm,Bstdm,Astdmt,Acorrs)[0]
            cy=cmaxmin_m(Astdmt,Bstdmt,Astdm,Acorrs)[0]

            cx_neg=c_neg_maxmin_m(Astdm,Bstdm,Astdmt,Acorrs)[0]
            cy_neg=c_neg_maxmin_m(Astdmt,Bstdmt,Astdm,Acorrs)[0]

            #corr_maxa=( Astdm*Astdmt*A.corrs*(1+cx*cy) + cx*Astdmt**2 + cy * Astdm**2) / (Bstdm * Bstdmt)
            limsa=( Astdm*Astdmt*A.corrs*(1+cx_neg*cy_neg) + cx_neg*Astdmt**2 + cy_neg * Astdm**2) / (Bstdm * Bstdmt)
            limsb=( Astdm*Astdmt*A.corrs*(1+cx*cy) + cx*Astdmt**2 + cy * Astdm**2) / (Bstdm * Bstdmt)

            lims_struct[a]['min']=np.minimum(limsa,limsb)
            lims_struct[a]['max']=np.maximum(limsa,limsb)
            lims_struct[a]['max'][lims_struct[a]['max']>1]=1
            lims_struct[a]['min'][lims_struct[a]['min']<-1]=-1
    
    if errdist:

        unshared_lims_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        corr_maxa_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        corr_mina_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        corr_min_Shared_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        corr_max_Shared_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        corr_min_Common_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        corr_max_Common_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))
     
        for xa in arange(shp[0]):
            # inv wishart distribution for covariance 
            
            #whA=scipy.stats.invwishart(dof,covsA[a,:,:]*(dof-1))
            #whB=scipy.stats.invwishart(dof,covsB[a,:,:]*(dof-1))
            
            whA=scipy.stats.wishart(dof,covsA[xa,:,:])
            whB=scipy.stats.wishart(dof,covsB[xa,:,:])

            covsA_sim=whA.rvs(errdist_perms)/(dof)
            covsB_sim=whB.rvs(errdist_perms)/(dof)

            ppA=zeros((1,errdist_perms))
            ppB=zeros((1,errdist_perms))
            
            whA=[]
            whB=[]

            for yb in arange(errdist_perms):
                whA.append(scipy.stats.wishart(dof,covsA_sim[yb,:,:]))
                ppA[0,yb]=whA[-1].pdf(covsA[xa,:,:]*dof)
                whB.append(scipy.stats.wishart(dof,covsB_sim[yb,:,:]))
                ppB[0,yb]=whB[-1].pdf(covsB[xa,:,:]*dof)
               
            ppA=ppA/sum(ppA)
            ppB=ppB/sum(ppB)

            ppA_cul=(dot(ppA,triu(ones(len(ppA.T)))).T) 
            ppB_cul=(dot(ppB,triu(ones(len(ppB.T)))).T) 
            rand_els = scipy.stats.uniform(0,1).rvs(errdist_perms) 
            els=sort(searchsorted(ppA_cul.flatten(),rand_els)) 

            for xb in arange(errdist_perms):

                A_sim=FC(whA[els[xb]].rvs(),cov_flag=True,dof=A.dof)
                B_sim=FC(whB[els[xb]].rvs(),cov_flag=True,dof=B.dof)
                out = corr_lims_all(A_sim,B_sim,errdist=False)
                # unshared
                unshared_lims_err[xb,xa,:,:] = out['unshared']['min']

                # shared
                tmp = corr_lims_shared_mat(A_sim,B_sim,dof=dof,sim_sampling=40) 
                corr_min_Shared_err[xb,xa,:,:]= out['shared']['min']
                corr_max_Shared_err[xb,xa,:,:]= out['shared']['max']

                # combined
                corr_min_Common_err[xb,xa,:,:]= out['combined']['min']
                corr_max_Common_err[xb,xa,:,:]= out['combined']['max']


                #tmp = corr_lims_mat(A_sim,B_sim,dof=dof) 
                #corr_maxa_err[xb,xa,:,:]=tmp[0].squeeze()
                #corr_mina_err[xb,xa,:,:]=tmp[1].squeeze()

        # unshared

        unshared_lims_err[abs(unshared_lims_err)>1]=sign(unshared_lims_err[abs(unshared_lims_err)>1]) 
        #lims_struct['unshared']['err']=unshared_lims_err

        # pctl_out = [percentile(unshared_lims_err,pctl,0),percentile(unshared_lims_err,100-pctl,0)]
        pctl_max = percentile(unshared_lims_err,100-pctl_shared/2.0,0)
        pctl_min = percentile(unshared_lims_err,pctl_shared/2.0,0)
        lims_struct['unshared']['pctls_min'] = pctl_min
        lims_struct['unshared']['pctls_max'] = pctl_max
        lims_struct['unshared']['pctls']=(Bcorrs> pctl_min) != (Bcorrs> pctl_max)


        # shared
        corr_max_Shared_err[abs(corr_max_Shared_err)>1]=sign(corr_max_Shared_err[abs(corr_max_Shared_err)>1]) 
        pctl_out_max = percentile(corr_max_Shared_err,100-pctl_shared/2.0, 0)
        lims_struct['shared']['pctls_max'] = pctl_out_max

        corr_min_Shared_err[abs(corr_min_Shared_err)>1]=sign(corr_min_Shared_err[abs(corr_min_Shared_err)>1]) 

        pctl_out_min = percentile(corr_min_Shared_err,pctl_shared/2.0, 0)
        lims_struct['shared']['pctls_min'] = pctl_out_min
        lims_struct['shared']['ccmax'] = corr_max_Shared_err
        lims_struct['shared']['ccmin'] = corr_min_Shared_err


        lims_struct['shared']['pctls'] = (Bcorrs> pctl_out_max) != (Bcorrs> pctl_out_min)

        # combined 
        #corr_mina_err[abs(corr_min_Common_err)>1]=sign(corr_min_Common_err[abs(corr_min_Common_err)>1]) 
        #lims_struct['combined']['min_err'] = corr_min_Common_err

        #pctl_out = [percentile(corr_min_Common_err,pctl,0),percentile(corr_min_Common_err,100-pctl, 0)]
        pctl_out_min = percentile(corr_min_Common_err,pctl,0)
        lims_struct['combined']['pctls_min'] =  pctl_out_min 
        
        #corr_maxa_err[abs(corr_max_Common_err)>1]=sign(corr_max_Common_err[abs(corr_max_Common_err)>1]) 
        #pctl_out_neg = [percentile(corr_maxa_err,pctl,0),percentile(corr_maxa_err,100-pctl, 0)]
        pctl_out_max = percentile(corr_max_Common_err,100-pctl,0)
        lims_struct['combined']['pctls_max'] =  pctl_out_max 

        #lims_struct['combined']['pctls'] = (Bcorrs> minimum(pctl_out_max[0] , pctl_out_max[1])) != (Bcorrs> maximum(pctl_out_min[0] ,  pctl_out_min[1]))
        lims_struct['combined']['pctls'] = (Bcorrs> pctl_out_max) != (Bcorrs> pctl_out_min)

    return lims_struct


# calculate amount of signal with correlation rho_xb to initial signal produces variance change from std_x to std_xb

def calc_weight(std_x,std_xb,rho_xb):
    tmp = sqrt((std_x*rho_xb)**2 + std_xb**2 -std_x**2)
    return(-std_x*rho_xb + tmp,-std_x*rho_xb - tmp)

# calculate max,min correlation of rho_by given rho_xy, rho_xb
def calc_pbv(rho_xy,rho_xb):
    min_cc =  (rho_xy*rho_xb) - sqrt(1 - rho_xy**2)*sqrt(1 - rho_xb**2)
    max_cc =  (rho_xy*rho_xb) + sqrt(1 - rho_xy**2)*sqrt(1 - rho_xb**2)
    #tmp = sqrt((rho_xy*rho_xb)**2 + 1 - rho_xy**2 -rho_xb**2)
    return(min_cc,max_cc)

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

# calc shared max range of change in signal
def  corr_lims_shared(std_x,std_xb,std_y,std_yb,rho_xy,sim_sampling=40):
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

def corr_lims_shared_mat(A,B,pcorrs=False,errdist=False,errdist_perms=300,dof=[],pctl=5,sim_sampling=101):

    if dof==[]:
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
        
    # loop over range of rho_xb shared 
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


    corrminShared=nanmin(nanmin(corr2,4),3) 
    corrmaxShared=nanmax(nanmax(corr2,4),3)

    corrminCommon=nanmin(nanmin(corr2Common,4),3) 
    corrmaxCommon=nanmax(nanmax(corr2Common,4),3)

    A_sim=[]
    B_sim=[]
 
    pctlsShared=[]
    corr_min_Shared_err=[]
    corr_max_Shared_err=[]

    pctlsCommon=[]
    corr_min_Common_err=[]
    corr_max_Common_err=[]

    if errdist:
        shp=covsA.shape

        corr_min_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        corr_max_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))

        corr_min_Shared_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        corr_max_Shared_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        
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

                A_sim=FC(whA[els[b]].rvs(),cov_flag=True,dof=A.dof)
                B_sim=FC(whB[els[b]].rvs(),cov_flag=True,dof=B.dof)

                tmp = corr_lims_shared_mat(A_sim,B_sim,dof=dof,sim_sampling=40.0) 
                corr_min_Shared_err[b,a,:,:]=tmp[0][0]
                corr_max_Shared_err[b,a,:,:]=tmp[0][1]
                corr_min_Common_err[b,a,:,:]=tmp[1][0]
                corr_max_Common_err[b,a,:,:]=tmp[1][1]

        corr_max_Shared_err[abs(corr_max_Shared_err)>1]=sign(corr_max_Shared_err[abs(corr_max_Shared_err)>1]) 
        pctl_out_max = [percentile(corr_max_Shared_err,pctl,0),percentile(corr_max_Shared_err,100-pctl, 0)]

        corr_min_Shared_err[abs(corr_min_Shared_err)>1]=sign(corr_min_Shared_err[abs(corr_min_Shared_err)>1]) 
        pctl_out_min = [percentile(corr_min_Shared_err,pctl,0),percentile(corr_min_Shared_err,100-pctl, 0)]
        #pctls_Shared = (Bcorrs> minimum(pctl_out_min[0] , pctl_out_min[1])) != (Bcorrs> maximum(pctl_out[0] ,  pctl_out[1]))


        pctl_out_max = [percentile(corr_max_Common_err,pctl,0),percentile(corr_max_Common_err,100-pctl, 0)]

        pctl_out_min = [percentile(corr_min_Common_err,pctl,0),percentile(corr_min_Common_err,100-pctl, 0)]
        #pctls_Shared = (Bcorrs> minimum(pctl_out_min[0] , pctl_out_min[1])) != (Bcorrs> maximum(pctl_out[0] ,  pctl_out[1]))

        return([[corrminShared,corrmaxShared],pctls_Shared,[corr_min_Shared_err, corr_max_Shared_err]],[[corrminCommon,corrmaxCommon],pctls_Common,[corr_min_Common_err, corr_max_Common_err]])
    else:
        return([[corrminShared,corrmaxShared],[corrminCommon,corrmaxCommon]])

def plot_class(A,B,thresh=1.96):
   
    corr_lims,corr_lims_TF = corr_lims_mat(A,B)
    shared,shared_TF = corr_lims_shared_mat(A,B)
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
    tmp[where(eye(cc.shape[1]))]=dd

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
    # calculate histograms of unshared, shared, and mixed effects.
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
    out_hist_shared_l=zeros((len(vv2),len(ccs)))
    out_hist_shared_u=zeros((len(vv2),len(ccs)))
    out_hist_lim_l=zeros((len(vv2),len(ccs)))
    out_hist_lim_u=zeros((len(vv2),len(ccs)))
    print('vv1:'+str(vv1))

    for vv2_cnt in range(len(vv2)):
        ooB[[0],[0]]=vvs_all[vv1]
        ooB[[1],[1]]=vvs_all[vv2[vv2_cnt]]
        ooB[[0,1],[1,0]]=ccs[cc]*(vvs_all[vv1]**.5)*(vvs_all[vv2[vv2_cnt]]**.5)

        tmpB=FC(ooB,cov_flag=True)
        
        out_unshared=corr_lims_unshared_mat(tmpA,tmpB,errdist=True,errdist_perms=errdist_perms,pctl=5,dof=dof)
        out_hist_unshared[vv2_cnt,:]=np.histogram(out_unshared[2][:,0,1,0],normed=True,range=[0,1],bins=r_[ccs,[1]])[0]
        out_shared=corr_lims_shared_mat(tmpA,tmpB,errdist=True,errdist_perms=errdist_perms,pctl=5,dof=dof)
        out_hist_shared_l[vv2_cnt,:]=np.histogram(out_shared[2][0][:,0,1,0],normed=True,range=[0,1],bins=r_[ccs,[1]])[0]
        out_hist_shared_u[vv2_cnt,:]=np.histogram(out_shared[2][1][:,0,1,0],normed=True,range=[0,1],bins=r_[ccs,[1]])[0]

        out_lims=corr_lims_mat(tmpA,tmpB,errdist=True,errdist_perms=errdist_perms,pctl=5,dof=dof)

        out_hist_lim_l[vv2_cnt,:]=np.histogram(out_lims[2][0][:,0,1,0],normed=True,range=[0,1],bins=r_[ccs,[1]])[0]
        out_hist_lim_u[vv2_cnt,:]=np.histogram(out_lims[2][1][:,0,1,0],normed=True,range=[0,1],bins=r_[ccs,[1]])[0]
        
    return(array(out_hist_unshared),array(out_hist_shared_l),array(out_hist_shared_u),array(out_hist_lim_l),array(out_hist_lim_u))


def calc_noerr(ccs,cc,vvs_all,vv1,vv2):
    # calculate histograms of unshared, shared, and shared effects.
    # out_hist_unshared=zeros((len(ccs),len(vvs_all),len(vvs_all),len(ccs)))
    ooA=ones((2,2))
    ooB=ones((2,2))
        
    if isinstance(vv2,numbers.Number):
        vv2=[vv2]
    elif vv2==[]:
        vv2=vvs_all
        vv1=vvs_all

    if isinstance(vv1,numbers.Number):
        vv1=[vv1]
    if isinstance(cc,numbers.Number):
        cc=[cc]

    if len(cc) < len(vv2):
        cc=cc*len(vv2)
    elif len(vv2) < len(cc):
        vv2=vv2*len(cc)

    if len(cc) < len(vv1):
        cc=cc*len(vv1)
    elif len(vv1) < len(cc):
        vv1=vv1*len(cc)

    if len(vv2) < len(vv1):
        vv2=vv2*len(vv1)
    elif len(vv1) < len(vv2):
        vv1=vv1*len(vv2)
  
    out_unshared=zeros(len(vv2))
    out_shared_l=zeros(len(vv2))
    out_shared_u=zeros(len(vv2))
    out_lim_l=zeros(len(vv2))
    out_lim_u=zeros(len(vv2))
    for cnt in range(len(vv2)):
        
        ooA[[0,1],[1,0]]=ccs[cc[cnt]]
        ooA[[0,1],[0,1]]=1
        tmpA=FC(ooA,cov_flag=True)

        print('cc[cnt]:'+str(cc[cnt]))
        ooB[[0],[0]]=vvs_all[vv1[cnt]]
        ooB[[1],[1]]=vvs_all[vv2[cnt]]
        ooB[[0,1],[1,0]]=ccs[cc[cnt]]*(vvs_all[vv1[cnt]]**.5)*(vvs_all[vv2[cnt]]**.5)

        tmpB=FC(ooB,cov_flag=True)
        
        out_unshared[cnt]=corr_lims_unshared_mat(tmpA,tmpB,errdist=False)[0,0,1]
        out_shared=corr_lims_shared_mat(tmpA,tmpB,errdist=False)
        #out_hist_shared_l[cnt]
        out_shared_u[cnt]=out_shared[1][1][0,0,1]
        out_shared_l[cnt]=out_shared[1][0][0,0,1]

        out_lims=corr_lims_mat(tmpA,tmpB,errdist=False)
        #out_lim_u[cnt]=out_shared[0][1][0,0,1]
        #out_lim_l[cnt]=out_shared[0][0][0,0,1]

        out_lim_u[cnt]=out_lims[1][0,0,1]
        out_lim_l[cnt]=out_lims[0][0,0,1]


        #out_hist_lim_l[cnt,:]=np.histogram(out_lims[2][0][:,0,1,0],normed=True,range=[0,1],bins=r_[ccs,[1]])[0]
        #out_hist_lim_u[cnt,:]=np.histogram(out_lims[2][1][:,0,1,0],normed=True,range=[0,1],bins=r_[ccs,[1]])[0]
        
    return(array(out_unshared),array(out_shared_l),array(out_shared_u),array(out_lim_l),array(out_lim_u))

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
        return calc_hists(*a_b)

def runwithvv2(c):
    vv1s=range(len(vvs_all))
    args=itertools.izip(itertools.repeat(ccs),itertools.repeat(ccval),itertools.repeat(vvs_all),vv1s,itertools.repeat(vv1s),itertools.repeat(100),itertools.repeat(100))
    pool=Pool()
    out=pool.imap(func_star,args)
    pool.close()
    pool.join()
    return(out,pool)

def plot_fb(ccs,low,high,vvs=[],cmap=cm.gray,colorbar=True):
   
    fig=pl.gcf()
    pl.plot(ccs,ccs,color=[0.55,0.55,0.55],label='Test',linewidth=5)

    for a in arange(len(low)):
        srted=np.sort(c_[low[a],high[a]])
        pl.fill_between(ccs,srted[:,0],srted[:,1],color=cmap(a*256/len(low)))

    ax=pl.gca()

    if vvs != []:
        cc=ax.pcolor(array([[.2]]),cmap=cmap,visible=False,vmin=min(vvs),vmax=max(vvs))      
        if colorbar:
            cbar=pl.colorbar(cc,shrink=0.5,ticks=vvs,fraction=0.1)
            cbar.set_label('Proportional change in variance \n in region one in second condition.')

    ax.set_ylim(-1,1)
    
    ax.set_xlabel('Correlation in condition 1')
    ax.set_ylabel('Correlation in condition 2')

    # ax.set_title('Change in correlation after adding/removing shared signal')

def plot_diffs(A,B,thr):

    ccstatsmat = ml_funcs.flattenall((stats.ttest_rel(rtoz(A.get_corrs()),rtoz(B.get_corrs()))[0]))

    lims=cf.corr_lims_mat(A,B)
    unshared=cf.corr_lims_unshared_mat(A,B)
    shared=cf.corr_lims_shared_mat(A,B)

    print(A)

def flatten_tcs(A):
    tcs_dm=pl.demean(A.tcs,2)
    shp=tcs_dm.shape
    out=FC(reshape(tcs_dm.swapaxes(0,1),[shp[1],-1]))
    # subjtract dofs from demeaning 
    out.dof=out.dof-shp[0]
    return(out)

def dr_loader(dir,prefix='dr_stage1',subj_ids=False,ROIs=[],subjorder=True):


    dr_files=sort(glob.glob(prefix+'_subject?????.txt'))    
    if not (subj_ids==''):
        dr_files=dr_files[subj_ids]

    subjs=len(dr_files)
    maskflag=False

    tmp=atleast_2d(loadtxt(dr_files[0]).T)
    if os.path.isfile(dir+'/goodnodes.txt'):
        goodnodes=loadtxt(dir+'/goodnodes.txt').astype(int)
    else:
        goodnodes=arange(tmp.shape[-2])

    tmp=tmp[goodnodes,:]
    shp=tmp.shape

    datamat=zeros((subjs,shp[0],shp[1]))
    mask=zeros((subjs,shp[0],shp[1]))
    cnt=0
    for a in arange(subjs):
            tmpdata=atleast_2d(loadtxt(dr_files[cnt]).T)[goodnodes,:]
            if tmpdata.shape[1] < shp[1]:
                mask[a,:,tmpdata.shape[1]:]=1
                maskflag=True
            datamat[a,:,:tmpdata.shape[1]]=tmpdata
            cnt+=1
    if maskflag:
        datamat=ma.masked_array(datamat,mask)
    
    if not (ROIs==''):
        datamat=datamat[:,ROIs,:]

    if ( not subjorder ):
        datamat=swapaxes(datamat,0,1)

    A=FC(datamat)

    return A


