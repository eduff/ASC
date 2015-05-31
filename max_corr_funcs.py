from numpy import *
import numpy as np
import numpy.linalg as la
import matplotlib.pylab as pl
import glob,os
import nibabel as nb
import scipy.stats

def corr_lims(std_xa,std_xb,std_ya,std_yb,rho_a):

           cx=cmaxmin(std_xa,std_xb,std_ya,rho_a)[0]
           cx_neg=c_neg_maxmin(std_xa,std_xb,std_ya,rho_a)[0]

           cy=cmaxmin(std_ya,std_yb,std_xa,rho_a)[0]
           cy_neg=c_neg_maxmin(std_ya,std_yb,std_xa,rho_a)[0]

           corr_maxa=( std_xa*std_ya*rho_a*(1+cx*cy) + cx*std_ya**2 + cy * std_xa**2) / (std_xb*std_yb)
           corr_maxa_neg=( std_xa*std_ya*rho_a*(1+cx_neg*cy_neg) + cx_neg*std_ya**2 + cy_neg * std_xa**2) / (std_xb*std_yb)
           corr_min=( std_xa*std_ya*rho_a*(1+cx*cy) + cx*std_ya**2 + cy * std_xa**2) / (std_xb*std_yb)
           return([corr_maxa_neg, corr_maxa],(rho_b> corr_maxa_neg) != (rho_b> corr_maxa))

## def corr_lims_mat(A,B,pcorrs=False):
def corr_lims_mat(A,B,errdist=False,pcorrs=False,errdist_perms=1000,dof=[],pctl=5):

    if dof==[]:
        if A.tcs==[]:
            raise ValueError('Need time courses or dof')
        else:
            dof=A.tcs.shape[-1]-1
       
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

    corr_maxa=( Astdm*Astdmt*A.corrs*(1+cx*cy) + cx*Astdmt**2 + cy * Astdm**2) / (Bstdm * Bstdmt)
    corr_maxa_neg=( Astdm*Astdmt*A.corrs*(1+cx_neg*cy_neg) + cx_neg*Astdmt**2 + cy_neg * Astdm**2) / (Bstdm * Bstdmt)
    corr_min=( Astdm*Astdmt*A.corrs*(1+cx*cy) + cx*Astdmt**2 + cy * Astdm**2) / (Bstdm * Bstdmt)

    A_sim=[]
    B_sim=[]
 
    corr_maxa_err=[]
    corr_maxa_neg_err=[]
    pctls=[]

    if errdist:
        shp=covsA.shape

        corr_maxa_neg_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        corr_maxa_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))

        for a in arange(shp[0]):
            whA=scipy.stats.wishart(dof,covsA[a,:,:])
            whB=scipy.stats.wishart(dof,covsB[a,:,:])
            
            covsA_sim=whA.rvs(errdist_perms)/(dof)
            covsB_sim=whB.rvs(errdist_perms)/(dof)

            ppA=zeros((1,errdist_perms))
            ppB=zeros((1,errdist_perms))
            
            whA=[]
            whB=[]

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

                A_sim=FC(whA[els[b]].rvs(),cov_flag=True)
                B_sim=FC(whB[els[b]].rvs(),cov_flag=True)

                tmp,_,_ = corr_lims_mat(A_sim,B_sim,dof=dof) 
                corr_maxa_neg_err[:,a,:,:]=tmp[0]
                corr_maxa_err[:,a,:,:]=tmp[1]
        corr_maxa_err[abs(corr_maxa_err)>1]=sign(corr_maxa_err[abs(corr_maxa_err)>1]) 
        pctl_out = [percentile(corr_maxa_err,pctl,0),percentile(corr_maxa_err,100-pctl, 0)]

        corr_maxa_neg_err[abs(corr_maxa_neg_err)>1]=sign(corr_maxa_neg_err[abs(corr_maxa_neg_err)>1]) 
        pctl_out_neg = [percentile(corr_maxa_neg_err,pctl,0),percentile(corr_maxa_neg_err,100-pctl, 0)]
        
        pctls = (Bcorrs> minimum(pctl_out_neg[0] , pctl_out_neg[1])) != (Bcorrs> maximum(pctl_out[0] ,  pctl_out[1]))
    ## return(unshared,unshared_lims_err,pctl_out)

    #return([corr_maxa_neg, corr_maxa],[corr_maxa_neg_err, corr_maxa_err],[pctl_out_neg,pctl_out])
    #return([corr_maxa_neg, corr_maxa],[corr_maxa_neg_err, corr_maxa_err],[pctl_out_neg,pctl_out])

    #return([corr_maxa_neg, corr_maxa], pctls,[corr_maxa_neg_err, corr_maxa_err])

    return([corr_maxa_neg, corr_maxa],pctls,[corr_maxa_neg_err, corr_maxa_err])

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
    def __init__(self,tcs,cov_flag=False):
    
        if cov_flag==True:
            self.tcs=[]
            if tcs.ndim==2:
                tcs=(atleast_3d(tcs).transpose(2,0,1))
            covs=tcs
        else:
            if tcs.ndim==2:
                tcs=(atleast_3d(tcs).transpose(2,0,1))

            self.tcs = tcs
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
                pinvA=linalg.pinv(self.corrs[a,:,:])
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

# calculate noise related change 
def corr_lims_unshared(std_x,std_xb,std_y,std_yb,rho_xy):

    corr_unshared= array((std_x*std_y*rho_xy) / (std_xb*std_yb))
    
    corr_unshared[abs(corr_unshared) > 1]=sign(corr_unshared[abs(corr_unshared) > 1])

    return corr_unshared

def corr_lims_unshared_mat(A,B,errdist=False,pcorrs=False,errdist_perms=1000,dof=[],pctl=5):

    if dof==[]:
        if A.tcs==[]:
            raise ValueError('Need time courses or dof')
        else:
            dof=A.tcs.shape[-1]-1
        
    covsA=A.get_covs(pcorrs=pcorrs)
    covsB=B.get_covs(pcorrs=pcorrs)
    Bcorrs=B.get_corrs(pcorrs=pcorrs)   
    Bstds=B.get_stds_m(pcorrs=pcorrs)
    Bstds_t=B.get_stds_m_t(pcorrs=pcorrs)

    unshared = covsA / (Bstds*Bstds_t)
    unshared[unshared>1]=1 
    unshared[abs(unshared)>1]=sign(unshared[abs(unshared)>1]) 
    
    # unshared = calc_rho_unshared(A,B)
    #corr_unshared[abs(corr_unshared) > 1]=sign(corr_unshared[abs(corr_unshared) > 1])

    A_sim=[]
    B_sim=[]
 
    unshared_lims_err=[]
    pctls=[]
    ppA=[]

    if errdist:
        shp=covsA.shape

        unshared_lims_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))

        for a in arange(shp[0]):
            # generate init samples
            whA=scipy.stats.wishart(10,covsA[a,:,:])
            whB=scipy.stats.wishart(10,covsB[a,:,:])
            
            covsA_sim=whA.rvs(errdist_perms)/(10)
            covsB_sim=whB.rvs(errdist_perms)/(10)

            for a in arange(errdir_perms):

            # A_sim=FC(covsA_sim,cov_flag=True)
            # B_sim=FC(covsB_sim,cov_flag=True)

            ppA=zeros((1,errdist_perms))
            ppB=zeros((1,errdist_perms))
            
            whA=[]
            whB=[]
            
            cnt=0

            while cnt < errdist_perms:

                rr = scipy.stats.uniform(0,1).rvs() 
                rint=np.random.randint(errdist_perms)
                whA = (scipy.stats.wishart(dof,covsA_sim[rint,:,:]))

                if whA.pdf(covsA[a,:,:]*dof) > rr:

                    whB=(scipy.stats.wishart(dof,covsB_sim[rint,:,:]))

                    pp=whB.pdf(covsB[a,:,:]*dof)

                    while  pp > rr:

                        whB=(scipy.stats.wishart(dof,covsB_sim[b,:,:]))
                        pp=whB.pdf(covsB[a,:,:]*dof)
                        rr = scipy.stats.uniform(0,1).rvs() 
                        rint=np.random.randint(errdist_perms)

                    A_sim=FC(whA.rvs(),cov_flag=True)
                    B_sim=FC(whB.rvs(),cov_flag=True)

                    unshared_lims_err[cnt,a,:,:],_,_ = corr_lims_unshared_mat(A_sim,B_sim,dof=dof)
                    cnt+=1
                    display(cnt)

            # ppA=ppA/sum(ppA)
            # ppB=ppB/sum(ppB)
            # select var
            #rand_els = scipy.stats.uniform(0,1).rvs(errdist_perms) 
            #elsA=sort(searchsorted(ppA_cul.flatten(),rand_els)) 
            #elsB=sort(searchsorted(ppB_cul.flatten(),rand_els)) 

            #for b in arange(err_dist_perms):
        unshared_lims_err[abs(unshared_lims_err)>1]=sign(unshared_lims_err[abs(unshared_lims_err)>1]) 
        pctl_out = [percentile(unshared_lims_err,pctl,0),percentile(unshared_lims_err,100-pctl,0)]
        pctls=(Bcorrs> pctl_out[0]) != (Bcorrs> pctl_out[1])

    return(unshared, pctls ,ppA)

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

# calc common max range of change in signal
def  corr_lims_common(std_x,std_xb,std_y,std_yb,rho_xy,sim_sampling=40.0):
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

#def corr_lims_common_mat(A,B,pcorrs=False):
def corr_lims_common_mat(A,B,errdist=False,pcorrs=False,errdist_perms=300,dof=[],pctl=5,sim_sampling=100):

    if dof==[]:
        if A.tcs==[]:
            raise ValueError('Need time courses or dof')
        else:
            dof=A.tcs.shape[-1]-1
     
    # calculate limits if change in A,B due to same source
    
    shp=A.covs.shape
    corr2=zeros((shp[0],shp[1],shp[2],sim_sampling,sim_sampling))
    aa=(arange(-1,1.000,2.0/sim_sampling))

    covsA=A.get_covs(pcorrs=pcorrs)
    covsB=B.get_covs(pcorrs=pcorrs)

    Acorrs=A.get_corrs(pcorrs=pcorrs)
    Astds=A.get_stds_m(pcorrs=pcorrs)
    Astds_t=A.get_stds_m_t(pcorrs=pcorrs)

    Bcorrs=B.get_corrs(pcorrs=pcorrs)
    Bstds=B.get_stds_m(pcorrs=pcorrs)
    Bstds_t=B.get_stds_m_t(pcorrs=pcorrs)

    inds = tile(arange(sim_sampling),(shp[0],shp[1],shp[1],1))
    
    for aaa in arange(sim_sampling/2.0)+sim_sampling/2.0:
        rho_xb=aa[aaa]
        (rho_yb_l,rho_yb_u)=calc_pbv(Acorrs,rho_xb)

        for rho_yb in arange(0,1-0.0001,2.0/sim_sampling):
            bbb = rho_yb*sim_sampling/2.0+sim_sampling/2.0
            wx = calc_weight(Astds,Bstds,rho_xb)[0]
            wy = calc_weight(Astds,Bstds,rho_yb)[0]
            corr2[:,:,:,aaa,bbb] = (Astds*Astds*Acorrs + wx*Astds*rho_xb + wy*Astds*rho_yb + wx*wy )/(Bstds*Bstds)

        inds_u=(floor(rho_yb_u*sim_sampling/2.0)+sim_sampling/2.0).astype(int)
        inds_u=tile(inds_u,(sim_sampling,1,1,1)).transpose(1,2,3,0)
        corr2[:,:,:,aaa,:][inds_u<inds]=nan

        inds_l=(floor(rho_yb_l*sim_sampling/2.0)+sim_sampling/2.0).astype(int)
        inds_l=tile(inds_l,(sim_sampling,1,1,1)).transpose(1,2,3,0)
        corr2[:,:,:,aaa,:][inds_l>inds]=nan

    corr2[corr2==0]=nan
    corr2[corr2<-1]=-1
    #return(corr2,rho_yb_l,rho_yb_u)
    #return([nanmin(nanmin(corr2,4),3), nanmax(nanmax(corr2,4),3)],)

    corrmin=nanmin(nanmin(corr2,4),3) 
    corrmax=nanmax(nanmax(corr2,4),3)

    A_sim=[]
    B_sim=[]
 
    unshared_lims_err=[]
    pctls=[]
    corr_min_err=[]
    corr_max_err=[]

    if errdist:
        shp=covsA.shape

        corr_min_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))
        corr_max_err=zeros((errdist_perms,shp[0],shp[1],shp[1]))

        for a in arange(shp[0]):
            # invWishhart flat prior
            # rpriors=rprior(r=shp[1]*3,n=errdist_perms,M=)

            #for b in arange(errdist_perms):
            #    whA=scipy.statswishart(

            # for b in rprior(n=100,M=covsA[
            
            # generate init samples
            whA=scipy.stats.wishart(dof,covsA[a,:,:])
            whB=scipy.stats.wishart(dof,covsB[a,:,:])
            covsA_sim=whA.rvs(errdist_perms)/(dof)
            covsB_sim=whB.rvs(errdist_perms)/(dof)
            ppA=zeros((1,errdist_perms))
            ppB=zeros((1,errdist_perms))
            
            whA=[]
            whB=[]

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

                A_sim=FC(whA[els[b]].rvs(),cov_flag=True)
                B_sim=FC(whB[els[b]].rvs(),cov_flag=True)

                tmp,_,_ = corr_lims_common_mat(A_sim,B_sim,dof=dof,sim_sampling=40.0) 
                corr_min_err[b,a,:,:]=tmp[0]
                corr_max_err[b,a,:,:]=tmp[1]

        corr_max_err[abs(corr_max_err)>1]=sign(corr_max_err[abs(corr_max_err)>1]) 
        pctl_out = [percentile(corr_max_err,pctl,0),percentile(corr_max_err,100-pctl, 0)]

        corr_min_err[abs(corr_min_err)>1]=sign(corr_min_err[abs(corr_min_err)>1]) 
        pctl_out_min = [percentile(corr_min_err,pctl,0),percentile(corr_min_err,100-pctl, 0)]
        pctls = (Bcorrs> minimum(pctl_out_min[0] , pctl_out_min[1])) != (Bcorrs> maximum(pctl_out[0] ,  pctl_out[1]))
    #return([corrmin, corrmax],[corr_min_erre corr_max_err],[pctl_out_min,pctl_out])
    #return([corrmin, corrmax], (Bcorrs> pctl_out_min) != (Bcorrs> pctl_out),[corr_min_err, corr_max_err])

    return([corrmin,corrmax],pctls,[corr_min_err, corr_max_err])

def plot_class(A,B,thresh=1.96):
   
    corr_lims,corr_lims_TF = corr_lims_mat(A,B)
    common,common_TF,_ = corr_lims_common_mat(A,B)
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
