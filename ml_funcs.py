import sys,os, getopt,itertools
import re,glob
import nibabel
from scipy.stats import wilcoxon
from scipy.linalg import pinv
from subprocess import call
import numpy as np
import pylab
import pickle
import sklearn
from sklearn import covariance, svm, grid_search
from functools import reduce

skcv=sklearn.cross_validation

def dr_loader(dir,conds=1,prefix='dr_stage1',subjs=[],subjorder=True):

    dr_files=sort(glob.glob(prefix+'_subject?????.txt'))    
    #if subjs != []:
    #    dr_files=dr_files[subj_ids]

    subjs=len(dr_files)/conds
    maskflag=False

    if ( not subjorder ):
        tmp=subjs
        subjs=conds
        conds=tmp

    tmp=atleast_2d(loadtxt(dr_files[0]).T)
    if os.path.isfile(dir+'/goodnodes.txt'):
        goodnodes=loadtxt(dir+'/goodnodes.txt').astype(int)
    else:
        goodnodes=arange(tmp.shape[-2])

    tmp=tmp[goodnodes,:]
    shp=tmp.shape

    datamat=zeros((subjs,conds,shp[0],shp[1]))
    mask=zeros((subjs,conds,shp[0],shp[1]))
    cnt=0
    for a in arange(subjs):
        for b in arange(conds):
            tmpdata=atleast_2d(loadtxt(dr_files[cnt]).T)[goodnodes,:]
            if tmpdata.shape[1] < shp[1]:
                mask[a,b,:,tmpdata.shape[1]:]=1
                maskflag=True
            datamat[a,b,:,:tmpdata.shape[1]]=tmpdata
            cnt+=1
    if maskflag:
        datamat=ma.masked_array(datamat,mask)

    if ( not subjorder ):
        datamat=swapaxes(datamat,0,1)
    return datamat

def load_con(confile):
    cnt=0
    ff=file(confile) 

    for line in ff.readlines():
        if '/Matrix' in line:
            break
        cnt+=1
    ff.close()
    condata=genfromtxt(confile,skip_header=cnt+1) 

    return condata

def nCk(n,k): 
    return int( reduce(mul, (Fraction(n-i, i+1) for i in range(k)), 1) )

def offdiag(x):
    b=eye(len(x))
    return x.flatten()[find(b!=1)]

def rtoz(x):
    return(0.5*(np.log(1+x) - np.log(1-x)))

def crossfunc(x,func):
    xx=np.tile(x,(len(x),1))
    yy=np.tile(x,(len(x),1)).T
    out=func(xx,yy)
    return out

def triu_all(x):
    return x.flatten()[find(triu(ones(x.shape),1))]

def tril_all(x):
    return x.flatten()[find(tril(ones(x.shape),1))]

def diag_all(x):
    triuu= x.flatten()[find(triu(ones(x.shape)))]
    trill= x.flatten()[find(tril(ones(x.shape)))]

    return intersect1d(triuu,trill)

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


def flattenall_inds(x,dims):
    if len(dims)==3:
        dims=dims[1:]
    out = where(triu(ones(dims),1))

    return(array([out[0][x],out[1][x]]))
    
def maxratio(xx,yy):
    return(maximum(abs(xx/yy),abs(yy/xx)))

def minratio(xx,yy):
    return(minimum(abs(xx/yy),abs(yy/xx)))


def transp21(xx):
    dims=ndim(xx)
    axx=arange(dims-2)
    return transpose(xx,tuple(arange(ndim(xx)-2)) + tuple((dims-1,dims-2)))

def gen_cv_scheme(data,design,con,conno,groups=[]):
    
    contr=con[conno,:]
    contr=dot(contr,design.T)
    evs=[]
    labels=[]
    for ll in unique(contr[contr!=0]):
        evs=r_[evs,where(contr==ll)[0]]
        labels=r_[labels,contr[contr==ll]]

    evs=evs.astype(int)

    if groups==[]:
        groups=ones((len(evs,)))
    else:
        groups=groups[evs]

    return (data[evs,:],labels,groups,evs)

def corr_calc(data, covtype,params=[[]],prec_flag=False,cc_flag=True,savefile=[],paramExt=[],corrmat={}):

    if isscalar(params):
            params=[params]
    elif params==[[]]:
        params=['noparam']

    if paramExt==[]:
        paramExt=''

    corrconds=['Corr','Cov','GraphLasso','GraphLassoCV','fast_mcd','LedoitWolf','OAS','TIK']

    if ndim(data)>2:
        shape_o=data.shape
        data=reshape(data,((prod(shape_o[:-2])),shape_o[-2],shape_o[-1]))
        outcorr=zeros( (prod(shape_o[:-2]),shape_o[-2],shape_o[-2]))

        if (prec_flag==True): 
            outprec=zeros( (prod(shape_o[:-2]),shape_o[-2],shape_o[-2]))
    else:
        shape_o=data.shape
        data=reshape(data,(1,)+data.shape)
        outcorr = zeros( (prod(shape_o[:-2]),shape_o[-2],shape_o[-2]))

        if (prec_flag==True): 
            outprec = zeros( (prod(shape_o[:-2]),shape_o[-2],shape_o[-2]))

    if corrmat=={}:
        outcorr_l={}
        outprec_l={}
    else:
        if prec_flag:
            outprec_l=corrmat
        else:
            outcorr_l=corrmat
    
    paramNames=[]

    for param in params:

        if param==[]:
            paramName=paramExt
            paramNames.append(paramName)
        else:
            paramName=str(param)+paramExt
            paramNames.append(paramName)

        for els in arange(shape(data)[0]):
            indata=data[els,:,:]

            if type(indata) == ma.masked_array:
                indata=array(indata[:,find(~indata.mask[0,:])])

            if (cc_flag==True) & (prec_flag==True): 
                (cc,prec)=corr_calc_base(indata,covtype,param,prec_flag=prec_flag,cc_flag=cc_flag)
                outcorr[els,:]=cc
                outprec[els,:]=prec
            elif (cc_flag==True):
                outcorr[els,:]=corr_calc_base(indata,covtype,param,prec_flag=prec_flag,cc_flag=cc_flag)
            else:
                outprec[els,:]=corr_calc_base(indata,covtype,param,prec_flag=prec_flag,cc_flag=cc_flag)


        if (cc_flag==True):
            outcorr_tmp=reshape(outcorr, tuple(shape_o[:-1])+(shape_o[-2],))
            #if isscalar(param):
            outcorr_l[paramName] = outcorr_tmp.copy()
            #else:
            #outcorr_l=outcorr_tmp.copy()

        if (prec_flag==True): 
            outprec_tmp=reshape(outprec, tuple(shape_o[:-1])+(shape_o[-2],))

            #if isscalar(param):
            outprec_l[paramName] = outprec_tmp.copy()
            #else:
            #outprec_l=outprec_tmp.copy()

    if cc_flag==True:
        if prec_flag==False: 
            corrmat=outcorr_l
        else:
            corrmat['cc']=outcorr_l
            corrmat['prec']=outprec_l
    else: 
        corrmat=outprec_l
        
    if savefile != []:
        if cc_flag==True: 
            f=file(covtype+'_cc.dat','w')
            pickle.dump(outcorr_l,f)
            f.close()
        if prec_flag==True:
            f=file(covtype+'_prec.dat','w')
            pickle.dump(outprec_l,f)
            f.close()

    return(corrmat,params,array(paramNames))

def corr_calc_base(data, covtype,param=[],prec_flag=False,cc_flag=True):

    if covtype == 'Corr':
        cc=corrcoef(data)

        if prec_flag==True:
            prec = pinv(cc)

    elif covtype == 'Cov':

        cc=cov(data)
        if prec_flag==True:
            prec = pinv(cc)

    elif covtype == 'EmpCov':

        fit_cc=covariance.EmpiricalCovariance().fit(data.T)
        prec  = fit_cc.precision_
        cc  = fit_cc.covariance_

    elif covtype=='GraphLasso':

        fit_cc=covariance.GraphLasso(param,max_iter=100).fit(data.T)
        prec  = fit_cc.precision_
        cc  = fit_cc.covariance_

    elif covtype == 'GraphLassoCV':

        fit_cc=covariance.GraphLassoCV(max_iter=100).fit(data.T)
        prec  = fit_cc.precision_
        cc  = fit_cc.covariance_

    elif covtype == 'fast_mcd':

        fit_cc=covariance.fast_mcd(data.T)[1]
        if prec_flag==True:
            prec = pinv(cc)

    elif covtype == 'LedoitWolf':
        fit_cc=covariance.LedoitWolf().fit(data.T)
        prec  = fit_cc.precision_
        cc  = fit_cc.covariance_

    elif covtype == 'OAS':
        fit_cc=covariance.LedoitWolf().fit(data.T)
        prec  = fit_cc.precision_
        cc  = fit_cc.covariance_

    elif covtype == 'TIK':
        if param == []:
            param = 0.1

        fit_cc=covariance.ShrunkCovariance(shrinkage=param).fit(data.T)

        prec  = fit_cc.precision_

        cc  = fit_cc.covariance_

    if prec_flag == True: 
        if cc_flag == True:
            out = (cc,prec)
        else: 
            out = prec
    else:
        out = cc

    #if isinstance(out,tuple)

    return out

def predict_DR(covtype,labels_full=[],groups_full=[],prefix='dr_stage1',params=[[]],subjs=1,ccsetting=[],prec_flag=False,contrs=[[0,1]],paired=True,subjorder=True,dirs=['.'],dirnames=['noparam']):
    (datamat,params,paramNames)=dr_loader_helper(covtype,labels_full,groups_full,prefix,params,subjs,ccsetting,prec_flag,contrs,paired,subjorder,dirs,dirnames)
    cnt=0
    accuracies={}
    if params==[[]]:
        if paramNames != []:
            params=paramNames
        else:
            params=['noparam']
 
    if isscalar(params):
        params=[params]
    
    if type(labels_full)==str:
        labels_full=loadtxt(labels_full)
    if type(groups_full)==str:
        groups_full=loadtxt(groups_full)

    if (not type(datamat)==dict):
        tmp={}
        tmp[params[0]]=datamat
        datamat=tmp

    for con in contrs:
        accuracies[cnt]={}

        for param in paramNames:

            data=datamat[param]

            shp=data.shape

            if shp[1]==1:
                inds=((labels_full==con[0])|(labels_full==con[1]))
                data=flattenall(data[inds,:,:,:])
                groups=groups_full[inds]
                labels=labels_full[inds]
            else:
                data=flattenall(data[:,con,:,:].reshape((shp[0]*2,shp[2],shp[2])))
                
            clf=svm.SVC(kernel='linear', C=1)
            
            if labels_full==[]:
                labels=tile(arange(2),(subjs,))

            if paired:

                if groups_full==[]:
                    groups=arange(shp[0]*2)/2

                cv=sklearn.cross_validation.LeavePLabelOut
                accuracies[cnt][param]=mean(skcv.cross_val_score(clf,groupwisedemean(data,groups),labels,cv=cv(groups,1)))
            else:
                cv=sklearn.cross_validation.StratifiedKFold
                n_folds=  (bincount(labels.astype(int)))
                n_folds= min(n_folds[n_folds>0])
                accuracies[cnt][param]=mean(skcv.cross_val_score(clf,groupwisedemean(data,groups),labels,cv=cv(  labels,n_folds=n_folds)))

        cnt+=1

    return accuracies

def predict_DR_cv(covtype,labels_full=[],groups_full=[],prefix='dr_stage1',params=[[]],subjs=1,ccsetting=[],prec_flag=False,contrs=[[0,1]],paired=True,subjorder=True,dirs=['.'],dirnames=[[]],groups_demean=[]):

    # load all datamats (across ICAs/params)
    (datamat,params,paramNames)=dr_loader_helper(covtype,labels_full,groups_full,prefix,params,subjs,ccsetting,prec_flag,contrs,paired,subjorder,dirs=dirs,dirnames=dirnames)

    data=datamat[paramNames[0]]
    shp=data.shape
    cnt=0
    accuracies={}

    
    if params==[[]]:
        if paramNames != []:
            params=paramNames
        else:
            params=['noparam']
    
    if isscalar(params):
        params=[params]
    
    #load labels,groups
    if type(labels_full)==str:
        labels_full=loadtxt(labels_full)
    if type(groups_full)==str:
        groups_full=loadtxt(groups_full)
    if groups_demean != []:
        groups_demean=loadtxt(groups_demean)
        for a in datamat.keys():
            datamat[a],groups_merge,labels_merge,groups_demean_merge=mergelikelabels(flattenall(datamat[a]),groups_full,labels_full,groups_demean)
            datamat[a]=groupwisedemean(datamat[a],groups_demean_merge)
        groups_full=groups_merge
        labels_full=labels_merge
        data=datamat[paramNames[0]]
        shp=data.shape

    paramNames=paramNames[argsort(params)]
    params=sort(params)

    subj_list=arange(shp[0])
    #preds = zeros((len(contrs),(shp[0]*2)))
    preds={}

    cnt=0
    for con in contrs:

        if labels_full!=[]:
            inds=((labels_full==con[0])|(labels_full==con[1]))
            labels=labels_full[inds]
            groups=groups_full[inds]


        preds[cnt]=zeros((len(labels),))
        cv_accs=zeros((len(params,)))

        if paired:
            cv=sklearn.cross_validation.LeaveOneLabelOut
            cvv=cv(groups,1)
        else:
            cv=sklearn.cross_validation.LeaveOneLabelOut
            cvv=cv(groups,1)
            #cv=sklearn.cross_validation.LeaveOneOut
            #n_folds= (bincount(labels.astype(int)))
            #n_folds= min(n_folds[n_folds>0]-2)
            #cvv=cv(len(labels))

        svr=svm.SVC(class_weight='auto')
        parameters = {'kernel':('linear', 'rbf'), 'C':logspace(0.1,10,4)}
        clf = grid_search.GridSearchCV(svr,parameters)
        clf=svm.SVC(kernel='linear', C=1)

        for train,test in cvv:

            for param in arange(len(params)):

                data=datamat[paramNames[param]]
                shp=data.shape
                lt=len(train)
                data=flattenall(data[inds,:])
                data_merge,groups_merge,labels_merge=mergelikelabels(data,groups,labels)

                if (len(shp)==2):
                    datatrain=(data_merge[train,:])  #.reshape((lt*2,shp[2],shp[2]))
                elif shp[1]==1:
                    datatrain=flattenall(data_merge[train,:])  #.reshape((lt*2,shp[2],shp[2]))
                else:
                    data=flattenall(data_merge[:,con,:,:].reshape((shp[0]*2,shp[2],shp[2])))

                if groups_full==[]:
                    groups=(arange(lt*2)/2)

                if labels_full==[]:
                    labels=tile(arange(2),(lt,))

                groups_test=zeros((2,))
                #datatrain=flattenall(data[:,label=con,:,:][train,:].reshape((lt*2,shp[2],shp[2])))
                #datatest=flattenall(data[:,con,:,:][test,:].reshape((2,shp[2],shp[2])))
           
                #datatrain,groupsout,labelsout=mergelikelabels(datatrain,groups[train],labels[train])
                groupsout=groups[train]
                labelsout=labels[train]
                if paired:
                    cv_accs[param]=mean(skcv.cross_val_score(clf,groupwisedemean(datatrain,groupsout),labelsout,cv=cv(groupsout,1)))
                else:
                    cv_accs[param]=mean(skcv.cross_val_score(clf,datatrain,labelsout,cv=cv(groupsout,1)))
                    #cv_accs[param]=mean(skcv.cross_val_score(clf,groupwisedemean(datatrain,groups[train]),labels[train],cv=cv(len(labels[train]))))

            bestparam=where(cv_accs==max(cv_accs))[0][-1]
            data=datamat[paramNames[bestparam]]
            shp=data.shape
            lt=len(train)
            data=flattenall(data[inds,:])
            data_merge,groups_merge,labels_merge=mergelikelabels(data,groups,labels)
            if (len(shp)==2):
                datatrain=(data_merge[train,:])  #.reshape((lt*2,shp[2],shp[2]))
                datatest=(data_merge[test,:])  #.reshape((lt*2,shp[2],shp[2]))
            elif shp[1]==1:
                datatrain=flattenall(data_merge[train,:])  #.reshape((lt*2,shp[2],shp[2]))
                datatest=flattenall(data_merge[test,:])  #.reshape((lt*2,shp[2],shp[2]))
            else:
                data=flattenall(data_merge[:,con,:,:].reshape((shp[0]*2,shp[2],shp[2])))
        
            if paired:
                clf.fit(groupwisedemean(datatrain,groups[train]),labels[train])
                preds[cnt][test]=clf.predict(groupwisedemean(datatest,groups[test]))
            else:
                clf.fit(datatrain,labels[train])
                preds[cnt][test]=clf.predict(datatest)
        
        accuracies[cnt] = mean(labels==preds[cnt])

        cnt=cnt+1
    return accuracies,preds


def mergelikelabels(datamat,groups,labels,secondgroup=[]):
    datamatout=zeros((0,datamat.shape[1]))
    labelsout=zeros((0,))
    groupsout=zeros((0,))
    secondgroupout=zeros((0,))

    for grp in unique(groups):
        inds=(grp==groups)
        for lab in unique(labels):
            likelabels=(grp==groups)&(lab==labels)
            
            if sum(likelabels)>1:
                datamatout=r_[datamatout,atleast_2d(mean(datamat[likelabels,:],0))]
            elif sum(likelabels)==0:
                continue
            else:
                datamatout=r_[datamatout,atleast_2d(datamat[likelabels,:])]
            labelsout=r_[labelsout,labels[likelabels][0]]
            groupsout=r_[groupsout,groups[likelabels][0]]
            if secondgroup != []:
                secondgroupout=r_[secondgroupout,secondgroup[likelabels][0]]
    if secondgroup != []:
        return datamatout,groupsout,labelsout,secondgroupout
    else:
        return datamatout,groupsout,labelsout

def groupwisedemean(datamat,groups):
    for grp in unique(groups):
        datamat[groups==grp,:]=datamat[groups==grp,:] - mean(datamat[groups==grp,:],0)

    return datamat

def dr_loader_helper(covtype,ev='design.mat',con='design.con',prefix='dr_stage1',params=[[]],subjs=1,ccsetting=[],prec_flag=False,contrs=[[0,1]],paired=True,subjorder=True,dirs=['.'],dirnames=[[]]):

    origdir=os.getcwd()
    corrmat={}
    paramsOrig=params
    paramsNew=array([])
    paramNamesNew=array([])

    for dd in arange(len(dirs)): 

        os.chdir(dirs[dd])
        print(dd)

        if prec_flag:
            filename = covtype + '_prec.dat'
        else:
            filename = covtype + '_cc.dat'

        if ( not os.path.isfile(filename)):
            print('Generating .dat file')
            datamat=dr_loader('.',prefix=prefix,subjs=subjs,subjorder=subjorder)
            (corrmat,params,paramNames)=corr_calc(datamat,covtype,prec_flag=prec_flag,cc_flag=(not prec_flag),params=paramsOrig,savefile=True,paramExt=dirnames[dd],corrmat=corrmat)

        else:
            f=file(filename)
            corrmat=pickle.load(f)

            # Recalculate if different params
            if type(corrmat) == dict:
                (params,paramNames)=make_paramNames(paramsOrig,dirnames)
                if len(setdiff1d(paramNames,corrmat.keys())) != 0:
                    print('Regenerating .dat file')
                    datamat=dr_loader('.',prefix=prefix,subjs=subjs,subjorder=subjorder)
                    (corrmat,params,paramNames)=corr_calc(datamat,covtype,prec_flag=prec_flag,cc_flag=(not prec_flag),params=paramsOrig,savefile=True,paramExt=dirnames[dd],corrmat=corrmat)


        paramsNew=r_[paramsNew,params]
        paramNamesNew=r_[paramNamesNew,paramNames]

        os.chdir(origdir)
    return(corrmat,paramsNew,paramNamesNew)

def make_paramNames(params,dirnames):
    paramout=[]
    out=[]
    if dirnames==[[]]:
        return(array(params),array(params))
    if params==[[]]:
        return(array(dirnames),array(dirnames))
    for aa in itertools.product(params,dirnames):
        print(aa)
        paramout.append(aa[0])
        out.append(str(aa[0])+'_'+aa[1]+'c')
    return(array(paramout),array(out))


#def std_ext(x,n):
#    return (tile(std(rnds,1),(230,1)).T))
