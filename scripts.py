
import scripts2
import ml_funcs
import scipy.signal as signal
import glob,re
import scipy.stats as stats
from matplotlib.pylab import find
import itertools

#
t21=ml_funcs.transp21
rtoz=ml_funcs.rtoz

from operator import mul
from fractions import Fraction

#conds=5
#contrs=[[0,1]]
#contrs=[] # [[0,4]]

# datamat load/preprocess

datamat=ml_funcs.dr_loader('.',subjs=conds,prefix=prefix)

if not('ROI_order' in locals()):
    ROI_order=arange(datamat[0].shape[1])
    
# reorganise
datamat=datamat[:,:,ROI_order,:]

# bp filter option
if filt:
    (b,a)=signal.filter_design.butter(5,[lf*TR,uf*TR],'bandpass')
    datamat=signal.filtfilt(b,a,datamat)

# plotDR(dirname,covtype='Corr',
#out=nitime.algorithms.multi_taper_psd(datamat,TR)

#(datamat,corrmat,varmat)=ml_funcs.dr_loader('.',types=conds,prefix='dr_stage1_nodemean')
#(datamat,corrmat,varmat)=ml_funcs.dr_loader('.',types=conds)

# calculate NETMAT 
covtype='Corr'
prec_flag=False

paramsOrig=[[]]
(corrmat,params,paramNames)=ml_funcs.corr_calc(datamat,covtype,prec_flag=prec_flag,cc_flag=(not prec_flag),params=paramsOrig)
corrmat=corrmat[params[0]]

covtype='Cov'
(varmat,params,paramNames)=ml_funcs.corr_calc(datamat,covtype,prec_flag=prec_flag,cc_flag=(not prec_flag),params=paramsOrig)
varmat=varmat[params[0]]
inds=arange(varmat.shape[-1])
varmat=varmat[:,:,inds,inds]

if contrs==[]:
    outsize=nCk(conds,2)
    for a in itertools.combinations(np.arange(conds),2):
        contrs.append(a)
else:
    outsize=len(contrs)

rois=corrmat.shape[-1]

varvarmat=zeros(varmat.shape + (varmat.shape[-1],))

for a in ndindex(varmat.shape[0:2]):
    varvarmat[a][ix_(arange(varmat.shape[-1]),arange(varmat.shape[-1]))]=tile(varmat[a],(rois,1))

sz=varmat.shape[-1]

#ccstatsmat=zeros((outsize,sz,sz))
#varvarstatsmat=zeros((outsize,sz,sz))

utrisz=((sz**2)-sz)/2

ccstatsmat=zeros((outsize,utrisz))
varstatsmat=zeros((outsize,utrisz))
varstatsmatt=zeros((outsize,utrisz))
vvs=zeros((outsize,utrisz))
vvst=zeros((outsize,utrisz))

ccmat1=zeros((outsize,utrisz))
ccmat2=zeros((outsize,utrisz))
vvmat1a=zeros((outsize,utrisz))
vvmat1b=zeros((outsize,utrisz))
vvmat2a=zeros((outsize,utrisz))
vvmat2b=zeros((outsize,utrisz))

diffsv=zeros((outsize,utrisz))
diffsvt=zeros((outsize,utrisz))
diffsc=zeros((outsize,utrisz))
diffscd=zeros((outsize,utrisz))

cnt=0
inds=[]

for val in contrs:
    ccstatsmat[cnt,:]=ml_funcs.flattenall((stats.ttest_rel(rtoz(corrmat[val[0],:,:,:]),rtoz(corrmat[val[1],:,:,:]))[0]))
    tmp=(stats.ttest_rel((varmat[val[0],:,:]**.5),(varmat[val[1],:,:]**.5))[0])
    #vvs[cnt,:] = ml_funcs.flattenall(ml_funcs.crossfunc((tmp),maximum))
    #vvst[cnt,:] = ml_funcs.flattenall(t21(ml_funcs.crossfunc((tmp),maximum)))
    vvs[cnt,:]=ml_funcs.triu_all(tile(tmp,(rois,1)))
    vvst[cnt,:]=ml_funcs.triu_all(tile(tmp,(rois,1)).T)

    # match sign of vvs to corr change

    vvs_s=vvs*sign(ccstatsmat)
    vvst_s=vvst*sign(ccstatsmat)

    diffsv[cnt,:]=ml_funcs.flattenall(mean(varvarmat[val[0],:,:],0)/mean(varvarmat[val[1],:,:],0))
    diffsvt[cnt,:]=ml_funcs.flattenall(t21(mean(varvarmat[val[0],:,:]/varvarmat[val[1],:,:],0)))
    diffsc[cnt,:]=ml_funcs.flattenall(mean(corrmat[val[0],:,:,:],0)-mean(corrmat[val[1],:,:,:],0))
    diffscd[cnt,:]=ml_funcs.flattenall(mean(corrmat[val[0],:,:,:],0)/mean(corrmat[val[1],:,:,:],0))
    
    ccmat1[cnt,:]=ml_funcs.flattenall(mean(corrmat[val[0],:,:,:],0))
    ccmat2[cnt,:]=ml_funcs.flattenall(mean(corrmat[val[1],:,:,:],0))
    vvmat1a[cnt,:]=ml_funcs.flattenall(mean(varvarmat[val[0],:,:],0))
    vvmat1b[cnt,:]=ml_funcs.flattenall(mean(varvarmat[val[0],:,:],0).T)
    vvmat2a[cnt,:]=ml_funcs.flattenall(mean(varvarmat[val[1],:,:],0))
    vvmat2b[cnt,:]=ml_funcs.flattenall(mean(varvarmat[val[1],:,:],0).T)
    
    inds.append(val)
    cnt+=1

inds=array(inds)

ids={}
bins=arange(0,9.2,1)
ppns=zeros((len(bins)-1,6))
la=logical_and
lo=logical_or
edge=0

# near zero testing
ccmat1_orig=ccmat1.copy()
ccmat2_orig=ccmat2.copy()

ccmat1nz= find((abs(ccmat1)<0.1) & (sign(ccmat1)==-sign(ccmat2)))
#ccmat1[0,ccmat1nz]=-ccmat1[0,ccmat1nz]*0.1
ccmat2nz= find((abs(ccmat2)<0.1) & (sign(ccmat1)==-sign(ccmat2)))
#ccmat2[0,ccmat2nz]=-ccmat2[0,ccmat2nz]*0.1

out=scripts2.range(ccmat1,ccmat2,vvmat1a**.5,vvmat2a**.5,vvmat1b**.5,vvmat2b**.5)

for a in arange(len(bins)-1):
    ids[a]=where(la(abs(ccstatsmat)>bins[a],abs(ccstatsmat)<bins[a+1]))

    # synchronisation
    couplingf_orig=out[2][0,:,:].reshape(ccmat1.shape)

    couplingf=out[2][0,:,:].reshape(ccmat1.shape)
    ppns[a,5]=sum(couplingf[ids[a]]==1)

    ids_tmp = where(couplingf[ids[a]]==0)
    ids_tmp=(ids[a][0][ids_tmp],ids[a][1][ids_tmp])

    # changes in signal
    ppns[a,0]= sum(minimum(vvs_s[ids_tmp],vvst_s[ids_tmp])>edge)
    # changes in noise
    ppns[a,2]= sum(maximum(vvs_s[ids_tmp],vvst_s[ids_tmp])<-edge)
    # changes in opposite dirs
    ppns[a,1]= sum(la(maximum(vvs_s[ids_tmp],vvst_s[ids_tmp])>edge,abs(minimum(vvs_s[ids_tmp],vvst_s[ids_tmp]))<edge))
    # 1 change in noise 
    ppns[a,3]= sum(la(minimum(vvs_s[ids_tmp],vvst_s[ids_tmp])<-edge,abs(maximum(vvs_s[ids_tmp],vvst_s[ids_tmp]))<edge))
    # 1 change in signal
    ppns[a,4]= sum(la(maximum(vvs_s[ids_tmp],vvst_s[ids_tmp])>edge,(minimum(vvs_s[ids_tmp],vvst_s[ids_tmp]))<-edge))

    labels=('Signal levels change (both)','Signal level change (one)','Noise level change (both)','Noise level change (one)','Signal and noise change','Synchronization change')
    
plotcolors=[[0.2,0.6,1],[0.62,0.82,0.98],[0.40,0.95,0.46],[0.6,0.95,0.6],[0.15,0.87,0.87],[0.8,0.8,0.8]]

#fig.clf()
#oo=histogram((abs(ccstatsmat).flatten()),bins=bins,normed=1)
#ax_ppns = fig.add_axes([0.05, 0.05, 0.75, .9])
#ax_x_marg = fig.add_axes([0.05, 0.8, 0.75, 0.13])

#ax_x_marg.hist((abs(ccstatsmat).flatten()),color=([0.4,0.4,0.4]),bins=bins,log=True)

#for a in arange(6)[::-1]:
#        sums = sum(ppns[:-1,:len(bins)],1)
#        sums[sums==0]=inf
#        ax_ppns.fill_between(bins[:-2],sum(ppns[:-1,:a],1)/sums,  sum(ppns[:-1,:a+1],1)/sums,color=plotcolors[a],label='Test')
#        p = plt.Rectangle((0, 0), 0, 0,color=plotcolors[a],label=labels[a])
#        ax_ppns.add_patch(p)
#
#plt.bar(arange(0,len(oo[0])),oo[0]/max(oo[0])*0.1)
#ax_ppns.set_xlim([0,7])
#ax_ppns.set_xlabel('Z-score for change in correlation')
#ax_ppns.set_ylabel('Proportions of types of signal change')
#
#ax_ppns.legend()
#
#fig.clf()
sp=contrs[0][0]
if sp == 1:
    sp = 2
elif sp == 2:
    sp = 1

ax=subplot(4,1,sp)
ccmatp2=ccmat2.copy()

ccmatp2[ccmat1<0]=ccmat2[ccmat1<0]*-1
ccmatp1=abs(ccmat1)

ccmatp2_orig=ccmat2_orig.copy()

ccmatp2_orig[ccmat1_orig<0]=ccmat2_orig[ccmat1_orig<0]*-1
ccmatp1_orig=abs(ccmat1_orig)


hist((ccmat1-ccmat2).T,bins=arange(-0.5,.5,0.04),normed=False,label='All connections')
hist((ccmat1_orig-ccmat2_orig)[0,find(couplingf_orig)].T,bins=arange(-0.5,.5,0.04),normed=False,label='Coupling change connections')
hist((ccmat1-ccmat2)[0,find(couplingf)].T,bins=arange(-0.5,.5,0.04),normed=False,label='Coupling change connections \n with nonzero correlation')

ax.set_ylim(0,120)

if sp==1:
    ax.legend(loc=2)

#if sp==4:
#    ax.set_xlabel('Change in correlation')
#    fig.savefig(cond_names[sp]+'_hist.pdf')
#ax_x_marg.get_yaxis().set_visible(False)
#ax_x_marg.get_xaxis().set_visible(False)

