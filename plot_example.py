import asc_funcs as cf
from asc_funcs import FC

# load some real data to estimate AR structure
A_real=cf.dr_loader('.',subj_inds=arange(15),dof='EstEff',prefix='real')
Af_real= cf.flatten_tcs(A_real)

# generate baseline covmat (two blocks)
covmat=random.random((10,10))*0.2+0.4
covmat[:5,:5]=random.random((5,5))*0.2+0.45
covmat[5:,5:]=random.random((5,5))*0.2+0.45
covmat[:5,5:]=(random.random((5,5))*0.02)-0.01

# make symmetric
covmat.T[triu_indices(10,k=1)]=covmat[triu_indices(10,k=1)]
covmat[where(eye(10))]=1

# time course for different conditions
tcsA=zeros(A_real.tcs.shape)
tcsB=zeros(A_real.tcs.shape)
tcsC=zeros(A_real.tcs.shape)
tcsD=zeros(A_real.tcs.shape)
tcsE=zeros(A_real.tcs.shape)
# tcsF=zeros(A_real.tcs.shape)
# tcsG=zeros(A_real.tcs.shape)

for a in arange(15):
    tcsA[a,:,:]=cf.gen_sim_data(A_real.tcs[a,:,:],covmat=covmat)
    tcsB[a,:,:]=cf.gen_sim_data(A_real.tcs[a,:,:],covmat=covmat)
    tcsC[a,:,:]=cf.gen_sim_data(A_real.tcs[a,:,:],covmat=covmat)
    tcsD[a,:,:]=cf.gen_sim_data(A_real.tcs[a,:,:],covmat=covmat)
    tcsE[a,:,:]=cf.gen_sim_data(A_real.tcs[a,:,:],covmat=covmat)
    #tcsF[a,:,:]=cf.gen_sim_data(A_real.tcs[a,:,:],covmat=covmat)
    #tcsG[a,:,:]=cf.gen_sim_data(A_real.tcs[a,:,:],covmat=covmat)

    # E: switch 
    tmp=-tcsE[a,0:3,:]*sqrt(0.7) - sqrt(0.3)*cf.gen_sim_data(tcsE[a,0:3,:])
    tcsE[a,0:3,:]=std(tcsA[a,0:3,:])*tmp/std(tmp)
    # tcsE[a,0:3,:]=tmp

    # C: unshared
    for b in arange(3):
        tcsC[a,b,:]=tcsC[a,b,:]+0.7*cf.gen_sim_data(tcsC[a,b,:])

    # D: unshared
    for b in arange(3):
        tcsD[a,b,:]=tcsD[a,b,:]+0.3*cf.gen_sim_data(tcsD[a,b,:])

# B: shared 
Badd=swapaxes(tile(cf.gen_sim_data(tcsA[:,0,:]),(3,1,1)),0,1)*0.85
Badd2=swapaxes(tile(cf.gen_sim_data(tcsA[:,0,:]),(3,1,1)),0,1)*0.85
tcsB[:,:3,:]=tcsB[:,:3,:]+Badd

# D: shared 
Dadd=swapaxes(tile(cf.gen_sim_data(tcsA[:,0,:]),(3,1,1)),0,1)*0.7
Dadd2=swapaxes(tile(cf.gen_sim_data(tcsA[:,0,:]),(3,1,1)),0,1)*0.7
tcsD[:,:3,:]=tcsD[:,:3,:]+Dadd

#Eadd1=swapaxes(tile(cf.gen_sim_data(tcsA[:,0,:]),(3,1,1)),0,1)*1.40
#Eadd2=swapaxes(tile(cf.gen_sim_data(tcsA[:,0,:]),(3,1,1)),0,1)*0.68

#tcsE[:,:3,:]=tcsE[:,:3,:]+Eadd1
#tcsE[:,2:5,:]=tcsE[:,2:5,:]+Eadd2

A=cf.FC(tcsA)
B=cf.FC(tcsB)
C=cf.FC(tcsC)
D=cf.FC(tcsD)
E=cf.FC(tcsE)

cf.dr_saver(A,'.',prefix='sim44',goodnodes=range(10))
cf.dr_saver(B,'.',prefix='sim44',goodnodes=range(10),aug=15)
cf.dr_saver(C,'.',prefix='sim44',goodnodes=range(10),aug=30)
cf.dr_saver(D,'.',prefix='sim44',goodnodes=range(10),aug=45)
cf.dr_saver(E,'.',prefix='sim44',goodnodes=range(10),aug=60)

#### 
##
##tcs = Af_real.tcs[0,:3,:]
##shp = tcs[:2,:].shape
##nreps=20
##errdist_perms=200
##
##a=0.57741163399813
##b=0.7948717948717948
##c=0.69230769230769229
##corrmat1=array([[1,a,b],[a,1,c],[b,c,1]])
##tmp=cf.gen_sim_data(tcs,covmat=corrmat1,nreps=nreps)
##
### initial condition
##Af=cf.FC(tmp[:,:2,:])
##
##tmp1=cf.gen_sim_data(tcs,covmat=corrmat1,nreps=nreps)
##zz1=zeros((nreps,shp[0],shp[1]))
##zz1[:,1,:]=tmp1[:,2,:]
##Bf=cf.FC(tmp1[:,:2,:]+0.6*zz1)
##
##b2=0.84615384615384615
##c2=0.0564102564102564
##corrmat2=array([[1,a,b2],[a,1,c2],[b2,c2,1]])
##corrmat2=array([[1.2**2,a,b2],[a,1.2**.5,c2],[b2,c2,1.2**.5]])
##tmp2=cf.gen_sim_data(tcs,covmat=corrmat2,nreps=nreps)
##
##
##zz2=zeros((nreps,shp[0],shp[1]))
##zz2[:,1,:]=tmp2[:,2,:]
##
##Cf=cf.FC(tmp2[:,:2,:]+0.6*zz2)
##
##b3=0.9999999
##c3=0.57741163399813
##corrmat3=array([[1,a,b3],[a,1,c3],[b3,c3,1]])
##tmp3=cf.gen_sim_data(tcs,covmat=corrmat3,nreps=nreps)
##
##zz3=zeros((nreps,shp[0],shp[1]))
##zz3[:,1,:]=tmp3[:,2,:]
##
##Df=cf.FC(tmp3[:,:2,:]+0.6*zz3)
##
##corrmat1=array([[1,a],[a,1]])
##corrmat2=array([[1.2**2,a],[a,1.2**2]])
##
##if gen_sim_data==True:
##
##    #lims1=cf.corr_lims_all(Af,Bf,errdist_perms=errdist_perms,show_pctls=True,pctl=5)
##    #lims2=cf.corr_lims_all(Af,Cf,errdist_perms=errdist_perms,show_pctls=True,pctl=5)
##    #lims3=cf.corr_lims_all(Af,Df,errdist_perms=errdist_perms,show_pctls=True,pctl=5)
##
##    lims1=cf.ASC_lims_all(A,B,errdist_perms=100,show_pctls=True,pctl=5)
##
