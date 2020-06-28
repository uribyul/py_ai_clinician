#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Re-implmentation of AI Clinician Matlab Code in Python 
# Author: KyungJoong Kim (GIST, South Korea)
# Date: 2020 June 2 
# 
# This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE


# Note 
#
# K-Means in scikit-learn will produce differnt outcome with Matlab's original K-means 
# The random number generator will produce different random number sequence 
# 
# Todo: 
# eICU was not included in this re-implementation 





## AI Clinician core code

# (c) Matthieu Komorowski, Imperial College London 2015-2019
# as seen in publication: https://www.nature.com/articles/s41591-018-0213-5

# version 16 Feb 19
# Builds 500 models using MIMIC-III training data
# Records best candidate models along the way from off-policy policy evaluation on MIMIC-III validation data
# Tests the best model on eRI data


# TAKES:
        # MIMICtable = m*59 table with raw values from MIMIC
        # eICUtable = n*56 table with raw values from eICU
        

# GENERATES:
        # MIMICraw = MIMIC RAW DATA m*47 array with columns in right order
        # MIMICzs = MIMIC ZSCORED m*47 array with columns in right order, matching MIMICraw
        # eICUraw = eICU RAW DATA n*47 array with columns in right order, matching MIMICraw
        # eICUzs = eICU ZSCORED n*47 array with columns in right order, matching MIMICraw
        # recqvi = summary statistics of all 500 models
        # idxs = state membership of MIMIC test records, for all 500 models
     	# OA = optimal policy, for all 500 models
        # allpols = detailed data about the best candidate models

# This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE

# Note: The size of the cohort will depend on which version of MIMIC-III is used.
# The original cohort from the 2018 Nature Medicine publication was built using MIMIC-III v1.3.

import pickle 
import numpy as np 
import pandas as pd 
from scipy.stats import zscore, rankdata
import math 
import scipy.io as sio 
import datetime 
from scipy.stats.mstats import mquantiles
from mdptoolbox.mdp import PolicyIteration
from reinforcement_learning_mp import offpolicy_multiple_eval_010518
from kmeans_mp import kmeans_with_multiple_runs 
from multiprocessing import freeze_support 

def my_zscore(x):
    return zscore(x,ddof=1),np.mean(x,axis=0),np.std(x,axis=0,ddof=1)


# In[ ]:


######### Functions used in Reinforcement Learning ######## 


class PolicyIteration_with_Q(PolicyIteration):
    def __init__(self, transitions, reward, discount, policy0=None,max_iter=1000, eval_type=0, skip_check=False):
        # Python MDP toolbox from https://github.com/sawcordwell/pymdptoolbox
        # In Matlab MDP Toolbox, P = (S, S, A), R = (S, A) 
        # In Python MDP Toolbox, P = (A, S, S), R= (S, A)

        transitions = np.transpose(transitions,(2,0,1)).copy() # Change to Action First (A, S, S)
        skip_check = True # To Avoid StochasticError: 'PyMDPToolbox - The transition probability matrix is not stochastic.'

        PolicyIteration.__init__(self, transitions, reward, discount, policy0=None,max_iter=1000, eval_type=0, skip_check=skip_check)
    
    def _bellmanOperator_with_Q(self, V=None):
        # Apply the Bellman operator on the value function.
        #
        # Updates the value function and the Vprev-improving policy.
        #
        # Returns: (policy, Q, value), tuple of new policy and its value
        #
        # If V hasn't been sent into the method, then we assume to be working
        # on the objects V attribute
        if V is None:
            # this V should be a reference to the data rather than a copy
            V = self.V
        else:
            # make sure the user supplied V is of the right shape
            try:
                assert V.shape in ((self.S,), (1, self.S)), "V is not the "                     "right shape (Bellman operator)."
            except AttributeError:
                raise TypeError("V must be a numpy array or matrix.")
        # Looping through each action the the Q-value matrix is calculated.
        # P and V can be any object that supports indexing, so it is important
        # that you know they define a valid MDP before calling the
        # _bellmanOperator method. Otherwise the results will be meaningless.
        Q = np.empty((self.A, self.S))
        for aa in range(self.A):
            Q[aa] = self.R[aa] + self.discount * self.P[aa].dot(V)
        # Get the policy and value, for now it is being returned but...
        # Which way is better?
        # 1. Return, (policy, value)
        return (Q.argmax(axis=0), Q, Q.max(axis=0))
        # 2. update self.policy and self.V directly
        # self.V = Q.max(axis=1)
        # self.policy = Q.argmax(axis=1)
    
    def run(self):
        # Run the policy iteration algorithm.
        self._startRun()

        while True:
            self.iter += 1
            # these _evalPolicy* functions will update the classes value
            # attribute
            if self.eval_type == "matrix":
                self._evalPolicyMatrix()
            elif self.eval_type == "iterative":
                self._evalPolicyIterative()
            # This should update the classes policy attribute but leave the
            # value alone
            policy_next, Q, null = self._bellmanOperator_with_Q()
            del null
            # calculate in how many places does the old policy disagree with
            # the new policy
            n_different = (policy_next != self.policy).sum()
            # if verbose then continue printing a table
            if self.verbose:
                _printVerbosity(self.iter, n_different)
            # Once the policy is unchanging of the maximum number of
            # of iterations has been reached then stop
            if n_different == 0:
                if self.verbose:
                    print(_MSG_STOP_UNCHANGING_POLICY)
                break
            elif self.iter == self.max_iter:
                if self.verbose:
                    print(_MSG_STOP_MAX_ITER)
                break
            elif self.iter > 20 and n_different <=5 : # This condition was added from the Nature Code 
                if self.verbose: 
                    print((_MSG_STOP))
                break 
            else:
                self.policy = policy_next

        self._endRun()
        
        return Q 
    


# In[ ]:


if __name__ == '__main__': 
    
    freeze_support()
    
    # To ignore 'Runtime Warning: Invalid value encountered in greater' caused by NaN 
    np.warnings.filterwarnings('ignore')

    # Load pickle 
    with open('step_4_start.pkl', 'rb') as file:
        MIMICtable = pickle.load(file)
    
    #############################  MODEL PARAMETERS   #####################################

    print('####  INITIALISATION  ####') 

    nr_reps=500               # nr of repetitions (total nr models) % 500 
    nclustering=32            # how many times we do clustering (best solution will be chosen) % 32
    prop=0.25                 # proportion of the data we sample for clustering
    gamma=0.99                # gamma
    transthres=5              # threshold for pruning the transition matrix
    polkeep=1                 # count of saved policies
    ncl=750                   # nr of states
    nra=5                     # nr of actions (2 to 10)
    ncv=5                     # nr of crossvalidation runs (each is 80% training / 20% test)
    OA=np.full((752,nr_reps),np.nan)       # record of optimal actions
    recqvi=np.full((nr_reps*2,30),np.nan)  # saves data about each model (1 row per model)
    # allpols=[]  # saving best candidate models
    
    
    # #################   Convert training data and compute conversion factors    ######################

    # all 47 columns of interest
    colbin = ['gender','mechvent','max_dose_vaso','re_admission'] 
    colnorm= ['age','Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','Temp_C','FiO2_1',        'Potassium','Sodium','Chloride','Glucose','Magnesium','Calcium',        'Hb','WBC_count','Platelets_count','PTT','PT','Arterial_pH','paO2','paCO2',        'Arterial_BE','HCO3','Arterial_lactate','SOFA','SIRS','Shock_Index','PaO2_FiO2','cumulated_balance'] 
    collog=['SpO2','BUN','Creatinine','SGOT','SGPT','Total_bili','INR','input_total','input_4hourly','output_total','output_4hourly']

    colbin=np.where(np.isin(MIMICtable.columns,colbin))[0]
    colnorm=np.where(np.isin(MIMICtable.columns,colnorm))[0]
    collog=np.where(np.isin(MIMICtable.columns,collog))[0]

    # find patients who died in ICU during data collection period
    # ii=MIMICtable.bloc==1&MIMICtable.died_within_48h_of_out_time==1& MIMICtable.delay_end_of_record_and_discharge_or_death<24;
    # icustayidlist=MIMICtable.icustayid;
    # ikeep=~ismember(icustayidlist,MIMICtable.icustayid(ii));
    reformat5=MIMICtable.values.copy() 
    # reformat5=reformat5(ikeep,:);
    icustayidlist=MIMICtable['icustayid']
    icuuniqueids=np.unique(icustayidlist) # list of unique icustayids from MIMIC
    idxs=np.full((icustayidlist.shape[0],nr_reps),np.nan) # record state membership test cohort

    MIMICraw=MIMICtable.iloc[:, np.concatenate([colbin,colnorm,collog])] 
    MIMICraw=MIMICraw.values.copy()  # RAW values
    MIMICzs=np.concatenate([reformat5[:, colbin]-0.5, zscore(reformat5[:,colnorm],ddof=1), zscore(np.log(0.1+reformat5[:, collog]),ddof=1)],axis=1)
    MIMICzs[:,3]=np.log(MIMICzs[:,3]+0.6)   # MAX DOSE NORAD 
    MIMICzs[:,44]=2*MIMICzs[:,44]   # increase weight of this variable


    # eICU section was not implemented 
    
    # compute conversion factors using MIMIC data
    a=MIMICraw[:, 0:3]-0.5 
    b= np.log(MIMICraw[:,3]+0.1)
    c,cmu,csigma = my_zscore(MIMICraw[:,4:36])
    d,dmu,dsigma = my_zscore(np.log(0.1+MIMICraw[:,36:47]))
    

    ####################### Main LOOP ###########################
    bestpol = 0 
    
    for modl in range(nr_reps):  # MAIN LOOP OVER ALL MODELS
        N=icuuniqueids.size # total number of rows to choose from
        grp=np.floor(ncv*np.random.rand(N,1)+1);  #list of 1 to 5 (20% of the data in each grp) -- this means that train/test MIMIC split are DIFFERENT in all the 500 models
        crossval=1;
        trainidx=icuuniqueids[np.where(grp!=crossval)[0]]
        testidx=icuuniqueids[np.where(grp==crossval)[0]]
        train=np.isin(icustayidlist,trainidx)
        test=np.isin(icustayidlist,testidx)
        X=MIMICzs[train,:]
        Xtestmimic=MIMICzs[~train,:]
        blocs=reformat5[train,0]
        bloctestmimic=reformat5[~train,0]
        ptid=reformat5[train,1]
        ptidtestmimic=reformat5[~train,1] 
        outcome=9 #   HOSP _ MORTALITY = 7 / 90d MORTA = 9
        Y90=reformat5[train,outcome];   


        print('########################   MODEL NUMBER : ',modl)
        print(datetime.datetime.now())

        #######   find best clustering solution (lowest intracluster variability)  ####################
        print('####  CLUSTERING  ####') # BY SAMPLING
        N=X.shape[0] #total number of rows to choose from
        sampl=X[np.where(np.floor(np.random.rand(N,1)+prop))[0],:]

        C = kmeans_with_multiple_runs(ncl,10000,nclustering,sampl) 
        idx = C.predict(X)

        ############################## CREATE ACTIONS  ########################
        print('####  CREATE ACTIONS  ####') 

        nact=nra*nra

        iol=MIMICtable.columns.get_loc('input_4hourly') 
        vcl=MIMICtable.columns.get_loc('max_dose_vaso') 

        a= reformat5[:,iol].copy()                   # IV fluid
        a= rankdata(a[a>0])/a[a>0].shape[0]   # excludes zero fluid (will be action 1)

        iof=np.floor((a+0.2499999999)*4)  #converts iv volume in 4 actions

        a= reformat5[:,iol].copy() 
        a= np.where(a>0)[0]  # location of non-zero fluid in big matrix

        io=np.ones((reformat5.shape[0],1))  # array of ones, by default     
        io[a]=(iof+1).reshape(-1,1)   # where more than zero fluid given: save actual action
        io = io.ravel() 

        vc=reformat5[:,vcl].copy() 
        vcr= rankdata(vc[vc!=0])/vc[vc!=0].size
        vcr=np.floor((vcr+0.249999999999)*4)  # converts to 4 bins
        vcr[vcr==0]=1
        vc[vc!=0]=vcr+1 
        vc[vc==0]=1

        ma1 = np.array([np.median(reformat5[io==1,iol]),np.median(reformat5[io==2,iol]),np.median(reformat5[io==3,iol]), np.median(reformat5[io==4,iol]),np.median(reformat5[io==5,iol])]) # median dose of drug in all bins
        ma2 = np.array([np.median(reformat5[vc==1,vcl]),np.median(reformat5[vc==2,vcl]),np.median(reformat5[vc==3,vcl]), np.median(reformat5[vc==4,vcl]),np.median(reformat5[vc==5,vcl])])

        med = np.concatenate([io.reshape(-1,1),vc.reshape(-1,1)],axis=1)
        uniqueValues,actionbloc = np.unique(med,axis=0,return_inverse=True)

        actionbloctrain=actionbloc[train] 

        ma2Values = ma2[uniqueValues[:,1].astype('int64')-1].reshape(-1,1)
        ma1Values = ma1[uniqueValues[:,0].astype('int64')-1].reshape(-1,1)

        uniqueValuesdose = np.concatenate([ma2Values,ma1Values],axis=1) # median dose of each bin for all 25 actions 

        ####################################################################################################################################
        print('####  CREATE QLDATA3  ####')

        r=np.array([100, -100]).reshape(1,-1)
        r2=r*(2*(1-Y90.reshape(-1,1))-1)
        # because idx and actionbloctrain are index, it's equal to (Matlab's original value -1)
        qldata=np.concatenate([blocs.reshape(-1,1), idx.reshape(-1,1), actionbloctrain.reshape(-1,1), Y90.reshape(-1,1), r2],axis=1)  # contains bloc / state / action / outcome&reward     
        # 0 = died in Python, 1 = died in Matlab 
        qldata3=np.zeros((np.floor(qldata.shape[0]*1.2).astype('int64'),4))
        c=-1
        abss=np.array([ncl+1, ncl]) #absorbing states numbers # 751, 750 

        for i in range(qldata.shape[0]-1):
            c=c+1
            qldata3[c,:]=qldata[i,0:4] 
            if(qldata[i+1,0]==1): #end of trace for this patient
                c=c+1     
                qldata3[c,:]=np.array([qldata[i,0]+1, abss[int(qldata[i,3])], -1, qldata[i,4]]) 

        qldata3=qldata3[:c+1,:]


        # ###################################################################################################################################
        print("####  CREATE TRANSITION MATRIX T(S'',S,A) ####")
        transitionr=np.zeros((ncl+2,ncl+2,nact))  #this is T(S',S,A)
        sums0a0=np.zeros((ncl+2,nact)) 

        for i in range(qldata3.shape[0]-1):    
            if (qldata3[i+1,0]!=1) : # if we are not in the last state for this patient = if there is a transition to make!
                S0=int(qldata3[i,1]) 
                S1=int(qldata3[i+1,1])
                acid= int(qldata3[i,2]) 
                transitionr[S1,S0,acid]=transitionr[S1,S0,acid]+1 
                sums0a0[S0,acid]=sums0a0[S0,acid]+1

        sums0a0[sums0a0<=transthres]=0  #delete rare transitions (those seen less than 5 times = bottom 50%!!)

        for i in range(ncl+2): 
            for j in range(nact): 
                if sums0a0[i,j]==0: 
                    transitionr[:,i,j]=0; 
                else:
                    transitionr[:,i,j]=transitionr[:,i,j]/sums0a0[i,j]


        transitionr[np.isnan(transitionr)]=0  #replace NANs with zeros
        transitionr[np.isinf(transitionr)]=0  #replace NANs with zeros

        physpol=sums0a0/np.sum(sums0a0, axis=1).reshape(-1,1)    #physicians policy: what action was chosen in each state


        print("####  CREATE TRANSITION MATRIX T(S,S'',A)  ####")

        transitionr2=np.zeros((ncl+2,ncl+2,nact))  # this is T(S,S',A)
        sums0a0=np.zeros((ncl+2,nact))

        for i in range(qldata3.shape[0]-1) : 
            if (qldata3[i+1,0]!=1) : # if we are not in the last state for this patient = if there is a transition to make!
                S0=int(qldata3[i,1])
                S1=int(qldata3[i+1,1])
                acid= int(qldata3[i,2]) 
                transitionr2[S0,S1,acid]=transitionr2[S0,S1,acid]+1;  
                sums0a0[S0,acid]=sums0a0[S0,acid]+1

        sums0a0[sums0a0<=transthres]=0;  #delete rare transitions (those seen less than 5 times = bottom 50%!!) IQR = 2-17

        for i in range(ncl+2): 
            for j in range(nact): 
                if sums0a0[i,j]==0:
                    transitionr2[i,:,j]=0 
                else: 
                    transitionr2[i,:,j]=transitionr2[i,:,j]/sums0a0[i,j]

        transitionr2[np.isnan(transitionr2)]=0 #replace NANs with zeros
        transitionr2[np.isinf(transitionr2)]=0 # replace infs with zeros

        print('####  CREATE REWARD MATRIX  R(S,A) ####')
        # CF sutton& barto bottom 1998 page 106. i compute R(S,A) from R(S'SA) and T(S'SA)
        r3=np.zeros((ncl+2,ncl+2,nact))
        r3[ncl,:,:]=-100
        r3[ncl+1,:,:]=100
        R=sum(transitionr*r3)
        R=np.squeeze(R)   #remove 1 unused dimension


        print('####  POLICY ITERATION   ####')


        pi = PolicyIteration_with_Q(transitionr2, R, gamma, np.ones((ncl+2,1))) 
        Qon = np.transpose(pi.run()) 
        OptimalAction=np.argmax(Qon,axis=1).reshape(-1,1)  #deterministic 
        OA[:,modl]=OptimalAction.ravel() #save optimal actions


        print('####  OFF-POLICY EVALUATION - MIMIC TRAIN SET ####')
        # create new version of QLDATA3

        r=np.array([100, -100]).reshape(1,-1)
        r2=r*(2*(1-Y90.reshape(-1,1))-1)
        # because idx and actionbloctrain are index, it's equal to (Matlab's original value -1)
        qldata=np.concatenate([blocs.reshape(-1,1), idx.reshape(-1,1), actionbloctrain.reshape(-1,1), Y90.reshape(-1,1),np.zeros((idx.size,1)), r2[:,0].reshape(-1,1), ptid.reshape(-1,1) ],axis=1)   # contains bloc / state / action / outcome&reward     
        # 0 = died in Python, 1 = died in Matlab 
        qldata3=np.zeros((np.floor(qldata.shape[0]*1.2).astype('int64'),8))
        c=-1
        abss=np.array([ncl+1, ncl]) #absorbing states numbers # 751, 750 

        for i in range(qldata.shape[0]-1):
            c=c+1
            qldata3[c,:]=qldata[i,[0,1,2,4,6,6,6,6]] 
            if(qldata[i+1,0]==1): #end of trace for this patient
                c=c+1     
                qldata3[c,:]=np.array([qldata[i,0]+1, abss[int(qldata[i,3])], -1, qldata[i,5],0,0,-1,qldata[i,6]]) 

        qldata3=qldata3[:c+1,:]

        #  add pi(s,a) and b(s,a)
        p=0.01 #softening policies 
        softpi=physpol.copy() # behavior policy = clinicians' 
        for i in range(ncl): 
            ii=softpi[i,:]==0    
            z=p/sum(ii)    
            nz=p/sum(~ii)    
            softpi[i,ii]=z;   
            softpi[i,~ii]=softpi[i,~ii]-nz;

        softb=np.abs(np.zeros((ncl+2,nact))-p/24) #"optimal" policy = target policy = evaluation policy 

        for i in range(ncl): 
            softb[i,OptimalAction[i]]=1-p


        for i in range(qldata3.shape[0]):  # adding the probas of policies to qldata3
            if qldata3[i,1]<ncl :
                qldata3[i,4]=softpi[int(qldata3[i,1]),int(qldata3[i,2])]
                qldata3[i,5]=softb[int(qldata3[i,1]),int(qldata3[i,2])]
                qldata3[i,6]=OptimalAction[int(qldata3[i,1])]  #optimal action

        qldata3train=qldata3.copy() 


        bootql,bootwis = offpolicy_multiple_eval_010518(qldata3,physpol, 0.99,1,6,750)

        recqvi[modl,0]=modl
        recqvi[modl,3]=np.nanmean(bootql)
        recqvi[modl,4]=mquantiles(bootql,0.99, alphap=0.5, betap=0.5)[0]
        recqvi[modl,5]=np.nanmean(bootwis) # we want this as high as possible
        recqvi[modl,6]=mquantiles(bootwis,0.05, alphap=0.5, betap=0.5)[0] #we want this as high as possible


        # testing on MIMIC-test
        print('####  OFF-POLICY EVALUATION - MIMIC TEST SET ####')

        # create new version of QLDATA3 with MIMIC TEST samples

        idxtest = C.predict(Xtestmimic)

        idxs[test,modl]=idxtest.ravel()  #important: record state membership of test cohort

        actionbloctest=actionbloc[~train] 
        Y90test=reformat5[~train,outcome]

        r=np.array([100, -100]).reshape(1,-1)
        r2=r*(2*(1-Y90test.reshape(-1,1))-1)
        # because idx and actionbloctrain are index, it's equal to (Matlab's original value -1)

        qldata=np.concatenate([bloctestmimic.reshape(-1,1), idxtest.reshape(-1,1), actionbloctest.reshape(-1,1), Y90test.reshape(-1,1),np.zeros((idxtest.size,1)), r2[:,0].reshape(-1,1), ptidtestmimic.reshape(-1,1) ],axis=1)   # contains bloc / state / action / outcome&reward     
        # 0 = died in Python, 1 = died in Matlab 
        qldata3=np.zeros((np.floor(qldata.shape[0]*1.2).astype('int64'),8))
        c=-1
        abss=np.array([ncl+1, ncl]) #absorbing states numbers # 751, 750 

        for i in range(qldata.shape[0]-1):
            c=c+1
            qldata3[c,:]=qldata[i,[0,1,2,4,6,6,6,6]] 
            if(qldata[i+1,0]==1): #end of trace for this patient
                c=c+1     
                qldata3[c,:]=np.array([qldata[i,0]+1, abss[int(qldata[i,3])], -1, qldata[i,5],0,0,-1,qldata[i,6]]) 

        qldata3=qldata3[:c+1,:]

        #  add pi(s,a) and b(s,a)
        p=0.01 # small correction factor #softening policies 
        softpi=physpol.copy() # behavior policy = clinicians' 
        for i in range(ncl): 
            ii=softpi[i,:]==0    
            z=p/sum(ii)    
            nz=p/sum(~ii)    
            softpi[i,ii]=z;   
            softpi[i,~ii]=softpi[i,~ii]-nz;

        softb=np.abs(np.zeros((ncl+2,nact))-p/24) #"optimal" policy = target policy = evaluation policy 

        for i in range(ncl): 
            softb[i,OptimalAction[i]]=1-p

        for i in range(qldata3.shape[0]):  # adding the probas of policies to qldata3
            if qldata3[i,1]<ncl :
                qldata3[i,4]=softpi[int(qldata3[i,1]),int(qldata3[i,2])]
                qldata3[i,5]=softb[int(qldata3[i,1]),int(qldata3[i,2])]
                qldata3[i,6]=OptimalAction[int(qldata3[i,1])]  #optimal action

        qldata3test=qldata3.copy() 

        bootmimictestql,bootmimictestwis = offpolicy_multiple_eval_010518(qldata3,physpol, 0.99,1,6,2000)

        recqvi[modl,18]=mquantiles(bootmimictestql,0.95, alphap=0.5, betap=0.5)[0] #PHYSICIANS' 95% UB
        recqvi[modl,19]=np.nanmean(bootmimictestql)
        recqvi[modl,20]=mquantiles(bootmimictestql,0.99, alphap=0.5, betap=0.5)[0]
        recqvi[modl,21]=np.nanmean(bootmimictestwis) 
        recqvi[modl,22]=mquantiles(bootmimictestwis,0.01, alphap=0.5, betap=0.5)[0] 
        recqvi[modl,23]=mquantiles(bootmimictestwis,0.05, alphap=0.5, betap=0.5)[0] #AI 95% LB, we want this as high as possible




        if recqvi[modl,23] > 40: #saves time if policy is not good on MIMIC test: skips to next model
            print('########################## eICU TEST SET #############################')
            # eICU part was not implemented 



        # eICU testing was not included 
        if recqvi[modl,23]>0 : #  & recqvi(modl,14)>0   # if 95% LB is >0 : save the model (otherwise it's pointless)
            print('####   GOOD MODEL FOUND - SAVING IT   ####' ) 

            # best pol 
            if(bestpol < recqvi[modl,23]):
                print('Best policy was replaced => 95% LB is ',recqvi[modl,23])
                bestpol = recqvi[modl,23] 
                
                # save to pickle 
                with open('bestpol.pkl', 'wb') as file:
                    pickle.dump(modl,file)
                    pickle.dump(Qon,file)
                    pickle.dump(physpol,file)
                    pickle.dump(transitionr,file) 
                    pickle.dump(transitionr2,file)
                    pickle.dump(R,file)
                    pickle.dump(C,file)
                    pickle.dump(train,file)
                    pickle.dump(qldata3train,file)
                    pickle.dump(qldata3test,file) 


    recqvi=recqvi[:modl+1,:]

    # save to pickle for visualization 
    with open('step_5_start.pkl', 'wb') as file:
        pickle.dump(MIMICzs,file)
        pickle.dump(actionbloc,file)
        pickle.dump(reformat5,file)
        pickle.dump(recqvi,file)

    # save recqvi in csv format 
    np.savetxt('recqvi.csv',recqvi,delimiter=',') 



# In[ ]:




