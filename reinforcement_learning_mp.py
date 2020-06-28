import numpy as np
from multiprocessing import Pool
import os 

def offpolicy_multiple_eval_010518(qldata3, physpol, gamma,do_ql,iter_ql,iter_wis): 

    # performs all off-policy algos in one run. bootstraps to generate CIs.

    # do_ql = False 
    
    num_processors = os.cpu_count() 
    p=Pool(processes = num_processors)
          
    if do_ql==1:  #to save time, do offpol q_learning or not
  
        args = [] 
        
        for i in range(iter_ql): 
            args.append([qldata3,physpol,gamma,i])
        bootql = p.starmap(offpolicy_eval_tdlearning,args)


    else: 
        bootql=55  #gives an approximate value

    print('   Mean value of physicians'' policy by TD Learning : %f \n'%np.nanmean(bootql))
   
    args = [] 
    for i in range(iter_wis): 
        args.append([qldata3,gamma,i])
    bootwis = p.starmap(offpolicy_eval_wis,args)

    p.close()
    p.join()

    print('   Mean value of AI policy by WIS : %f \n'%np.nanmean(bootwis))

    return bootql, bootwis 




def offpolicy_eval_wis(qldata3,gamma ,iterID): 
# WIS estimator of AI policy
# Thanks to Omer Gottesman (Harvard) for his assistance

    p=np.unique(qldata3[:,7]) 
    prop=25000/p.size #25000 patients of the samples are used
    prop=min([prop, 0.75])  # max possible value is 0.75 (75% of the samples are used)


    ii=np.floor(np.random.rand(p.shape[0],1)+prop)  # prop #of the samples are used
    p = p.reshape(-1,1)
    j=np.isin(qldata3[:,7],p[ii==1])
    q=qldata3[j==1,:]
    fence_posts=np.where(q[:,0]==1)[0] 
    num_of_trials=fence_posts.shape[0] 
    individual_trial_estimators = np.full((num_of_trials,1),np.nan)
    rho_array=np.full((num_of_trials,1),np.nan)
    c=0  # count of matching pairs pi_e + pi_b
     
    for i in range(num_of_trials-1):  
        rho=1
        for t in range(fence_posts[i],fence_posts[i+1]-1): #stops at -2
            rho=rho*q[t,5]/q[t,4]
            if rho>0: 
                c=c+1 
        
        rho_array[i]=rho  

    ii=np.isinf(rho_array) | np.isnan(rho_array)  # some rhos are INF
    normalization=np.nansum(rho_array[~ii])
        

    for i in range(num_of_trials-1): 
        current_trial_estimator = 0
        rho = 1
        discount = 1/gamma   
            
        for t in range(fence_posts[i],fence_posts[i+1]-1): # stops at -2 otherwise ratio zeroed
            rho=rho*q[t,5]/q[t,4]        
            discount = discount* gamma
            current_trial_estimator = current_trial_estimator+ discount * q[t+1,3]            
        
        individual_trial_estimators[i] =  current_trial_estimator*rho
     
    return np.nansum(individual_trial_estimators[~ii])/normalization

def OffpolicyQlearning150816( qldata3 , gamma, alpha, numtraces,ncl,nact): 
    # OFF POLICY Q LEARNING

    #initialisation of variables
    sumQ=np.zeros((numtraces,1))  #record sum of Q after each iteration
    Q=np.zeros((ncl, nact))  
    maxavgQ=1
    modu=100
    listi=np.where(qldata3[:,0]==1)[0]   # position of 1st step of each episodes in dataset
    nrepi=listi.size  # nr of episodes in the dataset
    jj=0 

    for j in range(numtraces): 
        i=listi[int(np.floor(np.random.rand()*(nrepi-2)))]  #pick one episode randomly (not the last one!)
        trace = []
        while qldata3[i+1,0]!=1 : 
            S1=int(qldata3[i+1,1])
            a1=int(qldata3[i+1,2]) 
            r1=int(qldata3[i+1,3]) 
            step = [ r1, S1, a1 ]
            trace.append(step) 
            i=i+1

        tracelength = len(trace) 

        return_t = trace[tracelength-1][0] # get last reward as return for penultimate state and action. 

        sumQ_delta = 0.0 
        for t in range(tracelength-2,-1,-1):       # Step through time-steps in reverse order
            s = int(trace[t][1]) # get state index from trace at time t
            a = int(trace[t][2]) # get action index
            delta = -alpha*Q[s,a] + alpha*return_t
            Q[s,a] += delta # update Q.
            sumQ_delta += delta 
            return_t = return_t*gamma + trace[t][0] # return for time t-1 in terms of return and reward at t

        if(jj==0): 
            sumQ[jj,0]+=sumQ_delta 
        else: 
            sumQ[jj,0]=sumQ[jj-1,0]+sumQ_delta
            
        jj=jj+1
  

        if  (j+1)%(500*modu)==0 : #check if can stop iterating (when no more improvement is seen)
            s=np.mean(sumQ[j-49999:j+1]) 
            d=(s-maxavgQ)/maxavgQ
            if abs(d)<0.001: 
                 break   #exit routine
            maxavgQ=s
 
    sumQ=sumQ[:jj] 
       
    return Q, sumQ        
 

def offpolicy_eval_tdlearning( qldata3, physpol, gamma, iterID): 
    # V value averaged over state population
    # hence the difference with mean(V) stored in recqvi(:,3)

    ncl=physpol.shape[0]-2
    nact = physpol.shape[1]
    
    p=np.unique(qldata3[:,7])
    prop=5000/p.size # 5000 patients of the samples are used
    prop=min([prop, 0.75])  #max possible value is 0.75 (75% of the samples are used)

    ii=qldata3[:,0]==1
    a=qldata3[ii,1]
    d=np.zeros((ncl,1))
    for i in range(ncl): 
        d[i]=sum(a==i)    # intitial state disctribution

    # print(iterID,'starts...')
    ii=np.floor(np.random.rand(p.shape[0],1)+prop)     # select a random sample of trajectories
    p = p.reshape(-1,1)
    jj=np.isin(qldata3[:,7],p[ii==1])
    q=qldata3[jj==1,0:4] 

    Qoff,_ = OffpolicyQlearning150816( q , gamma, 0.1, 300000,ncl,nact)
        

    V=np.zeros((ncl,nact))
    for k in range(ncl):  
        for j in range(nact): 
            V[k,j]=physpol[k,j]*Qoff[k,j]
        
        
    Vs = sum(np.transpose(V))
    
    # print(iterID,'ends...') 
    
    return np.nansum(Vs[:ncl].reshape(-1,1)*d)/sum(d)

# without parallelization 
def offpolicy_eval_tdlearning_with_morta( qldata3, physpol, ptid, idx, actionbloctrain, Y90, gamma, num_iter ):
    
    ncl=physpol.shape[0]-2
    nact = physpol.shape[1]
    
    bootql = np.full((num_iter,1),np.nan)
   

    p=np.unique(qldata3[:,7])
    prop=5000/p.size # 5000 patients of the samples are used
    prop=min([prop, 0.75])  #max possible value is 0.75 (75% of the samples are used)

    jprog = 0  
    prog = np.full((int(np.floor(ptid.shape[0]*1.01*prop*num_iter)),4),np.nan)
    

    ii=qldata3[:,0]==1
    a=qldata3[ii,1]
    d=np.zeros((ncl,1))
    for i in range(ncl): 
        d[i]=sum(a==i)    # intitial state disctribution

    for i in range(num_iter): 
        if(i%10==0):
            print(i)

        ii=np.floor(np.random.rand(p.shape[0],1)+prop)     # select a random sample of trajectories
        p = p.reshape(-1,1)
        jj=np.isin(qldata3[:,7],p[ii==1])
        q=qldata3[jj==1,0:4] 

        Qoff,_ = OffpolicyQlearning150816( q , gamma, 0.1, 300000,ncl,nact)
            

        V=np.zeros((ncl,nact))
        for k in range(ncl):  
            for j in range(nact): 
                V[k,j]=physpol[k,j]*Qoff[k,j]
            
            
        Vs = sum(np.transpose(V))
        
        bootql[i] = np.nansum(Vs[:ncl].reshape(-1,1)*d)/sum(d)
        jj = np.where(np.isin(ptid,p[ii==1]))[0]
        for ii in range(jj.size): # record offline Q value in training set & outcome - for plot
            prog[jprog,0] = Qoff[idx[jj[ii]],actionbloctrain[jj[ii]]]
            prog[jprog,1] = Y90[jj[ii]]
            prog[jprog,2] = ptid[jj[ii]]  # HERE EACH ITERATION GIVES A DIFFERENT PT_ID  //// if I just do rep*ptid it bugs and mixes up ids, for ex with id3 x rep 10 = 30 (which already exists)
            prog[jprog,3] = i 
            jprog = jprog+1 
        
        
    return bootql, prog 

# with parallelization 
def offpolicy_eval_tdlearning_with_morta_mp( qldata3, physpol, ptid, idx, actionbloctrain, Y90, gamma, num_iter ):
    
    num_processors = os.cpu_count() 
    p=Pool(processes = num_processors)
        
    args = [] 
        
    for i in range(num_iter): 
        args.append([qldata3,physpol,ptid, idx, actionbloctrain, Y90, gamma,i])
    
    results = p.starmap(offpolicy_eval_tdlearning_with_morta_worker,args)

    # bootql 
    bootql = np.full((num_iter,1),np.nan)
    for i in range(len(results)):
        bootql[i] = results[i][0]

    # prog
    cnt = 0  
    for i in range(len(results)):
        cnt+=results[i][1].shape[0]
    
    prog = np.full((cnt,4),np.nan)

    jprog = 0 
    for i in range(len(results)):
        for j in range(results[i][1].shape[0]):
            temp = results[i][1]
            prog[jprog] = temp[j] 
            jprog = jprog+1

    p.close()
    p.join()

    return bootql, prog 

# worker 
def offpolicy_eval_tdlearning_with_morta_worker( qldata3, physpol, ptid, idx, actionbloctrain, Y90, gamma, iterID ):
    # V value averaged over state population
    # hence the difference with mean(V) stored in recqvi(:,3)

    # print(iterID,' start')
    # print('|',end='')

    ncl=physpol.shape[0]-2
    nact = physpol.shape[1]
    
    p=np.unique(qldata3[:,7])
    prop=5000/p.size # 5000 patients of the samples are used
    prop=min([prop, 0.75])  #max possible value is 0.75 (75% of the samples are used)

    ii=qldata3[:,0]==1
    a=qldata3[ii,1]
    d=np.zeros((ncl,1))
    for i in range(ncl): 
        d[i]=sum(a==i)    # intitial state disctribution

    # print(iterID,'starts...')
    ii=np.floor(np.random.rand(p.shape[0],1)+prop)     # select a random sample of trajectories
    p = p.reshape(-1,1)
    jj=np.isin(qldata3[:,7],p[ii==1])
    q=qldata3[jj==1,0:4] 

    Qoff,_ = OffpolicyQlearning150816( q , gamma, 0.1, 300000,ncl,nact)
        

    V=np.zeros((ncl,nact))
    for k in range(ncl):  
        for j in range(nact): 
            V[k,j]=physpol[k,j]*Qoff[k,j]
        
        
    Vs = sum(np.transpose(V))

    bootql = np.nansum(Vs[:ncl].reshape(-1,1)*d)/sum(d)
    
    # print(iterID,'ends...') 

    jj = np.where(np.isin(ptid,p[ii==1]))[0]

    jprog = 0 
    prog = np.full((jj.size,4),np.nan)

    for ii in range(jj.size): # record offline Q value in training set & outcome - for plot
        prog[jprog,0] = Qoff[idx[jj[ii]],actionbloctrain[jj[ii]]]
        prog[jprog,1] = Y90[jj[ii]]
        prog[jprog,2] = ptid[jj[ii]]  # HERE EACH ITERATION GIVES A DIFFERENT PT_ID  //// if I just do rep*ptid it bugs and mixes up ids, for ex with id3 x rep 10 = 30 (which already exists)
        prog[jprog,3] = iterID  
        jprog = jprog+1 

    return bootql, prog  