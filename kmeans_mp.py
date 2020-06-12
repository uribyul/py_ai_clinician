from sklearn.cluster import KMeans
import numpy as np
from multiprocessing import Pool
import os 


def run(ncl,max_iter,sampl,clusteringID): 
    C = KMeans(n_clusters=ncl,max_iter=max_iter,n_init=1,verbose=False).fit(sampl)
    print('Clustering ID = ',clusteringID,'n_iter = ',C.n_iter_,'Inertia = ',C.inertia_)
    return C 


def kmeans_with_multiple_runs(ncl,max_iter,nclustering,sampl):
    
    num_processors = os.cpu_count() 
    p=Pool(processes = num_processors)
          
    args = [] 
        
    for i in range(nclustering): 
        args.append([ncl,max_iter,sampl,i])
    clusters = p.starmap(run,args)

    inertias = []
    for i in range(len(clusters)):
        inertias.append(clusters[i].inertia_)
    index = inertias.index(min(inertias))

    print('The best inertia = ',min(inertias))

    p.close()
    p.join()

    return clusters[index]
    

