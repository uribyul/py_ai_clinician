import numpy as np
from scipy.signal import lfilter

# based on 'rloess' method 
# specially x is uniform  
def smooth(x,y,span):
    is_x = 1 
    is_span = 1 
    idx = x 
    c = np.full_like(y,np.nan)
    ok = ~np.isnan(x)
    iter = 5 
    t = y.size 
    if(span<1):
        span = np.ceil(span*t)
    c = lowess(x[ok],y[ok],span,iter)[ok]
    c[idx] = c 
    return c 

def lowess(x,y,span,iter):
    n = y.size
    span = int(np.floor(span))
    span = min(span, n)
    c = y 
    diffx = np.diff(x)
    
    # x is uniform
    span = 2*np.floor(span/2)+1 
    c = unifloess(y,span)
    seps = np.sqrt(np.spacing(1))
    
    halfw = np.floor(span/2) 
    lbound = np.array(range(1,n+1))-halfw
    rbound = np.array(range(1,n+1))+halfw
    lbound[np.where(lbound>n+1-span)] = n+1-span 
    rbound[np.where(rbound<span)] = span 
    lbound[np.where(lbound<1)] = 1 
    rbound[np.where(rbound>n)] = n
    x = np.array(range(1,x.size+1))
    maxabsyXeps = max(np.abs(y))*np.spacing(1)    
    
    # robust fit 
    for k in range(1,iter+1):
        r = y-c 
        rweight = iBisquareWeights(r,maxabsyXeps)
        
        for i in range(n):
            if(i>0 and x[i]==x[i-1]): 
                c[i] = c[i-1] 
                continue 
            if(np.isnan(c[i])): 
                continue 
            
            idx = np.array(range(int(lbound[i])-1,int(rbound[i]))) 
                        
            if(any(rweight[idx]<=0)):
                idx = iKNearestNeighbours(span,i,x,(rweight>0))
            
            x1 = x[idx] - x[i] 
            d1 = np.abs(x1)
            y1 = y[idx] 
            weight = iTricubeWeights(d1)
            if all(weight<seps): 
                weight[:] = 1 
            v = np.array([np.ones(x1.size),x1]).T
            v = np.column_stack([v,np.power(x1,2).T])
            weight = weight * rweight[idx] 
            weights = weight.copy() 
            for m in range(v.shape[1]-1):
                weights = np.column_stack([weights,weight])
                    
            v = weights*v 
            y1 = weight * y1 
            b = np.linalg.lstsq(v,y1)[0]
            c[i] = b[0]
            
    return c

def iKNearestNeighbours(k,i,x,input): 
    
    if(np.count_nonzero(input)<= k):
        idx = np.where((input))
    else:
        d = abs(x-x[i])
        ds = np.sort(d[input])
        dk = ds[int(k-1)]
        close = (d <= dk)
        idx = np.where((close)&(input))

    return idx 
       
            
def iTricubeWeights(d):
    maxD = max(d)
    if maxD > 0: 
        d = d/max(d) 
    w = np.power(1-np.power(d,3),1.5)
    return w 
        
def iBisquareWeights(r, myeps):
    idx = ~np.isnan(r)
    s = max(1e8*myeps,np.median(np.abs(r[idx])))
    
    delta = iBisquare(r/(6*s))
    
    if(sum(np.isnan(r))!=0):
        delta[np.isnan(r)] = 0 
    
    return delta 

def iBisquare(x):
    b = np.zeros_like(x)
    idx = np.abs(x) <1 
    b[idx] = np.abs(1-np.power(x[idx],2))
    return b 
    
def unifloess(y,span):
    y = y.copy() 
    halfw = (span-1)/2 
    d = np.abs(np.array(range(int(1-halfw),int(halfw))))
    dmax = int(halfw) 
    x1 = np.array(range(2,int(span)))-(halfw+1)
    weight = np.power(1-np.power(d/dmax,3),1.5)
    v = np.array([np.ones(x1.size),x1]).T
    v = np.column_stack([v,np.power(x1,2).T])
    
    weights = weight.copy() 
    for i in range(v.shape[1]-1):
        weights = np.column_stack([weights,weight])
    
    V = v*weights
    Q,_ = np.linalg.qr(V)
    alpha = np.matmul(Q[int(halfw-1),:],Q.T)
    alpha = alpha * weight 
    ys = lfilter(alpha,1,y)
    ys[range(int(halfw),ys.shape[0]-int(halfw))] = ys[range(int(span-2),ys.shape[0]-1)]
    x1 = np.array(range(1,int(span))) 
    v = np.array([np.ones(x1.size),x1]).T
    v = np.column_stack([v,np.power(x1,2).T])
    
    for j in range(1,int(halfw+1)):
        d = np.abs(np.array(range(1,int(span)))-j)
        weight = np.power(1-np.power(d/(span-j),3),1.5)
        weights = weight.copy() 
        for k in range(v.shape[1]-1):
            weights = np.column_stack([weights,weight])
        V = v*weights
        Q,_ = np.linalg.qr(V)
        alpha = np.matmul(Q[j-1,:],Q.T)
        alpha = alpha * weight
        ys[j-1] = np.matmul(alpha,y[0:int(span)-1])
        ys[ys.shape[0]-j] = np.matmul(alpha,y[y.shape[0]:y.shape[0]-int(span):-1])
    
    return ys 