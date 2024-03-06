#------------------------------------------------------------------------------
# @author: Jakob Stoye, Institute for Numerical Analysis, TU Braunschweig
#          jakob.stoye@tu-braunschweig.de
#
# Associated with the publication
#
# create a random tangent vectors in the tangent space of U0 of pre-defined
# rank and norm 1.
#------------------------------------------------------------------------------
def create_random_rank_TangentVec(U0, rank, metric_alpha=0.0):
#------------------------------------------------------------------------------
    n = np.shape(U0)[0]
    p = np.shape(U0)[1]
    
    if (rank > p):
        print("ERROR: DEMANDED RANK LARGER THAN DIMENSION P")
        return 
    #end if
    
    # create pseudo-random tangent vector in T_U0 St(n,p)
    A = np.zeros((p,p))
    if rank == 1:
        rnkA = 0
    else:
        rnkA = random.randint(0,int(rank/2)+1);
    
    #for i in range(int(rank/2)):
    for i in range(rnkA):
        ua = random.randn(p,1)
        va = random.randn(p,1)
        A += np.dot(ua,va.transpose()) #add rank 1 matrix
    #end for i
    
    A = A-A.transpose()   # "random" p-by-p skew symmetric matrix of rank 2*int(rank/2)
    
    U,Sigma,VT = linalg.svd(A)
    T = np.zeros((n,p))
    
    if rnkA == 0:
        rnkB = rank
    else:
        rnkB = random.randint((rank-2*rnkA),rank)
    
    for j in range(rank-2*rnkA):
        ut = random.randn(n,1)
        vt = random.randn(p,1)
        T += np.dot(ut,vt.transpose())
    #end for j
    for j in range(rnkB-(rank-2*rnkA)):
        ut = random.randn(n,1)
        v = np.matrix(VT[j,:])
        T += np.dot(ut,v)
    #end for j
    
    Delta = np.dot(U0,A) + T - np.dot(U0,np.dot(U0.transpose(),T))
    
    #normalize Delta
    norm_Delta = scipy.sqrt(StAux.alphaMetric(Delta, Delta, U0,metric_alpha))
    Delta = (1/norm_Delta)*Delta
    return Delta
#------------------------------------------------------------------------------
