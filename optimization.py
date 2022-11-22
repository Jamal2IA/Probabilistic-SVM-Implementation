from Kernels import * 
import cvxopt
import cvxopt.solvers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers 
import math 
from cvxopt import matrix

def gm(y_predict,y_test):
    test_min=0
    test_max=0
    pred_min=0
    pred_max=0
    y_test=np.asarray(y_test)
    for i in range(0,154):
        if(y_test[i]==1):
             test_min=test_min+1
        else:
             test_max=test_max+1
    print("y_test min",test_min)       
    print("y_test max",test_max)
    for i in range(0,154):
        if(y_predict[i]==1 and y_predict[i]==y_test[i]):
             pred_min=pred_min+1
        elif(y_predict[i]==-1 and y_predict[i]==y_test[i]):
             pred_max=pred_max+1
    print("y_pred min",pred_min)       
    print("y_pred max",pred_max)
    se=pred_min/test_min
    sp=pred_max/test_max
    print(se,sp)
    gm=math.sqrt(se*sp)
    print("GM",gm)

 
global K,K1,sv,m,a,sv_y
kernel = linear_kernel
C = 100
def init_cst( kernel=gaussian_kernel, C_=None): 
    kernel = kernel
    C = C_
    if C is not None: C_ = float(C)
def m_func(X_train,X_test, y):
    
    n_samples, n_features = X_train.shape 
    nt_samples, nt_features= X_test.shape
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i,j] = gaussian_kernel(X_train[i], X_train[j])
           # print(K[i,j])
    X_train=np.asarray(X_train)
    X_test=np.asarray(X_test)
    K1 = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K1[i,j] = gaussian_kernel(X_train[i], X_train[j])
           # print(K[i,j])
    print(K1.shape)
    P = cvxopt.matrix(np.outer(y,y) * K)
    q = cvxopt.matrix(np.ones(n_samples) * -1)
    A = cvxopt.matrix(y, (1,n_samples))
    A = matrix(A, (1,n_samples), 'd') #changes done
    b = cvxopt.matrix(0.0)
    #print(P,q,A,b)
    if C is None:
        G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h = cvxopt.matrix(np.zeros(n_samples))

    else:
        tmp1 = np.diag(np.ones(n_samples) * -1)
        tmp2 = np.identity(n_samples)
        G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
        tmp1 = np.zeros(n_samples)
        tmp2 = np.ones(n_samples) * C
        h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
    # solve QP problem
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    print(solution['status'])
    # Lagrange multipliers
    a = np.ravel(solution['x'])
    a_org = np.ravel(solution['x'])
    # Support vectors have non zero lagrange multipliers
    sv_ = a > 1e-5 
    ind = np.arange(len(a))[sv_]
    a_org=a
    a = a[sv_]
    sv = X_train[sv_]
    sv_y = y[sv_]
    sv_yorg=y
    kernel = gaussian_kernel
    X_train=np.asarray(X_train)
    b = 0
    for n in range(len(a)):
        b += sv_y[n]
        b -= np.sum(a * sv_y * K[ind[n],sv_])
    b /= len(a)
    w_phi=0
    total=0
    for n in range(len(a_org)):
        w_phi = a_org[n] * sv_yorg[n] * K1[n] 
    d_hyp=np.zeros(n_samples)
    for n in range(len(a_org)):
        d_hyp += sv_yorg[n]*(w_phi+b)
    func=np.zeros((n_samples))
    func=np.asarray(func)
    typ=2
    if(typ==1):
        for i in range(n_samples):
            func[i]=1-(d_hyp[i]/(np.amax(d_hyp[i])+0.000001))
    beta=0.2
    if(typ==2):
        for i in range(n_samples):
            func[i]=2/(1+beta*d_hyp[i])
    r_max=268/500
    r_min=1
    m=func[0:268]*r_min
    print(m.shape)
    m=np.append(m,func[268:768]*r_max)
    print(m.shape)
