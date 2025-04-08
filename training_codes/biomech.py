# This is a python implementation of the biomechanical model for iris normalization
# All usage of this should cite: 
# Tomeo-Reyes, Inmaculada, Arun Ross, Antwan D. Clark, and Vinod Chandran. "A biomechanical approach to iris normalization." In 2015 International Conference on Biometrics (ICB), pp. 9-16. IEEE, 2015.

import numpy as np

def Biomech(ri0, ri1, numVertices):
    
    ro = .006
    
    vertices = np.linspace((ri1/ro),(ro/ro),numVertices)
    h = vertices[2]-vertices[1]
    
    elements = []
    
    boundaries = np.zeros((len(vertices), len(vertices)))
    
    Er = 4000
    Et = 2970
    nu = 0.49 # The Poisson variable.
    
    zeta = Et / Er

    eta = Er / (1 - zeta * (nu ** 2))

    chi = Et / (1 - zeta * (nu ** 2))
    
    U = np.zeros((len(vertices),1))
    U_linear = U
    
    resThresh = 1e-7
    
    eta2 = ri1/ro #Normalized final radius. This is different from the eta with the material parameters. 
    lamda = (ri1-ri0)/ro #This might be need to be modified to incorporate the various changes. This isn't quite fully the equation.
    
    for q in range(100):
        K = np.zeros((len(vertices), len(vertices)))
        K[0,0] = 1
        K[len(vertices)-1,len(vertices)-1] = 1
        
        f = np.zeros((len(vertices),1))
        f[0] = 1
        f[len(vertices)-1] = 0
        
        for i in range(1, len(vertices)-1):
            Umid = np.array([(U[i-1]+U[i])/2, (U[i]+U[i+1])/2])
            dU_dr = np.array([(U[i]-U[i-1])/(h), (U[i+1]-U[i])/(h)])
            
            for j in range(i-1, i+2):
               
                start = max(i-1,j-1)
                stop = min(i,j)
                if start <= stop:
                    step = 1
                else:
                    step = -1
                for k in range(start, stop+1, step):
                                    
                    basis_i = 0.5
                    basis_j = 0.5
  
                    if k == i-1:
                         dbi_dr = 1/h
                    else:
                         dbi_dr = -1/h
  
                    if k == j-1:
                         dbj_dr = 1/h
                    else:
                         dbj_dr = -1/h
  
                    r = vertices[k]+h/2
  
                    k_local = k - (i-1)
  
                    omega = (nu * eta * zeta - chi) / r
                    psi = (eta - nu * chi) / r
  
                    alpha = eta / r
  
                    beta = -chi / (r * r)
  
                    Kijk = basis_i * basis_j * (Umid[k_local] * (lamda*(nu+1)*zeta/(2*r**3))  - zeta/r**2) + \
                            basis_i * dbj_dr  * (dU_dr[k_local]*(-(1-nu*zeta)/(2*r)) - Umid[k_local] * (nu*zeta*lamda/r**2) + 1/r) + \
                            dbi_dr  * dbj_dr  * (-1 + lamda/2*dU_dr[k_local])
                    K[i,j] = K[i,j] + h * Kijk
        
        res = np.linalg.norm(K @ U - f) 
        if (res < resThresh):
            break
        
        Unew = np.linalg.inv(K) @ f
        U = U + (Unew - U)
        
        if np.linalg.norm(U_linear) == 0:
            U_linear = U   
    UR = U * (ri1 - ri0)
    R = np.linspace(ri0, ro, numVertices) + UR.reshape(1,numVertices)
    normR = R - R.min()
    normR = normR / normR.max()
    
    return normR
    
if __name__=='__main__':
    import sys
    
    print(Biomech(float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])))
        