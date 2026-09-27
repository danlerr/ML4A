
import numpy as np
from scipy.stats import multivariate_normal

# This function generate random data from K multivariate Gassian ditribuitions.

def generateRandomData(n, K): 
    Mu = np.zeros((K, 2))
    Sigma = []
    D = []

    for i in range(K):

        #generiamo una media casuale per il cluster i-esimo 
            # viene crato un array 1x2 con numeri casuali tra [0,1)
            # viene aggiunto l'offset (3i), in modo da avere i centroidi dei cluster in zone diverse dello spazio
            # -> ad esempio se k = 3, il primo cluster avrà valori nella zona [0,3] il secondo in [3,6] e il terzo in [6,9]
        mu_i = 3*np.random.rand(1, 2) + 3 * i 
            #flatten rende la matrice output di .rand (1x2) in array monodimensionale (2,)
        mu_i = mu_i.flatten()

        #matrice di covarianza simmetrica e definita positiva
            #temp: matrice 2x2 con valori random in [0,1)
        temp = np.random.rand(2, 2)
        # la matrice di covarianza di una distribuzione deve essere:
            # 1. simmetrica -> (temp + temp.T) garantisce la simmetria
            # 2. definita positiva -> np.dot(., .) (elevare al quadrato una matrice simmetrica garantisce che tutti gli autovalori siano positivi)
        sigma_i = 0.75 * np.dot((temp + temp.T), (temp + temp.T))

        # generiamo n punti nel cluster i 
        points = np.random.multivariate_normal(mu_i, sigma_i, n)

        # associamo ai punti un etichetta 
        labels = (i + 1) * np.ones((n,1))

        cluster_data = np.hstack((points, labels))
        D.append(cluster_data)
        Mu[i, :] = mu_i
        Sigma.append(sigma_i)

    D = np.vstack(D)
    return D, Mu, Sigma