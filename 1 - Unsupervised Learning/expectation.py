import numpy as np
from scipy.stats import multivariate_normal

"""Calcola il primo passo (Expectation) dell'algoritmo EM.

  Calcola la probabilità di ciascun punto di appartenere a ciascuno dei 3
  cluster.
"""

# funzione che calcola il primo passo dell'algoritmo EM
def expectation(Data, Param):
    # Estrae le coordinate x e y dai dati (colonne 0 e 1)
    points = Data[:, :2]

    # Calcola la probabilità non normalizzata per ciascun cluster
    # (Param.lambda contiene i pesi, Param.mu le medie, Param.sigma le covarianze)
    rv1 = multivariate_normal(mean=Param["mu1"], cov=Param["sigma1"])
    p_cluster1 = Param["lambda"][0] * rv1.pdf(points)

    rv2 = multivariate_normal(mean=Param["mu2"], cov=Param["sigma2"])
    p_cluster2 = Param["lambda"][1] * rv2.pdf(points)

    rv3 = multivariate_normal(mean=Param["mu3"], cov=Param["sigma3"])
    p_cluster3 = Param["lambda"][2] * rv3.pdf(points)

    # Impila le probabilità in una matrice n x 3
    p_clusters = np.column_stack((p_cluster1, p_cluster2, p_cluster3))

    # Normalizza Gamma dividendo per la somma delle probabilità di riga
    row_sums = p_clusters.sum(axis=1, keepdims=True)
    Gamma = p_clusters / row_sums

    # Trova l'indice del cluster con probabilità massima per ciascun punto
    # (in Python gli indici partono da 0, quindi aggiungiamo +1 per avere le label 1, 2, 3)
    max_indices = np.argmax(Gamma, axis=1) + 1
    Data[:, 2] = max_indices

    return Gamma, Data

def expectation_kmeans(X, Mu):
  """Questa funzione calcola il primo passo del K-means (Expectation).

  Associa ciascun punto al centroide più vicino.
  """
  num_points, num_clusters = X.shape[0], Mu.shape[0]

  # Calcola la distanza tra un insieme di punti e i centroidi
  dv = np.zeros((num_points, num_clusters))
  for jj in range(num_clusters):
    # Distanza euclidrea vettorizzata
    dv[:, jj] = np.sqrt(np.sum((X[:, :2] - Mu[jj, :]) ** 2, axis=1))

  # Trova l'indice del centroide minimo per ogni punto
  # In Python gli indici partono da 0, quindi aggiungiamo +1 per avere le label (1, 2, 3)
  idx = np.argmin(dv, axis=1) + 1
  X[:, 2] = idx
  return X