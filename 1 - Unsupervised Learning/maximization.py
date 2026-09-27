import numpy as np


def maximization(Gamma, Data, Param):
  """Questa funzione calcola il secondo passo dell'algoritmo EM, la Maximization.

  Aggiorna i parametri delle distribuzioni Normali in base al nuovo dataset
  etichettato.
  """
  # Somma delle probabilità Gamma per ciascuna colonna (cluster)
  n = np.sum(Gamma, axis=0)

  # Calcolo dei pesi (lambda)
  Param["lambda"] = n / np.sum(n)

  # Estrazione delle coordinate x e y dai dati
  coords = Data[:, :2]

  # Calcolo delle medie (mu) per ciascun cluster
  Param["mu1"] = np.dot(Gamma[:, 0], coords) / n[0]
  Param["mu2"] = np.dot(Gamma[:, 1], coords) / n[1]
  Param["mu3"] = np.dot(Gamma[:, 2], coords) / n[2]

  # Calcolo delle matrici di covarianza (sigma) per ciascun cluster
  diff1 = coords - Param["mu1"]
  Param["sigma1"] = np.dot(diff1.T, diff1 * Gamma[:, 0:1]) / n[0]

  diff2 = coords - Param["mu2"]
  Param["sigma2"] = np.dot(diff2.T, diff2 * Gamma[:, 1:2]) / n[1]

  diff3 = coords - Param["mu3"]
  Param["sigma3"] = np.dot(diff3.T, diff3 * Gamma[:, 2:3]) / n[2]

  return Param

def maximization_kmeans(X):
  """
  Questa funzione calcola il secondo passo del K-means (Maximization).
  Aggiorna i centroidi calcolando la media dei punti assegnati a ciascun
  cluster.
  """
  num_clusters = X.shape[1]  # Nota: strutturalmente basato sull'originale
  # Poiché X ha 3 colonne [x, y, label], contiamo i cluster unici o usiamo la dimensione nota
  unique_clusters = int(np.max(X[:, 2]))
  mu = np.zeros((unique_clusters, 2))

  for jj in range(unique_clusters):
    idx = X[:, 2] == (jj + 1)
    data = X[idx, :2]
    if len(data) > 0:
      mu[jj, :] = np.sum(data, axis=0) / len(data)

  return mu