import matplotlib.pyplot as plt
import numpy as np
import generateRandomData
from expectation import expectation_kmeans
from maximization import maximization_kmeans

# Parametri iniziali
numPoints = 500
numClusters = 3

# 1. Genera dati casuali con 3 cluster
Data_o, Mu_o, Sigma_o = generateRandomData.generateRandomData(
    numPoints, numClusters
)

# 2. Mescola le etichette iniziali (crea una copia per non sporcare i dati originali)[cite: 5]
Data_r = Data_o.copy()
# Assegna etichette casuali tra 1 e numClusters alla terza colonna
Data_r[:, 2] = np.random.randint(
    1, numClusters + 1, size=numClusters * numPoints
)

# 3. Scelta iniziale casuale per i centroidi (Mu)[cite: 5]
Mu = np.random.rand(3, 2)

# 4. Parametri per il ciclo di ottimizzazione[cite: 5]
iterazioni = 0
epsilon = 0.0001
shift = epsilon + 1  # Valore iniziale maggiore di epsilon
Data = Data_r.copy()

print("Avvio dell'algoritmo K-means...")

while shift > epsilon:
  iterazioni += 1

  # Expectation: associa i punti al centroide più vicino (richiama la funzione dal file expectation.py)[cite: 5]
  Data_u = expectation_kmeans(Data, Mu)

  # Maximization: ricalcola i centroidi come media dei punti assegnati (richiama la funzione dal file maximization.py)[cite: 5]
  Mu_u = maximization_kmeans(Data_u)

  # Calcola lo spostamento (differenza quadratica totale tra i vecchi e i nuovi centroidi)[cite: 5]
  shift = np.sqrt(np.sum((Mu - Mu_u) ** 2))

  print(
      f"Iterazione: {iterazioni}, Errore (shift): {shift:.4f}, mu1:"
      f" {Mu_u[0]}, mu2: {Mu_u[1]}, mu3: {Mu_u[2]}"
  )

  # Aggiorna per il giro successivo[cite: 5]
  Mu = Mu_u
  Data = Data_u

Data_f = Data
Mu_f = Mu

# 5. Plot dei risultati (equivalente ai subplot di MATLAB)[cite: 5]
plt.figure(figsize=(7, 7))

# Pannello 1: Raw data
plt.subplot(2, 2, 1)
plt.scatter(
    Data_o[:, 0],
    Data_o[:, 1],
    c="teal",
    s=10,
    edgecolors="darkcyan",
    linewidths=1.5,
)
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y")
plt.title("raw data")
plt.axis("square")

# Pannello 2: True values
plt.subplot(2, 2, 2)
colors = ["red", "green", "blue"]
for ii in range(numClusters):
  d_cluster = Data_o[Data_o[:, 2] == (ii + 1)]
  plt.scatter(
      d_cluster[:, 0],
      d_cluster[:, 1],
      c=colors[ii],
      s=10,
      linewidths=1.5,
  )
  plt.plot(
      Mu_o[ii, 0], Mu_o[ii, 1], marker="P", color="m", markersize=12, linewidth=2
  )
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y")
plt.title("true value")
plt.axis("square")

# Pannello 3: Data clustered by K-means (nostro)
plt.subplot(2, 2, 3)
for ii in range(numClusters):
  d_cluster = Data_f[Data_f[:, 2] == (ii + 1)]
  plt.scatter(
      d_cluster[:, 0],
      d_cluster[:, 1],
      c=colors[ii],
      s=10,
      linewidths=1.5,
  )
  plt.plot(
      Mu_f[ii, 0],
      Mu_f[ii, 1],
      marker="P",
      color="m",
      markersize=12,
      linewidth=2,
  )
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y")
plt.title("data clustered by k-means")
plt.axis("square")

plt.tight_layout()
plt.show()