import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.manifold import TSNE
from create_data import new_points

def process_batch(seed):
    X, y = new_points(seed)
    tsne = TSNE(n_components=2, init='pca', perplexity=80, n_iter=1500, random_state=seed)
    X2 = tsne.fit_transform(X)
    return X2, y

# Liste de seeds
seeds = [0]

# Mesurer le temps total (sans parallélisation)
start = time.time()
results = []
for s in seeds:
    results.append(process_batch(s))
end = time.time()

print(f"Temps d'exécution total (séquentiel) : {end - start:.2f} secondes")

# Fusionner les résultats
X2_total = np.vstack([res[0] for res in results])
y_total = np.hstack([res[1] for res in results])

# Visualisation
plt.figure(figsize=(8,6))
scatter = plt.scatter(X2_total[:,0], X2_total[:,1], c=y_total, cmap="tab10", s=10, alpha=0.7)
plt.colorbar(scatter, label="Labels")
plt.title("t-SNE des données fusionnées (séquentiel, 4 seeds)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()
