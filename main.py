import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from joblib import Parallel, delayed
from create_data import new_points, save_data
import random


def nb_parallelisation(n):
    return list(range(n))


# Génération des points en parallèle
def generate_points(seed, n_features, centers):
    n_samples = 500
    liste_noises = [3/100, 10/100, 20/100]
    n_noise = random.choice(liste_noises)
    centers = centers
    cluster_std = np.random.uniform(0.5, 2.5)
    return new_points(n_samples, n_noise, centers, n_features, seed, cluster_std)

def recolte_datas(seeds, n_features, centers):
    n_features = n_features
    results = Parallel(n_jobs=-1)(delayed(generate_points)(s, n_features, centers) for s in seeds)
    return results

def fusion_datas(results):
    X_total = np.vstack([res[0] for res in results])
    y_total = np.hstack([res[1] for res in results])
    return X_total, y_total


def begin_points():
    seeds = nb_parallelisation(4)
    n_features = 2
    centers = 4
    for i in range(30):
        n_features += 2
        centers += 2
        results = recolte_datas(seeds, n_features, centers)
        X_total, y_total = fusion_datas(results)
        save_data(X_total, y_total, i)






tsne = TSNE(n_components=2, init='pca', perplexity=80, n_iter=1500, random_state=42)
X2_total = tsne.fit_transform(X_total)

# Visualisation
#plt.figure(figsize=(8,6))
scatter = plt.scatter(X_total[:,0], X_total[:,1], c=y_total, cmap="tab10", s=10, alpha=0.7)
plt.colorbar(scatter, label="Labels")
plt.title("original")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")


plt.figure(figsize=(8,6))
scatter = plt.scatter(X2_total[:,0], X2_total[:,1], c=y_total, cmap="tab10", s=10, alpha=0.7)
plt.colorbar(scatter, label="Labels")
plt.title("t-SNE global sur les 4 seeds (points générés en parallèle)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()
