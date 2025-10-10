import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from joblib import Parallel, delayed
from create_data import new_points, save_data

import time

# def nb_parallelisation(n):
#     return list(range(n))


# Génération des points en parallèle
def generate_points(
        n_features: int,
        centers: int,   
        seed:int|None|np.random.RandomState=None,
        pct_noise: float=0,
        n_samples: int = 500,
        cluster_std = np.random.uniform(0.5, 2.5)
        ):
    """_summary_

    Args:
        seed (int|None|np.random.RandomState): _description_
        n_features (int): _description_
        centers (int): _description_
        pct_noise (float): percent of the cluster std used for the noise std
        n_samples (int): Total number of points
        cluster_std (float): Std dev among a cluster

    Returns:
        _type_: _description_
    """

    #liste_noises = [3/100, 10/100, 20/100]
    #n_noise = random.choice(liste_noises)

    return new_points(n_samples,
                      pct_noise,
                      centers,
                      n_features,
                      cluster_std,
                      seed
                      )

def recolte_datas(seeds, n_features, centers, cluster_std = np.random.uniform(0.5, 2.5), pct_noise = np.random.uniform(0, 1)):
    results = Parallel(n_jobs=-1)(delayed(generate_points)(n_features=n_features, centers=centers, cluster_std=cluster_std, pct_noise=pct_noise) for s in seeds)
    return results

def fusion_datas(results):
    X_total = np.vstack([res[0] for res in results])
    y_total = np.hstack([res[1] for res in results])
    return X_total, y_total


def begin_points(n=4):
    seeds = [i for i in range(n)]#nb_parallelisation(4)
    n_features = 2
    centers = 4
    for i in range(30):
        n_features += 2
        centers += 2
        results = recolte_datas(seeds, n_features, centers)
        X_total, y_total = fusion_datas(results)
        save_data(X_total, y_total, i)




if __name__=="__main__":

    n_dim = [2, 10, 50] # Mettre en variable globale ?
    centers = [3, 5, 10]
    X = {}
    y = {}
    for cent in centers:
        for dim in n_dim:
            X_arr, y_arr = generate_points(n_features=dim, centers=cent, cluster_std=1.5, pct_noise=0.2, n_samples=5000)
            X[f"{cent}"][f"{dim}"] = X_arr
            y[f"{cent}"][f"{dim}"] = y_arr
            np.save(f"datas/points_X_dim_cent_{dim}_{cent}.npy", X[f"{dim}"])
            np.save(f"datas/points_y_dim_cent_{dim}_{cent}.npy", y[f"{dim}"])
    
    print("start TSNE")



    perpl = [30, 50] # Potentiellement rajouter valeurs si temps
    exag = [12, 20] # Utile rajouter valeurs ?

    for perpx in perpl:
        for ex in exag:
            tsne = TSNE(perplexity=perpx, n_iter=1500)
            print("TSNE done")
            for dim in n_dim:
                res = tsne.fit_transform(X[f"{dim}"])
                np.save(f"datas/tsne_perp_ex_dim_{perpx}_{ex}_{dim}.npy", res)
                print("dim done")
    
    
    '''
    tsne = TSNE(perplexity=50, n_iter=1500, init="random")
    print(f"TSNE {time.time() - ori}")
    tsne_pca = TSNE(perplexity=50, n_iter=1500, init="pca")
    print(f"TSNE PCA {time.time() - ori}")
    result = tsne.fit_transform(X)
    print(f"TSNE out {time.time() - ori}")
    result_pca = tsne_pca.fit_transform(X)
    print(f"TSNE 2 out {time.time() - ori}")
    
    

    fig, ax = plt.subplots(ncols=3)
    
    ax[0].scatter(X[:, 0], X[:, 1], c=y)
    ax[0].set_title("Avant TSNE")
    


    ax[1].scatter(result[:, 0], result[:, 1], c=y)
    ax[1].set_title("Apres TSNE")

    ax[2].scatter(result_pca[:, 0], result_pca[:, 1], c=y)
    ax[2].set_title("Apres TSNE (PCA)")

    plt.show()
    '''


"""
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
"""