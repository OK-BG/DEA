import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from joblib import Parallel, delayed
from create_data import new_points, save_data


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
    X, y = generate_points(2, 5, cluster_std=1.5, pct_noise=0.2, n_samples=1000)
    
    tsne = TSNE(perplexity=50, n_iter=1500, init="random")
    tsne_pca = TSNE(perplexity=50, n_iter=1500, init="pca")
    result = tsne.fit_transform(X)
    result_pca = tsne_pca.fit_transform(X)
    

    fig, ax = plt.subplots(ncols=3)
    
    ax[0].scatter(X[:, 0], X[:, 1], c=y)
    ax[0].set_title("Avant TSNE")
    


    ax[1].scatter(result[:, 0], result[:, 1], c=y)
    ax[1].set_title("Apres TSNE")

    ax[2].scatter(result_pca[:, 0], result_pca[:, 1], c=y)
    ax[2].set_title("Apres TSNE (PCA)")


    plt.show()


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