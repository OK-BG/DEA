import numpy as np
import os
from sklearn.datasets import make_blobs

def save_data(X, y, iter):
    os.makedirs("datas", exist_ok=True)
    np.savez(f"datas/data_iter_{iter}.npz", X_all=X, y_all=y)

# centers = nombre de clusters, et y1 y recupere l'etiquette correspondant (0, 1 ou 2 en ft du cluster), n_features = nombre de dimensions, random_state = permet d'avoir tjrs le meme resultat quand on lance le code (42 est la convention)
def new_points(n_samples: int,
               pct_noise: float,
               centers: int,
               n_features: int,
               cluster_std: float,
               seed: int|None|np.random.RandomState = None
               ):
    """
    Summary TODO

    Args:
        n_samples (int): Total numbers of points
        n_noise (float): standard deviation of the noise ?
        centers (int): Number of clusters
        n_features (int): Number of dimensions
        cluster_std (float): Standard deviation among a cluster
        seed (int): seed used to generate the randomness

    Returns:
        X (np.array) : The generated clusters, of size (n_samples * n_features)
        y1 (np.array) : The number of the cluster to which the data belong, size (n_samples * 1)
    """
    X1, y1 = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features, random_state=seed, cluster_std=cluster_std) # caracteristiques : https://www.nature.com/articles/s41467-019-13056-x#:~:text=structure%20can%20be%20seen%20in,depending%20on%20the%20random%20seed
    # cluster_std a tester : [0.5, 1.0, 2.0], noise_std a tester : 0.5 Ã— cluster_std (faible bruit relatif), 1.0 Ã— cluster_std (bruit comparable), 3.0 Ã— cluster_std (bruit fort), 5.0 Ã— cluster_std (bruit tres fort / outliers)

    # Pour tester la robustesse : Si tu veux tester robustesse faible â†’ 500 (3%). Pour robustesse moderee â†’ 1500 (â‰ˆ10%). Pour robustesse forte â†’ 3000 (â‰ˆ20%).
    noise = np.random.normal(0,
                             pct_noise*cluster_std, 
                             (n_samples, n_features)) # Multiplying cluster std to get a percentage of noise ?


    # ajout donnees normales + bruit
    X = X1 + noise
    #X = np.vstack([X1, noise])
    #y = np.hstack([y1, [-1]*round(n_samples*n_noise)])  # -1 pour etiqueter le bruit


    return X, y1


