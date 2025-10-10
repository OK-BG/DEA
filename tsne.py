import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from sklearn.manifold import TSNE

def save_data(X, y, perp, exx, i = 0):
    os.makedirs("tsne", exist_ok=True)
    np.savez(f"tsne/tsne_perp_{perp}_exx_{exx}.npz", X_all=X, y_all=y)
    print(f"Sauvegarde {i + 1}...")

def go_TSNE(X_all, perplexity, early_exaggeration):
    tsne = TSNE(n_components=2, init='pca', perplexity=perplexity, n_iter=1500, early_exaggeration=early_exaggeration, random_state=42)
    X_tsne = tsne.fit_transform(X_all)
    return X_tsne


# Boucle sur tous les fichiers
for file in sorted(glob.glob("datas/data_iter_*.npz")):
    data = np.load(file, allow_pickle=True)
    X_all = np.vstack(data["X_all"])
    y_all = np.hstack(data["y_all"])

    perplexity = [30, 50]
    early_exaggeration = [12, 20]

    for i in perplexity:
        for y in early_exaggeration:
            X_tsne = go_TSNE(X_all, i, y)
            save_data(X_all, y_all,i, y)


