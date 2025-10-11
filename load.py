import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.util as ski

def load_data(file_name: str) -> np.array:
    return np.load(file_name, allow_pickle=True)


def load_all_file(list_file: list[str]) -> list[np.array]:
    res = []
    for fil in list_file:
        res.append(load_data(fil))
    return res

def list_file_name(dim_list = [2, 10, 50], perp_list = [30, 50], ex_list = [12, 20]):
    X_list = []
    y_list = []
    tsne_list = []

    for dim in dim_list:
        X_list.append(f"datas/points_X_dim_{dim}.npy")
        y_list.append(f"datas/points_y_dim_{dim}.npy")
        for perp in perp_list:
            for ex in ex_list:
                tsne_list.append(f"datas/tsne_perp_ex_dim_{perp}_{ex}_{dim}.npy")

    return X_list, y_list, tsne_list

def compute_dim(nb_img: int) -> tuple(int, int):
    x = round(np.sqrt(nb_img))
    y = nb_img/x
    return x, y

#def subplot a partir de list ndarray + dim determine par fonction au dessus ():


def show_montage():
    _, _, tsne_list = list_file_name()
    all_file = load_all_file(tsne_list)
    print(all_file[0].shape)
    montg_fil = ski.montage(all_file[0:len(all_file):2])
    print(f"month {montg_fil.shape}")
    plt.imshow(montg_fil)
    plt.show()
    return

if __name__=="__main__":

    show_montage()

    '''
    file_dir = os.listdir("datas")
    for f_in in file_dir:
        X = load_data(f"datas/{f_in}")
        if (len(X.shape) > 1):
            plt.scatter(X[:, 0], X[:, 1])
            plt.title(f"{f_in}")
            plt.show()
            '''