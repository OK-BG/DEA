import numpy as np
import os
import matplotlib.pyplot as plt
def load_data(file_name: str) -> np.array:
    return np.load(file_name, allow_pickle=True)

if __name__=="__main__":
    file_dir = os.listdir("datas")
    for f_in in file_dir:
        X = load_data(f"datas/{f_in}")
        if (len(X.shape) > 1):
            plt.scatter(X[:, 0], X[:, 1])
            plt.title(f"{f_in}")
            plt.show()