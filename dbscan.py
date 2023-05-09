#%%
import numpy as np
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from typing import *


class DBSCAN:

    def __init__(self, X: np.ndarray, eps: float, min_points: int) -> None:
        self.eps = eps
        self.min_points = min_points
        self.X = np.append(X, np.zeros((X.shape[0], 1)), axis=1)


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Run DBSCAN"""
        points = self.X
        current_label = 1
        for point in points:
            # label already assigned
            if point[-1] != 0:
                continue

            is_core, neighbors_idx = self.is_core_point(point)
            
            # check if is core
            if not is_core:
                continue

            self.assign_label(point, neighbors_idx, current_label)

            current_label += 1
        
        return self


    def assign_label(self, point: np.ndarray, neighbors_idx: np.ndarray, label: int):
        """Recursive label assigned"""
        point[-1] = label
        while True:
            next_neighbors = np.array([], dtype=int)
            for i in neighbors_idx:
                # if label has been set, continue
                if self.X[i, -1]:
                    continue

                self.X[i, -1] = label
                is_core, _neighbors_idx = self.is_core_point(self.X[i])
                if not is_core:
                    continue
                next_neighbors = np.concatenate((next_neighbors, _neighbors_idx), dtype=int)

            if next_neighbors.shape[0] == 0:
                break

            neighbors_idx = next_neighbors


    def is_core_point(self, point: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Returns if the given point is a core point"""
        neighbors_idx = self.find_neighbors(point)
        return (neighbors_idx.shape[0] >= self.min_points, neighbors_idx)


    def find_neighbors(self, _point: np.ndarray) -> np.ndarray:
        """
        Return a mask of the indexes of the points which are 
        neighbors to the given point
        """
        point = _point[:-1]
        diff = self.X[:, :-1] - point
        dists = np.linalg.norm(diff, axis=1)
        neighbors_idx = np.where(dists < self.eps)[0]
        return neighbors_idx


# %%
if __name__ == "__main__":
    X, y = make_circles(n_samples=1_000, factor=0.2, noise=0.1, random_state=0)

    plt.scatter(X[y == 0, 0], X[y == 0, 1], s=10, label="Cluster1")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], s=10, label="Cluster2")
    plt.title("Scattered data")
    plt.show()

    d = DBSCAN(X, 0.2, 15)()
    cluster = set(d.X[:, -1])
    print(len(cluster))
    for clust in cluster:
        plt.scatter(d.X[d.X[:, -1] == clust][:, 0], 
                    d.X[d.X[:, -1] == clust][:, 1], 
                    s=10, label=f"Cluster{clust}")
    # %%
    centers = [(4, 3), (6, 5) , (8,2)]
    cluster_std = [0.7, 0.8, 0.7]

    X, y = make_blobs(
        n_samples=3050, centers=centers, cluster_std=0.4, random_state=0
    )

    X = StandardScaler().fit_transform(X)

    plt.scatter(X[y == 0, 0], X[y == 0, 1], s=10, label="Cluster1")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], s=10, label="Cluster2")
    plt.scatter(X[y == 2, 0], X[y == 2, 1], s=10, label="Cluster3")
    plt.title("Scattered data")
    plt.show()

    d = DBSCAN(X, 0.2, 15)()
    cluster = set(d.X[:, -1])
    print(len(cluster))
    for clust in cluster:
        plt.scatter(d.X[d.X[:, -1] == clust][:, 0], 
                    d.X[d.X[:, -1] == clust][:, 1], 
                    s=10, label=f"Cluster{clust}")
    # %%
