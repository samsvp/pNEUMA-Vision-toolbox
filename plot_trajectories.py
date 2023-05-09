#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import *


class DBSCAN:

    def __init__(self, X: np.ndarray, eps: float, min_points: int) -> None:
        self.eps = eps
        self.min_points = min_points
        self.X = np.append(X, np.zeros((X.shape[0], 1)), axis=1)
        self.core_points = []


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
    

    def get_core_points(self) -> np.ndarray:
        """Returns all core points found after running the algorithm"""
        return self.X[self.X[:, -1] != 0]


#%%
def calculate_velocity(current_frame: int, vehicle_id: int, df: pd.DataFrame, 
                       last_velocity: Tuple[np.ndarray, float]) \
                            -> Tuple[np.ndarray, float]:
    current_vehicle = get_vehicle_frame(current_frame, vehicle_id, df)
    next_vehicle = get_vehicle_frame(current_frame + 1, vehicle_id, df)
    
    if current_vehicle is None or next_vehicle is None:
        return last_velocity
    
    v = next_vehicle[[2, 3]].values - current_vehicle[[2, 3]].values
    mag = np.linalg.norm(v)
    
    if mag == 0:
        return (np.array([[0, 0]]), 0)
    
    n_v = v / mag
    return (v, mag)


def get_vehicle_frame(frame: int, vehicle_id: int, df: pd.DataFrame):
    vehicles = df[df[0] == frame]
    vehicle = vehicles[vehicles[1] == vehicle_id]

    if vehicle.shape[0] == 0:
        return None
    
    return vehicle

#%%

filenames = ["0900_0930_D1_mot.txt", "0900_0930_D1_RM_mot.txt"]
filename = filenames[1]

df = pd.read_csv(filename, header=None)

plt.figure(figsize=(8, 6), dpi=80)
n_df = df[df[3] < 1200]
#n_df = n_df[n_df[2] < 3000]
n_df = n_df[n_df[3] > 1000]
n_df = n_df[n_df[0] < 500]
# add a column to count how many times the object with 
# an id has appeared up until the frame
n_df[10] = n_df.groupby(1).cumcount() + 1
plt.scatter(n_df[2], n_df[3], s=1)
plt.show()

# get the final and start locations of each car
idx = n_df.groupby(1)[10].idxmax()
plt.figure(figsize=(8, 6), dpi=80)
plt.scatter(n_df[n_df[10] == 1][2], n_df[n_df[10] == 1][3], s = 0.1)
plt.scatter(n_df.loc[idx][2], n_df.loc[idx][3], s = 0.1)

#%%
_n_df = n_df[list(range(6))]
_n_df.columns = ["frame", "id", "x", "y", "w", "h"]
# _n_df["w"] = _n_df["w"] / 3840
_n_df.loc[:, "x"] = _n_df["x"] + _n_df["w"] / 2
# _n_df["h"] = _n_df["h"] / 2160
_n_df.loc[:, "y"] = _n_df["y"] + _n_df["h"]
vehicles_dict = _n_df.sample(10_000).to_dict("list")
vehicles_dict["vx"] = []
vehicles_dict["vy"] = []
vehicles_dict["mag"] = []

for i in range(len(vehicles_dict["frame"])):
    frame = vehicles_dict["frame"][i]
    vehicle_id = vehicles_dict["id"][i]
    v, mag = calculate_velocity(frame, vehicle_id, n_df, (np.array([[0, 0]]), 0))
    vehicles_dict["vx"].append(v[0, 0])
    vehicles_dict["vy"].append(v[0, 1])
    vehicles_dict["mag"].append(mag)

plt.figure(figsize=(8, 6), dpi=80)
veh_df = pd.DataFrame(vehicles_dict)
mask = veh_df["vx"] ** 2 + veh_df["vy"] ** 2 >= 200

veh_df = veh_df[mask]
plt.scatter(veh_df["x"], veh_df["y"], s=1)

X = veh_df[["x", "y", "vx", "vy"]].values


# %%
eps = 10
dbscan = DBSCAN(X, eps, 10)()
cluster = set(dbscan.X[:, -1])
print(len(cluster))
plt.figure(figsize=(8, 6), dpi=80)
for clust in cluster:
    plt.scatter(dbscan.X[dbscan.X[:, -1] == clust][:, 0], 
                dbscan.X[dbscan.X[:, -1] == clust][:, 1], 
                s=1, label=f"Cluster{clust}")

# %%
