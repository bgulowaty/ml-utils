from sklearn.neighbors import NearestNeighbors


def find_nearest_neighbors_indices(arr, point, n):
    nn = NearestNeighbors(n_neighbors=n)
    nn.fit(arr)
    return nn.kneighbors([point], n)[1].reshape(-1)