from tqdm import tqdm
from itertools import combinations
from helperFunctions import *

from scipy.spatial.distance import euclidean

def euclidean_distance_time_points(x, y):
    """
    Calculate the Euclidean distance between two arrays x and y representing time series
    based on the time points.
    """
    return euclidean(x,y)


def euclidean_distance_slopes(x, y):
    """
    Calculate the Euclidean distance between two arrays x and y representing time series
    based on the time points.
    """

    x = differenced(x)
    y = differenced(y)

    return euclidean(x,y)


def euclidean_distance_concav(x, y):
    """
    Calculate the Euclidean distance between two arrays x and y representing time series
    based on the time points.
    """

    x = differenced(differenced(x))
    y = differenced(differenced(y))

    return np.sqrt(np.sum((x - y) ** 2))

# TODO add DTW as distance measures

# TODO think of early stopping

# TODO add possibility to change weights for tp's far in the future

# TODO consider other linkage methods p32 Book TS clustering

# TODO function (5.22) of p59 Book TS clustering

# TODO beschrijf p70 PACF clustering

def polygonal_distance(x, y):

    def polygonal_coefficients(x):
        A = x[0] + x[-1]
        b1 = 0.5 * (x[1] - x[0] + A)
        bt = 0.5 * ((x[2:] - x[1:-1]) - (x[1:-1] - x[:-2]))
        bT = 0.5 * (A - x[-2] + x[-1])
        return np.concatenate(([b1], bt, [bT]))

    px = polygonal_coefficients(x)
    py = polygonal_coefficients(y)
    return np.sqrt(np.sum((px - py) ** 2))


def calculate_distance_matrix(df, distance_function=euclidean):

    """
    Calculate pairwise distances between all pairs of stocks in the DataFrame df
    based on their time series using the polygonal distance measure.

    Returns a NumPy array where the indices correspond to stock pairs, and the values
    are the corresponding distances.
    """

    stocks = df.index.tolist()
    num_stocks = len(stocks)
    distance_matrix = np.zeros((num_stocks, num_stocks))

    #with tqdm(total=num_stocks * (num_stocks - 1) // 2) as pbar:
    for i in range(num_stocks):
        for j in range(i + 1, num_stocks):
            ts1, ts2 = df.loc[stocks[i], 'target'], df.loc[stocks[j], 'target']
            dist = distance_function(np.array(ts1), np.array(ts2))
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
                #pbar.update(1)

    return distance_matrix


def calculate_intra_subgroup_distance(subgroup_df, distance_matrix):
    """
    Calculate the average distance between all pairs of stocks in the given subgroup
    based on the pre-computed distance matrix.

    Returns a float representing the average distance.
    """
    subgroup_indices = subgroup_df.index.to_numpy()
    pairwise_distances = distance_matrix[np.ix_(subgroup_indices, subgroup_indices)]
    num_pairs = len(subgroup_indices) * (len(subgroup_indices) - 1) // 2
    total_distance = np.nansum(pairwise_distances) / 2
    intra_subgroup_distance = total_distance / max(1, num_pairs)
    return intra_subgroup_distance


def calculate_inter_subgroup_distance(subgroup_df, complement_df, distance_matrix):
    """
    Calculate the average distance between all pairs of stocks in the given subgroup and
    complement subgroup based on the pre-computed distance matrix.

    Returns a float representing the average distance.
    """
    subgroup_indices = subgroup_df.index.to_numpy()
    complement_indices = complement_df.index.to_numpy()
    pairwise_distances = distance_matrix[np.ix_(subgroup_indices, complement_indices)]
    num_pairs = len(subgroup_indices) * len(complement_indices)
    total_distance = np.nansum(pairwise_distances)
    inter_subgroup_distance = total_distance / max(1, num_pairs)
    return inter_subgroup_distance


def cluster_based_quality_measure(subgroup_df, col, complement_df, **kwargs):

    """calculates ratio between inter subgroup distance and intra subgroup distance,
    corrected for the size

    Based on Verhaegh 2022

    """

    n = len(subgroup_df)
    n_c = len(complement_df)
    N = n + n_c

    distance_matrix = kwargs['distance_matrix']
    size_correction = kwargs['correct_for_size']

    print('inter= ',calculate_inter_subgroup_distance(subgroup_df, complement_df, distance_matrix))
    print('intra+1= ',(calculate_intra_subgroup_distance(subgroup_df, distance_matrix) + 1 ))
    
    return size_correction(n, N) * calculate_inter_subgroup_distance(subgroup_df, complement_df, distance_matrix) \
        / (calculate_intra_subgroup_distance(subgroup_df, distance_matrix) + 1 )
