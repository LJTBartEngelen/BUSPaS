from tqdm import tqdm
from itertools import combinations
from Code.helperFunctions import *
from scipy.spatial.distance import euclidean
import random
import warnings


# TODO add DTW as distance measures

# TODO think of early stopping

# TODO add possibility to change weights for tp's far in the future

# TODO consider other linkage methods p32 Book TS clustering

# TODO function (5.22) of p59 Book TS clustering

# TODO beschrijf p70 PACF clustering

def eval_quality(sub_group, df, target, quality_measure, comparison_type='complement', **kwargs):
    if comparison_type == 'population':
        complement_df = df
    elif comparison_type == 'complement':
        complement_df = df.loc[~df.index.isin(sub_group.index)]
    else:
        complement_df = df

    phi = quality_measure(sub_group, target, complement_df, **kwargs)

    coverage = len(sub_group) / len(df)

    return phi, coverage


def euclidean_distance_time_points(x, y):
    """
    Calculate the Euclidean distance between two arrays x and y representing time series
    based on the time points.
    """
    return euclidean(x, y)


def euclidean_distance_slopes(x, y):
    """
    Calculate the Euclidean distance between two arrays x and y representing time series
    based on the time points.
    """

    x = differenced(x)
    y = differenced(y)

    return euclidean(x, y)


def euclidean_distance_concav(x, y):
    """
    Calculate the Euclidean distance between two arrays x and y representing time series
    based on the time points.
    """

    x = differenced(differenced(x))
    y = differenced(differenced(y))

    return np.sqrt(np.sum((x - y) ** 2))


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

    for i in range(num_stocks):
        for j in range(i + 1, num_stocks):
            ts1, ts2 = df.loc[stocks[i], 'target'], df.loc[stocks[j], 'target']
            dist = distance_function(np.array(ts1), np.array(ts2))
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix


def swap_randomize_symmetric(matrix):
    """
    Returns a swap-randomized version of a symmetric n x n matrix with zero diagonal.

    Parameters:
        matrix (np.array): Input symmetric matrix with zero diagonal.

    Returns:
        np.array: A swap-randomized symmetric matrix with zero diagonal.
    """
    n = matrix.shape[0]

    # Step 1: Get the upper triangle indices (excluding the diagonal)
    triu_indices = np.triu_indices(n, k=1)

    # Step 2: Extract and shuffle the upper triangle values
    upper_triangle_values = matrix[triu_indices]
    np.random.shuffle(upper_triangle_values)

    # Step 3: Create a new matrix and assign the shuffled values to the upper triangle
    result = np.zeros_like(matrix)  # Initialize with zeros to keep the diagonal zero
    result[triu_indices] = upper_triangle_values  # Set shuffled values in upper triangle
    result[(triu_indices[1], triu_indices[0])] = upper_triangle_values  # Reflect to lower triangle

    return result


def calculate_intra_subgroup_distance(subgroup_df, distance_matrix):
    """
    Calculate the average distance between all pairs of stocks in the given subgroup
    based on the pre-computed distance matrix.

    Returns a float representing the average distance.
    """
    subgroup_indices = subgroup_df.index.to_numpy()
    pairwise_distances = distance_matrix[np.ix_(subgroup_indices, subgroup_indices)].astype(np.float32)
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
    # Attempt to sum in float16
    num_pairs = len(subgroup_indices) * len(complement_indices)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        total_distance = np.nansum(pairwise_distances, dtype=np.float16)
        inter_subgroup_distance = total_distance / max(1, num_pairs)

    # If result is inf, try float32, then float64 if necessary
    if np.isinf(total_distance) or np.isinf(inter_subgroup_distance):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            total_distance = np.nansum(pairwise_distances, dtype=np.float32)
            inter_subgroup_distance = total_distance / max(1, num_pairs)
            if np.isinf(total_distance) or np.isinf(inter_subgroup_distance):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    total_distance = np.nansum(pairwise_distances, dtype=np.float64)
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
    inter = calculate_inter_subgroup_distance(subgroup_df, complement_df, distance_matrix)

    intra = calculate_intra_subgroup_distance(subgroup_df, distance_matrix) + 1

    phi = size_correction(n, N) * inter / intra

    return phi


#TODO Only for BUSPaS Algo
def difference_abs(subgroup_df, col, complement_df, **kwargs):

    """
    quality measure for numerical single targets absolute difference between two subgroups
    """

    n = len(subgroup_df)
    n_c = len(complement_df)

    size_correction = kwargs['correct_for_size']

    subgroup_average = sum(subgroup_df[col])/n
    complement_average = sum(complement_df[col])/n_c

    difference = subgroup_average - complement_average
    difference_abs = abs(difference)
    phi = difference_abs

    N = n + n_c

    return phi * size_correction(n, N)


def difference_pos(subgroup_df, col, complement_df, **kwargs):
    """
    quality measure for numerical single targets difference between two subgroups
    looks for largest difference where subgroup is larger
    """

    n = len(subgroup_df)
    n_c = len(complement_df)

    size_correction = kwargs['correct_for_size']

    subgroup_average = sum(subgroup_df[col])/n
    complement_average = sum(complement_df[col])/n_c

    difference = subgroup_average - complement_average
    phi = difference

    N = n + n_c

    return phi * size_correction(n, N)


def difference_neg(subgroup_df, col, complement_df, **kwargs):
    """
    quality measure for numerical single targets difference between two subgroups
    looks for largest difference where subgroup is smaller
    """

    n = len(subgroup_df)
    n_c = len(complement_df)

    size_correction = kwargs['correct_for_size']

    subgroup_average = sum(subgroup_df[col])/n
    complement_average = sum(complement_df[col])/n_c

    difference = subgroup_average - complement_average
    phi = -difference

    N = n + n_c

    return phi * size_correction(n, N)


def difference_profit(subgroup_df, col, complement_df, **kwargs):
    """
    quality measure for numerical single targets difference between two subgroups
    looks for largest difference where subgroup is smaller
    """

    revenue_column = col[0]
    cost_column = col[1]

    n = len(subgroup_df)
    n_c = len(complement_df)

    size_correction = kwargs['correct_for_size']

    if n>0:
        subgroup_average = (sum(subgroup_df[revenue_column])-sum(subgroup_df[cost_column]))/n
    else:
        subgroup_average = 0

    if n_c>0:
        complement_average = (sum(complement_df[revenue_column])-sum(complement_df[cost_column]))/n_c
    else:
        complement_average = 0

    difference = subgroup_average - complement_average
    phi = difference

    N = n + n_c

    return phi * size_correction(n, N)
