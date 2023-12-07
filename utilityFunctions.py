import numpy as np
import pandas as pd
import math
from fastdtw import fastdtw
from frechetdist import frdist
from scipy.spatial.distance import euclidean
from scipy.signal import periodogram, welch
# Function to standardize numerical variables
# Used for the arrhythmia and adult datasets


def standardize(df, num) : #from Haar et al.
    """
    df: DataFrame of the dataset
    num: List containing numerical column names
    """
    df_std = df.copy()
    for col in num :
        if col != 'target' :
            mean = df_std[col].mean()
            std = df_std[col].std()
            df_std[col] = (df_std[col]-mean)/std
    return df_std


def replace_nan(lst):
    # Convert the list to a numpy array

    arr = np.array(lst)
    # Find the indices of the nan values
    nan_indices = np.isnan(arr)

    # Find the indices of the first and last known values
    first_known_index = np.argmax(~nan_indices)

    last_known_index = len(lst) - 1 - np.argmax(~np.flip(nan_indices))

    # Replace the nan values with interpolated values
    for i in range(len(arr)):

        if nan_indices[i]:
            if i < first_known_index:
                arr[i] = arr[first_known_index]
            elif i > last_known_index:
                arr[i] = arr[last_known_index]
            else:
                prev_known_index = i - 1
                while np.isnan(arr[prev_known_index]):
                    prev_known_index -= 1
                next_known_index = i + 1
                while np.isnan(arr[next_known_index]):
                    next_known_index += 1
                prev_known_value = arr[prev_known_index]
                next_known_value = arr[next_known_index]
                interp_value = (prev_known_value + next_known_value) / 2
                arr[i] = interp_value

    # Convert the numpy array back to a list and return it
    return arr.tolist()


def remove_nans_at_edges(lst):
    start = 0
    end = len(lst) - 1
    while start <= end and math.isnan(lst[start]):
        start += 1
    while end >= start and math.isnan(lst[end]):
        end -= 1
    return lst[start:end+1]


def agg_ts(ts_df, agg_func="mean"):  # Isolates ts of a subgroup

    """Aggregation function for time-series in a dataframe.
    Isolates ts of a dataframe and combines the time-series into one ts in list format."""

    ts_df = pd.DataFrame(ts_df["target"])

    nr_tp = len(ts_df["target"][0])

    if agg_func != None:

        ts_df[[str("tp{t_point}".format(t_point=i)) for i in range(nr_tp)]] = pd.DataFrame(ts_df["target"].tolist(),
                                                                                           index=ts_df.index)

        if agg_func == "mean":
            ts_df = ts_df.drop("target", axis=1).mean(
                axis=0)

        elif agg_func == "sum":
            ts_df = ts_df.drop("target", axis=1).sum(axis=0)

        elif agg_func == "max":
            ts_df = ts_df.drop("target", axis=1).max(axis=0)

        elif agg_func == "min":
            ts_df = ts_df.drop("target", axis=1).min(axis=0)

    return list(ts_df)


def aggregate_timeseries_subgroup(df, col, agg_func="mean"):

    nan_mask = df[col].apply(lambda x: all(np.isnan(i) for i in x))
    df = df[~nan_mask]

    df[col] = df[col].apply(replace_nan)

    if agg_func == "sum":
        aggregated_ts = list(map(lambda x: sum(x), zip(*df[col])))

    else:

        aggregated_ts = list(map(lambda x: sum(x)/len(x), zip(*df[col])))

    return aggregated_ts


def min_max_norm(x):
    x = np.array(x)
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min)


def z_score_norm(x):
    x = np.array(x)

    mean = x.mean()
    std = x.std()
    return (x - mean) / std


def box_cox_transform(x, lambda_):

    x = np.array(x)
    if lambda_ == 0:
        return np.log(x)
    else:
        return (x ** lambda_ - 1) / lambda_


def percent_change_norm(x):

    x = replace_nan(x)
    x = np.array(x)
    if x[0] > 0:
        x_0 = x[0]
    else:
        # Find the indices of the nan values
        nan_indices = np.isnan(x)

        # Find the indices of the first and last known values
        first_known_index = np.argmax(~nan_indices)
        x_0 = x[first_known_index]

    return 100 * (x - x_0) / x_0


def differenced(lst):
    diff = [(lst[i] - lst[i-1]) for i in range(1, len(lst))]
    return diff


def percentual_differences(lst):
    pct_diff = [((lst[i] - lst[i-1]) / (lst[i-1]+0.00001)) * 100 for i in range(1, len(lst))]
    return pct_diff


def RMSE(x,y):
    return np.sqrt(np.mean((np.array(x) - np.array(y)) ** 2))

def dtw(x, y, window):
    x_new = []
    y_new = []

    for i, num in enumerate(x):
        x_new.append([num, i + 1])
    for i, num in enumerate(y):
        y_new.append([num, i + 1])
    distance, _ = fastdtw(x_new, y_new, dist=euclidean, radius=window)
    return distance


def frechet_distance(x, y):
    x_new = []
    y_new = []

    for i, num in enumerate(x):
        x_new.append([num, i + 1])
    for i, num in enumerate(y):
        y_new.append([num, i + 1])

    return frdist(x_new, y_new)


def _xlogx(x, base=2):
    """Returns x log_b x if x is positive, 0 if x == 0, and np.nan
    otherwise. This handles the case when the power spectrum density
    takes any zero value.
    """
    x = np.asarray(x)
    xlogx = np.zeros(x.shape)
    xlogx[x < 0] = np.nan
    valid = x > 0
    xlogx[valid] = x[valid] * np.log(x[valid]) / np.log(base)
    return xlogx


def spectral_entropy(x, sf = 1, method='fft', nperseg=None, normalize=False,
                     axis=-1):
    #https: // github.com / raphaelvallat / entropy / blob / master / entropy / entropy.py
    """The smaller the value the better to be predicted"""

    x = np.asarray(x)
    # Compute and normalize power spectrum
    if method == 'fft':
        _, psd = periodogram(x, sf, axis=axis)
    elif method == 'welch':
        _, psd = welch(x, sf, nperseg=nperseg, axis=axis)
    psd_norm = psd / psd.sum(axis=axis, keepdims=True)
    se = -_xlogx(psd_norm).sum(axis=axis)
    if normalize:
        se /= np.log2(psd_norm.shape[axis])

    return se


def calc_std(lst):
    return np.std(lst)


def entropy(subgroup_size, population_size):
    complement_size = population_size - subgroup_size
    return ((-subgroup_size / population_size) * math.log2(subgroup_size / population_size)) \
        - ((complement_size / population_size) * math.log2(complement_size / population_size))

def sqrt_size(subgroup_size, population_size):
    return math.sqrt((subgroup_size/population_size))

def no_size_corr(subgroup_size, population_size):
    return 1

def para_size(subgroup_size, population_size):
    return 1-((subgroup_size/population_size)-1)**10