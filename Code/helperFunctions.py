import pandas as pd
import numpy as np
import re
import math


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


def as_string(desc):
    # Adds ' and ' to <desc> such that selectors are properly separated when the refine function is used
    return ' and '.join(desc)


def update_std_dev(new_data, mean, std_dev, n): 
        new_mean = (mean * n + new_data) / (n + 1) 
        new_std_dev = math.sqrt(((n * std_dev**2) + (new_data - mean)**2) / (n + 1)) 
        return new_std_dev


def convert_columns_num_to_cat(df,nr_chunks=5):
    for col in df.columns:
        if (df[col].dtype == 'float64') or (df[col].dtype == 'float32') or (df[col].dtype == 'int64'):
            dat = np.sort(df[col])
            dat = dat[np.logical_not(np.isnan(dat))]
            for i in range(1, nr_chunks+1):
                # determine the number of chunks you want to divide your data in
                x = np.percentile(dat, (i-1)*100/nr_chunks)  #
                y = np.percentile(dat, (i)*100/nr_chunks)
                candidate = "{} <= {} <= {}".format(x ,col, y)
                df[col] = df[col].apply(lambda val: candidate if (not isinstance(val, str) and x <= val <= y) else val)

    return df


def unique_descriptions(descriptions):
    """transforms a list of descriptions into a list of unique descriptions"""
    unique_values = []
    unique_values_result = []
    for item in descriptions:
        if str(item) not in unique_values:
            unique_values.append(str(item))
            unique_values_result.append(item)
    return unique_values_result

