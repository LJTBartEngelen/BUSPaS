import math
import heapq
import numpy as np


# Functions
def refine(desc, more):
    # Creates a copy of the seed <desc> and adds it to the new selector <more>
    # Used to prevent pointer issues with selectors
    copy = desc[:]
    copy.append(more)
    return copy


def as_string(desc):
    # Adds ' and ' to <desc> such that selectors are properly separated when the refine function is used
    return ' and '.join(desc)


def eta(seed, df, features, n_chunks=5, allow_exclusion=True):
    # Returns a generator which includes all possible refinements of <seed> for the given <features> on dataset <df>
    # n_chunks refers to the number of possible splits we consider for numerical features

    print("eta ", seed)

    # TODO ETA implement that new descriptions are not too much overlapping with old descriptions
    # TODO ETA.1 no two same sided numerical operators from same attribute
    # TODO ETA.2 no exclusion of one (non-frequent) att-value for attr that contains large amount of values

    # TODO work out idea to give rewards or penalties to interesting attributes
    #  (country and sector more interesting than volume , inclusion more interesting than exclusion)

    if seed != []:  # we only specify more on the elements that are still in the subset
        d_str = as_string(seed)
        ind = df.eval(d_str)
        df_sub = df.loc[ind,]
    else:
        df_sub = df
    for f in features:
        column_data = df_sub[f]
        if (df_sub[f].dtype == 'float64') or (df_sub[f].dtype == 'float32'):
            # get quantiles here instead of intervals for the case that data are very skewed

            dat = np.sort(column_data)
            dat = dat[np.logical_not(np.isnan(dat))]
            try:
                for i in range(1, n_chunks + 1):
                    # determine the number of chunks you want to divide your data in
                    x = np.percentile(dat, i*100/n_chunks)  # 100/i
                    candidate = "{} <= {}".format(f, x)
                    if not candidate in seed:  # if not already there
                        yield refine(seed, candidate)
                    candidate = "{} > {}".format(f, x)
                    if not candidate in seed:  # if not already there
                        yield refine(seed, candidate)
            except IndexError:
                pass
        elif (df_sub[f].dtype == 'object'):

            uniq = column_data.dropna().unique()

            for i in uniq:
                candidate = "{} == '{}'".format(f, i)
                if not candidate in seed:  # if not already there
                    yield refine(seed, candidate)
                candidate = "{} != '{}'".format(f, i)
                if (not candidate in seed) and allow_exclusion:  # if not already there
                    yield refine(seed, candidate)
        elif df_sub[f].dtype == 'int64':

            dat = np.sort(column_data)
            dat = dat[np.logical_not(np.isnan(dat))]
            for i in range(1, n_chunks + 1):
                # determine the number of chunks you want to divide your data in
                x = np.percentile(dat, 100 / i)  #
                candidate = "{} <= {}".format(f, x)
                if not candidate in seed:  # if not already there
                    yield refine(seed, candidate)
                candidate = "{} > {}".format(f, x)
                if not candidate in seed:  # if not already there
                    yield refine(seed, candidate)
        elif df_sub[f].dtype == 'bool':

            uniq = column_data.dropna().unique()
            for i in uniq:
                candidate = "{} == '{}'".format(f, i)
                if not candidate in seed:  # if not already there
                    yield refine(seed, candidate)
                candidate = "{} != '{}'".format(f, i)  # TODO Exculsion hier zetten ipv los
                if (not candidate in seed) and allow_exclusion:  # if not already there
                    yield refine(seed, candidate)
        else:
            assert False