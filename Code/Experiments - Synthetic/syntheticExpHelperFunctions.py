from Code.qualityMeasure import *
from itertools import combinations
import bisect


def true_subgroup_description_analysis(true_subgroup_description, found_descriptions):
    """
    returns boolean indicating if a subgroup description is found or not and the rank of it

    """
    found_true_subgroup = set(true_subgroup_description) in [set(i) for i in found_descriptions]

    rank_ts_exact = next(
        (i for i, description in enumerate(found_descriptions) if set(description) == set(true_subgroup_description)),
        None)

    return found_true_subgroup, rank_ts_exact


def true_subgroup_description_analysis_for_small_d(true_subgroup_description, found_descriptions, d):
    """
    Analyze a list of found descriptions to locate the first description that contains
    at least one subset of size d from the true subgroup description.

    Parameters:
        true_subgroup_description (iterable): The true subgroup description from which
                                              smaller subsets will be derived.
        found_descriptions (list of iterables): A list of candidate descriptions to search through.
        d (int): The size of subsets to consider from the true_subgroup_description.

    Returns:
        tuple: A one-element tuple containing the index of the first found description that
               contains a subset of size d from the true subgroup description, or (None,)
               if no such description exists.
    """

    # Use a generator expression to iterate through found_descriptions with their indices.
    # For each description, check if any combination of size d from the true_subgroup_description
    # is fully contained within that description.
    rank_ts_subset = next(
        (
            i  # The index of the current description in found_descriptions.
            for i, description in enumerate(found_descriptions)
            # Check if at least one combination of size d from true_subgroup_description
            # is a subset of the current description.
            if any(set(subset).issubset(description) for subset in combinations(true_subgroup_description, d))
        ),
        None  # Default value if no such description is found.
    )

    # Return the result as a one-element tuple.
    return rank_ts_subset,


def true_subgroup_quality_analysis(true_subgroup_description, result, dataframe, matrix):
    true_subgroup_quality, _ = eval_quality(dataframe.query(as_string(true_subgroup_description)), dataframe, 'target',
                                         cluster_based_quality_measure, 'complement', distance_matrix=matrix,
                                         correct_for_size=no_size_corr)

    """
    returns the rank of the true subgroup description in the results or nan if true subgroup description is not found
    returns boolean indicating if all found subgroups have better quality then true subgroup quality 
    """

    found_qualities = [i[0] for i in result]

    true_subgroup_rank = bisect.bisect_left([-x for x in found_qualities], -true_subgroup_quality) + 1
    all_qualities_better_than_true_subgroup = true_subgroup_rank > len(found_qualities)
    if true_subgroup_rank > len(found_qualities):
        true_subgroup_rank = np.nan

    return true_subgroup_rank, all_qualities_better_than_true_subgroup