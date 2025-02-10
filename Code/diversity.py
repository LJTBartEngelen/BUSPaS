import numpy as np
import pandas as pd
import itertools as it
import re
from collections import defaultdict
from Code.qualityMeasure import *
from Code.helperFunctions import *

#TODO Add a distance based diversity measure (self-invented case specific)
#TODO Test dominance pruning

#__________ description based _____________#

def fixed_size_desc_selection(result, k, result_order_qdc=(0, 1, 2)):
    """
    Based on text of (van Leeuwen 2012)

    Perform fixed-size description-based subgroup selection.

    Transforms a result list in a pruned version of size k.

    Args:
    result: List of tuples, where each tuple contains (quality, description, coverage).
            - quality: A float representing the quality of the subgroup.
            - description: A list of conditions representing the subgroup description.
            - coverage: A set of elements representing the coverage of the subgroup.
    k: The number of subgroups to select.
    order: A tuple of three integers representing the order of (quality, description, coverage)
           in the input result tuples. Default is (0, 1, 2) for (quality, description, coverage).

    Returns:
    sel: A list of selected subgroups (quality, description, coverage).
    """

    # Extract ordered indices for quality, description, and coverage
    quality_idx, description_idx, coverage_idx = result_order_qdc

    # Sort the candidate subgroups in descending order of quality based on specified order
    sorted_result = sorted(result, key=lambda x: -x[quality_idx])

    # Initialize the selected subgroup set
    sel = []

    # Iterate over the sorted candidate subgroups
    for item in sorted_result:
        quality = item[quality_idx]
        description = item[description_idx]
        coverage = item[coverage_idx]

        # Check if the current candidate subgroup can be added to the selection
        should_select = True
        to_remove = None
        for selected in sel:
            selected_quality = selected[0]
            selected_description = selected[1]
            selected_coverage = selected[2]

            # If quality is the same and the overlap of conditions is max(len(selected_description), len(description)) - 1, skip the candidate
            if quality == selected_quality:
                len_desc = len(description)
                len_sel_desc = len(selected_description)
                max_len = max(len_desc, len_sel_desc)
                common_conditions = sum(1 for cond in description if cond in selected_description)

                if common_conditions == max_len - 1 or common_conditions == max_len:
                    should_select = False
                    if len_desc < len_sel_desc:
                        # Mark the previously selected subgroup for removal
                        to_remove = selected
                        should_select = True
                    break

        # Remove the previously selected subgroup if applicable
        if to_remove:
            sel.remove(to_remove)

        # If the candidate passes the test, add it to the selection
        if should_select:
            sel.append((quality, description, coverage))

        # Stop when we have selected k subgroups
        if len(sel) == k:
            break

    return sel


def extract_attribute(condition):
    """
    Extract the attribute part of a condition, ignoring the operator and value.
    Supports operators: =, <, >, <=, >=, !=

    Args:
    condition: A string representing a condition (e.g., "A=True", "B>5").

    Returns:
    attribute: The attribute part of the condition (e.g., "A", "B").
    """
    # Regular expression to match attribute followed by an operator and a value
    match = re.match(r"([a-zA-Z_]+)\s*(=|<|>|<=|>=|!=)", condition)
    if match:
        return match.group(1)  # Return the attribute (group 1 of the match)
    return condition  # Fallback, return the entire condition if no operator is found


def variable_size_desc_selection(result, c, l, result_order_qdc=(0, 1, 2)):
    """
    Based on text of (van Leeuwen 2012)

    Perform variable-size description-based subgroup selection.

    Transforms a result list in a pruned version of size l.

    Args:
    result: List of tuples, where each tuple is (quality, description, coverage).
            - quality: A float representing the quality of the subgroup.
            - description: A list of conditions representing the subgroup description.
            - coverage: A set of elements representing the coverage of the subgroup.
    c: The maximum number of times an attribute can appear in a condition in the subgroup set.
    l: The maximum length of a description.

    Returns:
    sel: A list of selected subgroups (quality, description, coverage).
    """

    quality_idx, description_idx, coverage_idx = result_order_qdc

    # Sort the candidate subgroups in descending order of quality based on specified order
    sorted_result = sorted(result, key=lambda x: -x[quality_idx])

    # Initialize the selected subgroup set
    sel = []

    # A dictionary to keep track of the number of times each attribute appears in the selected conditions
    attribute_usage = defaultdict(int)

    # Iterate over the sorted candidate subgroups
    for item in sorted_result:

        quality = item[quality_idx]
        description = item[description_idx]
        coverage = item[coverage_idx]

        # Check if any attribute in the subgroup's conditions has been used too many times
        can_add = True
        for condition in description:
            attribute = extract_attribute(condition)  # Extract the attribute from the condition
            if attribute_usage[attribute] >= c * l:
                can_add = False
                break
        
        # If the candidate passes the test, add it to the selection and update attribute usage
        if can_add:
            sel.append((quality, description, coverage))
            for condition in description:
                attribute = extract_attribute(condition)
                attribute_usage[attribute] += 1
        
        # Stop if all attributes have reached the maximum usage limit
        if all(usage >= c * l for usage in attribute_usage.values()):
            break

    return sel

#_____________ cover based ________________#


def cover_score(subgroup_coverage, selected_coverages, alpha=1.0):
    """
    Calculate the multiplicative weighted covering score for a subgroup.

    Args:
    subgroup_coverage: The coverage of the current subgroup (a set of covered elements).
    selected_coverages: A list of sets representing the coverage of already selected subgroups.
    alpha: A weighting parameter (default is 1.0, can range between 0 and 1).

    Returns:
    score: The covering score for the subgroup.
    """

    score = 0
    for t in subgroup_coverage:
        # Count how many times t is already covered by selected subgroups
        already_covered_count = sum(1 for coverage in selected_coverages if t in coverage)
        score += alpha ** already_covered_count

    return score / len(subgroup_coverage)


def fixed_size_cover_based_selection(result, df, k, alpha, result_order_qdc = (0, 1, 2)):
    """
    Based on text of (van Leeuwen 2012)

    Perform variable-size description-based subgroup selection.

    Transforms a result list in a pruned version of size k.

    Args:
    result: List of tuples, where each tuple is (quality, description, coverage).
            - quality: A float representing the quality of the subgroup.
            - description: A list of conditions representing the subgroup description.
            - coverage: A set of elements representing the coverage of the subgroup.
    k:
    alpha:

    Returns:
    sel: A list of selected subgroups (quality, description, coverage).
    """

    quality_idx, description_idx, coverage_idx = result_order_qdc

    # Sort the candidate subgroups in descending order of quality
    sorted_result = sorted(result, key=lambda x: -x[quality_idx])
    extents = []
    for i in sorted_result:
        extents.append(list(df.query(as_string(i[description_idx])).index))

    sel = [sorted_result[quality_idx]]
    sel_ext = [extents[0]]
    sorted_result.remove(sorted_result[quality_idx])
    extents.remove(extents[0])

    for _ in range(k-1):
        best_score = -float('inf')
        best_candidate = None
        best_candidate_extent = None
        best_candidate_i = None
        for i in range(len(sorted_result)):
            omega = cover_score(extents[i], sel_ext, alpha=alpha)

            quality = sorted_result[i][quality_idx]
            score = omega * quality
             
            if score > best_score:
                best_score = score
                best_candidate = sorted_result[i]
                best_candidate_extent = extents[i]
                best_candidate_i = i
                
        sel.append(best_candidate)
        sel_ext.append(best_candidate_extent)
        sorted_result.remove(best_candidate)
        del extents[best_candidate_i]
    
    return sel

    
#___________ dominance pruning ____________#

def get_new_descriptions(subgroup=None,description_idx=1):

    pruned_descriptions_ = []
    items_old_desc = subgroup[description_idx]
    #items_old_desc = old_desc.items()

    for r in np.arange(1, len(list(items_old_desc))):
        combs = list(it.combinations(items_old_desc, r=r))
        combs_r = [list(i) for i in combs]
        # combs_r = [{'description': list(desc)} for desc in combs]
        pruned_descriptions_.extend(combs_r)
        pruned_descriptions_.append(subgroup[description_idx])
    return sorted(pruned_descriptions_, key=len)


def get_quality_dp(desc,df,matrix):
    sg = df.query(as_string(desc))
    quality, coverage = eval_quality(sg, df, 'target', cluster_based_quality_measure,distance_matrix=matrix,
                                     correct_for_size=no_size_corr, comparison_type="complement")
    return quality


def dominance_pruning(results, df, matrix, result_order_qdc=(0, 1, 2)):

    """
    Based on text of (van Leeuwen 2012)

    Transforms a result list in a version with adapted descriptions that are less redundant.

    emm_result = [(quality, description, coverage),...,...,...] """

    quality_idx, description_idx, coverage_idx = result_order_qdc

    quals_dict = {}
    best_candidate_descs = []
    for desc in results:
        if len(desc[description_idx]) < 2:
            best_candidate_desc = desc[description_idx]
        else:
            best_quality = desc[quality_idx]
            best_candidate_desc = desc[description_idx]
            quals_dict[str(desc[description_idx])] = desc[quality_idx]
            candidate_set = get_new_descriptions(desc, description_idx)
            for desc_candidate in candidate_set:
                if str(desc_candidate) in quals_dict:
                    qual = quals_dict[str(desc_candidate)]
                else:
                    quals_dict[str(desc_candidate)] = get_quality_dp(desc_candidate, df, matrix)
                    qual = quals_dict[str(desc_candidate)]
                if qual >= best_quality and len(best_candidate_desc)>len(desc_candidate): ###TODO Change if we're also going to work qualities that improve by decreasing
                    best_candidate_desc = desc_candidate
        best_candidate_descs.append(best_candidate_desc)

    return unique_descriptions(best_candidate_descs)


#___________ local patterns, redundancy and diverstiy ____________#

def jaccard_similarity(list1, list2):
    """
    Calculate the Jaccard similarity between two lists of integers.

    Args:
    list1: First list of integers.
    list2: Second list of integers.

    Returns:
    float: The Jaccard similarity between the two lists.
    """
    set1 = set(list1)
    set2 = set(list2)

    # Calculate the intersection and union
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    # Handle the case where both sets are empty
    if union == 0:
        return 1.0  # Jaccard similarity is defined as 1 when both sets are empty

    # Calculate Jaccard similarity
    return intersection / union


def calculate_coverage_distance_matrix(extents):

    """returns the coverage distance matrix based on the extents of the found subgroups and the Jaccard similarity
    and the keys to understand which descriptions are compared"""

    # Extract keys and values from the dictionary
    keys = list(extents.keys())
    values = np.array(list(extents.values()))

    # Create an empty matrix to hold the distances
    num_items = len(keys)
    distance_matrix = np.zeros((num_items, num_items))

    # Fill the distance matrix
    for i in range(num_items):
        for j in range(i, num_items):
            distance = jaccard_similarity(values[i], values[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Ensure symmetry

    return distance_matrix, keys


def redundancy(results, df, similarity_threshold=0.5, result_order_qdc=(0, 1, 2),
               retrieve_neighbourhoods=False, coverage_distance_matrix=False):
    """
    Based on (Bosc 2018)'s Redundancy, filter, and neighborhood definitions

    Select only the best subgroup from each group with Jaccard similarity above a threshold,
    and group similar subgroups.

    Args:
    results: List of lists, where each list contains [quality, description, coverage].
    df: DataFrame containing the data for subgroup evaluation.
    similarity_threshold: A float value indicating the Jaccard similarity threshold.
    result_order_qdc: List indicating the order of [quality, description, coverage] in results.

    Returns:
    tuple: Redundancy score, list of the best subgroups, and list of grouped similar subgroups.
            High redundancy Dom[0-1] is worse.
    """

    if len(results) == 0:
        redundancy_score, result_local_optima, neighbourhoods, (coverage_distance_matrix, keys) \
            = None, None, None, (None,None)
    else:
        quality_idx, description_idx, coverage_idx = result_order_qdc

        # Sort results by quality in descending order
        results = sorted(results, key=lambda x: x[quality_idx], reverse=True)

        # Initialize lists for best subgroups and groups of similar subgroups
        local_optima = []
        neighbourhood_pairs = []
        extents = None
        if coverage_distance_matrix:
            extents = {}

        # Iterate through each subgroup
        for result in results:
            quality = result[quality_idx]
            description = result[description_idx]
            coverage = result[coverage_idx]
            extent = df.query(as_string(description)).index
            extent_set = set(extent)
            if extents:
                extents[description] = extent_set

            # Track if this subgroup is unique or better than similar ones in local_optima
            to_remove = []
            is_unique_or_better = True

            # Check for similarity with already selected subgroups
            for idx, (best_quality, best_extent, best_cov, best_desc) in enumerate(local_optima):
                best_extent_set = set(best_extent)
                if jaccard_similarity(extent_set, best_extent_set) > similarity_threshold:
                    neighbourhood_pairs.append([best_desc, description])  # Use lists instead of tuples

                    if quality > best_quality:
                        to_remove.append(idx)
                        is_unique_or_better = True
                        break
                    else:
                        is_unique_or_better = False
                        break

            # Remove overruled subgroups
            for idx in sorted(to_remove, reverse=True):  # Reverse to maintain index consistency
                del local_optima[idx]
            # Add to local_optima if it is unique or the best within its similarity group
            if is_unique_or_better:
                local_optima.append([quality, extent, coverage, description])

        # Extract the best subgroups in the required output format
        result_local_optima = [[quality, description, coverage] for quality, _, coverage, description in local_optima]
        redundancy_score = 1 - len(result_local_optima) / len(results)

        # Create the neighbourhoods list by clustering descriptions based on similarity
        neighbourhoods = []
        if retrieve_neighbourhoods:
            for quality, _, _, description in local_optima:
                current_neighbourhood = [description]
                for neighbourhood_pair in neighbourhood_pairs:
                    if neighbourhood_pair[1] == description:
                        current_neighbourhood.append(neighbourhood_pair[0])
                    elif neighbourhood_pair[0] == description:
                        current_neighbourhood.append(neighbourhood_pair[1])
                neighbourhoods.append(current_neighbourhood)
        else:
            neighbourhoods = None

        if coverage_distance_matrix:
            coverage_distance_matrix, keys = calculate_coverage_distance_matrix(extents)
        else:
            coverage_distance_matrix, keys = None, None

    return redundancy_score, result_local_optima, neighbourhoods, (coverage_distance_matrix, keys)


def diversity(result, result_is_local_optima=True, df=None, similarity_threshold=None, result_order_qdc=None):

    """
    (Bosc 2018: Definition 6) an aggregated quality of non-redundant (only the local optima) subgroups.

    result = result list
    result_is_local_optima: Boolean to communicate whether result list is local optima
    df: if result_is_local_optima=False: DataFrame containing the data for subgroup evaluation.
    similarity_threshold: define when result_is_local_optima=False, A float value indicating
                            the Jaccard similarity threshold, typically 0.5-0.95.
    result_order_qdc: define when result_is_local_optima=False, List indicating the
                            order of [quality, description, coverage] in results.

    return: float, the larger the better a.k.a. to be maximized
    """
    if result is None or len(result) == 0:
        return None, None

    elif result_is_local_optima:
        result_local_optima = result
        diversity = sum([result[0] for result in result_local_optima])
        return diversity

    else:
        redundancy_tuple = redundancy(result, df, similarity_threshold, result_order_qdc)
        _, result_local_optima, _, _ = redundancy_tuple
        diversity = sum([result[0] for result in result_local_optima])

        return diversity, redundancy_tuple


# def ground_truth_quality(df, descriptions_ground_truth, descriptions_found):
#
#     """(Bosc 2018: Definition 17) a relative quality of a subgroup set when a ground truth is known
#
#     Computes the average distance between a ground truth subgroups and the best matching subgroup found
#     returns a value between 0 and 1. Where 1 is the best possible outcome.
#
#     _________________________________________________________
#
#     df: DataFrame containing the data for subgroup evaluation.
#     descriptions_ground_truth: List of descriptions of the ground truth subgroups.
#     descriptions_found: List of descriptions of the found subgroups.
#
#     returns: A float value between 0 and 1.
#
#     """
#
#     extents_ground_truth = {description_ground_truth:set(df.query(as_string(description_ground_truth)).index)
#                             for description_ground_truth in descriptions_ground_truth}
#     extents_found = {description_found:set(df.query(as_string(description_found)).index)
#                      for description_found in descriptions_found}
#
#     similarities_max_all = []
#     for description_ground_truth in extents_ground_truth.keys():
#         similarities = []
#         for description_found in extents_found.keys():
#             similarities.append(jaccard_similarity(extents_ground_truth[description_ground_truth],
#                                                    extents_found[description_found]))
#         similarities_max_all.append(max(similarities))
#
#     return sum(similarities_max_all)/len(descriptions_ground_truth.keys())

def ground_truth_quality(df, descriptions_ground_truth, descriptions_found):
    """
    (Bosc 2018: Definition 17) a relative quality of a subgroup set when a ground truth is known

    Computes the average distance between a ground truth subgroups and the best matching subgroup found.
    Returns a value between 0 and 1, where 1 is the best possible outcome.
    If the size of ground truth descriptions is 1 the returned value will be be the similarity between the best matching
    found subgroup and the true subgroup.

    _________________________________________________________

    df: DataFrame containing the data for subgroup evaluation.
    descriptions_ground_truth: List of descriptions of the ground truth subgroups.
    descriptions_found: List of descriptions of the found subgroups.

    returns: A float value between 0 and 1.
    """

    # Convert list descriptions to tuple to use as dictionary keys
    extents_ground_truth = {
        tuple(description_ground_truth): set(df.query(as_string(description_ground_truth)).index)
        for description_ground_truth in descriptions_ground_truth
    }
    extents_found = {
        tuple(description_found): set(df.query(as_string(description_found)).index)
        for description_found in descriptions_found
    }

    similarities_max_all = []
    for description_ground_truth in extents_ground_truth:
        similarities = []
        for description_found in extents_found:
            # Compute Jaccard similarity between ground truth and found subgroup extents
            similarities.append(
                jaccard_similarity(extents_ground_truth[description_ground_truth],
                                   extents_found[description_found])
            )
        # Append the maximum similarity for this ground truth description
        similarities_max_all.append(max(similarities))

    # Calculate the average of the maximum similarities
    return sum(similarities_max_all) / len(extents_ground_truth)


def cover_redundancy(df, results, result_order_qdc=(0, 1, 2)):

    """(van Leeuwen 2012) metric to compare different subgroup sets found on the same dataset.

        "If we have several subgroup sets of (roughly) the same size and for the same
        dataset, a lower CR indicates that fewer tuples are covered by more subgroups than
        expected, and thus the subgroup set is more diverse/less redundant."
    """

    quality_idx, description_idx, coverage_idx = result_order_qdc

    # Sort results by quality in descending order
    results = sorted(results, key=lambda x: x[quality_idx], reverse=True)

    descriptions = [result[description_idx] for result in results]
    extents = {str(description):set(df.query(as_string(description)).index)
                     for description in descriptions}

    indices = [i for i in df.index]

    cover_counts = {index : sum([int(1 if index in extent else 0) for extent in extents.values()]) for index in indices}
    expected_cover_count = sum(cover_counts.values())/len(indices)

    CR = (sum([abs(cover_count-expected_cover_count) for cover_count in cover_counts.values()])
          / (len(indices)*expected_cover_count))  # rewritten algebraically to improve efficiency

    return CR, cover_counts
