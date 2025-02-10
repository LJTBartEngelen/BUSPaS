import numpy as np
import pandas as pd
import itertools as it
import re
from collections import defaultdict

#__________ description based _____________#
def fixed_size_desc_selection(result, k):
    """
    Perform fixed-size description-based subgroup selection.

    Args:
    result: List of tuples, where each tuple is (quality, description, coverage).
            - quality: A float representing the quality of the subgroup.
            - description: A list of conditions representing the subgroup description.
            - coverage: A set of elements representing the coverage of the subgroup.
    k: The number of subgroups to select.

    Returns:
    sel: A list of selected subgroups (quality, description, coverage).
    """

    # Sort the candidate subgroups in descending order of quality
    sorted_result = sorted(result, key=lambda x: -x[0])

    # Initialize the selected subgroup set
    sel = []

    # Iterate over the sorted candidate subgroups
    for quality, description, coverage in sorted_result:
        # Check if the current candidate subgroup can be added to the selection
        should_select = True
        to_remove = None
        for selected_quality, selected_description, selected_coverage in sel:
            # If quality is the same and the overlap of conditions is max(len(selected_description), len(description)) - 1, skip the candidate
            if quality == selected_quality:
                len_desc = len(description)
                len_sel_desc = len(selected_description)
                max_len = max(len_desc, len_sel_desc)
                common_conditions = sum(1 for cond in description if cond in selected_description)
                
                if common_conditions == max_len - 1 or common_conditions == max_len:  # TODO Mention in paper
                    should_select = False
                    if len_desc < len_sel_desc:
                        # Mark the previously selected subgroup for removal
                        to_remove = (selected_quality, selected_description, selected_coverage)
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

def variable_size_desc_selection(result, c, l):
    """
    Perform variable-size description-based subgroup selection.

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

    # Sort the candidate subgroups in descending order of quality
    sorted_result = sorted(result, key=lambda x: -x[0])

    # Initialize the selected subgroup set
    sel = []

    # A dictionary to keep track of the number of times each attribute appears in the selected conditions
    attribute_usage = defaultdict(int)

    # Iterate over the sorted candidate subgroups
    for quality, description, coverage in sorted_result:
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


def variable_size_desc_selection(result, df, k, alpha):
    """
    Perform variable-size description-based subgroup selection.

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
    # Sort the candidate subgroups in descending order of quality
    sorted_result = sorted(result, key=lambda x: -x[0])
    extents = []
    for i in sorted_result:
        extents.append(list(df.query(as_string(i[1])).index))

    sel = [ sorted_result[0] ]
    sel_ext = [ extents[0] ]
    sorted_result.remove(sorted_result[0])
    extents.remove(extents[0])

    for _ in range(k-1):
        best_score = -float('inf')
        best_candidate = None
        best_candidate_extent = None
        best_candidate_i = None
        for i in range(len(sorted_result)):
            omega = cover_score(extents[i], sel_ext, alpha=alpha)

            quality = sorted_result[i][0]
            score = omega * quality
             
            if score > best_score:
                best_score = score
                best_candidate = sorted_result[i]
                print('new best= ', best_candidate)
                best_candidate_extent = extents[i]
                best_candidate_i = i
                
        sel.append(best_candidate)
        sel_ext.append(best_candidate_extent)
        sorted_result.remove(best_candidate)
        del extents[best_candidate_i]
    
    return sel

    
#___________ dominance pruning ____________#

def get_new_descriptions(subgroup=None):

    pruned_descriptions_ = []
    items_old_desc = subgroup[1]
    #items_old_desc = old_desc.items()

    for r in np.arange(1, len(list(items_old_desc))):
        combs = list(it.combinations(items_old_desc, r=r))
        combs_r = [list(i) for i in combs]
        # combs_r = [{'description': list(desc)} for desc in combs]
        pruned_descriptions_.extend(combs_r)
        pruned_descriptions_.append(subgroup[1])
    return pruned_descriptions_

def get_quality(desc,df):
    sg = df.query(as_string(desc))
    quality, coverage = eval_quality(sg, df, 'target', cluster_based_quality_measure,distance_matrix=euclidean_slope_distance_matrix,correct_for_size=no_size_corr, comparison_type = "complement" )
    return quality

def dominance_pruning(emm_result): ###TODO Check if this actually works

    """ emm_result = [(quality, description, coverage),...,...,...] """
    
    quals_dict = {}
    best_candidate_descs = []
    for desc in emm_result:
        best_qual = 0
        best_candidate_desc = None
        quals_dict[str(desc[1])] = desc[0]
        candidate_set = get_new_descriptions(desc)
        for desc_candidate in candidate_set:
            if str(desc_candidate) in quals_dict:
                 qual = quals_dict[str(desc_candidate)]
            else:
                quals_dict[str(desc_candidate)] = get_quality(desc_candidate,df_work)
                qual = quals_dict[str(desc_candidate)]
            if qual > best_qual: ###TODO Change if we're also going to work qualities that improve by decreasing
                best_candidate_desc = desc_candidate
        best_candidate_descs.append(best_candidate_desc)
    return best_candidate_descs