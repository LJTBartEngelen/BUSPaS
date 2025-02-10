"""
The following code was adapted from W. Duivesteijn, T.C. van Dijk. (2021)
    Exceptional Gestalt Mining: Combining Magic Cards to Make Complex Coalitions Thrive.
    In: Proceedings of the 8th Workshop on Machine Learning and Data Mining for Sports Analytics.
    Available from http://wwwis.win.tue.nl/~wouter/Publ/J05-EMM_DMKD.pdf
"""

import heapq
import time
import numpy as np
import math
import pandas as pd

from Code.dataImporter import *
from Code.helperFunctions import *
from Code.qualityMeasure import *

# Classes
class BoundedPriorityQueue:
    """
    Used to store the <q> most promising subgroups
    Ensures uniqueness
    Keeps a maximum size (throws away value with least quality)
    """

    def __init__(self, bound):
        # Initializes empty queue with maximum length of <bound>
        self.values = []
        self.bound = bound
        self.entry_count = 0

    def add(self, element, quality, coverage, **adds):
        # Adds <element> to the bounded priority queue if it is of sufficient quality
        new_entry = (quality, coverage, self.entry_count, element, adds)

        if (len(self.values) >= self.bound):
            heapq.heappushpop(self.values, new_entry)
        else:
            heapq.heappush(self.values, new_entry)

        self.entry_count += 1

    def get_values(self):
        # Returns elements in bounded priority queue in sorted order
        for (q, coverage, _, e, x) in sorted(self.values, reverse=True):
            yield (q, coverage, e, x)

    def get_element_sets(self):
        # Returns elements in bounded priority queue in sorted order
        element_sets = []
        for (_, _, _, e, _) in sorted(self.values, reverse=True):
            element_sets.append(set(e))
        return element_sets

    def show_contents(self):
        # Prints contents of the bounded priority queue (used for debugging)
        print("show_contents")
        for (q, coverage, _, entry_count, e) in self.values:
            print(q, coverage, entry_count, e)


class Queue:
    """
    Used to store candidate solutions
    Ensures uniqueness
    """

    def __init__(self):  # Initializes empty queue
        self.items = []

    def is_empty(self):  # Returns True if queue is empty, False otherwise
        return self.items == []

    def enqueue(self, item):  # Adds <item> to queue if it is not already present
        if item not in self.items:
            self.items.insert(0, item)

    def dequeue(self):  # Pulls one item from the queue
        return self.items.pop()

    def size(self):  # Returns the number of items in the queue
        return len(self.items)

    def get_values(self):  # Returns the queue (as a list)
        return self.items

    def add_all(self, iterable):  # Adds all items in <iterable> to the queue, given they are not already present
        for item in iterable:
            self.enqueue(item)

    def clear(self):  # Removes all items from the queue
        self.items.clear()

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


def eta(seed, df, features, n_chunks=5, allow_exclusion=True, report_progress = False):
    # Returns a generator which includes all possible refinements of <seed> for the given <features> on dataset <df>
    # n_chunks refers to the number of possible splits we consider for numerical features

    if report_progress:
        print("eta ", seed)

    # TODO ETA implement that new descriptions are not too much overlapping with old descriptions
    # TODO ETA.2 no exclusion of one (non-frequent) att-value for attr that contains large amount of values

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
                candidate = "{} != '{}'".format(f, i)  # TODO Exclusion hier zetten ipv los
                if (not candidate in seed) and allow_exclusion:  # if not already there
                    yield refine(seed, candidate)
        else:
            assert False


# def desc_diverse(desc): ###TODO Revise if needed and where this is applied
#
#     # alerts when a description contains something in the form desc =
#     # ['marketCap > 687961248.0', 'fullTimeEmployees == 9800.0',
#     #       'marketCap > 23376277504.0', 'marketCap > 96089239552.0']
#     # where we see that some operators are redundant
#     desc_list = []
#     for i in desc:
#         words = i.split()
#         attribute = words[0]
#         operator = words[1]
#
#         desc_list.append((attribute, operator))
#
#     desc_list_num = [i for i in desc_list if i[1] not in ["!=", "=="]]
#
#     return len(desc_list_num) == len(set(desc_list_num))


def satisfies_all(desc, ind, len_df, threshold=0, threshold_absolute=2):

    # Function used to check if subgroup with pattern <desc> is sufficiently big relative to its dataset <df>
    # A subgroup is sufficiently big if the proportion of data included in it exceeds <threshold>

    sum_ind = len(ind)
    #sum_ind = sum(ind)

    return sum_ind >= len_df * threshold and sum_ind >= threshold_absolute #and desc_diverse(desc)


class BeamSearch:

    def __init__(self,df):
    
        self.df = df
        self.len_df = len(self.df)
        self.result = None

        self.avg_quality = None
        self.max_quality = None
        self.avg_coverage = None
        self.max_coverage = None
        self.descriptions = []

        self.duration = 0.0
        self.duration_quality = 0.0
        self.count_quality = 0

        self.result = []
        self.qualities = None
        self.coverages = None

        self.pvalues = None
        self.result_pvalues = None
        self.randomized_qualities = None

    def descriptions_(self):
        
        for desc in self.descriptions:
            print(desc)
    
    def analyseEMMresults(self):

        print('Outcome of EMM is:')
        print(' ')
        print('avg_quality = ', self.avg_quality)
        print('max_quality = ', self.max_quality)
        print(' ')
        print('avg_coverage = ', self.avg_coverage)
        print('max_coverage = ', self.max_coverage)
        print(' ')


        for z in self.result:
            conjunction = " Ʌ ".join(["(" + condition.replace(" == ", "=").strip() + ")" for condition in z[3]])
            print('description =',conjunction)
            print('quality =',round(z[0],3))
            print('coverage =',round(z[1],3))
            print(' ')

        print(' ')
        print(' ')
        self.descriptions_()
    

    def EMM(self, features, w=10, d=3, q=10, quality_measure=cluster_based_quality_measure, catch_all_description=[],
            comparison_type = 'complement',target='target', n_chunks=5, ensure_diversity=True,
            report_progress=False, allow_exclusion=True, min_coverage = 0.01,
                              min_coverage_abs = 5, min_error = 0.001, show_result = True, minutes=float('inf'), **kwargs):
        """
        w - width of beam, i.e. the max number of results in the beam
        d - num levels, i.e. how many attributes are considered
        q - max results, i.e. max number of results output by the algorithm
        eta - a function that receives a description and returns all possible refinements
        satisfies_all - a function that receives a description and verifies wheather it satisfies some requirements as needed
        eval_quality - returns a quality for a given description. This should be comparable to qualities of other descriptions
        catch_all_description - the equivalent of True, or all, as that the whole dataset shall match
        df - dataframe of mined dataset
        features - features in scope
        target - column name of target attribute in df
        report_progress = print progress in detail
        comparison_type = using 'complement' or 'population' to compare subgroup
        correct_for_size = correct the quality measure for subgroup size e.g. using 'entropy' score
        allow_exclusion = in eta allow != for cat attributes
        """

        self.features = features
        self.w = w
        self.d = d
        self.q = q
        self.quality_measure = quality_measure
        self.catch_all_description = catch_all_description
        self.comparison_type = comparison_type
        self.target = target
        self.n_chunks = n_chunks
        self.ensure_diversity = ensure_diversity
        self.report_progress = report_progress
        self.allow_exclusion = allow_exclusion
        self.min_coverage = min_coverage
        self.min_coverage_abs = min_coverage_abs
        self.min_error = min_error
        self.show_result = show_result
        self.max_running_time = minutes*60
        self.kwargs = kwargs

        time_limit_exceeded = False

        # Initialize variables
        df = self.df
        
        resultSet = BoundedPriorityQueue(q)  # Set of results, can contain results from multiple levels
        candidateQueue = Queue()  # Set of candidate solutions to consider adding to the ResultSet
        candidateQueue.enqueue(catch_all_description)  # Set of results on a particular level
        self.count_quality = 0
        self.duration_quality = 0
        self.duration = 0

        start_time_beam_search = time.time()

    
        error = min_error  # Allowed error margin (due to floating point error) when comparing the quality of solutions
        len_df = self.len_df
    
        # we calculate the quality measure for the complete population so that we won't have to do it many times later
    
        # Perform BeamSearch for <d> levels
        for level in range(d):

            if report_progress:
                print("level : ", level)
    
            # Initialize this level's beam
            beam = BoundedPriorityQueue(w)
    
            sizequeue = len(candidateQueue.get_values())
            count_queue = 0
    
            # Go over all rules generated on previous level, or 'empty' rule if level = 0
            for seed in candidateQueue.get_values():

                if time.time() - start_time_beam_search > self.max_running_time:
                    self.duration = None
                    time_limit_exceeded = True
                    break

                count_queue += 1

                if report_progress:
                    print("    seed : ", seed)
    
                # Start by evaluating the quality of the seed
                if seed != []:
                    ind_seed = df.query(as_string(seed))
                    seed_quality, _ = eval_quality(ind_seed, df, target, quality_measure, comparison_type, **kwargs)
                else:
                    seed_quality = 0
    
                # For all refinements created by eta function on descriptions (i.e features), which can be different types of columns
                # eta(seed) reads the dataset given certain seed (i.e. already created rules) and looks at new descriptions
    
                if report_progress:
    
                    try:
                        size = sum(1 for _ in (eta(seed, df, features, n_chunks, allow_exclusion)))
                    except:
                        size = 0
                    count_desc = 0

                    print(level, '/', d, 'queue=', count_queue, '/', sizequeue)
    
                for desc in eta(seed, df, features, n_chunks, allow_exclusion):
                    if report_progress:
    
                        count_desc += 1
                        print(level, '/', d, 'queue=', count_queue, '/', sizequeue, 'descr=', count_desc, '/', size)
                    else:
                        pass
    
                    # Check if the subgroup contains at least x% of data, proceed if yes
    
                    if set(desc) not in resultSet.get_element_sets():
    
                        ind = df.query(as_string(desc))
    
                        if satisfies_all(desc, ind, len_df, self.min_coverage, self.min_coverage_abs):
                            # added to exclude factual the same descriptions
    
                            # Calculate the new solution's quality

                            start_time_quality = time.time()
                            quality, coverage = eval_quality(ind, df, target, quality_measure, comparison_type,
                                                             **kwargs)
                            end_time_quality = time.time()

                            self.duration_quality += end_time_quality - start_time_quality
                            self.count_quality += 1
    
                            # Ensure diversity by forcing difference in quality when compared to its seed
                            # if <ensure_diversity> is set to True. Principle is based on:
                            # Van Leeuwen, M., & Knobbe, A. (2012), Diverse subgroup set discovery.
                            # Data Mining and Knowledge Discovery, 25(2), 208-242.
                            if ensure_diversity:
    
                                if quality < (seed_quality * (1 - error)) or quality > (seed_quality * (1 + error)):
                                    resultSet.add(desc, quality, coverage)
                                    beam.add(desc, quality, coverage)
                            else:
                                resultSet.add(desc, quality, coverage)
                                beam.add(desc, quality, coverage)
    
            # When all candidates for a search level have been explored,
            # the contents of the beam are moved into candidateQueue, to generate next level candidates
            candidateQueue = Queue()
            candidateQueue.add_all(desc for (_, _, desc, _) in beam.get_values())

            if time_limit_exceeded:
                break
            else:
                pass

        end_time_beam_search = time.time()
        if self.duration is not None:
            self.duration = end_time_beam_search - start_time_beam_search
        else:
            pass

        if report_progress:
            print('Checked ', self.count_quality, ' subgroups in ', self.duration, ' seconds')

        # Return the <resultSet> once the BeamSearch algorithm has completed
        self.result = resultSet
        self.result = sorted(self.result.values, key=lambda x: x[0], reverse=True)

        quals = [i[0] for i in self.result]
        self.qualities = quals
        covs = [i[1] for i in self.result]
        self.coverages = covs
        self.avg_quality = round(sum(quals) / len(quals), 3)
        self.avg_coverage = round(sum(covs) / len(covs), 3)
        self.max_quality = round(max(quals), 3)
        self.max_coverage = round(max(covs), 3)
        self.descriptions = [i[3] for i in self.result]

        if show_result:
            self.analyseEMMresults()

        return resultSet


    def run_statistical_test(self,k,show_progress_pvals=False, save_path='saved_results_statistical_test_beamsearch_'):

        start = time.time()
        save_path = save_path+str(start)+'.pkl'

        randomized_qualities = []

        for _ in range(k):
            if show_progress_pvals:
                print('Start beam search statistical test: ',_,'/',k)
            swapped_matrix = swap_randomize_symmetric(self.kwargs['distance_matrix'])

            bs = BeamSearch(self.df)
            bs.EMM(
                self.features,
                self.w,
                self.d,
                1,
                self.quality_measure,
                self.catch_all_description,
                self.comparison_type,
                self.target,
                self.n_chunks,
                self.ensure_diversity,
                self.report_progress,
                self.allow_exclusion,
                self.min_coverage,
                self.min_coverage_abs,
                self.min_error,
                self.show_result,
                distance_matrix=swapped_matrix,
                correct_for_size=no_size_corr)

            randomized_qualities += bs.qualities


        pvalues = []
        for quality in self.qualities:
            empirical_p_value = (sum([(comp > quality) for comp in randomized_qualities])+0.5*sum([(comp == quality) for comp in randomized_qualities])+1)/(len(randomized_qualities)+1)
            pvalues.append(empirical_p_value)
        self.pvalues = pvalues
        self.result_pvalues = []
        self.randomized_qualities = randomized_qualities

        #TODO Apply Multiple Comparison Correction

        for i in range(len(self.result)):

            self.result_pvalues.append(tuple(list(self.result[i])+[self.pvalues[i]]))

        with open(save_path, 'wb') as f:
            pickle.dump(self.result_pvalues, f)

        return self.result_pvalues


    
