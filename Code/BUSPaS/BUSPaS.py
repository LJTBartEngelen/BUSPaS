import heapq
import re
import time
from Code.qualityMeasure import *
from Code.helperFunctions import *
import pickle


def satisfies_all(desc, ind, len_df, threshold_coverage=0.001, threshold_absolute=2):  # TODO optimize threshold

    # Function used to check if subgroup with pattern <desc> has sufficient coverage w.r.t. <df>
    # A subgroup is sufficiently large if the proportion of data included in it exceeds <threshold>

    sum_ind = len(ind)
    #sum_ind = sum(ind)

    return sum_ind >= len_df * threshold_coverage and sum_ind >= threshold_absolute #and desc_diverse(desc) #TODO remove last


def as_string(desc):
    # Adds ' and ' to <desc> such that selectors are properly separated when the refine function is used
    return ' and '.join(desc)


class BUSPaS:
    
    def __init__(self, df, distance_matrix=[], number_of_row_pairs=100, depth=3, q=10, z=2, nr_chunks=5,
                 min_coverage_perc=0.01, min_coverage_abs=3):

        """ 
            df - dataframe descriptives and targets
            distance_matrix=[] - pre-calculated distance matrix, 
                                    calculates eucledian distance matrix based on last variable (target) if left empty,
                                    calculates distance matrix based on last variable (target) with the given distance metric if input is a string,
                                    ###TODO add distance metrics
            number_of_row_pairs=100 - number of closest couples in distance_matrix used to generate mini-clusters 
            depth=3 - max number of descriptors in generated descriptions 
            q=10 - number of reported most interesting subgroups
            nr_chunks=5 - number of chuncks in numerical data
                ###TODO check if still relevant
            min_coverage_perc=0.01 - min coverage as percentage of total set
            min_coverage_abs=3 - min number of items in subgroup
        """ 
        self.matrix = distance_matrix
        self.x = number_of_row_pairs
        self.d = depth
        self.df = df.copy()
        self.q = q
        self.min_coverage_perc = min_coverage_perc
        self.min_coverage_abs = min_coverage_abs
        self.nr_chunks = nr_chunks
        self.z = z

        if type(distance_matrix) is str:
            print('Calculating distance matrix')
            self.matrix = calculate_distance_matrix(self.df, distance_function=distance_matrix)
        elif type(distance_matrix) is np.ndarray:
            self.matrix = distance_matrix
        else:
            print('Calculating euclidean distance matrix')
            self.matrix = calculate_distance_matrix(self.df, distance_function=euclidean)

        self.avg_quality = None
        self.max_quality = None
        self.avg_coverage = None
        self.max_coverage = None
        self.descriptions = []

        self.duration = 0.0
        self.duration_quality = 0.0
        self.count_quality = 0

        self.result = []
        self.covs = None
        self.quals = None

        self.randomized_qualities = None
        self.pvalues = None
        self.result_pvalues = None

    def num_to_cat_attribute_converter(self):
        # steps to partition numerical columns into bins
        for col in self.df.columns:
            if (self.df[col].dtype == 'float64') or (self.df[col].dtype == 'float32') or (self.df[col].dtype == 'int64'):
                dat = np.sort(self.df[col])
                dat = dat[np.logical_not(np.isnan(dat))]
                for i in range(1, self.nr_chunks+1):
                    # determine the number of chunks you want to divide your data in
                    x = np.percentile(dat, (i-1)*100/self.nr_chunks)  #
                    y = np.percentile(dat, (i)*100/self.nr_chunks)
                    candidate = "{} <= {} <= {}".format(x, col, y)
                    self.df[col] = self.df[col].apply(lambda val: candidate if (not isinstance(val, str) and x <= val <= y) else val)
        pass

    def find_mini_clusters(self, matrix, x):
        """
        Finds the x smallest distances and returns tuples of the indexes i and j.
        
        matrix = distance matrix
        x = number of pairs we want to find
        """
        
        min_heap = []
        n = len(matrix)

        for i in range(n):
            for j in range(i+1, n):
                if len(min_heap) < x:
                    heapq.heappush(min_heap, (-matrix[i][j], (i, j)))
                else:
                    if matrix[i][j] < -min_heap[0][0]:
                        heapq.heappop(min_heap)
                        heapq.heappush(min_heap, (-matrix[i][j], (i, j)))

        return [idx for val, idx in min_heap]


    def get_common_attributes(self, rows, d_len):
        # TODO Improve for numerical attributes
        """
        Finds the attribute value combinations that are similar in all rows.

        rows = list of rows of a dataframe
        d_len = description length
        """

        common_attributes = []
        attribute_combinations = combinations(rows[0].items(), d_len)

        for combination in attribute_combinations:
            attributes = [f"{attribute} == '{value}'" for attribute, value in combination]
            if all(all(row.get(attribute) == value for attribute, value in combination) for row in rows[1:]):
                common_attributes.append(attributes)

        return common_attributes

    def get_unique_lists(self,list_of_lists):
        """
        Filters redundant descriptions, when they are exactly the same or in a different order.
        """
        
        unique_lists = [list(x) for x in set(tuple(set(sublist)) for sublist in list_of_lists)]
        return unique_lists

    def expand_mini_clusters(self, tuples_list, matrix):

        new_tuples_list = []

        for t in tuples_list:
            new_tuples = []

            for index in t:
                row = index
                new_tuple = list(t)  # Create a new tuple based on the existing one

                # Find the minimum value in the row that satisfies the conditions
                min_val = float('inf')
                min_index = None
                for i, val in enumerate(matrix[row]):
                    if val < min_val and i != row and i not in t:
                        min_val = val
                        min_index = i

                new_tuple.append(min_index)  # Add the new index to the tuple
                new_tuples.append(tuple(new_tuple))  # Convert the list back to a tuple and add it to the new list

            new_tuples_list.extend(new_tuples)

        return new_tuples_list

    def find_quality(self, quality_measure=cluster_based_quality_measure,
                     comparison_type="complement", size_corr=no_size_corr, minutes=float('inf')):
        
        self.running_time = time.time()
        self.quality_measure = quality_measure
        self.size_corr = size_corr
        self.count_quality = 0
        self.comparison_type = comparison_type
        self.max_running_time = minutes*60

        min_heap = []
        len_df = len(self.df)

        mini_clusters_z2 = self.find_mini_clusters(self.matrix, self.x)

        self.nr_mini_clusters_z2 = len(mini_clusters_z2)

        z_corr = self.z-2
        mini_clusters_z = mini_clusters_z2
        if self.z>2:
            for _ in range(z_corr):
                mini_clusters_z = self.expand_mini_clusters(mini_clusters_z, self.matrix)

        self.nr_mini_clusters_z = len(mini_clusters_z)
        mini_clusters_unique = [tuple(s) for s in {frozenset(t) for t in mini_clusters_z}]
        self.nr_mini_clusters_unique = len(mini_clusters_unique)

        #TODO Step: unique MC's optimizations

        candidate_descriptions = [self.get_common_attributes([self.df.iloc[i][:-1]
                                                                for i in index_tuple], desc_length)
                                                                for index_tuple in mini_clusters_unique
                                                                for desc_length in range(1,self.d+1)]
        self.nr_candidates = len(candidate_descriptions)

        # unique_candidate_descriptions gives the unique potential candidate descriptions
        #TODO check for optimization
        unique_candidate_descriptions = self.get_unique_lists([item for sublist in candidate_descriptions
                                                                    for item in sublist])
        self.nr_candidates_unique = len(unique_candidate_descriptions)

        # if time.time() - self.running_time > self.max_running_time:
        #     self.running_time = None
        #     return None
        #
        # else:
        for desc in unique_candidate_descriptions:
            # if time.time() - self.running_time > self.max_running_time:
            #     self.running_time = None
            #     break
            # else:
            ind = self.df.query(as_string(desc))
            # print(desc, 'to be checked')
            #checks if subgroups comply with size constrains
            if satisfies_all(desc, ind, len_df, self.min_coverage_perc, self.min_coverage_abs):
                # print(desc, ' satisfies contraints with len ', len(ind))
                start_time_quality = time.time()
                quality, coverage = eval_quality(ind, self.df, 'target', self.quality_measure, comparison_type,
                                                 distance_matrix=self.matrix,correct_for_size=size_corr)
                # print(desc,' qual= ',quality)
                end_time_quality = time.time()
                self.duration_quality += end_time_quality - start_time_quality
                self.count_quality += 1

                if len(min_heap) < self.q:
                    heapq.heappush(min_heap, (quality, desc, coverage))
                    #heapq.heapify(min_heap)
                else:

                    if -quality < -heapq.nsmallest(1, min_heap)[0][0]:

                        equal_quals = [i for i, x in enumerate(min_heap) if x[0] == quality]

                        #TODO Analyse if this is diversity check
                        #checks if set of records isn't already present as a result of a different description
                        if len(equal_quals) > 0:
                            for i in equal_quals:
                                comp = self.df.query(as_string(min_heap[i][1]))
                                if np.array_equal(comp.index, ind.index):
                                    pass
                                else:
                                    heapq.heappushpop(min_heap, (quality, desc, coverage))
                        else:
                            heapq.heappushpop(min_heap, (quality, desc, coverage))

        
        if self.running_time is not None:
            self.running_time = time.time() - self.running_time
            self.duration = self.running_time
        else:
            self.duration = 'NT'
        
        self.result = sorted(min_heap, key=lambda x: x[0], reverse=True)

        #part that changes the numerical propositions back into an evaluatable description instead of a string
        data = self.result
        for i, (value1, sublist, value2) in enumerate(data):
            for j, string in enumerate(sublist):
                
                if any(op in string for op in ['<', '<=', '>', '>=']):
                    # Extract the nested string without the outer quotes
                    try:
                        nested_string = string.split("'")[1]
                    # Replace the string with the extracted nested string
                        sublist[j] = nested_string
                    except:
                        pass
            # Update the modified sublist in the data
            data[i] = (value1, sublist, value2)

        self.result = [(quality, coverage, description) for (quality, description, coverage) in data]
        self.quals = [i[0] for i in self.result]
        self.covs = [i[1] for i in self.result]
        self.avg_quality = 0 if len(self.quals) == 0 else sum(self.quals) / len(self.quals)
        self.avg_coverage = 0 if len(self.covs) == 0 else sum(self.covs) / len(self.covs)
        self.descriptions = [i[2] for i in self.result]
        self.max_quality = 0 if len(self.quals) == 0 else round(max(self.quals), 3)
        self.max_coverage = 0 if len(self.covs) == 0 else round(max(self.covs), 3)

        pass
        
    def print_outcome(self):
        
        print('after checking ',self.count_quality,' potential subgroups')
        print('Outcome of bottumUpSearch is:')
        print(' ')
        print('avg_quality = ', round(self.avg_quality, 3))
        print('max_quality = ', round(max(self.quals), 3))
        print(' ')
        print('avg_coverage = ', round(self.avg_coverage, 3))
        print('max_coverage = ', round(max(self.covs), 3))
        print(' ')
                
        for z in self.result:
            conjunction = " Ʌ ".join(["(" + condition.replace(" == ", "=").strip() + ")" for condition in z[2]])
            print('description =',conjunction)
            print('quality =',round(z[0], 3))
            print('coverage =',round(z[1], 3))
            print(' ')

    def run_statistical_test(self, k, show_progress_pvals=False, save_path='saved_results_statistical_test_BUSPaS_'):

        start = time.time()
        save_path = save_path + str(start) + '.pkl'

        randomized_qualities = []

        for _ in range(k):
            if show_progress_pvals:
                print('Start buspas statistical test: ',_,'/',k)
            swapped_matrix = swap_randomize_symmetric(self.matrix)

            bus_swapped = BUSPaS(self.df, swapped_matrix, self.x, self.d, 1, self.z,
                                     nr_chunks=self.nr_chunks, min_coverage_perc=self.min_coverage_perc,
                                     min_coverage_abs=self.min_coverage_abs)
            bus_swapped.find_quality(quality_measure = self.quality_measure,
                                 comparison_type = self.comparison_type , size_corr = self.size_corr)

            randomized_qualities += bus_swapped.quals

        pvalues = []
        for quality in self.quals:
            empirical_p_value = (sum([(comp > quality) for comp in randomized_qualities]) + 0.5 * sum(
                [(comp == quality) for comp in randomized_qualities]) + 1) / (len(randomized_qualities) + 1)
            pvalues.append(empirical_p_value)
        self.randomized_qualities = randomized_qualities
        self.pvalues = pvalues
        self.result_pvalues = []

        # TODO Apply Multiple Comparison Correction

        for i in range(len(self.result)):
            self.result_pvalues.append(tuple(list(self.result[i]) + [self.pvalues[i]]))

        with open(save_path, 'wb') as f:
            pickle.dump(self.result_pvalues, f)

        return self.result_pvalues



#TODO Remove: old code
# Version before z parameter
#class BUSPaS:
#
#     def __init__(self, df, distance_matrix=[], number_of_row_pairs=100, depth=3, q=10, nr_chunks=5,
#                  min_coverage_perc=0.01, min_coverage_abs=3):
#
#         """
#             df - dataframe descriptives and targets
#             distance_matrix=[] - pre-calculated distance matrix,
#                                     calculates eucledian distance matrix based on last variable (target) if left empty,
#                                     calculates distance matrix based on last variable (target) with the given distance metric if input is a string,
#             number_of_row_pairs=100 - number of closest couples in distance_matrix used to generate mini-clusters
#             depth=3 - max number of descriptors in generated descriptions
#             q=10 - number of reported most interesting subgroups
#             nr_chunks=5 - number of chuncks in numerical data
#             min_coverage_perc=0.01 - min coverage as percentage of total set
#             min_coverage_abs=3 - min number of items in subgroup
#         """
#         self.matrix = distance_matrix
#         self.x = number_of_row_pairs
#         self.d = depth
#         self.df = df.copy()
#         self.q = q
#         self.min_coverage_perc = min_coverage_perc
#         self.min_coverage_abs = min_coverage_abs
#         self.nr_chunks = nr_chunks
#
#         if type(distance_matrix) == str:
#             print('Calculating distance matrix')
#             self.matrix = calculate_distance_matrix(self.df, distance_function=distance_matrix)
#         elif type(distance_matrix) == np.ndarray:
#             self.matrix = distance_matrix
#         else:
#             print('Calculating euclidean distance matrix')
#             self.matrix = calculate_distance_matrix(self.df, distance_function=euclidean)
#
#         self.avg_quality = None
#         self.max_quality = None
#         self.avg_coverage = None
#         self.max_coverage = None
#         self.descriptions = []
#
#         self.duration = 0.0
#         self.duration_quality = 0.0
#         self.count_quality = 0
#
#         self.randomized_qualities = None
#         self.pvalues = None
#         self.result_pvalues = None
#
#     def num_to_cat_attribute_converter(self):
#         # steps to partition numerical columns into bins
#         for col in self.df.columns:
#             if (self.df[col].dtype == 'float64') or (self.df[col].dtype == 'float32') or (
#                     self.df[col].dtype == 'int64'):
#                 dat = np.sort(self.df[col])
#                 dat = dat[np.logical_not(np.isnan(dat))]
#                 for i in range(1, self.nr_chunks + 1):
#                     # determine the number of chunks you want to divide your data in
#                     x = np.percentile(dat, (i - 1) * 100 / self.nr_chunks)  #
#                     y = np.percentile(dat, (i) * 100 / self.nr_chunks)
#                     candidate = "{} <= {} <= {}".format(x, col, y)
#                     self.df[col] = self.df[col].apply(
#                         lambda val: candidate if (not isinstance(val, str) and x <= val <= y) else val)
#         pass
#
#     def find_min_indices(self, matrix, x):
#         """
#         Finds the x smallest distances and returns tuples of the indexes i and j.
#
#         matrix = distance matrix
#         x = number of pairs we want to find
#         """
#
#         min_heap = []
#         n = len(matrix)
#
#         for i in range(n):
#             for j in range(i + 1, n):
#                 if len(min_heap) < x:
#                     heapq.heappush(min_heap, (-matrix[i][j], (i, j)))
#                 else:
#                     if matrix[i][j] < -min_heap[0][0]:
#                         heapq.heappop(min_heap)
#                         heapq.heappush(min_heap, (-matrix[i][j], (i, j)))
#
#         return [idx for val, idx in min_heap]
#
#     def get_common_attributes(self, row1, row2, d_len):
#         """
#         Finds the attribute value combinations that the two rows have in common.
#
#         row1, row2 = row of a dataframe
#         d_len = description length
#         """
#
#         common_attributes = []
#         for combination in combinations(row1.items(),
#                                         d_len):  # for combination in combinations(row1.iteritems(), d_len):
#             attributes = [f"{attribute} == '{value}'" for attribute, value in combination]
#             if all(row2.get(attribute) == value for attribute, value in combination):
#                 common_attributes.append(attributes)
#         return common_attributes
#
#     def get_unique_lists(self, list_of_lists):
#         """
#         Filters redundant descriptions, when they are exactly the same or in a different order.
#         """
#
#         unique_lists = [list(x) for x in set(tuple(set(sublist)) for sublist in list_of_lists)]
#         return unique_lists

#     def findQuality(self, quality_measure=cluster_based_quality_measure, comparison_type="complement",
#                     size_corr=no_size_corr):
#
#         self.running_time = time.time()
#         self.quality_measure = quality_measure
#         self.size_corr = size_corr
#         self.count_quality = 0
#         self.comparison_type = comparison_type
#
#         min_heap = []
#         len_df = len(self.df)
#
#         promising_combinations = self.find_min_indices(self.matrix, self.x)
#
#         #############################
#
#         candidate_descriptions = [self.get_common_attributes(self.df.iloc[promising_combinations[i][0]][:-1],
#                                                              self.df.iloc[promising_combinations[i][1]][:-1], d)
#                                   for i in range(len(promising_combinations)) for d in range(1, self.d + 1)]
#
#         # unique_candidate_descriptions gives the unique potential candidate descriptions
#         unique_candidate_descriptions = self.get_unique_lists(
#             [item for sublist in candidate_descriptions for item in sublist])
#
#         #############################
#
#         for desc in unique_candidate_descriptions:
#             ind = self.df.query(as_string(desc))
#
#             # checks if subgroups comply with size constrains
#             if satisfies_all(desc, ind, len_df, self.min_coverage_perc, self.min_coverage_abs):
#
#                 start_time_quality = time.time()
#                 quality, coverage = eval_quality(ind, self.df, 'target', self.quality_measure, comparison_type,
#                                                  distance_matrix=self.matrix, correct_for_size=size_corr)
#                 end_time_quality = time.time()
#                 self.duration_quality += end_time_quality - start_time_quality
#                 self.count_quality += 1
#
#                 if len(min_heap) < self.q:
#                     heapq.heappush(min_heap, (quality, desc, coverage))
#                     # heapq.heapify(min_heap)
#                 else:
#
#                     if -quality < -heapq.nsmallest(1, min_heap)[0][0]:
#
#                         equal_quals = [i for i, x in enumerate(min_heap) if x[0] == quality]
#
#                         # checks if set of records isn't already present as a result of a different description
#                         if len(equal_quals) > 0:
#                             for i in equal_quals:
#                                 comp = self.df.query(as_string(min_heap[i][1]))
#                                 if np.array_equal(comp.index, ind.index):
#                                     pass
#                                 else:
#                                     heapq.heappushpop(min_heap, (quality, desc, coverage))
#                         else:
#                             heapq.heappushpop(min_heap, (quality, desc, coverage))
#
#         self.running_time = time.time() - self.running_time
#         self.duration = self.running_time
#
#         self.result = sorted(min_heap, key=lambda x: x[0], reverse=True)
#
#         # part that changes the numerical propositions back into an evaluatable thing instead of a string
#         data = self.result
#         for i, (value1, sublist, value2) in enumerate(data):
#             for j, string in enumerate(sublist):
#
#                 if any(op in string for op in ['<', '<=', '>', '>=']):
#                     # Extract the nested string without the outer quotes
#                     try:
#                         nested_string = string.split("'")[1]
#                         # Replace the string with the extracted nested string
#                         sublist[j] = nested_string
#                     except:
#                         pass
#             # Update the modified sublist in the data
#             data[i] = (value1, sublist, value2)
#
#         self.result = [(quality, coverage, description) for (quality, description, coverage) in data]
#         self.quals = [i[0] for i in self.result]
#         self.covs = [i[1] for i in self.result]
#         self.avg_quality = sum(self.quals) / len(self.quals)
#         self.avg_coverage = sum(self.covs) / len(self.covs)
#         self.descriptions = [i[2] for i in self.result]
#         self.max_quality = round(max(self.quals), 3)
#         self.max_coverage = round(max(self.covs), 3)
#
#         pass
#
#     def print_outcome(self):
#
#         print('after checking ', self.count_quality, ' potential subgroups')
#         print('Outcome of bottumUpSearch is:')
#         print(' ')
#         print('avg_quality = ', round(self.avg_quality, 3))
#         print('max_quality = ', round(max(self.quals), 3))
#         print(' ')
#         print('avg_coverage = ', round(self.avg_coverage, 3))
#         print('max_coverage = ', round(max(self.covs), 3))
#         print(' ')
#
#         for z in self.result:
#             conjunction = " Ʌ ".join(["(" + condition.replace(" == ", "=").strip() + ")" for condition in z[2]])
#             print('description =', conjunction)
#             print('quality =', round(z[0], 3))
#             print('coverage =', round(z[1], 3))
#             print(' ')
#
#     def randomize_symmetric_matrix(matrix):
#         """
#         Takes a distance matrix.
#         Returns a swap randomized symmmetric version of the matrix.
#         """
#
#         n = matrix.shape[0]
#         triu_indices = np.triu_indices(n, k=1)  # Upper triangular indices excluding the diagonal
#         randomized_values = np.random.permutation(matrix[triu_indices])  # Randomize the upper triangular values
#         result = np.zeros_like(matrix)  # Initialize the result matrix with zeros
#         result[triu_indices] = randomized_values  # Assign the randomized values to the upper triangular indices
#         result.T[
#             triu_indices] = randomized_values  # Assign the same values to the lower triangular indices (keeping it symmetric)
#
#         return result
#
#     def run_statistical_test(self, k, show_progress_pvals=False,
#                              save_path='saved_results_statistical_test_BUSPaS_'):
#
#         start = time.time()
#         save_path = save_path + str(start) + '.pkl'
#
#         randomized_qualities = []
#
#         for _ in range(k):
#             if show_progress_pvals:
#                 print('Start buspas statistical test: ', _, '/', k)
#             swapped_matrix = swap_randomize_symmetric(self.matrix)
#
#             bus_swapped = BUSPaS(self.df, swapped_matrix, self.x, self.d, 1,
#                                  nr_chunks=self.nr_chunks, min_coverage_perc=self.min_coverage_perc,
#                                  min_coverage_abs=self.min_coverage_abs)
#             bus_swapped.findQuality(quality_measure=self.quality_measure,
#                                     comparison_type=self.comparison_type, size_corr=self.size_corr)
#
#             randomized_qualities += bus_swapped.quals
#
#         pvalues = []
#         for quality in self.quals:
#             empirical_p_value = (sum([(comp > quality) for comp in randomized_qualities]) + 0.5 * sum(
#                 [(comp == quality) for comp in randomized_qualities]) + 1) / (len(randomized_qualities) + 1)
#             pvalues.append(empirical_p_value)
#         self.randomized_qualities = randomized_qualities
#         self.pvalues = pvalues
#         self.result_pvalues = []
#
#
#         for i in range(len(self.result)):
#             self.result_pvalues.append(tuple(list(self.result[i]) + [self.pvalues[i]]))
#
#         with open(save_path, 'wb') as f:
#             pickle.dump(self.result_pvalues, f)
#
#         return self.result_pvalues
#
#     def statistical_test(self, k, save_title="./temp_statistical_test_results.pkl"):
#
#         """
#         Creates quality measure outputs of k randomized experiments
#         to use for statistical testing of the original output.
#         """
#
#         bus_df = self.df.reset_index(drop=True)
#
#         results = []
#
#         for _ in range(k):
#             matrix = BUSPaS.randomize_symmetric_matrix(self.matrix)
#
#             bus_rand = BUSPaS(bus_df, matrix, self.x, self.d, 1,
#                               nr_chunks=self.nr_chunks, min_coverage_perc=self.min_coverage_perc,
#                               min_coverage_abs=self.min_coverage_abs)
#             bus_rand.findQuality(quality_measure=self.quality_measure,
#                                  comparison_type=self.comparison_type, size_corr=self.size_corr)
#             res = bus_rand.quals
#             results = results + res
#
#             self.random_output = results
#             np.save(save_title, results)
#
#         pvalues = []
#         for x in self.quals:
#             empirical_p_value = (sum([(comp > x) for comp in self.random_output]) + 0.5 * sum(
#                 [(comp == x) for comp in self.random_output]) + 1) / (len(self.random_output) + 1)
#             pvalues.append(empirical_p_value)
#         self.pvalues = pvalues
#
#         print('after checking ', self.count_quality, ' potential subgroups')
#         print('Outcome of bottumUpSearch is:')
#         print(' ')
#         print('avg_quality = ', round(self.avg_quality, 3))
#         print('max_quality = ', round(max(self.quals), 3))
#         print(' ')
#         print('avg_coverage = ', round(self.avg_coverage, 3))
#         print('max_coverage = ', round(max(self.covs), 3))
#         print(' ')
#
#         p = 0
#         for z in self.result:
#             conjunction = " Ʌ ".join(["(" + condition.replace(" == ", "=").strip() + ")" for condition in z[2]])
#             print('description =', conjunction)
#             print('quality =', round(z[0], 3), " with p-value= ", round(self.pvalues[p], 4))
#             print('coverage =', round(z[1], 3))
#             print(' ')
#             p += 1
#         to_be_saved = [0, 0]
#         to_be_saved[0] = self.pvalues
#         to_be_saved[1] = self.random_output
#
#         np.save(save_title, to_be_saved)
#
#     def save(self, title, **kwargs):
#
#         self.title = title
#
#         try:
#             pickle_results = pd.read_pickle('results_hierarchical.pkl')
#         except:
#             pickle_results = pd.DataFrame(
#                 columns=['title', 'running_time', "subgroups_checked", 'descs', 'avg_quality', 'avg_coverage',
#                          'quality_measure', 'x', 'd', 'q',
#                          'correct_for_size_var', 'result'])
#
#         pickle_results.loc[len(pickle_results)] = [self.title, self.running_time, self.count_quality,
#                                                    self.descriptions, self.avg_quality, self.avg_coverage,
#                                                    self.quality_measure, self.x, self.d, self.q,
#                                                    self.size_corr, self.result]
#
#         pickle_results.to_pickle('results_hierarchical.pkl')
#
#         try:
#             try:
#                 pickle_results = pd.read_pickle('random_results_hierarchical.pkl')
#             except:
#                 pickle_results = pd.DataFrame(columns=['title', 'p-values', 'results'])
#
#             pickle_results.loc[len(pickle_results)] = [self.title, self.pvalues, self.random_output]
#
#             pickle_results.to_pickle('random_results_hierarchical.pkl')
#         except:
#             pass