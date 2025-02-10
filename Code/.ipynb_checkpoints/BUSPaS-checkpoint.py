import heapq
import re
import time
from qualityMeasure import *
from helperFunctions import *


def desc_diverse(desc):
    # alerts when a description contains something in the form desc =
    # ['marketCap > 687961248.0', 'fullTimeEmployees == 9800.0',
    #       'marketCap > 23376277504.0', 'marketCap > 96089239552.0']
    # where we see that some operators are redundant
    desc_list = []
    for i in desc:
        words = i.split()
        attribute = words[0]
        operator = words[1]

        desc_list.append((attribute, operator))

    desc_list_num = [i for i in desc_list if i[1] not in ["!=", "=="]]

    return len(desc_list_num) == len(set(desc_list_num))


def satisfies_all(desc, ind, len_df, threshold=0.0001, threshold_absolute=5):  # TODO optimize threshold

    # Function used to check if subgroup with pattern <desc> has sufficient coverage w.r.t. <df>
    # A subgroup is sufficiently large if the proportion of data included in it exceeds <threshold>

    sum_ind = len(ind)
    #sum_ind = sum(ind)

    return sum_ind >= len_df * threshold and sum_ind >= threshold_absolute and desc_diverse(desc)


def as_string(desc):
    # Adds ' and ' to <desc> such that selectors are properly separated when the refine function is used
    return ' and '.join(desc)


def eval_quality(ind, df, target, quality_measure, comparison_type='complement',**kwargs):

    #sub_group = df[ind]
    sub_group = ind

    if comparison_type == 'population':
        complement_df = df
    elif comparison_type == 'complement':
        complement_df = df.loc[~df.index.isin(sub_group.index)]
    else:
        complement_df = df

    phi = quality_measure(sub_group, target, complement_df, **kwargs)

    coverage = len(sub_group)/len(df)

    return phi, coverage

    
class BUSPaS:
    
    def __init__(self,df,distance_matrix=[],number_of_row_pairs=100,depth=3,q=10,nr_chunks=5,min_coverage_perc=0.01,min_coverage_abs=3):

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

        if type(distance_matrix) == str:
            print('Calculating distance matrix')
            self.matrix = calculate_distance_matrix(self.df, distance_function=distance_matrix)
        elif type(distance_matrix) == np.ndarray:
            self.matrix = distance_matrix
        else:
            print('Calculating euclidean distance matrix')
            self.matrix = calculate_distance_matrix(self.df, distance_function=euclidean)

    
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
                    candidate = "{} <= {} <= {}".format(x ,col, y)
                    self.df[col] = self.df[col].apply(lambda val: candidate if (not isinstance(val, str) and x <= val <= y) else val)
        pass

                    
    #TODO increase combination of two to combination of x
    def find_min_indices(matrix, x):
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
    
    
    def get_common_attributes(row1, row2, d_len):
        """
        Finds the attribute value combinations that the two rows have in common.
        
        row1, row2 = row of a dataframe
        d_len = description length
        """
        
        common_attributes = []
        for combination in combinations(row1.items(), d_len): # for combination in combinations(row1.iteritems(), d_len):
            attributes = [f"{attribute} == '{value}'" for attribute, value in combination]
            if all(row2.get(attribute) == value for attribute, value in combination):
                common_attributes.append(attributes)
        return common_attributes

    
    def get_unique_lists(list_of_lists):
        """
        Filters redundant descriptions, when they are exactly the same or in a different order.
        """
        
        unique_lists = [list(x) for x in set(tuple(set(sublist)) for sublist in list_of_lists)]
        return unique_lists

    
    def findQuality(self, quality_measure = cluster_based_quality_measure, comparison_type = "complement" ,size_corr = no_size_corr):
        
        self.running_time = time.time()
        self.quality_measure = quality_measure
        self.size_corr = size_corr
        self.quality_measure_counter = 0
        self.comparison_type = comparison_type
        
        min_heap = []
        len_df = len(self.df)
        
        promising_combinations = BUSPaS.find_min_indices(self.matrix, self.x)
        
        #TODO in line below it only works when target attribute is the last attribute
        candidate_descriptions = [BUSPaS.get_common_attributes(self.df.iloc[promising_combinations[i][0]][:-1],self.df.iloc[promising_combinations[i][1]][:-1],d) for i in range(len(promising_combinations)) for d in range(1,self.d+1)]
        
        # unique_candidate_descriptions gives the unique potential candidate descriptions
        unique_candidate_descriptions = BUSPaS.get_unique_lists([item for sublist in candidate_descriptions for item in sublist])
        
        for desc in unique_candidate_descriptions:
            ind = self.df.query(as_string(desc))
            
            #checks if subgroups comply with size constrains
            if satisfies_all(desc, ind, len_df, self.min_coverage_perc, self.min_coverage_abs):
                
                quality, coverage = eval_quality(ind, self.df, 'target', self.quality_measure, comparison_type,distance_matrix=self.matrix,correct_for_size=size_corr)
                self.quality_measure_counter += 1
                
                if len(min_heap) < self.q:
                    heapq.heappush(min_heap, (quality, desc, coverage))
                    #heapq.heapify(min_heap)
                else:
                    
                    if -quality < -heapq.nsmallest(1, min_heap)[0][0]:

                        equal_quals = [i for i, x in enumerate(min_heap) if x[0] == quality]
                        
                        # checks if set of records isn't already present as a result of a different description
                        if len(equal_quals)>0:
                            for i in equal_quals:
                                comp = self.df.query(as_string(min_heap[i][1]))
                                if np.array_equal(comp.index, ind.index):
                                    pass
                                else:
                                    heapq.heappushpop(min_heap, (quality, desc, coverage))
                        else:
                            heapq.heappushpop(min_heap, (quality, desc, coverage))

        
        self.running_time = time.time() - self.running_time
        
        self.result = sorted(min_heap, key=lambda x: x[0], reverse=True)
        
        
        #part that changes the numerical propositions back into a evaluatable thing instead of a string
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
            data[i] = ( value1, sublist, value2 )
            
        self.result = data
        self.quals = [i[0] for i in min_heap]
        self.covs = [i[2] for i in min_heap]
        self.avg_quality = sum(self.quals) / len(self.quals)
        self.avg_coverage = sum(self.covs) / len(self.covs)
        self.descs = [i[1] for i in min_heap]
        pass
        
    def print_outcome(self):
        
        print('after checking ',self.quality_measure_counter,' potential subgroups')
        print('Outcome of bottumUpSearch is:')
        print(' ')
        print('avg_quality = ',round(self.avg_quality,3))
        print('max_quality = ',round(max(self.quals),3))
        print(' ')
        print('avg_coverage = ',round(self.avg_coverage,3))
        print('max_coverage = ',round(max(self.covs),3))
        print(' ')
                
        for z in self.result:
            conjunction = " Ʌ ".join(["(" + condition.replace(" == ", "=").strip() + ")" for condition in z[1]])
            print('description =',conjunction)
            print('quality =',round(z[0],3))
            print('coverage =',round(z[2],3))
            print(' ')
            
    def randomize_symmetric_matrix(matrix):
        """
        Takes a distance matrix. 
        Returns a swap randomized symmmetric version of the matrix.
        """
        
        n = matrix.shape[0]
        triu_indices = np.triu_indices(n, k=1)  # Upper triangular indices excluding the diagonal
        randomized_values = np.random.permutation(matrix[triu_indices])  # Randomize the upper triangular values
        result = np.zeros_like(matrix)  # Initialize the result matrix with zeros
        result[triu_indices] = randomized_values  # Assign the randomized values to the upper triangular indices
        result.T[triu_indices] = randomized_values  # Assign the same values to the lower triangular indices (keeping it symmetric)
        
        return result
            
    def statistical_test(self,k,save_title="./temp_statistical_test_results.pkl"):
        
        """
        Creates quality measure outputs of k randomized experiments 
        to use for statistical testing of the original output.
        """
        
        bus_df = self.df.reset_index(drop=True)

        results = []

        for _ in range(k):
            
            matrix = BUSPaS.randomize_symmetric_matrix(self.matrix)

            bus_rand = BUSPaS(bus_df, matrix, self.x, self.d, 1,
                                     nr_chunks=self.nr_chunks, min_coverage_perc=self.min_coverage_perc,
                                     min_coverage_abs=self.min_coverage_abs)
            bus_rand.findQuality(quality_measure = self.quality_measure, 
                                 comparison_type = self.comparison_type , size_corr = self.size_corr)
            res = bus_rand.quals
            results = results+res
        
            self.random_output = results
            np.save(save_title, results)
            
        pvalues = []
        for x in self.quals:
            empirical_p_value = (sum([(comp > x) for comp in self.random_output])+0.5*sum([(comp == x) for comp in self.random_output])+1)/(len(self.random_output)+1)
            pvalues.append(empirical_p_value)
        self.pvalues = pvalues
        
        print('after checking ',self.quality_measure_counter,' potential subgroups')
        print('Outcome of bottumUpSearch is:')
        print(' ')
        print('avg_quality = ',round(self.avg_quality,3))
        print('max_quality = ',round(max(self.quals),3))
        print(' ')
        print('avg_coverage = ',round(self.avg_coverage,3))
        print('max_coverage = ',round(max(self.covs),3))
        print(' ')
            
        p=0
        for z in self.result:
            
            conjunction = " Ʌ ".join(["(" + condition.replace(" == ", "=").strip() + ")" for condition in z[1]])
            print('description =',conjunction)
            print('quality =',round(z[0],3)," with p-value= ",round(self.pvalues[p],4))
            print('coverage =',round(z[2],3))
            print(' ')
            p+=1
        to_be_saved = [0,0]
        to_be_saved[0] = self.pvalues
        to_be_saved[1] = self.random_output

        np.save(save_title, to_be_saved)
        
    
    def save(self, title, **kwargs):
        
        self.title = title
            
        try:
            pickle_results = pd.read_pickle('results_hierarchical.pkl')
        except:
            pickle_results = pd.DataFrame(columns=['title', 'running_time', "subgroups_checked", 'descs', 'avg_quality', 'avg_coverage',
                                                   'quality_measure', 'x', 'd', 'q',
                                                   'correct_for_size_var', 'result'])

        pickle_results.loc[len(pickle_results)] = [self.title, self.running_time, self.quality_measure_counter, self.descs, self.avg_quality, self.avg_coverage,
                                                   self.quality_measure, self.x, self.d, self.q,
                                                   self.size_corr, self.result]

        pickle_results.to_pickle('results_hierarchical.pkl')
        
        try:
            try:
                pickle_results = pd.read_pickle('random_results_hierarchical.pkl')
            except:
                pickle_results = pd.DataFrame(columns=['title', 'p-values','results'])

            pickle_results.loc[len(pickle_results)] = [self.title, self.pvalues, self.random_output]

            pickle_results.to_pickle('random_results_hierarchical.pkl')
        except:
            pass

        