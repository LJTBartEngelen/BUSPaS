import time
import pandas as pd

from beamSearch import *
from utilityFunctions import *
from clusterFunctions import *
from tsFeatures import *
#from afterAnalysis import *
import pickle


class Experiment:

    def __init__(self, title, df, features,
                 w_var=50, d_var=3, q_var=50, target_var='target',
                 n_chunks_var=5, normalization_var = None, comparison_type = 'complement',
                 ts_feature = None, ts_feature_var = None, report_progress = False, allow_exclusion = False,
                 min_coverage = 0.0001, min_coverage_abs = 5, min_error = 0.0001, **kwargs):
        self.title = title  # of saved experiment
        self.df = df.copy()  # that is used for EMM
        self.features = features  # in the descriptive space
        self.w_var = w_var  # width of Beam
        self.d_var = d_var  # max length of a description (nr levels)
        self.q_var = q_var  # number of results
        self.target_var = target_var  # name of the column that contains the target
        self.n_chunks_var = n_chunks_var  # number of chunks in which numerical attributes are split each level
        self.normalization_var = normalization_var  # function to normalize numerical data
        self.comparison_type = comparison_type  # whether we compare a sample with its complement or the population
        self.report_progress = report_progress  # True when we want elaborate updates on progress
        self.allow_exclusion = allow_exclusion  # True when we want exclusive operators in descriptions
        self.min_coverage = min_coverage  # minimum coverage required for a subgroup to be examined
        self.min_coverage_abs = min_coverage_abs  # minimum subgroup size required for a subgroup to be examined
        self.min_error = min_error

        self.ts_feature = ts_feature  # used if we calculate value for each single row as the target attribute
        self.ts_feature_var = ts_feature_var  # if needed to use an extra variable in the ts feature

        self.output = []  # placeholder
        self.running_time = 0  # placeholder
        self.quality_measure = None  # placeholder
        self.kwargs = kwargs  # placeholder

        self.distance_matrix = None
        self.correct_for_size_var = None  # function to correct for sample size

        self.quality_measure_kwargs = None  # placeholder

        self.__dict__.update(kwargs)

    def prepare(self,**kwargs):

        nan_mask = self.df[self.target_var].apply(lambda x: all(np.isnan(i) for i in x))
        same_values_mask = self.df[self.target_var].apply(lambda x: len(set(x)) == 1 )

        exclude_mask = np.logical_or(nan_mask, same_values_mask)

        self.df = self.df[~exclude_mask]

        if self.normalization_var is not None:
            self.df[self.target_var] = self.df[self.target_var].apply(self.normalization_var)

        if self.ts_feature is not None:
            self.df[self.target_var] = self.df[self.target_var]\
                .apply(lambda x: self.ts_feature(x, **kwargs))

        if 'distance_measure_matrix' in kwargs:
            self.distance_matrix = calculate_distance_matrix(self.df, distance_function=kwargs['distance_measure_matrix'])


    def mine( self, quality_measure = None, **kwargs ): #EMM

        #  quality_measure = function to calculate quality of a subgroup
        #  kwargs are used for variables in quality_measure
        self.quality_measure = quality_measure
        self.quality_measure_kwargs = kwargs

        start = time.time()

        if 'pre_calculated_distance_matrix' in kwargs:
            self.distance_matrix = kwargs['pre_calculated_distance_matrix']

        if self.distance_matrix is not None:
            kwargs['distance_matrix'] = self.distance_matrix

        self.output = EMM(self.df, self.features, w=self.w_var, d=self.d_var, q=self.q_var, catch_all_description=[],
                     comparison_type=self.comparison_type, target=self.target_var, n_chunks=self.n_chunks_var,
                     ensure_diversity=True, quality_measure=quality_measure, report_progress= self.report_progress,
                     allow_exclusion=self.allow_exclusion, min_coverage = self.min_coverage,
                          min_coverage_abs = self.min_coverage_abs, min_error = self.min_error, **kwargs)

        self.running_time = (time.time() - start)

        print("completed: ",self.title," in ", self.running_time, ' seconds')
        print(self.output.values)

    def save(self, **kwargs):

        if 'title' in kwargs:
            title_holder = self.title + ' ' + kwargs['title']
        else:
            title_holder = self.title

        pickle_results = pickle.load(open("results.pkl", "rb"))

        quality_measure_kwargs_placeholder = self.quality_measure_kwargs

        if 'distance_matrix' in quality_measure_kwargs_placeholder:
            del quality_measure_kwargs_placeholder['distance_matrix']

        pickle_results.loc[len(pickle_results)] = [title_holder, self.running_time, self.output.values,
                                                   self.quality_measure, quality_measure_kwargs_placeholder,
                                                   self.kwargs, self.w_var, self.d_var, self.q_var, self.n_chunks_var,
                                                   self.allow_exclusion, self.normalization_var,
                                                   self.correct_for_size_var]

        pickle_results.to_pickle('results.pkl')
        pickle_results.to_excel('/Users/bengelen003/OneDrive - pwc/Code/ESTM/results.xlsx', index=False)
