from beamSearch import *
from qualityMeasures import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle


def select_subgroup(df, desc):

    """selects subgroup based on description
    df = complete dataset
    desc = description for the to be retrieved subgroup. Conjunction of attribute-value (refined) pairs."""

    sub_group = df[df.eval(as_string(desc))]

    return sub_group


def analyse_output(df, output):

    subgroup_comb = output
    avg_coverage = sum([i[1] for i in subgroup_comb]) / len([i[1] for i in subgroup_comb])
    avg_quality = sum([i[0] for i in subgroup_comb]) / len([i[0] for i in subgroup_comb])

    set_individuals = set()
    tot_individuals = 0

    subgroup_comb = sorted(subgroup_comb, key=lambda x: x[0], reverse=True)

    for i in subgroup_comb:
        print('quality= ',round(i[0],5),' coverage= ',round(i[1],5),'\n',' description: ','\n',i[3],'\n')
        set_individuals = set(list(select_subgroup(df, i[3]).index)) | set_individuals
        tot_individuals = tot_individuals + len(select_subgroup(df, i[3]))
    nr_results = len([i[0] for i in subgroup_comb])

    print('\n')
    print('nr of subgroups= ', nr_results)
    print('avg. coverage= ', round(avg_coverage,5))
    print('tot. individuals over {} subgroups = '.format(nr_results), tot_individuals)
    print('tot. unique individuals over {} subgroups = '.format(nr_results), len(set_individuals))
    print('avg. quality measure= ', round(avg_quality,5))


def analyse_result(title, df):
    "old version of analysis"

    results = pickle.load(open("results.pkl", "rb")).copy()
    result = results[results['title'] == title]

    subgroup_comb = result['output'].reset_index(drop=True)[0]
    avg_coverage = sum([i[1] for i in subgroup_comb]) / len([i[1] for i in subgroup_comb])
    avg_quality = sum([i[0] for i in subgroup_comb]) / len([i[0] for i in subgroup_comb])

    set_individuals = set()
    tot_individuals = 0

    subgroup_comb = sorted(subgroup_comb, key=lambda x: x[0], reverse=True)

    for i in subgroup_comb:
        print('quality= ', round(i[0], 5), ' coverage= ', round(i[1], 5), '\n', ' description: ', '\n', i[3], '\n')
        set_individuals = set(list(select_subgroup(df, i[3]).index)) | set_individuals
        tot_individuals = tot_individuals + len(select_subgroup(df, i[3]))
    nr_results = len([i[0] for i in subgroup_comb])

    print('\n')
    print('nr of subgroups= ', nr_results)
    print('avg. coverage= ', round(avg_coverage, 5))
    print('tot. individuals over {} subgroups = '.format(nr_results), tot_individuals)
    print('tot. unique individuals over {} subgroups = '.format(nr_results), len(set_individuals))
    print('avg. quality measure= ', round(avg_quality, 5))


def subgroup_df_index_based(df, subgroup_index_list, col):
    distribution_values = df.loc[subgroup_index_list]
    d = distribution_values[col]
    leng1 = len(d)
    d = distribution_values.loc[(distribution_values[col] >= -1) & (distribution_values[col] <= 1)]
    print(leng1 - len(d))

    d = list(d[col])

    return d


def plot_subgroup_hist(df, desc, col, title):
    indexes = list(select_subgroup(df, desc).index)

    d = subgroup_df_index_based(df, indexes, col)

    # f = Fitter(d)#, distributions=['gamma','dweibull', 'erlang', 'norm', 'weibull_max', 'weibull_min'])
    # f.fit()
    # print(f.summary())

    h = plt.hist(d, bins=30, color="k")
    plt.xlabel('Partial autocorrelation lag=12')
    plt.ylabel('Occurrences')
    plt.title(title)
    # plt.savefig('C:/Users/bengelen003/OneDrive - pwc/Code/AE Haar - ETSM version/pacf3_pop.png')
    plt.show()


def analyse(original_df, index_result, results_file='results.pkl'):
    df = original_df

    results = pd.read_pickle(results_file).copy()
    result = results.iloc[index_result]
    subgroup_comb = result['output']  # .reset_index(drop=True) #[0]

    avg_coverage = sum([i[1] for i in subgroup_comb]) / len([i[1] for i in subgroup_comb])
    avg_quality = sum([i[0] for i in subgroup_comb]) / len([i[0] for i in subgroup_comb])

    set_individuals = set()
    tot_individuals = 0

    subgroup_comb = sorted(subgroup_comb, key=lambda x: x[0], reverse=True)

    for i in subgroup_comb:
        print('quality= ', round(i[0], 5), ' coverage= ', round(i[1], 5), '\n', ' description: ', '\n', i[3], '\n')
        set_individuals = set(list(select_subgroup(df, i[3]).index)) | set_individuals
        tot_individuals = tot_individuals + len(select_subgroup(df, i[3]))
    nr_results = len([i[0] for i in subgroup_comb])

    print('\n')
    print('nr of subgroups= ', nr_results)
    print('avg. coverage= ', round(avg_coverage, 5))
    print('tot. individuals over {} subgroups = '.format(nr_results), tot_individuals)
    print('tot. unique individuals over {} subgroups = '.format(nr_results), len(set_individuals))
    print('avg. quality measure= ', round(avg_quality, 5))
