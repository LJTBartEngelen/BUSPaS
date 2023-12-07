from tsFeatures import *

# class Quality_measure:
#
#     def __init__(self):
#


def average_list_difference(df, col, **kwargs):

    #TODO add variable for average list difference

    """
    Calculate the difference between the last and 32th last elements of a list in each row of a column
    and return the average of these differences using vectorized operations.
    """

    diff_list = df[col].apply(lambda x: x[-1]-x[-32]) #-32 is corona crisis
    return diff_list.mean()


def slope_aggregated(df, col, **kwargs):

    ts_agg = aggregate_timeseries_subgroup(df, col, agg_func='mean')
    beta = slope(ts_agg)

    return beta


#TODO auto correlation max afmaken
# def max_ac(df, col):
#
#     try:
#         return np.max(sm.tsa.pacf(x)[2:])
#
#     except:
#         pass


def mean_sq_squared(df, col, **kwargs):

    diff_list = df[col].apply(lambda x: x**2)

    return diff_list.mean()


# TODO mean: come up with statistically compensated version
def mean(df, col, **kwargs):
    avg = df[col].mean()
    return avg

# TODO median: come up with statistically compensated version
def median(df, col, complement_df,**kwargs):
    med = df[col].median()
    return med


def max_partial_autocorrelation(df, col, complement_df,**kwargs):

    diff_list = df[col].apply(lambda x: max_par_ac(x))

    return diff_list.mean()


def pacf_aggregated(df, col,complement_df, **kwargs):

    ts_agg = aggregate_timeseries_subgroup(df, col, agg_func='mean')

    pacf = par_ac_lag(ts_agg, lag=kwargs['lag'])
    #TODO dit geeft soms none values, uitzoeken hoe dit komt
    #TODO uitzoeken hoe het kan dat ik oa pacfs van boven de 1 krijg
    if pacf is not None:
        return pacf
    else:
        return 0


#TODO rmse quality measure
def rmse_quality_measure(df, col, **kwargs):
    return df[col].mean()


def as_string(desc):
    # Adds ' and ' to <desc> such that selectors are properly separated when the refine function is used
    return ' and '.join(desc)


def hellinger2(p, q, **kwargs):
    """Hellinger distance between two discrete distributions.
       input two arrays with discrete distributions e.g. p = [0, 0.3, 0.2, 0.1, 0.4]
    """
    return math.sqrt(sum([ (math.sqrt(p_i) - math.sqrt(q_i))**2 for p_i, q_i in zip(p, q) ]) / 2)


def hellinger_dot(p, q, **kwargs):
    """Hellinger distance between two discrete distributions.
    input two arrays with discrete distributions e.g. p = [0, 0.3, 0.2, 0.1, 0.4]
    """
    z = np.sqrt(p) - np.sqrt(q)
    return np.sqrt(z @ z / 2)


def stl_strength_of_trend_aggregated(df, col,complement_df, **kwargs):
    # ts = ts.tolist()
    # ts = ts.apply(lambda x: remove_nans_at_edges(x))
    ts = aggregate_timeseries_subgroup(df, col, agg_func='mean')
    ts = remove_nans_at_edges(ts)
    ts = pd.Series(ts, index=pd.date_range("1-11-2017", periods=len(ts), freq="M"), name="ts")
    try:
        stl = STL(ts)
        res = stl.fit()
        return max(0,( 1 - ( statistics.variance(res.resid) / statistics.variance(res.trend + res.resid) ) ) )
    except:
        return np.nan


def stl_strength_of_seasonality_aggregated(df,col, complement_df,**kwargs):
    # ts = ts.tolist()
    # ts = ts.apply(lambda x: remove_nans_at_edges(x))
    ts = aggregate_timeseries_subgroup(df, col, agg_func='mean')

    ts = remove_nans_at_edges(ts)
    ts = pd.Series(ts, index=pd.date_range("1-11-2017", periods=len(ts), freq="M"), name="ts")
    try:
        stl = STL(ts)
        res = stl.fit()
        return max(0, (1 - (statistics.variance(res.resid) / statistics.variance( res.seasonal + res.resid ))))

    except:
        return np.nan


# TODO make a valid qm from avg std
def avg_std(df, col,complement_df, **kwargs):

    """calculates the std for each time in the ts's and averages them"""

    lists = pd.DataFrame(df[col].tolist())
    std_list = lists.apply(calc_std)
    std_list = np.array(std_list).tolist()

    print(sum(std_list) / len(std_list))

    return - sum(std_list)/len(std_list) # - is so that lowest std is regarded as best


def wracc(df, col, complement, **kwargs):
    # Function used to calculate the solution's WRAcc

    sub_group = df
    df = complement

    prop_p_sg = len(sub_group[sub_group[col] == 1])/len(sub_group)
    prop_p_df = len(df[df[col] == 1])/len(df)
    wracc = ((len(sub_group)/len(df))**1) * (prop_p_sg - prop_p_df)  # for WRAcc a=1

    return wracc