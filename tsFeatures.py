import statsmodels.api as sm
import statistics
from scipy import stats
from statsmodels.tsa.seasonal import STL
from utilityFunctions import *


def max_par_ac(x, **kwargs):
    try:
        return np.max( sm.tsa.pacf(x)[2:] )

    except:
        pass


def max_ac_lag(x, **kwargs):
    try:
        return sm.tsa.acf(x, nlags=kwargs['lag'])[-1]

    except:
        pass


def par_ac_lag(ts, **kwargs):
    try:
        return sm.tsa.pacf(ts, nlags=kwargs['lag'])[-1]

    except:
        pass


def seasonality_boolean(x, **kwargs):

    """returns a boolean that answers the question whether there exists seasonlity in the time-series
    based on the partial autocorrelation of a time-series (ts)"""

    lag = kwargs['lag']

    lower_limit = -2 / math.sqrt(len(x))
    upper_limit = 2 / math.sqrt(len(x))

    print(x)

    if lag == None:
        pacfs = sm.tsa.pacf(x, nlags=None)
        return int(any(val < lower_limit or val > upper_limit for val in pacfs))

    else:
        corr = sm.tsa.pacf(x, nlags=lag)[-1]
        return int(corr < lower_limit or corr > upper_limit)


def seasonality_boolean_with_conf(x, **kwargs):

    lag = kwargs['lag']

    lower_limit = -2 / math.sqrt(len(x))
    upper_limit = 2 / math.sqrt(len(x))

    if lag == None:
        pacfs = sm.tsa.pacf(x, nlags=None)
        return int(any(val < lower_limit or val > upper_limit for val in pacfs))

    else:
        outc = sm.tsa.pacf(x, nlags=lag, alpha=0.05)
        l, u = outc[1][-1]
        return int((l < lower_limit and u < lower_limit) or (l > upper_limit and u > upper_limit))


def stl_strength_of_trend(ts, **kwargs):
    # ts = ts.tolist()
    # ts = ts.apply(lambda x: remove_nans_at_edges(x))
    ts = remove_nans_at_edges(ts)
    ts = pd.Series(ts, index=pd.date_range("1-11-2017", periods=len(ts), freq="M"), name="ts")
    try:
        stl = STL(ts)
        res = stl.fit()
        return max(0,( 1 - ( statistics.variance(res.resid) / statistics.variance(res.trend + res.resid) ) ) )
    except:
        return np.nan


def stl_strength_of_seasonality(ts, **kwargs):
    # ts = ts.tolist()
    # ts = ts.apply(lambda x: remove_nans_at_edges(x))
    ts = remove_nans_at_edges(ts)
    ts = pd.Series(ts, index=pd.date_range("1-11-2017", periods=len(ts), freq="M"), name="ts")
    try:
        stl = STL(ts)
        res = stl.fit()
        return max(0, (1 - (statistics.variance(res.resid) / statistics.variance( res.seasonal + res.resid ))))

    except:
        return np.nan


def slope(ts, **kwargs):
    beta1, intercept, r, p, std_err = stats.linregress([i for i in range(len(ts))], ts)
    return beta1