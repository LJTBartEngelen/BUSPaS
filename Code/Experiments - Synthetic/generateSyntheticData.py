import random
import pandas as pd
import time
import functools
from scipy.stats import cauchy

class ts_generator:
    """Allows user to create a time-series with a certain description:
    nr_tp = number of time-points in a time-series
    mu = starting point / average (only in random white noise)
    mu_err =
    sig_err =

    """

    def __init__(self, nr_tp, mu=100, mu_err=0, sig_err=1):
        self.nr_tp = nr_tp
        self.mu = mu
        self.mu_err = mu_err
        self.sig_err = sig_err

    def ts_random_whitenoise(self):
        """random white noise, points in the series diverge with a normal distributed random value from the mu"""

        return [round(self.mu + random.gauss(mu=self.mu_err, sigma=self.sig_err), 1) for i in range(self.nr_tp)]

    def ts_random_wn_shift(self):
        """random walk, points in the series diverge with a normal distributed random value from the previous value"""

        ts = [self.mu]
        for _ in range(self.nr_tp - 1):
            ts.append(round(ts[-1] + random.gauss(mu=self.mu_err, sigma=self.sig_err), 1))

        return ts

    def ts_random_chauchy_shift_prod(self):
        """random walk, points in the series diverge with a normal distributed random value from the previous value"""

        ts = [self.mu]
        for _ in range(self.nr_tp - 1):
            ts.append(round(ts[-1] * ((cauchy.rvs(loc=-0.5, scale=6, random_state=None)/100)+1), 1))

        return ts

    def ts_random_wns_season(self, alpha, season):
        """random walk dependent on last value and value a season back
        alpha = fraction explained by values of one season back
        season = number of time-points in a period / number of seasons"""

        ts = ts_generator(season, self.mu, self.mu_err, self.sig_err).ts_random_wn_shift()

        # ts_wns = [round(mu + random.gauss(mu=mu_err, sigma=sig_err),1) for _ in range(season)]
        for _ in range(self.nr_tp - season):
            ts.append(
                round((1 - alpha) * ts[-1] + alpha * ts[-season] + random.gauss(mu=self.mu_err, sigma=self.sig_err), 1))

        return ts


def MA(arr, window_size):
    """converts array into a Moving average version of the array"""

    i = 0
    moving_averages = []

    while i < len(arr) - window_size + 1:
        window = arr[i: i + window_size]
        window_average = round(sum(window) / window_size, 2)
        moving_averages.append(window_average)
        i += 1
    return moving_averages


def synthetic_data_generator(nr_inst=100, nr_attr=5, cats_low=2, cats_high=10, nr_tp=48, mu=50, mu_err=0, sig_err=1,
                             season=None, alpha=None, **kwargs):
    start_time = time.time()
    #####################

    data = []

    # Create descriptive attributes
    meta_attr = [random.randint(cats_low, cats_high) for i in range(
        nr_attr)]  # determines how many unique values per attributes, returns list with integers per attribute

    for i in range(nr_inst):
        r = []  # record
        for i in range(nr_attr):
            r.append(str(random.randint(1, meta_attr[
                i - 1])))  # creates descriptives #[random.randint(ts_low,ts_high) for i in range(nr_tp)] ts_random_wns_season(nr_tp,mu,mu_err,sig_err,alpha,season)
        try:
            ts_type = kwargs['ts_type']
            if ts_type == 'cauchy_rw':
                r.append(ts_generator(nr_tp, mu, mu_err,sig_err).ts_random_chauchy_shift_prod())
            elif ts_type == 'rw':
                r.append(ts_generator(nr_tp, mu, mu_err,sig_err).ts_random_wn_shift())
            else:
                r.append(ts_generator(nr_tp, mu, mu_err, sig_err).ts_random_wn_shift())
        except:
            r.append(ts_generator(nr_tp, mu, mu_err,
                                  sig_err).ts_random_wn_shift())# add a time-series for one record and add this to the record ###ts_random_wns_season(alpha,season)
        data.append(r)

        # naming of descriptve attributes
    cols = []
    for i in range(nr_attr):
        cols.append("A{att}".format(att=i + 1) + "numbOfVal{q}".format(q=meta_attr[i - 1]))

    cols.append("ts")

    df = pd.DataFrame(data, columns=cols)

    return df


def test_subgroup_generator(df, mu, mu_err, sig_err, alpha=0.3, season=4,  size_new_sg=50, descr='auto',**kwargs):

    """builds a test subgroup based on parameter values and columns in the existing dataframe,
    outputs the subgroup in list"""

    nr_tp = len(df['ts'][0])

    if descr == 'auto':
        new_sg_descr = []
        for i in df.columns.drop('ts'):
            temp = df[i].apply(lambda x: int(x))
            new_sg_descr.append(str(max(temp) + 1))
    else:
        new_sg_descr = descr

    new_sg = []
    for i in range(size_new_sg):

        ts_type = kwargs['ts_type']

        if ts_type == 'cauchy_rw':
            ts = ts_generator(nr_tp, mu, mu_err, sig_err).ts_random_chauchy_shift_prod()

        elif ts_type == 'rw':
            ts = ts_generator(nr_tp, mu, mu_err, sig_err).ts_random_wn_shift()

        else:
            ts = ts_generator(nr_tp, mu, mu_err, sig_err).ts_random_wns_season(alpha, season)

        temp = new_sg_descr.copy()
        temp.append(ts)

        new_sg.append(temp)
    return new_sg


def add_new_subgroup(df, sg):
    """adds new generated test subgroup (list) to an existing dataframe"""
    return pd.concat([df, pd.DataFrame(sg, columns=df.columns)], ignore_index=True)


def create_cluster_sg(sg, multiplications=5, deviation=1, true_sg_type="error_shift"):
    """takes a subgroup and multiplies its rows with time series with a deviation of the original time-series"""

    sample = sg[0]
    ts = sample[-1]
    length = len(ts)
    for _ in range(multiplications):

        #rand = [random.gauss(mu=0, sigma=deviation) for i in range(length)]

        rand = [0] * length  # Initialize a list of zeros with the desired length
        prev = 0  # Initialize a variable to store the previous value
        rand[0] = 0

        for i in range(1,length):
            val = random.gauss(mu=0, sigma=deviation)  # Generate a new random value
            prev += val  # Add the new value to the previous value
            if true_sg_type == "error_shift":
                rand[i] = prev  # Store the new sum in the list
            elif true_sg_type == "error_no_shift":
                  # Store the new sum in the list
                rand[i] = val
            else:
                rand[i] = prev

        new_ts = [x + y for x, y in zip(ts, rand)]
        new_entry = sample[:-1] + [new_ts]
        sg = sg + [new_entry]
    return sg


def create_true_sg(df, true_subgroup_attributes, sg_format, deviation=0.5, outliers=0, true_sg_type="error_shift"):
    """uses an example of a time series, creates a true subgroup and replaces an original subgroup"""

    conditions = [(df[col] == str(1)) for col in df.columns[:true_subgroup_attributes]]
    condition = functools.reduce(lambda a, b: a & b, conditions)
    true_sg = df[condition][:int(round((1-outliers)*len(df[condition])))]

    true_sg_len = len(true_sg)

    sg = create_cluster_sg(sg_format, multiplications=true_sg_len - 1, deviation=deviation, true_sg_type=true_sg_type)

    new_ts = pd.DataFrame(sg, columns=df.columns, index=true_sg.index)
    new_ts = new_ts.rename(index=dict(zip(list(true_sg.index), list(new_ts.index))))
    true_sg['ts'] = pd.DataFrame(new_ts).iloc[:, -1]
    df.update(true_sg)
    df = df.rename(columns={'ts': 'target'})

    return df


def generator_synthetic_data_with_true_subgroup(n,
                                                nr_attr,
                                                delta_mu_in_true_subgroup,
                                                nr_tp,
                                                true_subgroup_attributes,
                                                deviation_in_true_subgroup,
                                                outliers_perc,
                                                ts_type='cauchy_rw',
                                                true_sg_type="error_shift",
                                                typical_behavior_sig=2,
                                                typical_behavior_mu=-0.5):


    df = synthetic_data_generator(nr_inst=n, nr_attr=nr_attr, mu=100, mu_err=typical_behavior_mu, sig_err=typical_behavior_sig, cats_high=2, nr_tp=nr_tp,
                                  ts_type=ts_type)  # rw cauchy_rw
    pos_neg = random.choice([-1, 1])
    sg_format = test_subgroup_generator(df, mu=100, mu_err=typical_behavior_mu + (delta_mu_in_true_subgroup * pos_neg),
                                        sig_err=typical_behavior_sig, size_new_sg=1,
                                        descr=[str(1) for i in range(nr_attr)],ts_type=ts_type)  # rw cauchy_rw
    df = create_true_sg(df, true_subgroup_attributes, sg_format, deviation=deviation_in_true_subgroup,
                        outliers=outliers_perc, true_sg_type=true_sg_type)  # error_no_shift error_shift
    features = list(df.columns[:-1])

    true_gr_desc = []
    for i in range(true_subgroup_attributes):
        true_gr_desc.append("A{}numbOfVal2 == '1'".format(i + 1))

    return df, features, true_gr_desc
