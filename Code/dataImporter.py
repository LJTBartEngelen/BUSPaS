# Package imports
import pandas as pd
import numpy as np
import pickle

path = "C:/Users/bengelen004/Documents/TUe/BUSPas/New Submission WIP/Data/Stock/"

def make_naive(df, cat, num, features, naive_attributes=None, countries=None):
    """selects only a small set of well interpretable (naive) attributes
    and stocks from specific countries to experiment with
    """

    if naive_attributes is None:
        naive_attributes = ['enterpriseToEbitda', 'sector', 'industry',
                            'currency', 'fullTimeEmployees', 'exchange', 'marketCap',
                            'averageVolume10days',
                            'country'] 
    else:
        naive_attributes = naive_attributes

    if countries is None:
        countries = ['United Kingdom', 'United States', 'Russia', 'France', 'Germany', 'Netherlands',
                     'China']  # ['United Kingdom', 'Canada', 'Australia', 'Japan', 'South Korea','Luxembourg', 'Hungary', 'Bulgaria',
        # 'Iceland', 'Estonia', 'Lithuania', 'Latvia', 'Greece', 'Poland', 'Finland', 'Austria', 'Ireland',
        # 'Portugal', 'Norway', 'Russia', 'Netherlands', 'Sweden', 'Switzerland', 'Spain', 'Italy',
        # 'Belgium', 'Denmark', 'Qatar', 'United States', 'Turkey', 'New Zealand', 'France',
        # 'Germany', ]
    else:
        countries = countries

    cat_naive = list(set(naive_attributes) & set(cat))
    num_naive = list(set(naive_attributes) & set(num))

    features_naive = cat_naive + num_naive

    df_naive = df[features_naive + ['target']]

    df_naive = df_naive[df_naive['country'].isin(countries)]

    return cat_naive, num_naive, df_naive, features_naive


def comb_descr_targ(descriptive_space, target_space):
    """"Combines a descriptive space and a target space (column with lists that represent time-series) into one df"""

    # transforms targetspace into one column with lists that represent time-series
    selection = target_space[list(target_space.columns)].transpose()
    selection_add = selection.astype(str).agg(', '.join, axis=1)
    rows = []
    for i in selection_add:
        ts = [float(x) for x in list(i.split(", "))]
        rows.append(ts)
    selection['ts'] = rows

    combined_df = descriptive_space.copy()

    combined_df['target'] = selection['ts']

    return combined_df


# Function used to import a specified dataset
def getData(dataset, **kwargs):
    """" dataset = string name of dataset"""

    conversion_rates = pd.DataFrame({'currency': ['EUR',
                                                  'TRY',
                                                  'CNY',
                                                  'MXN',
                                                  'GBp',
                                                  'TWD',
                                                  'CAD',
                                                  'KRW',
                                                  np.nan,
                                                  'INR',
                                                  'USD',
                                                  'HKD',
                                                  'AUD',
                                                  'BRL',
                                                  'SEK',
                                                  'RUB',
                                                  'ILA',
                                                  'NOK',
                                                  'THB',
                                                  'IDR',
                                                  'MYR',
                                                  'DKK',
                                                  'NZD',
                                                  'SGD',
                                                  'ARS',
                                                  'CHF',
                                                  'ISK',
                                                  'QAR', 'GBP', 'JPY'],
                                     'rate': [1.03544, 0.05363, 0.13942, 0.05201, 1.19872, 0.03229, 0.74023,
                                              0.00075, 0, 0.01225, 1
                                         , 0.12802, 0.66997, 0.1876, 0.09486, 0.01638, 0.29051, 0.10014, 0.02813,
                                              0.00006,
                                              0.22177, 0.13923, 0.62078, 0.72744, 0.006, 1.05187, 0.00702, 0.26906,
                                              1.19872, 0.00722]}).set_index('currency')  # ,'currencyToUSD'
    to_be_converted = ['revenuePerShare', 'enterpriseValue', 'operatingCashflow', 'freeCashflow',
                       'totalCashPerShare', 'ebitda', 'totalAssets', 'totalDebt', 'totalCash', 'totalRevenue',
                       'marketCap', 'netIncomeToCommon',
                       'annualHoldingsTurnover', 'grossProfits', 'ytdReturn', 'dayHigh']

    if dataset == "synthetic":

        df = pd.read_pickle('C:/Users/bengelen003/OneDrive - pwc/Code/AE Haar - ETSM version/synthetic.pkl')
        features = list(df.columns)
        features.remove('target')
        cat = features
        num = []

    # Stock5YLarge on 5 year monthly selected only the right columns
    elif dataset == "Stock5YLarge":
        # descriptive_df = pickle.load(
        #     open(path + "descriptive_attributes_5yData_large.pkl", "rb")).copy()
        # target_att_5y_large = pickle.load(
        #     open(path + "target_attributes_5yData_large.pkl", "rb")).copy()

        descriptive_df = pd.read_pickle(path + "descriptive_attributes_5yData_large.pkl")
        target_att_5y_large = pd.read_pickle(path + "target_attributes_5yData_large.pkl")

        # exclude attributes that only contains NaN's
        descriptive_df = descriptive_df.dropna(axis=1, how='all')

        # exclude attributes that only contains same values or all different values
        not_incl_diversity = ['equityHoldings', 'sectorWeightings', 'bondRatings', 'holdings', 'bondHoldings',
                              'companyOfficers']

        # exclude attributes that will not contribute to insights for subgroups
        # TODO elaborate on describing why
        not_incl_by_hand = ['zip', 'trailingPE', 'trailingAnnualDividendYield', 'priceToSalesTrailing12Months',
                            'gmtOffSetMilliseconds', 'lastCapGain', 'err', 'address3', 'preMarketSource',
                            'impliedSharesOutstanding', 'postMarketSource', 'openInterest', 'marketState',
                            'underlyingSymbol', 'preferredPosition', 'uuid', 'maxAge', 'address1', 'messageBoardId',
                            'address2', 'logo_url', 'fax', 'currencySymbol', 'symbol', 'website', 'shortName', 'phone',
                            'category', 'fundFamily', 'morningStarOverallRating', 'longName']

        # exclude categorical attributes that are redundant as they are highly correlated with other attributes
        # based on Theils correlation
        not_incl_by_Theils = ['tradeable', 'quoteSourceName', 'regularMarketSource']

        # exclude numerical attributes that are redundant as they are highly correlated with other attributes
        # based on Pearson correlation
        not_incl_by_Pearson = ['averageDailyVolume10Day', 'averageDailyVolume10Day', 'regularMarketPrice',
                               'regularMarketDayHigh', 'dayLow', 'dayLow', 'regularMarketDayLow',
                               'twoHundredDayAverage',
                               'fiveYearAverageReturn', 'nextFiscalYearEnd', 'nextFiscalYearEnd', 'open',
                               'regularMarketOpen', 'dayLow', 'fiveYearAverageReturn', 'open', 'previousClose',
                               'regularMarketDayLow', 'regularMarketOpen', 'regularMarketPreviousClose',
                               'previousClose',
                               'previousClose', 'regularMarketPreviousClose', 'regularMarketDayHigh',
                               'regularMarketDayLow', 'dayLow', 'regularMarketDayLow', 'regularMarketOpen', 'open',
                               'regularMarketOpen', 'regularMarketPreviousClose', 'regularMarketPreviousClose',
                               'previousClose', 'regularMarketPrice', 'regularMarketVolume',
                               'twoHundredDayAverage', 'regularMarketVolume']

        # excluded attributes as they result in errors
        # TODO process longBusinessSummary into new attributes
        other = ['yield', '52WeekChange', 'longBusinessSummary']

        drop_attributes = not_incl_diversity + not_incl_by_hand + not_incl_by_Theils + not_incl_by_Pearson + other
        descriptive_df = descriptive_df.drop(drop_attributes, axis=1)

        descriptive_df = descriptive_df.replace("'", '', regex=True)

        cat = ['city', 'country', 'currency', 'exchange', 'exchangeName', 'exchangeTimezoneName',
               'exchangeTimezoneShortName', 'financialCurrency', 'industry', 'isEsgPopulated', 'lastSplitFactor',
               'legalType', 'market', 'quoteType', 'recommendationKey', 'sector', 'state']
        num = ['SandP52WeekChange', 'annualHoldingsTurnover', 'annualReportExpenseRatio', 'ask', 'askSize',
               'averageDailyVolume3Month', 'averageVolume', 'averageVolume10days', 'beta', 'beta3Year', 'bid',
               'bidSize', 'bondPosition', 'bookValue', 'cashPosition', 'convertiblePosition', 'currentPrice',
               'currentRatio', 'dateShortInterest', 'dayHigh', 'debtToEquity', 'dividendRate', 'dividendYield',
               'earningsGrowth', 'earningsQuarterlyGrowth', 'ebitda', 'ebitdaMargins', 'enterpriseToEbitda',
               'enterpriseToRevenue', 'enterpriseValue', 'exDividendDate', 'exchangeDataDelayedBy', 'fiftyDayAverage',
               'fiftyTwoWeekHigh', 'fiftyTwoWeekLow', 'fiveYearAvgDividendYield', 'floatShares', 'forwardEps',
               'forwardPE', 'freeCashflow', 'fullTimeEmployees', 'fundInceptionDate', 'grossMargins', 'grossProfits',
               'heldPercentInsiders', 'heldPercentInstitutions', 'lastDividendDate', 'lastDividendValue',
               'lastFiscalYearEnd', 'lastSplitDate', 'marketCap', 'morningStarRiskRating', 'mostRecentQuarter',
               'navPrice', 'netIncomeToCommon', 'numberOfAnalystOpinions', 'operatingCashflow', 'operatingMargins',
               'otherPosition', 'payoutRatio', 'pegRatio', 'preMarketPrice', 'priceHint', 'priceToBook',
               'profitMargins', 'quickRatio', 'recommendationMean', 'regularMarketChange',
               'regularMarketChangePercent', 'regularMarketTime', 'returnOnAssets', 'returnOnEquity', 'revenueGrowth',
               'revenuePerShare', 'sharesOutstanding', 'sharesPercentSharesOut', 'sharesShort',
               'sharesShortPreviousMonthDate', 'sharesShortPriorMonth', 'shortPercentOfFloat', 'shortRatio',
               'stockPosition', 'targetHighPrice', 'targetLowPrice', 'targetMeanPrice', 'targetMedianPrice',
               'threeYearAverageReturn', 'totalAssets', 'totalCash', 'totalCashPerShare', 'totalDebt', 'totalRevenue',
               'trailingAnnualDividendRate', 'trailingEps', 'trailingPegRatio', 'volume', 'ytdReturn']

        # ['52WeekChange', 'SandP52WeekChange', 'annualHoldingsTurnover', 'annualReportExpenseRatio',
        # 'ask', 'askSize', 'averageDailyVolume3Month', 'averageVolume', 'averageVolume10days', 'beta', 'beta3Year',
        # 'bid','bidSize', 'bondPosition', 'bookValue', 'cashPosition', 'convertiblePosition', 'currentPrice',
        # 'currentRatio', 'dateShortInterest', 'dayHigh', 'debtToEquity', 'dividendRate', 'dividendYield',
        # 'earningsGrowth', 'earningsQuarterlyGrowth', 'ebitda', 'ebitdaMargins', 'enterpriseToEbitda',
        # 'enterpriseToRevenue', 'enterpriseValue', 'exDividendDate', 'exchangeDataDelayedBy',
        # 'fiftyDayAverage', 'fiftyTwoWeekHigh', 'fiftyTwoWeekLow', 'fiveYearAvgDividendYield',
        # 'floatShares', 'forwardEps', 'forwardPE', 'freeCashflow', 'fullTimeEmployees', 'fundInceptionDate',
        # 'grossMargins', 'grossProfits', 'heldPercentInsiders', 'heldPercentInstitutions',
        # 'lastDividendDate', 'lastDividendValue', 'lastFiscalYearEnd', 'lastSplitDate', 'marketCap',
        # 'morningStarRiskRating', 'mostRecentQuarter', 'navPrice', 'netIncomeToCommon', 'numberOfAnalystOpinions',
        # 'operatingCashflow', 'operatingMargins', 'otherPosition', 'payoutRatio', 'pegRatio',
        # 'preMarketPrice', 'priceHint', 'priceToBook', 'profitMargins', 'quickRatio', 'recommendationMean',
        # 'regularMarketChange', 'regularMarketChangePercent', 'regularMarketTime', 'returnOnAssets',
        # 'returnOnEquity', 'revenueGrowth', 'revenuePerShare', 'sharesOutstanding',
        # 'sharesPercentSharesOut', 'sharesShort', 'sharesShortPreviousMonthDate', 'sharesShortPriorMonth',
        # 'shortPercentOfFloat', 'shortRatio', 'stockPosition', 'targetHighPrice', 'targetLowPrice',
        # 'targetMeanPrice', 'targetMedianPrice', 'threeYearAverageReturn', 'totalAssets', 'totalCash',
        # 'totalCashPerShare', 'totalDebt', 'totalRevenue', 'trailingAnnualDividendRate', 'trailingEps',
        # 'trailingPegRatio', 'volume', 'yield', 'ytdReturn']

        # combines descriptive and ts target space

        to_drop = ['currentPrice', 'fiftyDayAverage', 'fiftyTwoWeekLow', 'bid', 'targetHighPrice', 'beta',
                   'preMarketPrice', 'ask', 'targetMedianPrice', 'targetMeanPrice']

        df_merge = pd.merge(descriptive_df, conversion_rates, left_on='currency', right_index=True)

        for col in to_be_converted:
            df_merge[col] = df_merge[col] * df_merge['rate']
            df_merge[col].replace(0, np.nan, inplace=True)

        descriptive_df = df_merge.drop(to_drop, axis=1).drop('rate', axis=1)

        features = num + cat
        df = comb_descr_targ(descriptive_df, target_att_5y_large)

        nan_mask = df['target'].apply(lambda x: all(np.isnan(i) for i in x))
        same_values_mask = df['target'].apply(lambda x: len(set(x)) == 1 )
    
        exclude_mask = np.logical_or(nan_mask, same_values_mask)
    
        df = df[~exclude_mask]

    elif dataset == "Stock5YSmall":
        df, cat, num, features = getData("Stock5YLarge")

        if kwargs['countries'] == 'all':
            countries = list(df['country'].unique())  # all countries
        elif kwargs['countries'] == 'all_excl_nan':
            countries = list(df['country'].dropna(axis=0).unique())
        elif kwargs['countries'] == 'large_economies':
            countries = ['United Kingdom', 'United States', 'France', 'Switzerland', 'Germany', 'Netherlands', 'Japan',
                         'Italy', 'Spain', 'Canada', 'Australia', 'South Korea']
        elif kwargs['countries'] == 'western':
            countries = ['United Kingdom', 'Canada', 'Australia', 'Japan', 'South Korea', 'Luxembourg', 'Hungary',
                         'Bulgaria', 'Iceland', 'Estonia', 'Lithuania', 'Latvia', 'Greece', 'Poland', 'Finland',
                         'Austria', 'Ireland', 'Portugal', 'Norway', 'Russia', 'Netherlands', 'Sweden', 'Switzerland',
                         'Spain', 'Italy', 'Belgium', 'Denmark', 'Qatar', 'United States', 'Turkey', 'New Zealand',
                         'France', 'Germany']
        elif kwargs['countries'] == 'europe':
            countries = ['United Kingdom', 'Luxembourg', 'Hungary', 'Bulgaria',
                         'Iceland', 'Estonia', 'Lithuania', 'Latvia', 'Greece', 'Poland', 'Finland', 'Austria',
                         'Ireland',
                         'Portugal', 'Norway', 'Russia', 'Netherlands', 'Sweden', 'Switzerland', 'Spain', 'Italy',
                         'Belgium',
                         'Denmark', 'Turkey', 'France', 'Germany', ]
        else:
            countries = kwargs['countries']

        try:
            attributes = kwargs['attributes']
            if attributes == 'expertBased':
                attributes = ['averageVolume10days', 'beta', 'debtToEquity', 'enterpriseToEbitda', 'fullTimeEmployees',
                              'marketCap', 'industry', 'sector', 'country', 'currency', 'exchange',
                              'exchangeTimezoneName']
        except:
            attributes = features
        cat, num, df, features = make_naive(df, cat, num, features, naive_attributes=attributes, countries=countries)

        nan_mask = df['target'].apply(lambda x: all(np.isnan(i) for i in x))
        same_values_mask = df['target'].apply(lambda x: len(set(x)) == 1 )
    
        exclude_mask = np.logical_or(nan_mask, same_values_mask)
    
        df = df[~exclude_mask]

    

    elif dataset == "Stock2YSmall":
        cat = ['city', 'country', 'currency', 'exchange', 'exchangeName', 'exchangeTimezoneName',
               'exchangeTimezoneShortName', 'financialCurrency', 'industry', 'isEsgPopulated', 'lastSplitFactor',
               'legalType', 'market', 'quoteType', 'recommendationKey', 'sector', 'state']
        num = ['SandP52WeekChange', 'annualHoldingsTurnover', 'annualReportExpenseRatio', 'ask', 'askSize',
               'averageDailyVolume3Month', 'averageVolume', 'averageVolume10days', 'beta', 'beta3Year', 'bid',
               'bidSize', 'bondPosition', 'bookValue', 'cashPosition', 'convertiblePosition', 'currentPrice',
               'currentRatio', 'dateShortInterest', 'dayHigh', 'debtToEquity', 'dividendRate', 'dividendYield',
               'earningsGrowth', 'earningsQuarterlyGrowth', 'ebitda', 'ebitdaMargins', 'enterpriseToEbitda',
               'enterpriseToRevenue', 'enterpriseValue', 'exDividendDate', 'exchangeDataDelayedBy', 'fiftyDayAverage',
               'fiftyTwoWeekHigh', 'fiftyTwoWeekLow', 'fiveYearAvgDividendYield', 'floatShares', 'forwardEps',
               'forwardPE', 'freeCashflow', 'fullTimeEmployees', 'fundInceptionDate', 'grossMargins', 'grossProfits',
               'heldPercentInsiders', 'heldPercentInstitutions', 'lastDividendDate', 'lastDividendValue',
               'lastFiscalYearEnd', 'lastSplitDate', 'marketCap', 'morningStarRiskRating', 'mostRecentQuarter',
               'navPrice', 'netIncomeToCommon', 'numberOfAnalystOpinions', 'operatingCashflow', 'operatingMargins',
               'otherPosition', 'payoutRatio', 'pegRatio', 'preMarketPrice', 'priceHint', 'priceToBook',
               'profitMargins', 'quickRatio', 'recommendationMean', 'regularMarketChange',
               'regularMarketChangePercent', 'regularMarketTime', 'returnOnAssets', 'returnOnEquity', 'revenueGrowth',
               'revenuePerShare', 'sharesOutstanding', 'sharesPercentSharesOut', 'sharesShort',
               'sharesShortPreviousMonthDate', 'sharesShortPriorMonth', 'shortPercentOfFloat', 'shortRatio',
               'stockPosition', 'targetHighPrice', 'targetLowPrice', 'targetMeanPrice', 'targetMedianPrice',
               'threeYearAverageReturn', 'totalAssets', 'totalCash', 'totalCashPerShare', 'totalDebt', 'totalRevenue',
               'trailingAnnualDividendRate', 'trailingEps', 'trailingPegRatio', 'volume', 'ytdReturn']

        try:
            start_date = kwargs['start_date']
        except:
            start_date = 0

        try:
            end_date = kwargs['end_date']
        except:
            end_date = 0

        descriptive_df_2y = pickle.load(
            open("C:/Users/bengelen003/OneDrive - pwc/Code/HP/descriptive_attributes_2yData_large.pkl", "rb")).copy()
        target_att_2y_large = pickle.load(
            open("C:/Users/bengelen003/OneDrive - pwc/Code/HP/target_attributes_2yData_large.pkl", "rb")).copy()

        if start_date == 0 and end_date == 0:
            time_series = target_att_2y_large
        elif start_date == 0:
            time_series = target_att_2y_large[:end_date]
        elif end_date == 0:
            time_series = target_att_2y_large[start_date:]
        else:
            time_series = target_att_2y_large[start_date:end_date]

        time_series = time_series.interpolate(method='linear', limit_direction='both', limit=7)

        try:
            if kwargs['countries'] == 'all':
                countries = list(descriptive_df_2y['country'].unique())  # all countries
            elif kwargs['countries'] == 'all_excl_nan':
                countries = list(descriptive_df_2y['country'].dropna(axis=0).unique())
            elif kwargs['countries'] == 'large_economies':
                countries = ['United Kingdom', 'United States', 'France', 'Switzerland', 'Germany', 'Netherlands',
                             'Japan',
                             'Italy', 'Spain', 'Canada', 'Australia', 'South Korea']
            elif kwargs['countries'] == 'western':
                countries = ['United Kingdom', 'Canada', 'Australia', 'Japan', 'South Korea', 'Luxembourg', 'Hungary',
                             'Bulgaria', 'Iceland', 'Estonia', 'Lithuania', 'Latvia', 'Greece', 'Poland', 'Finland',
                             'Austria', 'Ireland', 'Portugal', 'Norway', 'Russia', 'Netherlands', 'Sweden',
                             'Switzerland',
                             'Spain', 'Italy', 'Belgium', 'Denmark', 'Qatar', 'United States', 'Turkey', 'New Zealand',
                             'France', 'Germany']
            elif kwargs['countries'] == 'europe':
                countries = ['United Kingdom', 'Luxembourg', 'Hungary', 'Bulgaria',
                             'Iceland', 'Estonia', 'Lithuania', 'Latvia', 'Greece', 'Poland', 'Finland', 'Austria',
                             'Ireland',
                             'Portugal', 'Norway', 'Russia', 'Netherlands', 'Sweden', 'Switzerland', 'Spain', 'Italy',
                             'Belgium',
                             'Denmark', 'Turkey', 'France', 'Germany', ]
            else:
                countries = kwargs['countries']
            descriptive_df_2y = descriptive_df_2y[descriptive_df_2y['country'].isin(countries)]
        except:
            pass

        try:
            attributes = kwargs['attributes']
        except:
            attributes = None

        if attributes == None:

            # exclude attributes that only contains NaN's
            descriptive_df = descriptive_df_2y.dropna(axis=1, how='all')

            # exclude attributes that only contains same values or all different values
            not_incl_diversity = ['equityHoldings', 'sectorWeightings', 'bondRatings', 'holdings', 'bondHoldings',
                                  'companyOfficers']

            # exclude attributes that will not contribute to insights for subgroups
            # TODO elaborate on describing why
            not_incl_by_hand = ['zip', 'trailingPE', 'trailingAnnualDividendYield', 'priceToSalesTrailing12Months',
                                'gmtOffSetMilliseconds', 'lastCapGain', 'err', 'address3', 'preMarketSource',
                                'impliedSharesOutstanding', 'postMarketSource', 'openInterest', 'marketState',
                                'underlyingSymbol', 'preferredPosition', 'uuid', 'maxAge', 'address1', 'messageBoardId',
                                'address2', 'logo_url', 'fax', 'currencySymbol', 'symbol', 'website', 'shortName',
                                'phone',
                                'category', 'fundFamily', 'morningStarOverallRating', 'longName']

            # exclude categorical attributes that are redundant as they are highly correlated with other attributes
            # based on Theils correlation
            not_incl_by_Theils = ['tradeable', 'quoteSourceName', 'regularMarketSource']

            # exclude numerical attributes that are redundant as they are highly correlated with other attributes
            # based on Pearson correlation
            not_incl_by_Pearson = ['averageDailyVolume10Day', 'averageDailyVolume10Day', 'regularMarketPrice',
                                   'regularMarketDayHigh', 'dayLow', 'dayLow', 'regularMarketDayLow',
                                   'twoHundredDayAverage',
                                   'fiveYearAverageReturn', 'nextFiscalYearEnd', 'nextFiscalYearEnd', 'open',
                                   'regularMarketOpen', 'dayLow', 'fiveYearAverageReturn', 'open', 'previousClose',
                                   'regularMarketDayLow', 'regularMarketOpen', 'regularMarketPreviousClose',
                                   'previousClose',
                                   'previousClose', 'regularMarketPreviousClose', 'regularMarketDayHigh',
                                   'regularMarketDayLow', 'dayLow', 'regularMarketDayLow', 'regularMarketOpen', 'open',
                                   'regularMarketOpen', 'regularMarketPreviousClose', 'regularMarketPreviousClose',
                                   'previousClose', 'regularMarketPrice', 'regularMarketVolume',
                                   'twoHundredDayAverage', 'regularMarketVolume']

            # excluded attributes as they result in errors
            # TODO process longBusinessSummary into new attributes
            other = ['yield', '52WeekChange', 'longBusinessSummary']

            drop_attributes = not_incl_diversity + not_incl_by_hand + not_incl_by_Theils + not_incl_by_Pearson + other
            drop_attributes = list(set(drop_attributes) & set(descriptive_df.columns))

            descriptive_df = descriptive_df.drop(drop_attributes, axis=1)

            descriptive_df = descriptive_df.replace("'", '', regex=True)

            cat = ['city', 'country', 'currency', 'exchange', 'exchangeName', 'exchangeTimezoneName',
                   'exchangeTimezoneShortName', 'financialCurrency', 'industry', 'isEsgPopulated', 'lastSplitFactor',
                   'legalType', 'market', 'quoteType', 'recommendationKey', 'sector', 'state']
            num = ['SandP52WeekChange', 'annualHoldingsTurnover', 'annualReportExpenseRatio', 'ask', 'askSize',
                   'averageDailyVolume3Month', 'averageVolume', 'averageVolume10days', 'beta', 'beta3Year', 'bid',
                   'bidSize', 'bondPosition', 'bookValue', 'cashPosition', 'convertiblePosition', 'currentPrice',
                   'currentRatio', 'dateShortInterest', 'dayHigh', 'debtToEquity', 'dividendRate', 'dividendYield',
                   'earningsGrowth', 'earningsQuarterlyGrowth', 'ebitda', 'ebitdaMargins', 'enterpriseToEbitda',
                   'enterpriseToRevenue', 'enterpriseValue', 'exDividendDate', 'exchangeDataDelayedBy',
                   'fiftyDayAverage',
                   'fiftyTwoWeekHigh', 'fiftyTwoWeekLow', 'fiveYearAvgDividendYield', 'floatShares', 'forwardEps',
                   'forwardPE', 'freeCashflow', 'fullTimeEmployees', 'fundInceptionDate', 'grossMargins',
                   'grossProfits',
                   'heldPercentInsiders', 'heldPercentInstitutions', 'lastDividendDate', 'lastDividendValue',
                   'lastFiscalYearEnd', 'lastSplitDate', 'marketCap', 'morningStarRiskRating', 'mostRecentQuarter',
                   'navPrice', 'netIncomeToCommon', 'numberOfAnalystOpinions', 'operatingCashflow', 'operatingMargins',
                   'otherPosition', 'payoutRatio', 'pegRatio', 'preMarketPrice', 'priceHint', 'priceToBook',
                   'profitMargins', 'quickRatio', 'recommendationMean', 'regularMarketChange',
                   'regularMarketChangePercent', 'regularMarketTime', 'returnOnAssets', 'returnOnEquity',
                   'revenueGrowth',
                   'revenuePerShare', 'sharesOutstanding', 'sharesPercentSharesOut', 'sharesShort',
                   'sharesShortPreviousMonthDate', 'sharesShortPriorMonth', 'shortPercentOfFloat', 'shortRatio',
                   'stockPosition', 'targetHighPrice', 'targetLowPrice', 'targetMeanPrice', 'targetMedianPrice',
                   'threeYearAverageReturn', 'totalAssets', 'totalCash', 'totalCashPerShare', 'totalDebt',
                   'totalRevenue',
                   'trailingAnnualDividendRate', 'trailingEps', 'trailingPegRatio', 'volume', 'ytdReturn']

            # combines descriptive and ts target space

            to_drop = ['currentPrice', 'fiftyDayAverage', 'fiftyTwoWeekLow', 'bid', 'targetHighPrice', 'beta',
                       'preMarketPrice', 'ask', 'targetMedianPrice', 'targetMeanPrice']

            df_merge = pd.merge(descriptive_df, conversion_rates, left_on='currency', right_index=True)

            for col in to_be_converted:
                df_merge[col] = df_merge[col] * df_merge['rate']
                df_merge[col].replace(0, np.nan, inplace=True)

            descriptive_df_2y = df_merge.drop(to_drop, axis=1).drop('rate', axis=1)
            # descriptive_df = descriptive_df.loc[:, descriptive_df.isnull().mean() <= 0.8]


        else:
            if attributes == "expertBased":
                attributes = ['averageVolume10days','beta','debtToEquity','enterpriseToEbitda','fullTimeEmployees',
                               'marketCap','industry','sector','country','currency','exchange','exchangeTimezoneName']
            descriptive_df_2y = descriptive_df_2y.replace("'", '', regex=True)
            attributes_app = attributes.copy()
            attributes_app.append('currency')
            attributes_app = list(set(attributes_app))
            descriptive_df_2y = descriptive_df_2y[attributes_app]

            to_be_converted = list(set(to_be_converted) & set(attributes))
            df_merge = pd.merge(descriptive_df_2y, conversion_rates, left_on='currency', right_index=True)
            for col in to_be_converted:
                df_merge[col] = df_merge[col] * df_merge['rate']
                df_merge[col].replace(0, np.nan, inplace=True)
            descriptive_df_2y = df_merge
            descriptive_df_2y = descriptive_df_2y[attributes]

        # try:
        #     attributes = kwargs['attributes']
        #     if attributes == 'expertBased':
        #         attributes = ['averageVolume10days','beta','debtToEquity','enterpriseToEbitda','fullTimeEmployees',
        #                       'marketCap','industry','sector','country','currency','exchange','exchangeTimezoneName']
        # except:

        df = comb_descr_targ(descriptive_df_2y, time_series)
        
        nan_mask = df['target'].apply(lambda x: all(np.isnan(i) for i in x))
        same_values_mask = df['target'].apply(lambda x: len(set(x)) == 1 )
    
        exclude_mask = np.logical_or(nan_mask, same_values_mask)
    
        df = df[~exclude_mask]

    else:
        num = []
        cat = []
        df = pd.DataFrame()
        features = num + cat

    cat = list(set(df.columns) & set(cat))
    num = list(set(df.columns) & set(num))
    features = num + cat

    # Return the dataset <df>, a list of categorical features <cat>,
    # a list of numerical features <num> and a total set of features

    return df, cat, num, features
