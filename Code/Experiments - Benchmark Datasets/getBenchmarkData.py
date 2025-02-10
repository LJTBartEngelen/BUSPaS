from Code.qualityMeasure import *
from Code.helperFunctions import *
import os
import pandas as pd
from scipy.io import arff
import re
from Code.dataImporter import getData
import pickle
import ast

def normalize_choice(value, nr_targets):
    # Extract the integer from the byte string
    choice = int(value.decode('utf-8').strip("b").strip("'"))
    # Normalize it to a scale of 0-1 (where 1 represents the maximum choice of 16)
    return round((choice - 1) / (nr_targets - 1), 3)


def convert_to_binary(df):
    # Loop through each column in the DataFrame
    for col in df.iloc[:,:-1].columns:
        # Check if the column has exactly two unique values
        if df[col].nunique() == 2:
            # Map the two unique values to 0 and 1
            unique_vals = df[col].unique()
            df[col] = df[col].map({unique_vals[0]: str(0), unique_vals[1]: str(1)})
    return df


def ranking_to_columns(ranking):
    # Split the ranking string into a list
    letters = ranking.split('>')
    # Create a dictionary mapping each letter to its rank
    return {letter: rank + 1 for rank, letter in enumerate(letters)}


def get_data_wisconsin():

    """"all descriptives are num"""

    nr_targets = 16

    data_wisconsin, meta = arff.loadarff(
        os.path.join(os.getcwd(), '..', '..', 'Data', 'Wisconsin', 'wisconsin_pref.arff'))
    data_wisconsin = pd.DataFrame(data_wisconsin)

    for col in data_wisconsin.columns[-nr_targets:]:
        data_wisconsin[col] = data_wisconsin[col].apply(lambda x: normalize_choice(x, nr_targets))

    columns_to_transform = [f'L{i}' for i in range(1, nr_targets+1)]

    data_wisconsin[columns_to_transform] = (data_wisconsin[columns_to_transform]
                                            .apply(lambda x: (x - x.min()) / (x.max() - x.min())))

    # Create the 'target' column as a list of values from columns L1 to L16
    data_wisconsin['target'] = data_wisconsin[columns_to_transform].values.tolist()

    # Drop the original columns L1 to L16
    data_wisconsin = data_wisconsin.drop(columns=columns_to_transform)

    data_wisconsin = data_wisconsin.reset_index(drop=True)

    path_post = 'euclidean_distance_matrix_wisconsin.npy'
    path = os.path.join(os.getcwd(), '..', '..', 'Data', 'Wisconsin', path_post)

    if not os.path.exists(path):
        distance_matrix = calculate_distance_matrix(data_wisconsin, distance_function=euclidean)
        np.save(path, distance_matrix)
    else:
        distance_matrix = np.load(path)

    return data_wisconsin, 'target', distance_matrix


def get_data_elevators():

    """"all descriptives are num"""

    nr_targets = 9

    elevators, meta = arff.loadarff(
        os.path.join(os.getcwd(), '..', '..', 'Data', 'elevators', 'elevators.arff'))
    elevators = pd.DataFrame(elevators)

    for col in elevators.columns[-nr_targets:]:
        elevators[col] = elevators[col].apply(lambda x: normalize_choice(x, nr_targets))

    columns_to_transform = [f'L{i}' for i in range(1, nr_targets+1)]

    elevators[columns_to_transform] = (elevators[columns_to_transform]
                                            .apply(lambda x: (x - x.min()) / (x.max() - x.min())))

    # Create the 'target' column as a list of values from columns L1 to L16
    elevators['target'] = elevators[columns_to_transform].values.tolist()

    # Drop the original columns L1 to L16
    elevators = elevators.drop(columns=columns_to_transform)

    elevators = elevators.reset_index(drop=True)

    path_post = 'euclidean_distance_matrix_elevators.npy'
    path = os.path.join(os.getcwd(), '..', '..', 'Data', 'elevators', path_post)

    if not os.path.exists(path):
        distance_matrix = calculate_distance_matrix(elevators, distance_function=euclidean)
        distance_matrix = np.float16(distance_matrix)
        np.save(path, distance_matrix)
    else:
        distance_matrix = np.load(path)

    return elevators, 'target', distance_matrix


def get_data_wine():

    nr_targets = 3

    data_wine, meta = arff.loadarff(os.path.join(os.getcwd(), '..', '..', 'Data', 'wine', 'wine.arff'))
    data_wine = pd.DataFrame(data_wine)

    for col in data_wine.columns[-nr_targets:]:
        data_wine[col] = data_wine[col].apply(lambda x: normalize_choice(x, nr_targets))

    columns_to_transform = [f'L{i}' for i in range(1, nr_targets + 1)]

    data_wine[columns_to_transform] = (data_wine[columns_to_transform]
                                            .apply(lambda x: (x - x.min()) / (x.max() - x.min())))

    # Create the 'target' column as a list of values from columns L1 to L16
    data_wine['target'] = data_wine[columns_to_transform].values.tolist()

    # Drop the original columns L1 to L16
    data_wine = data_wine.drop(columns=columns_to_transform)

    data_wine = data_wine.reset_index(drop=True)

    path_post = 'euclidean_distance_matrix_wine.npy'
    path = os.path.join(os.getcwd(), '..', '..', 'Data', 'wine', path_post)

    if not os.path.exists(path):
        distance_matrix = calculate_distance_matrix(data_wine, distance_function=euclidean)
        np.save(path, distance_matrix)
    else:
        distance_matrix = np.load(path)

    return data_wine, 'target', distance_matrix


def get_data_ecoli():

    """"all descriptives are num"""

    nr_targets = 8

    ecoli, meta = arff.loadarff(
        os.path.join(os.getcwd(), '..', '..', 'Data', 'ecoli', 'ecoli.arff'))
    ecoli = pd.DataFrame(ecoli)

    for col in ecoli.columns[-nr_targets:]:
        ecoli[col] = ecoli[col].apply(lambda x: normalize_choice(x, nr_targets))

    columns_to_transform = [f'L{i}' for i in range(1, nr_targets+1)]

    ecoli[columns_to_transform] = (ecoli[columns_to_transform]
                                            .apply(lambda x: (x - x.min()) / (x.max() - x.min())))

    # Create the 'target' column as a list of values from columns L1 to L16
    ecoli['target'] = ecoli[columns_to_transform].values.tolist()

    # Drop the original columns L1 to L16
    ecoli = ecoli.drop(columns=columns_to_transform)

    ecoli = ecoli.reset_index(drop=True)

    path_post = 'euclidean_distance_matrix_ecoli.npy'
    path = os.path.join(os.getcwd(), '..', '..', 'Data', 'ecoli', path_post)

    if not os.path.exists(path):
        distance_matrix = calculate_distance_matrix(ecoli, distance_function=euclidean)
        np.save(path, distance_matrix)
    else:
        distance_matrix = np.load(path)

    ecoli = convert_to_binary(ecoli)

    return ecoli, 'target', distance_matrix


#OLD DO NOT USE
# def get_data_sushi():
#
#     """attributes are mostly binary and will be converted likewise in this code"""
#
#     nr_targets = 10
#
#     data_sushi, meta = arff.loadarff(os.path.join(os.getcwd(), '..', '..', 'Data', 'Sushi', 'dataset_.arff'))
#     data_sushi = pd.DataFrame(data_sushi)
#
#     for col in data_sushi.columns[-nr_targets:]:
#         data_sushi[col] = data_sushi[col].apply(lambda x: normalize_choice(x, nr_targets))
#
#     columns_to_transform = [f'L{i}' for i in range(1, nr_targets + 1)]
#
#     data_sushi[columns_to_transform] = (data_sushi[columns_to_transform]
#                                             .apply(lambda x: (x - x.min()) / (x.max() - x.min())))
#
#     # Create the 'target' column as a list of values from columns L1 to L16
#     data_sushi['target'] = data_sushi[columns_to_transform].values.tolist()
#
#     # Drop the original columns L1 to L16
#     data_sushi = data_sushi.drop(columns=columns_to_transform)
#
#     data_sushi = data_sushi.reset_index(drop=True)
#
#     path_post = 'euclidean_distance_matrix_sushi.npy'
#     path = os.path.join(os.getcwd(), '..', '..', 'Data', 'Sushi', path_post)
#
#     if not os.path.exists(path):
#         distance_matrix = calculate_distance_matrix(data_sushi, distance_function=euclidean)
#         np.save(path, distance_matrix)
#     else:
#         distance_matrix = np.load(path)
#
#     data_sushi = convert_to_binary(data_sushi)
#
#     return data_sushi, 'target', distance_matrix


def get_data_sushi():
    data_sushi = pd.read_csv(os.path.join(os.getcwd(), '..', '..', 'Data', 'Sushi', 'sushi.txt'), delimiter=',')

    nr_targets = 10

    # Apply the function to the DataFrame and expand the dictionary into columns
    ranking_expanded = data_sushi['ranking'].apply(ranking_to_columns).apply(pd.Series)

    # Combine the new columns with the original DataFrame (optional)
    df = pd.concat([data_sushi, ranking_expanded], axis=1)

    # Drop the original ranking column if no longer needed
    data_sushi = df.drop(columns=['ranking'])

    columns = data_sushi.columns[-nr_targets:]

    # Create the 'target' column as a list of values from columns L1 to L16
    data_sushi['target'] = data_sushi[columns].values.tolist()

    # Drop the original columns L1 to L16
    data_sushi = data_sushi.drop(columns=columns)

    data_sushi = data_sushi.reset_index(drop=True)

    path_post = 'euclidean_distance_matrix_sushi.npy'
    path = os.path.join(os.getcwd(), '..', '..', 'Data', 'Sushi', path_post)

    if not os.path.exists(path):
        distance_matrix = calculate_distance_matrix(data_sushi, distance_function=euclidean)
        np.save(path, distance_matrix)
    else:
        distance_matrix = np.load(path)

    data_sushi = convert_to_binary(data_sushi)

    return data_sushi, 'target', distance_matrix

def get_data_cpu():

    """attributes are nominal"""

    nr_targets = 5

    data_cpu, meta = arff.loadarff(os.path.join(os.getcwd(), '..', '..', 'Data', 'cpu', 'cpu.arff'))
    data_cpu = pd.DataFrame(data_cpu)

    for col in data_cpu.columns[-nr_targets:]:
        data_cpu[col] = data_cpu[col].apply(lambda x: normalize_choice(x, nr_targets))

    columns_to_transform = [f'L{i}' for i in range(1, nr_targets + 1)]
    data_cpu[columns_to_transform] = (data_cpu[columns_to_transform]
                                      .apply(lambda x: (x - x.min()) / (x.max() - x.min())))
    data_cpu['target'] = data_cpu[columns_to_transform].values.tolist()
    data_cpu = data_cpu.drop(columns=columns_to_transform)

    data_cpu = data_cpu.reset_index(drop=True)

    path_post = 'euclidean_distance_matrix_cpu.npy'
    path = os.path.join(os.getcwd(), '..', '..', 'Data', 'cpu', path_post)

    if not os.path.exists(path):
        distance_matrix = calculate_distance_matrix(data_cpu, distance_function=euclidean)
        distance_matrix = np.float16(distance_matrix)
        np.save(path, distance_matrix)
    else:
        distance_matrix = np.load(path)

    return data_cpu, 'target', distance_matrix


def get_data_voting():

    data_voting = pd.read_pickle(os.path.join(os.getcwd(), '..', '..', 'Data', 'Voting', 'dutch_voting_db.pkl'))
    data_voting = data_voting.drop(
        columns=['OngeldigeStemmen', 'BlancoStemmen', 'GeldigeStemmen', 'Region code', 'Neighbourhoods (nr.)',
                 'Kiesgerechtigden', 'Opkomst'])
    data_voting = data_voting.rename(columns=lambda x: re.sub(r'\W+', '', x))
    data_voting = data_voting.rename(columns=lambda x: re.sub(r'^(\d+)(.*)$', r'\2\1', x))
    data_voting = data_voting.reset_index(drop=True)
    # Convert the DataFrame columns into a single 'target' column with arrays
    targets = data_voting.columns[-37:]
    features = data_voting.columns[:-37]

    targets_l = list(targets)

    data_voting['target'] = data_voting[targets_l].values.tolist()
    data_voting = data_voting.drop(columns=targets)

    data_voting = data_voting.reset_index(drop=True)

    path_post = 'euclidean_distance_matrix_voting.npy'
    path = os.path.join(os.getcwd(), '..', '..', 'Data', 'Voting', path_post)

    if not os.path.exists(path):
        distance_matrix = calculate_distance_matrix(data_voting, distance_function=euclidean)
        np.save(path, distance_matrix)
    else:
        distance_matrix = np.load(path)

    return data_voting, 'target', distance_matrix


def get_student_mat():

    nr_targets = 3

    file_path = os.path.join(os.getcwd(), '..', '..', 'Data', 'students', 'student-mat.csv')
    file_path = os.path.abspath(file_path)
    data_student_math = pd.read_csv(file_path, sep=';')  # ,header=True)

    columns_to_transform = data_student_math.columns[-nr_targets:]

    data_student_math[columns_to_transform] = (data_student_math[columns_to_transform]
                                      .apply(lambda x: (x - x.min()) / (x.max() - x.min())))

    data_student_math['target'] = data_student_math[columns_to_transform].values.tolist()
    data_student_math = data_student_math.drop(columns=columns_to_transform)

    data_student_math = data_student_math.reset_index(drop=True)

    path_post = 'euclidean_distance_matrix_student-mat.npy'
    path = os.path.join(os.getcwd(), '..', '..', 'Data', 'students', path_post)

    if not os.path.exists(path):
        distance_matrix = calculate_distance_matrix(data_student_math, distance_function=euclidean)
        np.save(path, distance_matrix)
    else:
        distance_matrix = np.load(path)

    return data_student_math, 'target', distance_matrix


def get_student_por():

    nr_targets = 3

    file_path = os.path.join(os.getcwd(), '..', '..', 'Data', 'students', 'student-por.csv')
    file_path = os.path.abspath(file_path)
    data_student_por = pd.read_csv(file_path, sep=';')  # ,header=True)

    columns_to_transform = data_student_por.columns[-nr_targets:]

    data_student_por[columns_to_transform] = (data_student_por[columns_to_transform]
                                      .apply(lambda x: (x - x.min()) / (x.max() - x.min())))

    data_student_por['target'] = data_student_por[columns_to_transform].values.tolist()
    data_student_por = data_student_por.drop(columns=columns_to_transform)

    data_student_por = data_student_por.reset_index(drop=True)

    path_post = 'euclidean_distance_matrix_student-por.npy'
    path = os.path.join(os.getcwd(), '..', '..', 'Data', 'students', path_post)

    if not os.path.exists(path):
        distance_matrix = calculate_distance_matrix(data_student_por, distance_function=euclidean)
        np.save(path, distance_matrix)
    else:
        distance_matrix = np.load(path)

    return data_student_por, 'target', distance_matrix


def get_data_countries(country_selections=[['Netherlands'], ['France'], ['Germany'], ['India'],
                                           ['Australia'], ['South Korea'],['Indonesia'], ['Brazil'],
                                           'large_economies']):

    if country_selections == 'countries_only':
        country_selections = [['Netherlands'], ['France'], ['Germany'], ['India'],
         ['Australia'], ['South Korea'], ['Indonesia'], ['Brazil']]

    elif country_selections == 'small_countries_only':
        country_selections = [['Netherlands'], ['Australia'], ['South Korea']]

    path_post_data = 'stock_datasets_' + str(country_selections) + '.pkl'
    path_data = os.path.join(os.getcwd(), '..', '..', 'Data', 'Stock', path_post_data)

    if os.path.exists(path_data):

        with open(path_data, 'rb') as f:
            datasets = pickle.load(f)

    else:
        datasets = {}

        for country_selection in country_selections:

            df_original, cat, num, features = getData("Stock5YSmall", countries=country_selection, attributes='expertBased')
            df = df_original.copy()
            df = df.reset_index(drop=True)

            nan_mask = df['target'].apply(lambda x: all(np.isnan(i) for i in x))
            same_values_mask = df['target'].apply(lambda x: len(set(x)) == 1)
            exclude_mask = np.logical_or(nan_mask, same_values_mask)

            df['target'] = df['target'][~exclude_mask]
            df['target'] = df['target'].apply(percent_change_norm)

            if country_selection != 'large_economies':

                path_post = 'euclidean_slope_distance_matrix_' + str(country_selection) + '.npy'
                path = os.path.join(os.getcwd(), '..', '..', 'Data', 'Stock', 'distance_matrices', path_post)

                if not os.path.exists(path):
                    euclidean_slope_distance_matrix = calculate_distance_matrix(df,
                                                                                
                                                                                distance_function=euclidean_distance_slopes)
                    np.save(path, euclidean_slope_distance_matrix)
                else:
                    euclidean_slope_distance_matrix = np.load(path)

            else:
                path = os.path.join(os.getcwd(), '..', '..', 'Data', 'Stock', 'distance_matrices')
                euclidean_slope_distance_matrix = np.load(path + '/euclidean_slope_distance_matrix.npy')

            df = df.reset_index(drop=True)

            datasets[str(country_selection)] = (df, 'target', euclidean_slope_distance_matrix)

        if not os.path.exists(path_data):
            with open(path_data, 'wb') as f:
                pickle.dump(datasets, f)

    return datasets


def get_data(data_selection='all'):

    """
    data_selection = 'all' -> gives all data
    data_selection = 'benchmark' -> gives benchmark data only, not the (memory expensive) stock data
    """

    data = {
        'countries': get_data_countries(),
        'wisconsin': get_data_wisconsin(),
        'cpu': get_data_cpu(),
        'voting': get_data_voting(),
        'student_math': get_student_mat(),
        'student_por': get_student_por(),
        'elevators': get_data_elevators(),
        'ecoli': get_data_ecoli(),
        'wine': get_data_wine(),
        'sushi': get_data_sushi()}

    if data_selection == 'all':
        data = handle_stock_dataset_for_countries(data)

        return data

    elif data_selection == 'benchmark':

        data_exclusion = ['countries']
        data = {key: value for key, value in data.items() if key not in data_exclusion}

        return data

    elif data_selection == 'benchmark_small':

        data_exclusion = ['countries', 'elevators', 'sushi', 'voting']
        data = {key: value for key, value in data.items() if key not in data_exclusion}

        return data

    elif data_selection == 'feasible':

        data_exclusion = ['sushi']
        data = {key: value for key, value in data.items() if key not in data_exclusion}
        data = handle_stock_dataset_for_countries(data)

        return data

    elif data_selection == 'stocks_all':

        data = {'countries': get_data_countries()}

        data = handle_stock_dataset_for_countries(data)

        return data

    elif data_selection == 'countries_only':

        data = {'countries': get_data_countries('countries_only')}

        data = handle_stock_dataset_for_countries(data)

        return data

    elif data_selection == 'small_countries_only':
        data = {'countries': get_data_countries('small_countries_only')}
        data = handle_stock_dataset_for_countries(data)
        return data

    else:

        data = {key: value for key, value in data.items() if key in data_selection}
        data = handle_stock_dataset_for_countries(data)

        return data


def handle_stock_dataset_for_countries(datasets):
    dict_final = datasets.copy()

    for data_set_key in datasets.keys():

        if data_set_key == 'countries':

            for dataset_key_country in datasets['countries'].keys():
                try:
                    key = ast.literal_eval(dataset_key_country)[0]
                except:
                    key = dataset_key_country
                data, target, matrix = datasets['countries'][dataset_key_country]
                dict_final[key] = (data, target, matrix)
            del dict_final['countries']

    return dict_final
