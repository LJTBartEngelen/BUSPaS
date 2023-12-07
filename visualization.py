import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def bresenham_line(x1, y1, x2, y2):
    # calculate the differences between the x and y coordinates
    dx = x2 - x1
    dy = y2 - y1

    # determine the sign of the slope
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1

    # calculate the absolute differences between the x and y coordinates
    dx = abs(dx)
    dy = abs(dy)

    # initialize the variables for the loop
    x = x1
    y = y1

    err = dx - dy

    # loop until the end of the line is reached

    while True:

        # yield the current x and y coordinates
        yield (x, y)

        # break the loop if the end of the line has been reached
        if x == x2 and y == y2:
            break
        # calculate the next error value
        e2 = 2 * err
        # check if it's time to move in the y direction
        if e2 > -dy:
            err = err - dy
            x = x + sx
        # check if it's time to move in the x direction
        if e2 < dx:
            err = err + dx
            y = y + sy


def TS_heatmap(df, ts_att='target', resolution_y=200, normalize='y', resolution_mult_x=5, min_val_occurrence=10):
    """"
    plots a heatmap from a dataframe of lists that represent time-series

    df = dataframe of population or subgroup
    ts_att = target attribute. column with ts in shape of lists.
    resolution_y = number of bins that the prices are subdivided in
    normalize = 'y'/'n' normalize the image in columns of later generated matrix. Especially usefull if most frequent
                prices at a certain moment are needed.
    resolution_mult_x = stretches the matrix
    min_val_occurrence = starting value when a bin in y direction is hit at least once
    """

    df = pd.DataFrame(df[ts_att].to_list()).transpose()

    min_y = df.min().min() - (df.max().max() - df.min().min()) * 0.01
    max_y = df.max().max() + (df.max().max() - df.min().min()) * 0.01

    matrix = np.zeros((resolution_y, len(df)))

    col_new = [i for i in range(len(df.columns))]
    df.columns = col_new

    for ts in range(len(df.columns)):
        for tp in range(len(df)):
            try:
                val = round(((df[ts][tp]) - min_y) / (max_y - min_y) * resolution_y)
                if tp < len(df) - 1:
                    val_next = round(((df[ts][tp + 1]) - min_y) / (max_y - min_y) * resolution_y)
                    if val_next != val:
                        between_vals = [*range(min([val, val_next]), max([val, val_next]), 1)]
                    else:
                        between_vals = [val]
                else:
                    between_vals = [val]
                for val_use in between_vals:
                    y = val_use
                    if matrix[y][tp] == 0:
                        matrix[y][tp] = min_val_occurrence
                    else:
                        matrix[y][tp] += 1
            except:
                pass

    if normalize == 'y':
        matrix = matrix / matrix.max(axis=0)

    matrix = np.repeat(matrix, resolution_mult_x, axis=1)

    plt.figure(figsize=(13, 13), dpi=80)
    plt.imshow(matrix, cmap='hot', interpolation='bilinear', origin='lower')

    plt.show()


def TS_heatmap_Bresenham_additive(df, ts_att='target', resolution_y=200, normalize='y', resolution_mult_x=5,
                                  min_val_occurrence=None, title=None, subgroup_only=False):
    """"
    plots a heatmap from a dataframe of lists that represent time-series. Uses Bresenham algorithm to draw lines
        between two points. Preferred over TS_heatmap when trend between two points is interesting.

    df = dataframe of population or subgroup
    ts_att = target attribute. column with ts in shape of lists.
    resolution_y = number of bins that the prices are subdivided in
    normalize = 'y'/'n' normalize the image in columns of later generated matrix. Especially usefull if most frequent
                prices at a certain moment are needed.
    resolution_mult_x = stretches the matrix
    min_val_occurrence = starting value when a bin in y direction is hit at least once
    title = title in the plot

    """

    df = pd.DataFrame(df[ts_att].to_list()).transpose()

    # sets boundaries for binning prices

    min_y = df.min().min() - (df.max().max() - df.min().min()) * 0.01
    max_y = df.max().max() + (df.max().max() - df.min().min()) * 0.01

    if min_val_occurrence == None:
        min_val_occurrence = len(df) * 0.05

    matrix = np.zeros((resolution_y, len(df) * resolution_mult_x - resolution_mult_x))

    col_new = [i for i in range(len(df.columns))]
    df.columns = col_new

    # for each time-series and time-point counts the frequency of one place in the matrix being overstepped

    if subgroup_only is not False:
        df = subgroup_only

    for ts in range(len(df.columns)):

        temp_matrix = np.zeros((resolution_y, len(df) * resolution_mult_x - resolution_mult_x))

        for tp in range(len(df)):

            try:
                val = round(((df[ts][tp]) - min_y) / (max_y - min_y) * resolution_y)

                if tp < len(df) - 1:
                    val_next = round(((df[ts][tp + 1]) - min_y) / (max_y - min_y) * resolution_y)

                    # uses bresenham algorithm to retrieve coordinates for points between two time+price combinations
                    coordinates = []
                    for x, y in bresenham_line(tp * resolution_mult_x, val, (tp + 1) * resolution_mult_x, val_next):
                        coordinates.append((x, y))

                        try:

                            if temp_matrix[y][x] == 0:
                                temp_matrix[y][x] = min_val_occurrence
                            else:
                                temp_matrix[y][x] += 1
                            temp_matrix[y][x] -= 1
                        except:
                            pass
            except:
                pass
        temp_matrix = [[1 if value else 0 for value in row] for row in temp_matrix]
        matrix = temp_matrix + matrix
    if min_val_occurrence > 1:
        # matrix = np.where(matrix < min_val_occurrence, min_val_occurrence, matrix)
        matrix = [[min_val_occurrence if 0 < value < min_val_occurrence else value for value in row] for row in matrix]
        # matrix = np.where(matrix > min_val_occurrence+2, matrix-1, matrix)
        # matrix = [[1 if value else 0 for value in row] for row in matrix]
    else:
        pass

    # normalizes matrix in column
    if normalize == 'y':
        matrix = np.array(matrix)
        matrix = matrix / matrix.max(axis=0)

    # prints matrix as heatmap
    plt.figure(figsize=(14, 8), dpi=80)
    im = plt.imshow(matrix, cmap='Reds', origin='lower', aspect='auto')  # Reds, Greys, hot

    plt.title(title)  # Add title
    plt.xlabel('Time')  # Add x-label
    plt.ylabel('Percentual Change')  # Add y-label

    # Set x axis range and labels
    x_ticks = np.linspace(0, len(df) * resolution_mult_x - resolution_mult_x - 1, 10)
    x_labels = np.round(np.linspace(0, len(df), 10), decimals=1)
    plt.xticks(x_ticks, x_labels)

    # Set y axis range and labels
    y_ticks = np.linspace(0, resolution_y - 1, 10)  # Show only 10 equally distributed values
    y_labels = np.round(np.linspace(min_y, max_y, 10), decimals=1)  # Round the labels to 2 decimal places
    plt.yticks(y_ticks, y_labels)

    cbar = plt.colorbar(im, fraction=0.025)  # Set colorbar fraction to 0.05
    cbar.set_label("Relative frequency", rotation=270, labelpad=5)

    plt.show()

def plot_ts(df, col, title_plot='   ', title_y_axis='Percentual change stock price',
            title_x_axis='Time', y_min=None, y_max=None):

    s = pd.Series(df[col])

    # Set figure size and dpi
    plt.figure(figsize=(14, 8), dpi=80)

    # Plot each list as a line
    for row in s:
        plt.plot(row)

    # Set the x-axis and y-axis labels
    plt.xlabel(title_x_axis)
    plt.ylabel(title_y_axis)

    # Set the title
    plt.title(title_plot)

    # Set the y-axis limits if provided
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)

    # Display the plot
    plt.show()

# def TS_heatmap_Bresenham_additive(df, ts_att='target', resolution_y=200, normalize='y', resolution_mult_x=5,
#                                   min_val_occurrence=None, title=None, subgroup_only=False):
#     """"
#     plots a heatmap from a dataframe of lists that represent time-series. Uses Bresenham algorithm to draw lines
#         between two points. Preferred over TS_heatmap when trend between two points is interesting.
#
#     df = dataframe of population or subgroup
#     ts_att = target attribute. column with ts in shape of lists.
#     resolution_y = number of bins that the prices are subdivided in
#     normalize = 'y'/'n' normalize the image in columns of later generated matrix. Especially usefull if most frequent
#                 prices at a certain moment are needed.
#     resolution_mult_x = stretches the matrix
#     min_val_occurrence = starting value when a bin in y direction is hit at least once
#     title = title in the plot
#
#     """
#
#     df = pd.DataFrame(df[ts_att].to_list()).transpose()
#
#     # sets boundaries for binning prices
#
#     min_y = df.min().min() - (df.max().max() - df.min().min()) * 0.01
#     max_y = df.max().max() + (df.max().max() - df.min().min()) * 0.01
#
#     if min_val_occurrence == None:
#         min_val_occurrence = len(df) * 0.05
#
#     matrix = np.zeros((resolution_y, len(df) * resolution_mult_x - resolution_mult_x))
#
#     col_new = [i for i in range(len(df.columns))]
#     df.columns = col_new
#
#     # for each time-series and time-point counts the frequency of one place in the matrix being overstepped
#
#     if subgroup_only is not False:
#         df = subgroup_only
#
#     for ts in range(len(df.columns)):
#
#         temp_matrix = np.zeros((resolution_y, len(df) * resolution_mult_x - resolution_mult_x))
#
#         for tp in range(len(df)):
#
#             try:
#                 val = round(((df[ts][tp]) - min_y) / (max_y - min_y) * resolution_y)
#
#                 if tp < len(df) - 1:
#                     val_next = round(((df[ts][tp + 1]) - min_y) / (max_y - min_y) * resolution_y)
#
#                     # uses bresenham algorithm to retrieve coordinates for points between two time+price combinations
#                     coordinates = []
#                     for x, y in bresenham_line(tp * resolution_mult_x, val, (tp + 1) * resolution_mult_x, val_next):
#                         coordinates.append((x, y))
#
#                         if temp_matrix[y][x] == 0:
#                             temp_matrix[y][x] = min_val_occurrence
#                         else:
#                             temp_matrix[y][x] += 1
#                     temp_matrix[y][x] -= 1
#             except:
#                 pass
#         temp_matrix = [[1 if value else 0 for value in row] for row in temp_matrix]
#         matrix = temp_matrix + matrix
#
#     # normalizes matrix in column
#     if normalize == 'y':
#         matrix = matrix / matrix.max(axis=0)
#
#     # prints matrix as heatmap
#     plt.figure(figsize=(14, 8), dpi=80)
#     im = plt.imshow(matrix, cmap='hot', origin='lower', aspect='auto')
#
#     plt.title(title)  # Add title
#
#     # Set x axis range and labels
#     x_ticks = np.linspace(0, len(df) * resolution_mult_x - resolution_mult_x - 1, 10)
#     x_labels = np.round(np.linspace(0, len(df), 10), decimals=1)
#     plt.xticks(x_ticks, x_labels)
#
#     # Set y axis range and labels
#     y_ticks = np.linspace(0, resolution_y - 1, 10)  # Show only 10 equally distributed values
#     y_labels = np.round(np.linspace(min_y, max_y, 10), decimals=1)  # Round the labels to 2 decimal places
#     plt.yticks(y_ticks, y_labels)
#
#     cbar = plt.colorbar(im, fraction=0.025)  # Set colorbar fraction to 0.05
#     cbar.set_label("Relative frequency", rotation=270, labelpad=5)
#
#     plt.show()