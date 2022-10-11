import numpy as np


def get_num_cols(df, col_to_plot):
    if col_to_plot == "default":
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        if not isinstance(col_to_plot, list):
            num_cols = [col_to_plot]
        else:
            num_cols = col_to_plot
    try:
        columns = [n for n in range(1, 10) if len(
            num_cols) % n == 0][1]  # columns for plot
    except:
        columns = 1
    return num_cols, columns


def get_cat_cols(df, col_to_plot):
    if col_to_plot == "default":
        cat_cols = df.select_dtypes(
            include=['category', 'object']).columns.tolist()  # columns in df
    else:
        if not isinstance(col_to_plot, list):
            cat_cols = [col_to_plot]
        else:
            cat_cols = col_to_plot
    try:
        columns = [n for n in range(1, 10) if len(
            cat_cols) % n == 0][1]  # columns for plot
    except:
        columns = 1
    return cat_cols, columns
