import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse


def haversine_dist(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    radius = 6371  # Earth's radius taken from google
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng / 2) ** 2
    h = 2 * radius * np.arcsin(np.sqrt(d))
    return h


def moscow_dist(lat1, lng1, lat2, lng2):
    """
    calculating two haversine distances by,
     - avoiding Latitude of one point
     - avoiding Longitude of one point
    and adding it together.
    """
    a = haversine_dist(lat1, lng1, lat1, lng2)
    b = haversine_dist(lat1, lng1, lat2, lng1)
    return a + b


def check_pass_obj(df):
    if isinstance(df, pd.DataFrame):
        return True
    else:
        return False


def change_missing_values_non_numeric_by_const(df, cols_to_fill):
    if check_pass_obj(df):
        for col in cols_to_fill:
            df[col] = df[col].fillna('NA')
    else:
        raise TypeError("Data must be a Pandas DataFrame")


def change_missing_values_numeric_by_const(df, cols_to_fill):
    if check_pass_obj(df):
        for col in cols_to_fill:
            df[col] = df[col].fillna(0)
    else:
        raise TypeError("Data must be a Pandas DataFrame")


def fill_nan_by_default(train):
    lst_numeric_for_fill = []
    lst_non_numeric_for_fill = []

    lst_input = list(train.columns)
    numeric_cols = train.select_dtypes(include=['number']).columns
    non_numeric_cols = train.select_dtypes(exclude=['number']).columns

    for col in lst_input:
        if col in non_numeric_cols:
            lst_non_numeric_for_fill.append(col)
        if col in numeric_cols:
            lst_numeric_for_fill.append(col)
        if len(lst_non_numeric_for_fill) > 0:
            change_missing_values_non_numeric_by_const(train, lst_non_numeric_for_fill)
            change_missing_values_non_numeric_by_const(train, lst_non_numeric_for_fill)
        if len(lst_numeric_for_fill) > 0:
            change_missing_values_numeric_by_const(train, lst_numeric_for_fill)
            change_missing_values_numeric_by_const(train, lst_numeric_for_fill)
    return train


def int8_type(number):
    if -128 <= number <= 127:
        return True
    return False


def int16_type(number):
    if -32768 <= number <= 32767:
        return True
    return False


def int32_type(number):
    if -2.1 * 10 ** 9 <= number <= 2.1 * 10 ** 9:
        return True
    return False


def int64_type(number):
    if -9.2 * 10 ** 18 <= number <= 9.2 * 10 ** 18:
        return True
    return False


def cast_numeric_column(df_col):
    max_val = df_col.max()
    min_val = df_col.min()
    if int8_type(min_val) and int8_type(max_val):
        return df_col.astype(np.int8)
    if int16_type(min_val) and int16_type(max_val):
        return df_col.astype(np.int16)
    if int32_type(min_val) and int32_type(max_val):
        return df_col.astype(np.int32)
    if int64_type(min_val) and int64_type(max_val):
        return df_col.astype(np.int64)


def get_pos(lat, lng):
    return lat, lng


def level_floor(floor):
    if floor <= 3:
        return 0
    elif floor <= 6:
        return 1
    else:
        return 2


def accuracy_report(y_true, y_pred, make_plot=True):
    print('MSE: {:.3f}'.format(mse(y_true, y_pred)))
    print('RMSE: {:.3f}'.format(np.sqrt(mse(y_true, y_pred))))
    print('MAE: {:.3f}'.format(mae(y_true, y_pred)))
    if make_plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred)
        plt.xlabel('Reality')
        plt.ylabel('Prediction')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')

