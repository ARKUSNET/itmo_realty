import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split

import src.utils_function as uf

default_data_frame_path = 'default_data_frame.csv'

model_path = 'rf_fitted.pkl'


def prepare_data():
    train = pd.read_csv("data/realty_data.csv")  # Read data

    # Find and drop duplicates
    train_temp1 = train[train.drop('price', axis=1).duplicated(keep=False)]
    train_temp2 = train_temp1.drop('price', axis=1).drop_duplicates(keep='last')  # Save only last
    mask_duple_tag = np.invert(train_temp1.index.isin(train_temp2.index))  # Get difference index
    train = train.drop(train_temp1.index[mask_duple_tag])

    # Fill nan value for integer - 0 and for object - NA

    train = uf.fill_nan_by_default(train)

    # Change type of column

    numeric_cols = train.select_dtypes(include=['number']).columns
    for col_name in numeric_cols:
        train[col_name] = uf.cast_numeric_column(train[col_name])

    # Create new feature

    train['is_studio'] = [1 if 'Студия' in row.product_name else 0 for row in train.itertuples()]

    train['is_studio'] = train['is_studio'].astype('bool')  # замена типа данных на булев, так как

    train['floor_category'] = train['floor'].apply(uf.level_floor)

    train = train.drop(columns=["city", "settlement", "district", "source", "postcode", "product_name", "period",
                                "address_name", "object_type", "area", "description"])

    # Transform categorical features to binary by use BinaryEncoder

    train['price'] = np.log1p(train['price'])

    make_default_dataframe(train)

    return train


def train_model(train, regression_model):
    X, y = train.drop("price", axis=1), train['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2024, test_size=0.25)

    model_reg = None

    if regression_model == 'Linear':
        model_reg = LinearRegression()
        model_reg.fit(X_train, y_train)

    if regression_model == 'Ridge':
        model_reg = RidgeCV(alphas=(0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10))
        model_reg.fit(X_train, y_train)

    if regression_model == 'XGB':
        model_reg = xgb.XGBRegressor(n_estimators=340, max_depth=2, learning_rate=0.2)
        model_reg.fit(X_train, y_train)

    uf.accuracy_report(y_test, model_reg.predict(X_test))

    with open('rf_fitted.pkl', 'wb') as file:
        pickle.dump(model_reg, file)


def read_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not exists")

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    return model


def make_default_dataframe(train):
    if not os.path.exists(default_data_frame_path):
        df_one_row = train.head(1)
        df_one_row = df_one_row.drop(columns='price')
        numeric_cols = list(df_one_row.select_dtypes(include=['number']).columns)
        non_numeric_cols = list(df_one_row.select_dtypes(exclude=['number']).columns)
        for idx, row in df_one_row.iterrows():
            change_row_value(idx, numeric_cols, non_numeric_cols, df_one_row)
        df_one_row.to_csv('default_data_frame.csv')


def change_row_value(idx, numeric_cols, non_numeric_cols, df_one_row):
    for col in numeric_cols:
        df_one_row.loc[idx, col] = 0
    for col in non_numeric_cols:
        df_one_row.loc[idx, col] = False


def remove_pkl_model():
    if os.path.exists(model_path):
        os.remove(path=model_path)
