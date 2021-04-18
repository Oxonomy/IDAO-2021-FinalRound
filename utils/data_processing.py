import hashlib
import os
import fnmatch
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

import config as c


def sort_data_by_datetime(df: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
    df.tran_time = pd.to_datetime(df[datetime_column])
    df = df.sort_values(by=[datetime_column])
    return df


def funnel_preprocessing(df: pd.DataFrame):
    return df


def appl_preprocessing(df: pd.DataFrame):
    df = sort_data_by_datetime(df, 'month_end_dt')
    return df


def aum_preprocessing(df: pd.DataFrame):
    df = sort_data_by_datetime(df, 'month_end_dt')
    return df


def balance_preprocessing(df: pd.DataFrame):
    df = df.drop_duplicates()
    df.prod_cat_name = df.prod_cat_name.fillna('unknown')
    df.prod_group_name = df.prod_group_name.fillna('unknown')

    df.prod_cat_name = df.prod_cat_name.astype("category")
    df.prod_group_name = df.prod_group_name.astype("category")

    df = sort_data_by_datetime(df, 'month_end_dt')
    return df


def client_preprocessing(df: pd.DataFrame):
    return df


def com_preprocessing(df: pd.DataFrame):
    df = sort_data_by_datetime(df, 'month_end_dt')
    df.month_end_dt = pd.to_datetime(df.month_end_dt)
    return df


def deals_preprocessing(df: pd.DataFrame):
    df = sort_data_by_datetime(df, 'agrmnt_start_dt')
    return df


def dict_mcc_preprocessing(df: pd.DataFrame):
    return df


def payments_preprocessing(df: pd.DataFrame):
    df = sort_data_by_datetime(df, 'day_dt')
    return df


def trxn_preprocessing(df: pd.DataFrame):
    df['txn_city'] = df['txn_city'].fillna('unknown')
    df.txn_city = df.txn_city.astype("category")
    df.txn_country = df.txn_country.astype("category")
    #df.txn_comment_1 = df.txn_comment_1.astype("category")

    #df.tran_amt_rur = df.tran_amt_rur.astype('float16')

    df = sort_data_by_datetime(df, 'tran_time')
    return df


def string_columns_to_int(df: pd.DataFrame):
    categorical_columns = [c for c in df.columns if (df[c].dtype.name == 'object') or (df[c].dtype.name == 'object')]
    print(categorical_columns)
    for column in categorical_columns:
        df[column] = df[column].map(lambda x: int(hashlib.sha1(str(x).encode("utf-8")).hexdigest(), 16) % (10 ** 8))
    return df
