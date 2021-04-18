import os
import numpy as np
import pandas as pd

import config as c
from utils.features import *
from utils.load_data import *
from utils.data_processing import *


def get_dataset(is_return_all_table=False) -> pd.DataFrame:
    """
    Загружает все таблицы и генерирует из них фичи
    :return: итоговая таблица
    """
    # Загрузка и предобработка таблиц
    df_funnel = load_funnel()
    df_appl = load_appl()
    df_aum = load_aum()
    df_balance = load_balance()
    df_client = load_client()
    df_com = load_com()
    df_deals = load_deals()
    df_dict_mcc = load_dict_mcc()
    df_payments = load_payments()
    df_trxn = load_trxn()

    # Генерация фич
    # df_funnel = get_mean_month_costs(df_funnel, df_trxn)
    df_funnel = get_month_payments(df_funnel, df_payments)
    df_funnel = get_funnel_features(df_funnel)
    df_funnel = get_client_features(df_funnel, df_client)
    df_funnel = get_transactions_features(df_funnel, df_trxn, df_dict_mcc)

    if not is_return_all_table:
        return df_funnel
    else:
        return df_funnel, df_appl, df_aum, df_balance, df_client, df_com, df_deals, df_payments


if __name__ == '__main__':
    print(get_dataset().head())
