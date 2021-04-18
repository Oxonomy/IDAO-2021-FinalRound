import gc


import pandas as pd
from utils.du import sleep
import config as c
from utils.features import *
from utils.load_data import *
from utils.data_processing import *



def get_dataset(is_return_all_table=False, do_features=True) -> pd.DataFrame:
    """
    Загружает все таблицы и генерирует из них фичи
    :return: итоговая таблица
    """
    df_funnel = load_funnel()
    df_trxn = load_trxn()
    df_dict_mcc = load_dict_mcc()
    if do_features:
        df_funnel = get_transactions_features(df_funnel, df_trxn, df_dict_mcc)
        df_funnel = get_trxn_features(df_funnel, df_trxn)

    del df_dict_mcc
    del df_trxn
    gc.collect()
    # Загрузка и предобработка таблиц
    #sleep(100)

    df_appl = load_appl()
    df_aum = load_aum()
    df_balance = load_balance()
    df_client = load_client()
    df_com = load_com()
    df_deals = load_deals()
    df_payments = load_payments()
    # df_trxn = load_trxn()

    # Генерация фич
    if do_features:
        df_funnel = get_month_payments(df_funnel, df_payments)
        df_funnel = get_funnel_features(df_funnel)
        df_funnel = get_aum_features(df_funnel, df_aum)
        df_funnel = get_appl_features(df_funnel, df_appl)
        df_funnel = get_client_features(df_funnel, df_client)
        # df_funnel = get_transactions_features(df_funnel, df_trxn, df_dict_mcc)
        df_funnel = get_comm_features(df_funnel, df_com)
        df_funnel = get_pensioner(df_funnel, df_payments)
        df_funnel = get_deals_features(df_funnel, df_deals)
        df_funnel = string_columns_to_int(df_funnel)


    if not is_return_all_table:
        del df_payments
        del df_balance
        gc.collect()

        return df_funnel
    else:
        return df_funnel, df_appl, df_aum, df_balance, df_client, df_com, df_deals, df_payments



if __name__ == '__main__':
    print(get_dataset().head())
