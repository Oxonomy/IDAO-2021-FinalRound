import os
import numpy as np
import pandas as pd

import config as c
from utils.data_processing import *


def get_mean_month_costs(df_funnel, df_trxn) -> pd.DataFrame:
    df_trxn['year_month'] = df_trxn.tran_time.dt.month + df_trxn.tran_time.dt.year * 10
    month_costs = df_trxn.groupby(['client_id', 'year_month']).sum()['tran_amt_rur'].reset_index()
    month_costs = month_costs.rename(columns={'tran_amt_rur': 'mean_month_costs'})
    month_costs = month_costs.groupby('client_id').mean()['mean_month_costs'].reset_index()

    df_funnel = pd.concat([df_funnel.set_index('client_id'), month_costs.set_index('client_id')], axis=1).reset_index()
    return df_funnel


def get_month_payments(df_funnel, df_payments) -> pd.DataFrame:
    month_payments = df_payments.groupby('client_id').mean()['sum_rur'].reset_index()
    month_payments = month_payments.rename(columns={'sum_rur': 'month_payments'})
    df_funnel = pd.concat([df_funnel.set_index('client_id'), month_payments.set_index('client_id')], axis=1).reset_index()
    return df_funnel
