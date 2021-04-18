import hashlib
import os
import numpy as np
import pandas as pd

import config as c
from utils.data_processing import *
from .features_transactions import pipeline


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


def get_funnel_features(funel) -> pd.DataFrame:
    funel['client_segment'] = funel['client_segment'].astype(int)
    funel['region_cd'] = funel['region_cd'].fillna(-1).astype(int)

    funel['feature_7'] = funel['feature_7'].fillna(-1)
    funel['feature_10div7'] = funel.feature_10 / funel.feature_7
    funel['feature_8'] = funel['feature_8'].fillna(-1)
    funel['feature_9'] = funel['feature_9'].fillna(-1)
    funel['feature_10'] = funel['feature_10'].fillna(-1)

    funel['freq_client_segment'] = funel.groupby('client_segment')['client_segment'].transform('count')
    funel['freq_feature_1'] = funel.groupby('feature_1')['feature_1'].transform('count')
    funel['freq_feature_9'] = funel.groupby('feature_9')['feature_9'].transform('count')
    funel['feature_1_client_segment'] = funel['feature_1'].astype(str) + '_' + funel['client_segment'].astype(str)
    funel['freq_feature_1_client_segment'] = funel.groupby('feature_1_client_segment')['feature_1_client_segment'].transform('count')
    return funel

def get_aum_features(funnel, aum):
    funnel = funnel.set_index('client_id')
    gb = aum.groupby('client_id')['balance_rur_amt']
    mean_balance = gb.mean()
    std_balance = gb.std()

    funnel['mean_balance'] = 0
    funnel['std_balance'] = 0
    funnel.loc[mean_balance.index, 'mean_balance'] = mean_balance
    funnel.loc[std_balance.index, 'std_balance'] = std_balance

    return funnel.reset_index()

def get_appl_features(funnel, appl):
    funnel = funnel.set_index('client_id')

    total_count = appl.groupby('client_id')['appl_prod_type_name'].count()
    funnel['appl_total_count'] = 0
    funnel.loc[total_count.index, 'appl_total_count'] = total_count

    for feature in ('appl_prod_group_name', 'appl_stts_name_dc', 'appl_sale_channel_name'):
        count_helper = appl.groupby('client_id')[feature].value_counts().rename('count').reset_index()
        top = count_helper.groupby('client_id').head(1).set_index('client_id')

        top_value = top[feature]
        funnel[f'top_value_{feature}'] = 0
        funnel.loc[top_value.index, f'top_value_{feature}'] = top_value

        top_rate = top['count'] / count_helper.groupby('client_id').sum()['count']
        funnel[f'top_rate_{feature}'] = 0
        funnel.loc[top_rate.index, f'top_rate_{feature}'] = top_rate

    appl['month_end'] = appl['month_end_dt'].str[5:7].astype(int).apply(lambda m: m if m < 9 else m - 12)
    last_month = appl.groupby('client_id')['month_end'].max()
    funnel['appl_last_month'] = 0
    funnel.loc[last_month.index, 'appl_last_month'] = last_month

    return funnel.reset_index()


def get_client_features(funel, client) -> pd.DataFrame:
    client['gender'] = client['gender'].fillna('UNK')
    client['age'] = client['age'].fillna(client['age'].median()).astype(int)
    client['education'] = client['education'].fillna('UNK')
    client['job_type'] = client['job_type'].fillna("UNK")
    client = client[['client_id', 'gender', 'age', 'education', 'job_type', 'region', 'city']]

    funel = funel.merge(client, on='client_id', how='left')
    return funel


def get_comm_features(df_funnel, df_com) -> pd.DataFrame:
    ring_up_flg_sum = df_com.groupby('client_id')[['ring_up_flg']].sum()
    ring_up_flg_sum = ring_up_flg_sum.rename(columns={'ring_up_flg': 'ring_up_flg_sum'})
    df_funnel = pd.concat([df_funnel.set_index('client_id'), ring_up_flg_sum], axis=1).reset_index()

    return df_funnel


def get_pensioner(df_funnel, df_payments):
    df_payments = df_payments.copy()[['client_id', 'pmnts_name']]
    df_payments.pmnts_name = (df_payments.pmnts_name == 'Pension receipts').astype(int)
    df_payments = df_payments.rename(columns={'pmnts_name': 'is_pensioner'})
    df_payments = df_payments.groupby('client_id').max()
    df_payments.fillna(0)

    df_funnel = pd.concat([df_funnel.set_index('client_id'), df_payments], axis=1).reset_index()
    return df_funnel


def get_transactions_features(df_funnel, df_trxn, df_dict_mcc):
    client_info = pipeline(df_trxn, df_dict_mcc)
    client_info = client_info.reset_index().rename({'index': 'client_id'}, axis=1)
    df_funnel = pd.merge(df_funnel, client_info, on='client_id', how='left')

    return df_funnel


def get_deals_features(df_funnel, df_deals):
    salary_cards_agrmnt_sum_rur_mean = df_deals[df_deals.prod_type_name == 'Salary cards'].groupby('client_id').mean()[['agrmnt_sum_rur']]
    salary_cards_agrmnt_sum_rur_mean = salary_cards_agrmnt_sum_rur_mean.rename(columns={'agrmnt_sum_rur': 'salary_cards_agrmnt_sum_rur_mean'})
    df_funnel = pd.concat([df_funnel.set_index('client_id'), salary_cards_agrmnt_sum_rur_mean], axis=1).reset_index()

    salary_cards_agrmnt_sum_rur_sum = df_deals[df_deals.prod_type_name == 'Salary cards'].groupby('client_id').sum()[['agrmnt_sum_rur']]
    salary_cards_agrmnt_sum_rur_sum = salary_cards_agrmnt_sum_rur_sum.rename(columns={'agrmnt_sum_rur': 'salary_cards_agrmnt_sum_rur_sum'})
    df_funnel = pd.concat([df_funnel.set_index('client_id'), salary_cards_agrmnt_sum_rur_sum], axis=1).reset_index()

    credit_cards_agrmnt_rate_active = df_deals[df_deals.prod_type_name == 'Credit cards'].groupby('client_id').mean()[['agrmnt_rate_active']]
    credit_cards_agrmnt_rate_active = credit_cards_agrmnt_rate_active.rename(columns={'agrmnt_rate_active': 'credit_cards_agrmnt_rate_active'})
    df_funnel = pd.concat([df_funnel.set_index('client_id'), credit_cards_agrmnt_rate_active], axis=1).reset_index()

    credit_cards_agrmnt_sum_rur = df_deals[df_deals.prod_type_name == 'Credit cards'].groupby('client_id').mean()[['agrmnt_sum_rur']]
    credit_cards_agrmnt_sum_rur = credit_cards_agrmnt_sum_rur.rename(columns={'agrmnt_sum_rur': 'credit_cards_agrmnt_sum_rur'})
    df_funnel = pd.concat([df_funnel.set_index('client_id'), credit_cards_agrmnt_sum_rur], axis=1).reset_index()

    return df_funnel
