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
    funel['region_cd'] = funel['region_cd'].fillna(0)
    funel['region_cd'] = funel['region_cd'].astype(int)

    funel['feature_7'] = funel['feature_7'].fillna(3642369.0)
    funel['feature_10div7'] = funel.feature_10 / funel.feature_7
    funel['feature_7'] = np.sqrt(funel['feature_7'])

    funel['feature_8'] = funel['feature_8'].fillna(3314257.0)

    funel['feature_9'] = funel['feature_9'].fillna(7.9)
    funel['feature_9'] = (funel['feature_9'] * 10).astype(int)

    funel['feature_10'] = funel['feature_10'].fillna(48149.0)
    funel['feature_10'] = np.log1p(funel['feature_10'])

    funel['freq_client_segment'] = funel.groupby('client_segment')['client_segment'].transform('count')
    funel['freq_feature_1'] = funel.groupby('feature_1')['feature_1'].transform('count')
    funel['freq_feature_9'] = funel.groupby('feature_9')['feature_9'].transform('count')
    funel['feature_1_client_segment'] = funel['feature_1'].astype(str) + '_' + funel['client_segment'].astype(str)
    funel['freq_feature_1_client_segment'] = funel.groupby('feature_1_client_segment')['feature_1_client_segment'].transform('count')
    return funel


def get_client_features(funel, client) -> pd.DataFrame:
    client['gender'] = client['gender'].fillna('M').map(lambda x: int(hashlib.sha1(x.encode("utf-8")).hexdigest(), 16) % (10 ** 8))
    client['age'] = client['age'].fillna(client['age'].median()).astype(int)
    client['citizenship'] = client['citizenship'].fillna('RUSSIA').map(lambda x: int(hashlib.sha1(x.encode("utf-8")).hexdigest(), 16) % (10 ** 8))
    client['education'] = client['education'].fillna('HIGHER_PROFESSIONAL').map(lambda x: int(hashlib.sha1(x.encode("utf-8")).hexdigest(), 16) % (10 ** 8))
    client['job_type'] = client['job_type'].fillna("USELESS_JOB").map(lambda x: int(hashlib.sha1(x.encode("utf-8")).hexdigest(), 16) % (10 ** 8))
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
