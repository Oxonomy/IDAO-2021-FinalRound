import datetime
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


def get_balance_features(funnel, balance):
    funnel = funnel.set_index('client_id')

    feature = 'prod_cat_name'
    feature_values = ['KN', 'CURRENT ACCOUNTS', 'SALARY CARDS', 'DEPOSITS', 'DEBIT CARDS',
                      'CREDIT CARDS', 'MORTGAGE', 'TECHNICAL CARDS', 'CAR LOANS',
                      'CREDITS IN CASH OR', 'Cash on demand', 'CASH CREDITS (X-SALE)']
    features_df = create_cat_feature_rates(balance, feature, feature_values)
    funnel.loc[features_df.index, features_df.columns] = features_df

    feature = 'prod_group_name'
    feature_values = ['Salary cards', 'Cash on demand', 'Debit cards', 'PILS',
                       'Time deposits', 'Credit card other', 'Open_card credit card',
                       'Mortgage', 'Technical cards', 'POS', 'Credit card 120 days',
                       'Car loans', 'Prepaid cards']
    features_df = create_cat_feature_rates(balance, feature, feature_values)
    funnel.loc[features_df.index, features_df.columns] = features_df

    return funnel.reset_index()


def create_cat_feature_rates(balance, feature, feature_values):
    feature2idx = dict(zip(feature_values, range(len(feature_values))))

    gb = balance.groupby('client_id')[feature]
    vc = gb.value_counts().rename('count').reset_index()
    vc['rates'] = vc['count'] / vc.groupby('client_id')['count'].transform('sum')

    top1_vc = vc.groupby('client_id').head(1).set_index('client_id')
    top1_feature = top1_vc[feature].rename(f'top_feature_{feature}')
    top1_count = top1_vc['count'].rename(f'top_count_{feature}')
    top1_rate = top1_vc['rates'].rename(f'top_rate_{feature}')

    def construct_feature_vector(sub_df):
        vector = np.zeros(len(feature_values) + 1)
        idxs = sub_df[feature].map(lambda v: feature2idx[v] if v in feature2idx else -1).values
        vector[idxs] = sub_df['rates']
        return vector

    rates = vc.groupby('client_id').apply(construct_feature_vector)
    rates = pd.DataFrame(np.stack(rates.values), index=rates.index,
                 columns=[f'rate_value{i}_{feature}' for i in range(len(feature2idx)+1)])

    features_df = pd.concat([rates, top1_feature, top1_count, top1_rate], axis=1)
    return features_df


def get_client_features(funel, client) -> pd.DataFrame:
    client['gender'] = client['gender'].fillna('UNK')
    client['age'] = client['age'].fillna(client['age'].median()).astype(int)
    client['education'] = client['education'].fillna('UNK')
    client['job_type'] = client['job_type'].fillna("UNK")
    client = client[['client_id', 'gender', 'age', 'education', 'job_type', 'region', 'city']]

    funel = funel.merge(client, on='client_id', how='left')
    return funel


def get_comm_features(df_funnel, df_com) -> pd.DataFrame:
    coms = df_com[df_com['channel'] == 'CALL'].groupby('client_id').sum()[['agr_flg', 'otkaz', 'dumaet', 'ring_up_flg', 'not_ring_up_flg']]
    df_funnel = pd.concat([df_funnel.set_index('client_id'), coms], axis=1).reset_index()

    # Количество звонков в последнем месяце
    calls_number_in_recent_months = df_com[(df_com.month_end_dt.dt.month == 8) & (df_com.month_end_dt.dt.year == 2019)].groupby('client_id').sum()[['count_comm']]
    calls_number_in_recent_months = calls_number_in_recent_months.rename(columns={'count_comm': 'calls_number_in_recent_months'})
    df_funnel = pd.concat([df_funnel.set_index('client_id'), calls_number_in_recent_months], axis=1).reset_index()

    # За сколько дней до конца общего периода был последний звонок
    last_coll = df_com.groupby('client_id')[['month_end_dt']].max()
    last_coll = last_coll.rename(columns={'month_end_dt': 'last_coll'})
    df_funnel = pd.concat([df_funnel.set_index('client_id'), last_coll], axis=1).reset_index()
    df_funnel.last_coll = df_funnel.last_coll.fillna(datetime.datetime.combine(datetime.date(2018, 1, 1), datetime.time(0, 0)))
    df_funnel['last_coll_days_delta'] = df_funnel.last_coll.map(lambda x: (datetime.datetime.combine(datetime.date(2020, 1, 1), datetime.time(0, 0)) - x).days)
    df_funnel = df_funnel.drop(columns=['last_coll'])

    df_funnel = df_funnel.set_index('client_id')
    products = ['Cash Loan', 'Credit Card', 'Debit Card', 'Investment bundle',
                'Mortgage', 'Currency exchange', 'Not applicable', 'Investment product']
    for product in products:
        gb = df_com[df_com['prod'] == product].groupby('client_id')
        feature_agr = (gb['agr_flg'].sum() > 0).astype(int).rename(f'agr_prod_{product}')
        feature_otkaz = (gb['otkaz'].sum() > 0).astype(int).rename(f'otkaz_prod_{product}')

        df_funnel[f'agr_prod_{product}'] = 0
        df_funnel[f'otkaz_prod_{product}'] = 0
        df_funnel.loc[feature_agr.index, f'agr_prod_{product}'] = feature_agr
        df_funnel.loc[feature_otkaz.index, f'otkaz_prod_{product}'] = feature_otkaz

    return df_funnel.reset_index()


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


def get_trxn_features(df_funnel, df_trxn):
    df_trxn['very_small_purchase'] = (df_trxn.tran_amt_rur < 150).astype('int16')
    df_trxn['small_purchase'] = (df_trxn.tran_amt_rur < 1500).astype('int16')
    df_trxn['average_purchase'] = ((df_trxn.tran_amt_rur > 1500) & (df_trxn.tran_amt_rur < 10000)).astype('int16')
    df_trxn['big_purchase'] = (df_trxn.tran_amt_rur > 10000).astype('int16')
    purchase = df_trxn.groupby('client_id').sum()[['very_small_purchase', 'small_purchase', 'average_purchase', 'big_purchase']]
    df_funnel = pd.concat([df_funnel.set_index('client_id'), purchase], axis=1).reset_index()

    # Повторяющаяся покупки
    recurring_purchases = (df_trxn.groupby('client_id').count() - df_trxn.groupby('client_id').nunique())[['tran_amt_rur', 'tsp_name']]
    recurring_purchases = recurring_purchases.rename(columns={'tran_amt_rur': 'nonunique_tran_amt_rur', 'tsp_name': 'nonunique_tsp_name'})
    df_funnel = pd.concat([df_funnel.set_index('client_id'), recurring_purchases], axis=1).reset_index()

    # Среднее арифметическое время платежа
    df_trxn['hour'] = df_trxn.tran_time.dt.hour
    mean_hour_trxn = df_trxn[['client_id', 'hour']].groupby('client_id').mean()
    df_funnel = pd.concat([df_funnel.set_index('client_id'), mean_hour_trxn], axis=1).reset_index()

    # Среднее арифметическое покупок за день
    df_trxn['date'] = df_trxn.tran_time.dt.date
    average_purchases_per_day = df_trxn[['client_id', 'date']].groupby('client_id').count() / df_trxn[['client_id', 'date']].groupby('client_id').nunique()
    df_funnel = pd.concat([df_funnel.set_index('client_id'), average_purchases_per_day], axis=1).reset_index()


    # За сколлько дней до конца общего периода была последняя трата
    last_transaction = df_trxn.groupby('client_id')[['tran_time']].max()
    last_transaction = last_transaction.rename(columns={'tran_time': 'last_transaction'})
    df_funnel = pd.concat([df_funnel.set_index('client_id'), last_transaction], axis=1).reset_index()
    df_funnel.last_transaction = df_funnel.last_transaction.fillna(datetime.datetime.combine(datetime.date(2019, 1, 1), datetime.time(0, 0)))
    df_funnel['last_transaction_days_delta'] = df_funnel.last_transaction.map(lambda x: (datetime.datetime.combine(datetime.date(2020, 1, 1), datetime.time(0, 0)) - x).days)
    df_funnel = df_funnel.drop(columns=['last_transaction'])

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
