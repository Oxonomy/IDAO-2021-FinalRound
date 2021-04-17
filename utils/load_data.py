import os
import numpy as np
import pandas as pd

import config as c
from utils.data_processing import *


def load_funnel():
    """
    Воронка продаж PACL. Содержит следующие столбцы:
        client_id : ID клиента.
        sale_flg : Продан флаг.
        sale_amount : Сумма заработанных банком денег в рублях.
        contacts : Количество контактов на предложение.
        client_segment: Бизнес-сегмент клиента.
        region_cd: ID региона.
    """
    path = os.path.join(c.DATASET_DIR, 'funnel.csv')
    df = pd.read_csv(path)
    df = funnel_preprocessing(df)
    return df


def load_appl():
    """
    ?
    :return:
    """
    path = os.path.join(c.DATASET_DIR, 'appl.csv')
    df = pd.read_csv(path)
    df = funnel_preprocessing(df)
    return df


def load_aum():
    """
    Данные AUM (активы под управлением). Содержит следующие столбцы:
        client_id: ID клиента.
        month_end_dt: Дата (в формате «конец месяца»).
        product_code: Тип аккаунта.
        balance_rur_amt: EOP (End Of Period) сумма остатка в рублях.
    """
    path = os.path.join(c.DATASET_DIR, 'aum.csv')
    df = pd.read_csv(path)
    df = aum_preprocessing(df)
    return df


def load_balance():
    """
    Данные баланса. Содержит следующие столбцы:
        client_id: Уникальный идентификатор клиента.
        prod_cat_nanme: Категория продукта.
        prod_group_name: Группа товаров.
        crncy_cd: Идентификатор валюты.
        eop_bal_sum_rur: EOP (End Of Period) сумма остатка в текущем месяце в рублях.
        min_bal_sum_rur: Минимальная сумма остатка в текущем месяце в рублях.
        max_bal_sum_rur: Максимальная сумма остатка в текущем месяце в рублях.
        avg_bal_sum_rur: Средняя сумма остатка в текущем месяце в рублях.
    """
    path = os.path.join(c.DATASET_DIR, 'balance.csv')
    df = pd.read_csv(path)
    df = balance_preprocessing(df)
    return df


def load_client():
    """
    Социально-демографические данные. Содержит следующие столбцы:
        client_id: ID клиента.
        gender: Пол.
        age: Возраст.
        region: ID региона.
        city: ID города.
        education: Уровень образования.
        citizenship: Клиентское гражданство.
        job_type: Рабочий статус.
    """
    path = os.path.join(c.DATASET_DIR, 'client.csv')
    df = pd.read_csv(path)
    df = client_preprocessing(df)
    return df


def load_com():
    """
    Данные кампании. Содержит следующие столбцы:
        client_id: ID клиента.
        channel: Канал.
        prod: Название предлагаемого продукта.
        agr_flg: клиент согласился принять предложение (бинарная переменная).
        otkaz: клиент отклонил предложение (двоичная переменная).
        dumaet: клиент не уверен в предложении (двоичная переменная).
        ring_up_flg: количество случаев дозвона.
        not_ring_up_flg: количество случаев отсутствия дозвона.
        count_comm: количество коммуникаций.
    """
    path = os.path.join(c.DATASET_DIR, 'com.csv')
    df = pd.read_csv(path)
    df = com_preprocessing(df)
    return df


def load_deals():
    """
    Данные о сделках. Содержит следующие столбцы:
        client_id: ID клиента.
        prod_type_name: Тип продукта.
        agrmnt_start_dt: Дата начала сделки.
        agrmnt_close_dt: Дата закрытия сделки.
        crncy_cd: Идентификатор валюты.
        agrmnt_rate_active: Процентная ставка по сделке.
        agrmnt_rate_passive: Процентная ставка по сделке.
        agrmnt_sum_rur: Сумма сделки.
    """
    path = os.path.join(c.DATASET_DIR, 'deals.csv')
    df = pd.read_csv(path)
    df = deals_preprocessing(df)
    return df


def load_dict_mcc():
    """
    Словарь MCC (Merchant Category Code). Содержит следующие столбцы:
        mcc_cd: Код MCC.
        brs_mcc_group: MCC group.
        brs_mcc_subgroup: Подгруппа MCC.
    """
    path = os.path.join(c.DATASET_DIR, 'dict_mcc.csv')
    df = pd.read_csv(path)
    df = dict_mcc_preprocessing(df)
    return df


def load_payments():
    """
    Заработная плата и пенсионные выплаты. Содержит следующие столбцы:
        client_id : ID клиента.
        day_dt : Дата платежа.
        sum_rur : Сумма платежа в рублях.
        pmnts_name : Способ оплаты.
    """
    path = os.path.join(c.DATASET_DIR, 'payments.csv')
    df = pd.read_csv(path)
    df = payments_preprocessing(df)
    return df


def load_trxn():
    """
    Данные по карточным операциям. Содержит следующие столбцы:
        client_id: ID клиента.
        card_id: ID карты.
        tran_time: Дата и время операции.
        tran_amt_rur: Сумма операции в рублях.
        mcc_cd: MCC - Код категории продавца.
        merchant_cd: Торговый код.
        txn_country: Страна, где произошла операция.
        txn_city: Город, в котором произошла операция.
        tsp_name: Имя продавца.
        txn_comment_1: Комментарий # 1
        txn_comment_2: Комментарий # 2
    """
    path = os.path.join(c.DATASET_DIR, 'trxn.csv')
    df = pd.read_csv(path)
    df = trxn_preprocessing(df)
    return df
