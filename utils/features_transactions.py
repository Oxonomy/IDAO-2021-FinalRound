import numpy as np
import pandas as pd


def pipeline(df, mcc):
    preprocess_trxn(df, mcc)

    client_info = pd.DataFrame(index=df['client_id'].unique())

    # COUNT, TOP, RATE categorical features
    feature_count_top_rate(client_info, df, 'mcc_cd')
    frequent_mcc_cd = [5411, 6011, 6536, 5499, 4812, 5814, 6538, 5921, 5541,
                       6012, 4131, 4121, 5331, 5816, 5812, 4814, 5399]
    client_info.loc[~client_info['top_mcc_cd'].isin(frequent_mcc_cd), 'top_mcc_cd'] = -1

    feature_count_top_rate(client_info, df, 'mcc_cd_humanized')
    feature_count_top_rate(client_info, df, 'merchant_cd')
    frequent_merchant_cd = [3967370607447365704, 6303729926858726613, 3956921063892053726,
                            -5649346468447594695, 6943930447919418153]
    client_info.loc[~client_info['top_merchant_cd'].isin(frequent_merchant_cd), 'top_merchant_cd'] = -1

    feature_count_top_rate(client_info, df, 'tran_month')
    feature_count_top_rate(client_info, df, 'tran_day')
    client_info.drop('n_tran_day', axis=1, inplace=True)

    feature_count_top_rate(client_info, df, 'txn_comment_1')
    feature_count_top_rate(client_info, df, 'txn_comment_2')
    client_info['top_txn_comment_2'].fillna('unknown', inplace=True)

    feature_count_top_rate(client_info, df, 'card_id')
    client_info.drop('top_card_id', axis=1, inplace=True)

    # TOTAL SPENT on specific category
    total_spend_mcc_labels = ['Cash_Cash', 'Finance_Money transfer', 'Electronics_Electronics',
                              'Cars_Car services', 'Housing_Furniture and equipment', 'Housing_Repair',
                              'Finance_Financial services', 'Health_Drugs', 'Health_Medical services']
    for label in total_spend_mcc_labels:
        df_mcc = df[df['mcc_cd_humanized'] == label]
        client_info[f'total_spend_{label}'] = df_mcc.groupby('client_id')['tran_amt_rur'].sum()

    # COUNTRY features
    df['n_txn_country'] = df.groupby('client_id')['txn_country'].nunique()
    feature_num_of_countries(client_info, df)

    # COUNT values of specific categorical column
    feature_count_values_per_client(client_info, df, 'txn_comment_1', values=df['txn_comment_1'].unique())
    feature_count_values_per_client(client_info, df, 'txn_comment_2',
                                    values=('Purchase payment (web)', 'Opening Online', '<manual mode>'))
    feature_count_manualmode_mcccd(client_info, df)

    # MERGE ["SPB", "ST. PETER"] like entities
    df = fix_duplicated_values(df)

    # COUNT, TOP, RATE for company and city
    feature_count_top_rate(client_info, df, 'txn_city')
    feature_count_top_rate(client_info, df, 'tsp_name')

    # ACTIVITY of user
    feature_user_activity(client_info, df)
    client_info['activity_window'] = client_info['last_active_month'] - client_info['first_active_month']

    # TIME SERIES features of monthly actions. VERY SLOW!
    #feature_action_seq_by_label(client_info, df, 'mcc_cd')

    # TIME SERIES features of spends
    mcc_labels_mean = ('General consumption_Food', 'Restaurants_Fast food',
                       'Restaurants_Restaurants', 'Clothes_Clothes', 'Health_Drugs')
    for mcc_label in mcc_labels_mean:
        feature_mean_mounthly_check_for_mcc(client_info, df, mcc_label, mean=True)

    mcc_labels_sum = ('General consumption_Food', 'Cash_Cash', 'Restaurants_Fast food', 'Finance_Money transfer',
                      'Electronics_Electronics', 'Cars_Car services', 'Health_Drugs',
                      'General consumption_Department stores',
                      'Clothes_Clothes', 'General consumption_Other goods', 'Restaurants_Restaurants', 'Transport_Bus',
                      'Transport_Taxi', 'Housing_Furniture and equipment', 'Housing_Repair',
                      'Finance_Financial services',
                      'Beauty_Cosmetics', 'Transport_Local transport', 'Health_Medical services', 'Luxury_Cigars',
                      'Restaurants_Bars', 'Beauty_Barber and beauty', 'Travel_Airlines', 'Travel_Hotels',
                      'Education_Education')
    for mcc_label in mcc_labels_sum:
        feature_mean_mounthly_check_for_mcc(client_info, df, mcc_label, mean=False)

    client_info['mean_mcc_sum_trend'] = 0
    for mcc_label in mcc_labels_sum:
        client_info['mean_mcc_sum_trend'] += client_info[f'trend_monthly_sum_{mcc_label}'] / len(mcc_labels_sum)

    client_info.drop('min_monthly_sum_Travel_Hotels', axis=1, inplace=True)

    # TIME SERIES features for different money transfers
    feature_client_money_transfer(
        client_info, df,
        tr_types=['Payment for goods and services', 'Cashless transfer', 'Payment by card (bank transfer)'],
        tr_label='outcome'
    )
    feature_client_money_transfer(
        client_info, df,
        tr_types=['Return of goods / services', 'Cash deposit by card'],
        tr_label='income'
    )
    feature_client_money_transfer(
        client_info, df,
        tr_types=['Cash withdrawal through an ATM', 'Cash withdrawal'],
        tr_label='cash'
    )

    return client_info


def preprocess_trxn(df, mcc):
    df['tsp_name'] = df['tsp_name'].fillna('unknown')
    df['txn_city'] = df['txn_city'].fillna('unknown')
    # не бейте плиз
    df['txn_city'] = df['txn_city'].apply(lambda s: s[2:] if s.startswith('G ') else s)
    df['txn_city'] = df['txn_city'].apply(lambda s: s[3:] if s.startswith('G. ') else s)
    df['txn_city'] = df['txn_city'].apply(lambda s: s[2:] if s.startswith('G.') else s)
    df['txn_city'] = df['txn_city'].apply(lambda s: s[:-2] if s.endswith(' G') else s)
    df['txn_city'] = df['txn_city'].apply(lambda s: s[:-3] if s.endswith(' G.') else s)
    df['txn_city'] = df['txn_city'].apply(lambda s: s[:-2] if s.endswith('G.') else s)

    df['tran_year'] = df['tran_time'].dt.year
    df['tran_month'] = df['tran_time'].dt.month
    df['tran_day'] = df['tran_time'].dt.day

    mcc = mcc.set_index('mcc_cd')
    mcc['group'] = mcc['brs_mcc_group'] + '_' + mcc['brs_mcc_subgroup']
    df['mcc_cd_humanized'] = df['mcc_cd'].map(mcc['group'])


def feature_count_top_rate(client_info, df, column):
    gb = df.groupby('client_id')[column]
    client_info[f'n_{column}'] = gb.nunique()
    mcc_client = gb.value_counts().rename('count').reset_index().groupby('client_id').head(1).set_index('client_id')
    client_info[f'top_{column}'] = mcc_client[column]
    client_info[f'rate_{column}'] = mcc_client['count'] / gb.count()


def feature_num_of_countries(client_info, df):
    client_info['num_of_rich_countries'] = 0

    poor_countries = set(['RUS', 'BLR', 'TUR', 'KAZ', 'UKR', 'UZB'])
    rich_countries = set(df['txn_country'].unique()).difference(poor_countries)
    poor_countries.remove('RUS')

    num_of_poor_countries = df.groupby('client_id')['txn_country'].unique().apply(
        lambda arr: len(set(arr).intersection(poor_countries))
    )
    client_info.loc[num_of_poor_countries.index, 'num_of_poor_countries'] = num_of_poor_countries

    num_of_rich_countries = df.groupby('client_id')['txn_country'].unique().apply(
        lambda arr: len(set(arr).intersection(rich_countries))
    )
    client_info.loc[num_of_rich_countries.index, 'num_of_rich_countries'] = num_of_rich_countries


def feature_count_values_per_client(client_info, df, column, values):
    for value in values:
        client_counts = df.groupby(['client_id'])[column].value_counts().rename('count')
        a = client_counts.reset_index()
        a = a[a[column] == value][['client_id', 'count']].set_index('client_id')

        client_info['count_' + value] = 0
        client_info.loc[a.index, 'count_' + value] = a['count']


def feature_count_manualmode_mcccd(client_info, df):
    cnt = df[df['txn_comment_2'] == '<manual mode>'].groupby(
        ['client_id', 'mcc_cd']).count()['card_id'].rename('count').reset_index()

    for mcc_cd in (6536, 6538, 6012, 4111, 6051, 6540, 4829):
        a = cnt[cnt['mcc_cd'] == mcc_cd].set_index('client_id')

        client_info[f'count_<manual mode>_{mcc_cd}'] = 0
        client_info.loc[a.index, f'count_<manual mode>_{mcc_cd}'] = a['count']


def fix_duplicated_values(df):
    tsp_name_fixes = {
        'KRASNOE BELOE': ['KRASNOEBELOE', 'KRASNOE I BELOE', 'KRASNOE BELOE.', 'KRASNOE I BELOE.',
                          'KRANSOE BELOE', 'KRACNOE BELOE'],
        'PYATEROCHKA': ['MAGAZIN PYATEROCHKA', 'MN PYATEROCHKA', 'SUPERMARKET PYATEROCHKA', 'PYATEROCHKA .'
                                                                                            'MAG. PYATEROCHKA PLYUS',
                        'MAGAZIN PYATYOROCHKA', 'PYATEROCHKA N', 'MAGAZINPYATEROCHKA PLY',
                        'SUPERMARKET PYATYOROCHKA', 'PYATEROCHKA MSPK', 'PAVILON PYATEROCHKA', 'PYATEROCHKA L',
                        'PYATEROCHKA SPASIBO', 'PYATEROCHKA A', 'MSPK PYATEROCHKA', 'PYATEROCHKA NA USHAKOV',
                        'PYATEROCHKA BELENOGOVA', 'BRN VY PYATEROCHKA', 'UNIVERSAM PYATEROCHKA', 'PYATEROCHKA M KM'
                                                                                                 'MARI MYASO PYATEROCHKA',
                        'PETEROCHKA', 'PYATEROCHKA SHOP'],
        'OPEN.RU': ['OPEN.RU CARDCARD', 'APP.OPEN.RU CARDCARD', 'APP OPEN.RU', 'WWW.OPENONLINE.RU',
                    'OPENBANK.RU CARDCARD', ],
        'OTKRYTIE': ['OP OTKRYTIE', 'OOO OTKRYTIE', 'PAO BANK FK OTKRYTIE', 'ADM ZDANIE BANK OTKRYTIE',
                     'BANK FK OTKRYTIE', 'OFIS BANKA OTKRYTIE', 'BANK OTKRYTIE', 'ZDANIE BANKA OTKRYTIE',
                     'OFIS FK OTKRYTIE', 'OFISY BANKA OTKRYTIYA', 'BANK OTKRITIE'],
        'MONETKA': ['MAGAZIN MONETKA', 'MONETKA.', 'SUPERMARKET MONETKA', 'TTS MONETKA', 'MONETKA .',
                    'MN MONETKA', 'IP ISAKOVA YN MONETKA', 'OOO MONETKA', 'TC MONETKA', 'UNIVERSAM MONETKA',
                    'UM MONETKA'],
        'YANDEX.EDA': ['YANDEX EDA', 'KWBYANDEX.EDA'],
        'YANDEX.MONEY': ['YANDEXMONEY', 'YM YANDEX.MONEY', 'MONEY.YANDEX', 'YMYANDEX.MONEY', 'Y.MYANDEX.MONEY'],
        'YANDEX.TAXI': ['YANDEX TAXI', 'WWW.TAXI.YANDEX.RU', 'YANDEX.TAXI.', 'Y.MYANDEX.TAXI.'],
        'LENTA': ['TK LENTA', 'GIPERMARKET LENTA', 'LENTA LLC', 'TTS LENTA', 'GM LENTA', 'LENTA .',
                  'SUPERMARKET LENTA', 'OOO LENTA', 'TC LENTA', 'GIPERT LENTA DOVATORA', 'LENTA GIPERMARKET'],
        'LUKOIL': ['LUKOIL.AZS U', 'LUKOIL.AZS Y', 'LUKOIL.AZS C', 'LUKOIL.AZS S', 'LUKOIL.AZK S',
                   'OOO LUKOILVNP', 'OFIS LUKOIL', 'AZS LUKOIL', 'OFFICE OOO LUKOIL', 'OFFICE LUKOIL',
                   'LUKOIL.DD AZS Y', 'LUKOIL AZS', 'LUKOIL.AAZS S', 'LUKOIL.AZS V'],
        'FIXPRICE': ['FIXPRICE .', 'FIXPRICE TC GRAND', 'FIXPRICE K', 'FIXPRICE BALTIYSK', 'MAGAZIN FIXPRICE'],
        'GAZPROMNEFT AZS': ['GAZPROMNEFT AZS', 'AZS N GAZPROM', 'AZS GAZPROM', 'GAZPROM AZS N', 'GAZPROMNEFTAZS',
                            'GAZPROM AZS N GAZPR', 'GAZPROM AZS', 'GAZPROM NEFT AZS', 'AZS GAZPROMNEFT',
                            'AZS GAZPROMNEFT N', 'AGZS GAZPROM SZH.GAZ', 'GAZPROM AZS GAZPRO',
                            'GAZPROM AZS N GAZPROM'],
        'BRISTOL': ['BRISTOL .', 'BRISTOL KONFETY', 'BRISTOLALKO', 'BRISTOL.', 'BRISTOL KV.'],
        'ALIEXPRESS': ['YMALIEXPRESS', 'WWW.ALIEXPRESS.COM', 'ALIEXPRESS.COM', 'YM.ALIEXPRESS',
                       'ALIEXPRESS.COM DS', 'YMALIEXPRESS.', 'ALIEXPRESS.COM ALIEXPR',
                       'ALIEXPRESS.COM ALIEXPRESS', 'YM ALIEXPRESS'],
        'YARCHE': ['MAG. YARCHE', 'YARCHE KIROVA G', 'YARCHE ASINO', 'YARCHE YURGA', 'YARCHE SIBIRSKAYA B',
                   'YARCHE MYUNNIKHA A', 'YARCHE LEN', 'YARCHE.', 'SUPERMARKET YARCHE', 'YARCHE LAZO',
                   'YARCHE TSENTR', 'YARCHE LYTKINA', 'YARCHE KIEVS', 'YARCHE ZONALN', 'YARCHE TRANSPA',
                   'YARCHE ARMIYA', 'YARCHE NOVOSIBA', 'YARCHE GAGARINA', 'YARCHE PUSHKINA', 'YARCHE KOMSOMOL',
                   'YARCHE IRKUTSKIY A', 'YARCHE TOREZA', 'SP YARCHE', 'YARCHE KIROVA', 'YARCHE CHERNYKH',
                   'YARCHE PLYUS', 'YARCHE ZORGE A', 'YARCHE GRUZINSKAYA', 'YARCHE KOZHEVNIKOVO',
                   'YARCHE SOVETSKAYA', 'YARCHE MELN', 'YARCHE VZLETNAYA', 'YARCHE BOGASHEVO MIRA',
                   'YARCHE PROF', 'YARCHE SEVAST', 'YARCHE KUT', 'YARCHE BOLSHAYA', 'YARCHE SHAKHTA'],
        'OKEY': ['TORGOVIY CENTR OKEY', 'GIPERMARKET OKEY', 'OOO OKEY', 'SUPERMARKET OKEY',
                 'OKEY ZHUKOVA', 'TC OKEY'],
        'BROKER': ['OTKRITIE BROKER JSC', 'PAWOPENBROKER', 'MY.BROKER.RU MY.BROKER', 'AO OTKRYTIE BROKER',
                   'TINKOFF INVESTMENT'],
        'WWW.RZD.RU': ['TICKET.RZD.RU', 'BILETI RZD'],
        'SUNLIGHT': ['SUNLIGHT BRILLIANT', 'SUNLIGHT', 'SUNLIGHT BRILLANT', 'YMSUNLIGHT'],
        'AZBUKA VKUSA': ['AV AZBUKAVKUSA', 'MARKET AZBUKA VKUSA', 'AV AZBUKAVKUSA.', 'AV AZBUKAVKUSA..',
                         'SP AV AZBUKAVKUSA', 'AV .AZBUKAVKUSA', 'AV AZBUKA VKUSA', 'OOO AZBUKA VKUSA'],
        'GASTRONOM': ['MAGAZIN GASTRONOM'],
        'UBER': ['YM UBER', 'UBER TRIP'],
        'GETT': ['GETTAXI.RU'],
        'CITYMOBIL': ['CITYMOBIL.', 'MMRCITYMOBIL'],
    }
    names_to_fix = ('PEREKRESTOK', 'MCDONALDS', 'DETSKIY MIR', 'GALAMART', 'VKUSVILL', 'GASTRONOM',
                    'KFC ', 'MAGNIT', 'OPENFOOD', 'BURGER KING', 'DNS', 'ODNOKLASSNIKI', 'PRODUKTY', 'AUTO')

    txn_city_fixes = {
        'MOSCOW': ['MOSKVA', 'MOSKOW', 'MOSKVAG', 'MOSKVA.', 'MSK'],
        'MOSCOW REGION': ['MOSCOW OBL L', 'MOSKOVSKAYA OBLAST', 'MOSCOW REG.',
                          'MOSCOW OBL.', 'MOSKOVSKAYA OBL.', 'MOSKOVSKAYA O',
                          'MOSCOWSKAYA O', 'MOSCOW R', 'MOSKOVSK.OBL.',
                          'MOSKOW REGION', ],
        'MOSKOVSKIJ': ['MOSKOVSKIY'],
        'NIZHNEVARTOVSK': ['NIZHNEVARTOVS'],
        'EKATERINB': ['EKB'],
        'PETER': ['SPB'],
        'NEFTEYUGANSK': ['NEFTEJUGANSK', 'NEFTEUGANSK', 'NEFTEYUANSK', 'NEFTEIUGANSK', 'NEFTEYUGANS'],
        'URAJ': ['URAY', 'URAI'],
        'PYTYAKH': ['PYTYAH', 'PYTJAH'],
    }
    cities_to_fix = ('PETER', 'TYUMEN', 'VOLGOGRA', 'EKATERINB', 'SARATOV', 'NOVGOROD',)

    def fix_duplicates(df, column, fixes, names_to_fix):
        fix_mapping = dict()
        for good_key, keys_to_replace in fixes.items():
            for key_to_replace in keys_to_replace:
                fix_mapping[key_to_replace] = good_key
        df[column] = df[column].apply(lambda name: fix_mapping[name] if name in fix_mapping else name)

        for name_to_fix in names_to_fix:
            df[column] = df[column].apply(lambda name: name_to_fix if name_to_fix in name else name)

    fix_duplicates(df, 'tsp_name', tsp_name_fixes, names_to_fix)
    fix_duplicates(df, 'txn_city', txn_city_fixes, cities_to_fix)

    return df


def construct_user_time_series(sub_df, label, fill=-1):
    seq = np.full(12, fill)
    for _, row in sub_df.iterrows():
        seq[int(row.tran_month) - 1] = row[label]

    seq = np.concatenate([seq[-4:], seq[:-4]])
    return seq


def trend_statistic(seq):
    n = len(seq)
    if n < 2:
        return 0
    x = range(n)
    return np.corrcoef(x, seq)[0, 1]


def feature_user_activity(client_info, df):
    n_active_days = df.groupby(['client_id', 'tran_month'])['tran_day'].nunique()
    seq_active_days = n_active_days.reset_index().groupby('client_id').apply(
        lambda df_: construct_user_time_series(df_, 'tran_day', fill=0)
    )

    client_info['first_active_month'] = seq_active_days.apply(lambda seq: np.nonzero(seq)[0][0] + 1)
    client_info['last_active_month'] = seq_active_days.apply(lambda seq: np.nonzero(seq)[0][-1] + 1)

    activity_df = pd.concat([client_info['first_active_month'], seq_active_days], axis=1)
    activity_df['non_empty_seq'] = activity_df.apply(lambda row: row[0][row['first_active_month'] - 1:], axis=1)
    activity_df['n_active_months'] = activity_df['non_empty_seq'].apply(lambda seq: len(np.nonzero(seq)[0]))

    client_info['year_activity_rate'] = (activity_df['n_active_months'] / (13 - activity_df['first_active_month']))
    client_info['mean_monthly_activity'] = activity_df['non_empty_seq'].apply(np.mean)
    client_info['std_monthly_activity'] = activity_df['non_empty_seq'].apply(np.std)
    client_info['max_monthly_activity'] = activity_df['non_empty_seq'].apply(np.max)
    client_info['min_monthly_activity'] = activity_df['non_empty_seq'].apply(np.min)
    client_info['trend_monthly_activity'] = activity_df['non_empty_seq'].apply(trend_statistic)


def feature_action_seq_by_label(client_info, df, label):
    n_actions_monthly = df.groupby(['client_id', 'tran_month'])[label].value_counts().rename('count')
    n_actions_monthly = n_actions_monthly.reset_index().groupby(['client_id', 'tran_month'])

    seq_n_actions = n_actions_monthly.sum()['count'].reset_index().groupby('client_id').apply(
        lambda df_: construct_user_time_series(df_, 'count', fill=0)
    ).rename('seq_n_actions')
    seq_n_top_actions = n_actions_monthly.head(1).groupby('client_id').apply(
        lambda df_: construct_user_time_series(df_, 'count', fill=0)
    ).rename('seq_n_top_actions')
    seq_top_actions = n_actions_monthly.head(1).groupby('client_id').apply(
        lambda df_: construct_user_time_series(df_, label, fill=0)
    ).rename('seq_top_actions')

    actions_df = pd.concat([
        client_info['first_active_month'], seq_n_actions, seq_n_top_actions, seq_top_actions
    ], axis=1)
    for i, col in enumerate(['seq_n_actions', 'seq_n_top_actions', 'seq_top_actions']):
        actions_df[f'non_empty_{col}'] = actions_df.apply(
            lambda row: row[col][row['first_active_month'] - 1:], axis=1)
    actions_df['non_empty_seq_top_action_rate'] = actions_df.apply(
        lambda row: row['non_empty_seq_n_top_actions'] / (row['non_empty_seq_n_actions'] + 1e-9),
        axis=1)

    for col in ('n_actions', 'n_top_actions', 'top_action_rate'):
        ts_col = f'non_empty_seq_{col}'
        client_info[f'mean_monthly_{col}_{label}'] = actions_df[ts_col].apply(np.mean)
        client_info[f'std_monthly_{col}_{label}'] = actions_df[ts_col].apply(np.std)
        client_info[f'max_monthly_{col}_{label}'] = actions_df[ts_col].apply(np.max)
        client_info[f'min_monthly_{col}_{label}'] = actions_df[ts_col].apply(np.min)
        client_info[f'trend_monthly_{col}_{label}'] = actions_df[ts_col].apply(trend_statistic)

    actions_df['counts'] = actions_df['non_empty_seq_top_actions'].apply(lambda seq: np.unique(seq, return_counts=True))
    client_info[f'top_{col}_monthly'] = actions_df['counts'].apply(lambda c: c[0][np.argmax(c[1])])
    client_info[f'rate_top_{col}_monthly'] = actions_df['counts'].apply(lambda c: np.max(c[1])) / actions_df[
        'non_empty_seq_top_actions'].apply(len)


def infer_statistic_from_seq(client_info, df_with_seq, label, fn):
    statistic = df_with_seq['non_empty_seq'].apply(fn)
    client_info[label] = 0
    client_info.loc[statistic.index, label] = statistic


def feature_mean_mounthly_check_for_mcc(client_info, df, mcc_label, mean=True):
    df_mcc = df[df['mcc_cd_humanized'] == mcc_label]
    gb = df_mcc.groupby(['client_id', 'tran_month'])['tran_amt_rur']
    if mean:
        mean_mcc_spend = gb.mean()
    else:
        mean_mcc_spend = gb.sum()
    mcc_seq = mean_mcc_spend.reset_index().groupby('client_id').apply(
        lambda df_: construct_user_time_series(df_, 'tran_amt_rur', fill=0)
    )

    mcc_seq_df = pd.concat([client_info['first_active_month'], mcc_seq], axis=1)
    mcc_seq_df.dropna(inplace=True)
    mcc_seq_df['non_empty_seq'] = mcc_seq_df.apply(lambda row: row[0][row['first_active_month'] - 1:], axis=1)

    prefix = 'mean' if mean else 'sum'
    infer_statistic_from_seq(client_info, mcc_seq_df, f'min_monthly_{prefix}_{mcc_label}', np.min)
    infer_statistic_from_seq(client_info, mcc_seq_df, f'max_monthly_{prefix}_{mcc_label}', np.max)
    infer_statistic_from_seq(client_info, mcc_seq_df, f'mean_monthly_{prefix}_{mcc_label}', np.mean)
    infer_statistic_from_seq(client_info, mcc_seq_df, f'std_monthly_{prefix}_{mcc_label}', np.std)
    infer_statistic_from_seq(client_info, mcc_seq_df, f'trend_monthly_{prefix}_{mcc_label}', trend_statistic)


def feature_client_money_transfer(client_info, df, tr_types: list, tr_label):
    df_outcome = df[df['txn_comment_1'].isin(tr_types)]

    monthly_outcome = df_outcome.groupby(['client_id', 'tran_month'])['tran_amt_rur'].sum()
    seq_outcome = monthly_outcome.reset_index().groupby('client_id').apply(
        lambda df_: construct_user_time_series(df_, 'tran_amt_rur', fill=0)
    )

    outcome_df = pd.concat([client_info['first_active_month'], seq_outcome], axis=1)
    outcome_df.dropna(inplace=True)
    outcome_df['non_empty_seq'] = outcome_df.apply(lambda row: row[0][row['first_active_month'] - 1:], axis=1)

    infer_statistic_from_seq(client_info, outcome_df, f'min_monthly_{tr_label}', np.min)
    infer_statistic_from_seq(client_info, outcome_df, f'max_monthly_{tr_label}', np.max)
    infer_statistic_from_seq(client_info, outcome_df, f'mean_monthly_{tr_label}', np.mean)
    infer_statistic_from_seq(client_info, outcome_df, f'std_monthly_{tr_label}', np.std)
    infer_statistic_from_seq(client_info, outcome_df, f'trend_monthly_{tr_label}', trend_statistic)
