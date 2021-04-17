from sklearn.metrics import mean_squared_error

import config as c
from utils.dataset import *
from utils.metrics import NIC
from utils.model import Model, get_combine_predictions


def main():
    df = get_dataset()
    x = df.drop(columns=['client_id']).to_numpy()

    model_sale_amount = Model.load('model_sale_amount')
    model_sale_flg = Model.load('model_sale_flg')


    target = get_combine_predictions(x, model_sale_flg, model_sale_amount)
    df['target'] = target
    df[['client_id', 'target']].to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
