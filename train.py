from sklearn.metrics import mean_squared_error
import logging
import config as c
from utils.dataset import *
from utils.metrics import NIC
from utils.model import Model, get_combine_predictions
logging.basicConfig(format='%(asctime)s %(message)s', filename='training.log', level=logging.DEBUG)

def main():
    df = get_dataset()

    x = df[df['sale_amount'].fillna(0) > 0].drop(columns=['client_id', 'sale_flg', 'sale_amount', 'contacts']).to_numpy()
    y = df[df['sale_amount'].fillna(0) > 0]['sale_amount'].to_numpy().reshape(-1, 1)

    model_sale_amount = Model('model_sale_amount')
    model_sale_amount.fit(x, y)
    model_sale_amount.save()
    print(mean_squared_error(y, model_sale_amount.predict(x)))


    x = df.drop(columns=['client_id', 'sale_flg', 'sale_amount', 'contacts']).to_numpy()
    y = df['sale_flg'].to_numpy().reshape(-1, 1)

    model_sale_flg = Model('model_sale_flg')
    model_sale_flg.fit(x, y)
    model_sale_flg.save()
    print(mean_squared_error(y, model_sale_flg.predict(x)))

    x = df.drop(columns=['client_id', 'sale_flg', 'sale_amount', 'contacts']).to_numpy()
    target = get_combine_predictions(x, model_sale_flg, model_sale_amount)
    print('NIC:', NIC(target, df['sale_amount'], df['contacts']))

    #df['target'] = target
    #df[['client_id', 'target']].to_csv('submission.csv', index=False)

    logging.info("model was trained")


if __name__ == '__main__':
    main()
