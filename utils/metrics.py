import numpy as np
import config as c


def NIC(y_predict: np.array, sale_amount: np.array, contacts: np.array) -> float:
    return float(np.sum(y_predict * (sale_amount.fillna(0).to_numpy() - (c.CALL_COST * contacts)) / len(y_predict)))
