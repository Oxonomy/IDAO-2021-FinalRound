import numpy as np
import pandas as pd
from functools import partial
from multiprocessing import Pool


def parallelize(data, func, num_of_processes=8):
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


def run_on_subset(func, data_subset):
    return data_subset.apply(func, axis=1)


def parallelize_on_rows(data: pd.DataFrame, func, num_of_processes=8):
    """
    Функция для вызова многопроцессорной обработки DataFrame
    :param data: исходная таблица
    :param func: функция, применяемая к каждому элементу
    :param num_of_processes: количество запускаемых процессов (лучше всего брать n*2-1, где n - колличество ядер процессора)
    :return: обработанная таблица
    """
    return parallelize(data, partial(run_on_subset, func), num_of_processes)
