import tensorflow as tf

import config as c


def kappa_loss(y_pred, y_true, y_pow=2, eps=1e-10, N=c.CLASS_COUNT, bsize=c.BATCH_SIZE, name='kappa'):
    """
    Квадратичная взвешаная капа
    :param y_pred: 2D-тензор
    :param y_true: 2D-тензор
    :param y_pow: y_pow
    :param eps: предотвращает деление на ноль
    :param N: количество классов
    :param bsize: размер батча
    :param name: назавние
    :return: величина ошибки
    """
    with tf.name_scope(name):
        y_true = tf.to_float(y_true)
        repeat_op = tf.to_float(tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]))
        repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
        weights = repeat_op_sq / tf.to_float((N - 1) ** 2)

        pred_ = y_pred ** y_pow
        try:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))
        except Exception:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [bsize, 1]))

        hist_rater_a = tf.reduce_sum(pred_norm, 0)
        hist_rater_b = tf.reduce_sum(y_true, 0)

        conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)

        nom = tf.reduce_sum(weights * conf_mat)
        denom = tf.reduce_sum(weights * tf.matmul(
            tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])) /
                              tf.to_float(bsize))

        return nom / (denom + eps)
