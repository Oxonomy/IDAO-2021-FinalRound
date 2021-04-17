import os


def turn_off_gpu():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
