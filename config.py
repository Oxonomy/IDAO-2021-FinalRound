import json
import os

__json = json.load(open('config.submission.json', 'r'))
if os.path.isfile('config.json'):
    __json = json.load(open('config.json', 'r'))


DATASET_DIR = __json.get('dataset_dir')
MODEL_DIR = __json.get('model_dir')
CALL_COST = 400 / 0.1

SEED = 7
