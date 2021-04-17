import json

__json = json.load(open('config.json', 'r'))

DATASET_DIR = __json.get('dataset_dir')
TEST_CSV = __json.get('test_csv')
TRAIN_CSV = __json.get('train_csv')
SUBMISSION_CSV = __json.get('submission_csv')

TEST_SIZE = __json.get('test_size')
BATCH_SIZE = __json.get('batch_size')

REMOVE_PATH_VARS = __json.get('remove_path_vars')

if bool(REMOVE_PATH_VARS):
    import sys
    try:
        sys.path.remove('D:\\Projects\\TFModels\\models\\research')
        sys.path.remove('D:\\Projects\\TFModels\\models\\research\\slim')
        sys.path.remove('D:\\Projects\\TFModels\\models\\research\\object_detection')
    except:
        pass

SEED = 7
