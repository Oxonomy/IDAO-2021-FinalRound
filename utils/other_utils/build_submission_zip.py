from zipfile import ZipFile
import os


def get_all_file_paths(directory):
    file_paths = []

    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths


def main():
    file_paths = []
    file_paths += get_all_file_paths('mcc')
    file_paths += get_all_file_paths('utils')
    file_paths += get_all_file_paths('model')
    file_paths += get_all_file_paths('debugg_data')
    file_paths += ['Makefile', 'run.sh', 'train.sh', 'archive_name', 'config.submission.json', 'config.py',
                   'train.py', 'generate_submission.py', 'debugging.py', 'debugging.sh']

    for file_name in file_paths:
        print(file_name)

    with ZipFile('submission.zip', 'w') as zip:
        for file in file_paths:
            zip.write(file)

    print('All files zipped successfully!')


if __name__ == "__main__":
    main()
