import zipfile
import os


if __name__ == '__main__':
    root = 'E:\\Datasets\\NinaproDataset'
    database_names = os.listdir(root)
    for database_name in database_names:
        sub_root = os.path.join(root, database_name)
        patients_names = os.listdir(sub_root)
        for patients_name in patients_names:
            data_root = os.path.join(sub_root, patients_name)
            if os.path.exists(data_root) and data_root[-4:] == '.zip':
                file = zipfile.ZipFile(data_root, 'r')
                file.extractall(path=data_root[:-4])
                file.close()
                os.remove(data_root)
        print(database_name + ' done.')
