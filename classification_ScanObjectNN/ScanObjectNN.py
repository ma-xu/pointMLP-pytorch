"""
ScanObjectNN download: http://103.24.77.34/scanobjectnn/h5_files.zip
"""

import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


import os

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    H5_DIR = os.path.join(DATA_DIR, 'h5_files')
    
    expected_file = os.path.join(H5_DIR, "main_split", "test_objectdataset_augmentedrot_scale75.h5")
    
    if os.path.exists(expected_file):
        print("Dataset already downloaded. Skipping download.")
        return
    
    if not os.path.exists(H5_DIR):
        os.makedirs(H5_DIR)
    
    www = 'https://github.com/ma-xu/pointMLP-pytorch/releases/download/dataset/h5_files.zip'
    zipfile = os.path.basename(www)
    
    os.system(f'wget {www} --no-check-certificate')
    os.system(f'unzip {zipfile} -d {H5_DIR}')
    os.system(f'rm {zipfile}')
    os.system(f'rm -rf "{os.path.join(H5_DIR, "main_files")}"')
    os.system(f'rm -rf "{os.path.join(H5_DIR, "__MACOSX")}"')

    print("Download and extraction complete.")


def load_scanobjectnn_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    all_data = []
    all_label = []

    h5_name = BASE_DIR + '/data/h5_files/main_split/' + partition + '_objectdataset_augmentedrot_scale75.h5'
    f = h5py.File(h5_name, mode="r")
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


class ScanObjectNN(Dataset):
    def __init__(self, num_points, partition='training'):
        self.data, self.label = load_scanobjectnn_data(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'training':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ScanObjectNN(1024)
    test = ScanObjectNN(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label)
