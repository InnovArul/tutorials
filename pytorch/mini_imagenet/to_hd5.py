import h5py
import numpy as np
import os, os.path as osp
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="images to hdf5")
parser.add_argument("--split", default="1", help="dataset split")
this_path = osp.dirname(osp.abspath(__file__))
args = parser.parse_args()

base_path = osp.join(this_path, args.split)  # dataset path
save_path = './dataset_' + args.split + '.hdf5'  # path to save the hdf5 file

hf = h5py.File(save_path, 'a')  # open the file in append mode

for i in os.listdir(base_path):   # read all the train/val/test dirs
    split_dir = os.path.join(base_path, i)
    print("digesting ", split_dir)
    grp = hf.create_group(i)  # create a hdf5 group

    for class_name in tqdm(os.listdir(split_dir)):  # read all classnames
        class_dir = os.path.join(split_dir, class_name)
        subgrp = grp.create_group(class_name)  # create a subgroup for the above created group.

        for imgname in os.listdir(class_dir):   # find all images inside class dir
            img_path = os.path.join(class_dir, imgname)
            with open(img_path, 'rb') as img_f:  # open images as python binary
                binary_data = img_f.read()

            binary_data_np = np.asarray(binary_data)
            dset = subgrp.create_dataset(imgname, data=binary_data_np) # save it in the subgroup. each a-subgroup contains all the images.

hf.close()