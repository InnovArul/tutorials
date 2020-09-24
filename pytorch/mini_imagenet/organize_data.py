import torch
import os, os.path as osp
import shutil
import random, more_itertools as mit

create_renamed_data = False
split_data = True
num_splits = 5
train_num, val_num, test_num = 400, 100, 100

def copy_files_into_folder(src_class_folder, split_folder, files):
    if not os.path.exists(split_folder): os.makedirs(split_folder)

    for filename in files:
        current_file = osp.join(src_class_folder, filename)
        shutil.copy(current_file, split_folder)

def do_partition(index, split):
    split_folder = os.path.join('./data', str(index))
    if not os.path.exists(split_folder): os.makedirs(split_folder)
    print('performing split ', index)

    for classname in split:
        src_class_folder = osp.join(renamed_data_dir, classname)
        all_files = os.listdir(src_class_folder)
        print('class ', classname, ' has ', len(all_files), ' files')

        # split into train, val, test
        train_files, test_files, val_files = mit.split_into(all_files, [train_num, test_num, val_num])

        split_train_folder = os.path.join('./data', str(index), 'train', classname)
        split_val_folder = os.path.join('./data', str(index), 'val', classname)
        split_test_folder = os.path.join('./data', str(index), 'test', classname)
        copy_files_into_folder(src_class_folder, split_train_folder, train_files)
        copy_files_into_folder(src_class_folder, split_test_folder, test_files)
        copy_files_into_folder(src_class_folder, split_val_folder, val_files)

def read_classmapping():
    class_mapping_file = 'classes.txt'
    with open(class_mapping_file, 'r') as f:
        all_classmapping = f.readlines()
    
    nxx_to_english = {}
    for line in all_classmapping:
        nxx, english = line.strip().split(' ')
        nxx_to_english[nxx] = english
    
    return nxx_to_english

if __name__ == '__main__':
    all_data_dir = './data/all_data'
    renamed_data_dir = './data/renamed_data'

    # collect folders
    all_classnames_nxx = os.listdir(all_data_dir)
    print(len(all_classnames_nxx), ' classes found in ', all_data_dir)

    classnamesnxx_to_english = read_classmapping()

    if not osp.exists(renamed_data_dir): os.makedirs(renamed_data_dir)

    if create_renamed_data:
        # rename folders
        for classname in all_classnames_nxx:
            current_classnxx_folder = osp.join(all_data_dir, classname)
            assert classname in classnamesnxx_to_english

            # move the folder 
            shutil.copytree(current_classnxx_folder, osp.join(renamed_data_dir, classnamesnxx_to_english[classname]))
    
    all_classes = list(classnamesnxx_to_english.values())
    random.shuffle(all_classes)

    # split into num_splits sets 
    class_splits = mit.divide(num_splits, all_classes)
    class_splits = [list(x) for x in class_splits]
    print('splitted classes:', class_splits)

    # for each split, perform train, val, test partition
    for i, split in enumerate(class_splits):
        do_partition(i, split)