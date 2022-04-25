""" 
Splits the cutouts into a train, val and test set
The split goes over all available data (thus over all fields) irrespective of whether 
fits extraction would be succesful
"""

import os
import pickle
import sys
from glob import glob

import numpy as np

sys.path.insert(0, os.environ['PROJECTPATH'])
from sys import argv
from cv2 import imread


def get_mean_and_std_of_images(image_directory, save_name, n_channels=3):
    """Given a directory, opens all pngs and returns the mean and std value of each rgb-channel."""
    image_paths = glob(os.path.join(image_directory, '*_radio_DR2_rotated0deg.png'))
    print(image_directory)
    means = []
    stds = []
    for image_path in image_paths:
        # open image
        img = imread(image_path)
        m = [np.mean(img[:, :, i]) for i in range(n_channels)]
        s = [np.std(img[:, :, i]) for i in range(n_channels)]
        means.append(m)
        stds.append(s)
    mean = np.mean(means, axis=0)
    std = np.std(stds, axis=0)
    print(mean, std)
    with open(save_name + '.txt', 'w') as f:
        f.writelines([str(mean), str(std)])


# Train, val, test split
# (expressed in terms of their boundaries)
split = [0, 0.6, 0.8, 1]

dataset_name = argv[1]
# dataset_name = "LGZ_v4_rotations_small"
try:
    IMAGEDIR = os.environ['IMAGEDIR']
    CACHE_DIR = os.environ['CACHE_PATH']
except:
    IMAGEDIR = '/data/mostertrij/data/frcnn_images'
    CACHE_DIR = '/data/mostertrij/data/cache'
root_directory_pytorch = os.path.join(IMAGEDIR, dataset_name)
os.makedirs(root_directory_pytorch, exist_ok=True)
cutout_list_path = os.path.join(root_directory_pytorch, 'labeled_annotated_cutouts.pkl')

print("image dir:", IMAGEDIR)
print("Cache dir:", CACHE_DIR)
print("dataset name:", dataset_name)

# Get mean and stds
train_dir = os.path.join(IMAGEDIR, dataset_name, 'LGZ_COCOstyle/train')
save_name = os.path.join(CACHE_DIR, f'DR1_large_and_bright_balanced_source_list_train_mean_std')
# get_mean_and_std_of_images(train_dir,save_name, n_channels=3)

# Get cutout list
with open(cutout_list_path, 'rb') as f:
    print(cutout_list_path)
    cutout_list = np.array(pickle.load(f))

# Get lists of sources per cutout
# Single comp sources
c_names_single = [c.c_source.sname for c in cutout_list if len(c.get_related_comp()[0]) == 0]
cutout_names_single = np.array(sorted(set(c_names_single), key=c_names_single.index))
print(f"number of single comp cutout names before {len(c_names_single)} and after set",
      f" {len(cutout_names_single)}")
# Multi comp sources
c_names_multi = [c.c_source.sname for c in cutout_list if len(c.get_related_comp()[0]) > 0]
cutout_names_multi = np.array(sorted(set(c_names_multi), key=c_names_multi.index))
print(f"number of multi comp cutout names before {len(c_names_multi)} and after set",
      f" {len(cutout_names_multi)}")

lsplit = len(split) - 1
ls = np.array(range(len(cutout_names_single)))
lm = np.array(range(len(cutout_names_multi)))
np.random.seed(42)
np.random.shuffle(ls)
np.random.shuffle(lm)
splitted_image_names_single = []
splitted_image_names_multi = []
# calculate indx borders
ls_indices = []
lm_indices = []
ls_index_borders = list(map(int, np.array(split) * len(cutout_names_single)))
lm_index_borders = list(map(int, np.array(split) * len(cutout_names_multi)))
# get a list of indices for test/train/val
for i in range(lsplit):
    ls_indices.append(ls[ls_index_borders[i]:ls_index_borders[i + 1]])
    lm_indices.append(lm[lm_index_borders[i]:lm_index_borders[i + 1]])
# apply indices to image and object lists to get splits
for ls_ind, lm_ind, suffix in zip(ls_indices, lm_indices, ['train', 'val', 'test']):
    print(suffix)
    save_path = os.path.join(CACHE_DIR, f'DR1_large_and_bright_balanced_source_list_{suffix}.npy')

    final_names = [n for n in cutout_names_single[ls_ind]] + [n for n in cutout_names_multi[lm_ind]]
    np.random.shuffle(final_names)
    np.save(save_path, final_names)
    print(f"Length {suffix} is {len(ls_ind) + len(lm_ind)}")

print(f'Single+multi clist = {len(cutout_names_single) + len(cutout_names_multi)}')
print(f'Single+multi indices = {np.sum([len(s) for s in ls_indices]) + np.sum([len(s) for s in lm_indices])}')

"""
#c_names =[c.c_source.sname for c in cutout_list]
#print(f"len cnames {len(c_names)}")
#cutout_names = np.array(sorted(set(c_names), key=c_names.index))
#print(f"len cutout_names setted {len(cutout_names)}")
lsplit = len(split)-1
l = np.array(range(len(cutout_names)))
np.random.seed(42)
np.random.shuffle(l)
splitted_image_names = []
# calculate indx borders
l_indices = []
l_index_borders = list(map(int, np.array(split)*len(cutout_names)))
# get a list of indices for test/train/val
for i in range(lsplit):
    l_indices.append(l[l_index_borders[i]:l_index_borders[i+1]])
# apply indices to image and object lists to get splits
for l_ind, suffix in zip(l_indices,['train','val','test']):
    print(suffix)
    save_path = os.path.join(CACHE_DIR, f'DR1_large_and_bright_balanced_source_list_{suffix}.npy')
    np.save(save_path, [n for n in cutout_names[l_ind]])
    print(f"Length {suffix} is {len(cutout_names[l_ind])}")
"""
