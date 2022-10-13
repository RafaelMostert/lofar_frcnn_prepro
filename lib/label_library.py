""" Take image label and convert to json format for VIA and XML for the CNNs.
We will first populate a python dictionary for the JSON and a fake dictionary
for the XML.
"""
import copy
import itertools
import os
import pickle
from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from itertools import chain
from shutil import copyfileobj

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from astropy.visualization import MinMaxInterval, SqrtStretch
from astropy.wcs import WCS
from cv2 import imread
from matplotlib.path import Path
from scipy.spatial import ConvexHull
from skimage.transform import rotate
from skimage.util import crop


class FakeDict(dict):
    """Used to circumvent the fact that a regular Python dict
    can not contain duplicate keys."""

    def __init__(self, items):
        self['something'] = 'something'
        self._items = items

    def items(self):
        return self._items


def load_fits(fits_filepath, dimensions_normal=True, verbose=False):
    if verbose:
        print('Loading the following fits file:', fits_filepath)
    # Load first fits file
    with fits.open(fits_filepath) as hdulist:
        if dimensions_normal:
            hdu = hdulist[0].data
        else:
            hdu = hdulist[0].data[0, 0]

        # Header
        hdr = hdulist[0].header
    return hdu, hdr


def fits_to_image(cutout, fits_filepath, image_filepath,
                  clip_image, lower_clip, upper_clip, fixed_cutout_size, overwrite=True,
                  scaling='', imsize: int = None, contours=None, rms_filepath: str = None,
                  rotation_angle_deg: int = 0):
    """Given the filepath to a FITS file containing a mosaic of the 
    entire field, returns the clipped version as png.
    If clip_image is False no clipping is applied, else lower_clip and upper_clip (sigma) are used
    to clip the image."""

    assert fits_filepath.endswith('.fits')
    assert os.path.exists(fits_filepath), f"{fits_filepath}"
    if not overwrite and os.path.exists(image_filepath):
        return True

    # Load the fits files
    # from astropy.io.fits import getdata, getheader
    # field, hdr = getdata(fits_filepath), getheader(fits_filepath)
    # print("fits_filepath:", fits_filepath)
    field, hdr = load_fits(fits_filepath)
    field_untouched = copy.deepcopy(field)
    # Load the rms 
    rms, rms_hdr = load_fits(rms_filepath)
    # Discard cutouts that contain NaNs
    if np.any(np.isnan(field)) or np.any(np.isnan(rms)):
        # print( f'Discarding cutout as it contains NaNs ({image_filepath})')
        return False
    if not fixed_cutout_size:
        # np.shape(field)[0] !=
        raise NotImplementedError('fixed_cutout_size=False is not implemented')
        pass

    if clip_image:
        # Clip field in sigma space 
        field = np.clip(field / rms, lower_clip, upper_clip)
    # Ensures that the image color-map is within the clip values for all images
    interval = MinMaxInterval()
    # Normalize values to lie between 0 and 1 and then apply a stretch
    if scaling == 'sqrt':
        stretch = SqrtStretch()
        field = stretch(interval(field))

    # Overlay contours on image
    if not contours is None:

        # Rotate data and rms
        field = rotate(field, -rotation_angle_deg, resize=False, center=None,
                       order=1, mode='constant', cval=0, clip=True,
                       preserve_range=True)
        if field_untouched.dtype.byteorder == '>':
            field_untouched = field_untouched.byteswap().newbyteorder()
        field_untouched = rotate(field_untouched, -rotation_angle_deg, resize=False, center=None,
                                 order=1, mode='constant', cval=0, clip=True,
                                 preserve_range=True)
        if rms.dtype.byteorder == '>':
            rms = rms.byteswap().newbyteorder()
        rms = rotate(rms, -rotation_angle_deg, resize=False, center=None,
                     order=1, mode='constant', cval=0, clip=True,
                     preserve_range=True)

        full_pixel_size = np.shape(field)[0]
        crop_amount = (full_pixel_size - cutout.size_pixels) / 2
        field = crop(field, crop_amount)
        field_untouched = crop(field_untouched, crop_amount)
        rms = crop(rms, crop_amount)

        if not imsize is None:
            # Resize array to required size
            # TODO: this step uses interpolation, might be unavoidable,
            # might be worth looking into alternative or move contour making before the resize
            field = resize_array(field, imsize, imsize)
            field_untouched = resize_array(field_untouched, imsize, imsize)
            rms = resize_array(rms, imsize, imsize)

        # Create borderless png with contours encoded in gb of rgb
        save_contour_image_in_rgb_channels(image_filepath, field,
                                           field_untouched, rms,
                                           imsize, contours=contours)

        """
        dpi=196
        w,h = 600,600
        plt.figure()
        fig = plt.figure()#frameon=False)
        fig.set_size_inches(w/dpi,h/dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(field, cmap='viridis',origin='lower')
        ax.contour(field_untouched/rms, levels=[3,4,5],
                colors=['black','grey','red'], origin='lower', alpha=0.8)
        fig.savefig(image_filepath, dpi=dpi)
        """
    else:
        # Save image
        assert angles_deg == [0], 'Rotation without contours is not implemented'
        plt.imsave(image_filepath, field, cmap='magma', origin='lower')
    plt.close()
    return True


def resize_array(arr, width, height, interpolation=Image.BILINEAR):
    """Resizes numpy array to a specified width and height using specified interpolation"""
    return np.array(Image.fromarray(arr).resize((width, height), interpolation))


def save_contour_image_in_rgb_channels(image_filepath: str, field: np.ndarray,
                                       field_untouched: np.ndarray, rms: np.ndarray,
                                       imsize: int, contours: list = [3, 5]):
    """Takes a stretched and minmaxed intensity field and an untouched intensity field and its
    untouched pybdsf rms-map, resizes it to imsize x imsize and encodes two contourfill-levels in the
    br of the rgb image. Then saves to image_filepath"""

    # Get filled contours
    assert len(contours) == 2, 'exactly two contour levels must be given'
    assert contours[0] < contours[1], 'first contour level must be lower than latter'
    contour1 = np.where(field_untouched / rms > contours[0], 0, 1)
    contour2 = np.where(field_untouched / rms > contours[1], 0, 1)

    # Create rgb channel image: red=field, green=3sigma contour, blue=5sigma contour
    rgb = np.array([contour2, field, contour1])
    # Save image
    plt.imsave(image_filepath, np.rot90(rgb.T))
    # plt.imsave(image_filepath, rgb)
    plt.close()


def convert_fits_to_images(rotation_angles: list, cache_dir: str, fits_file_paths: str,
                           image_names, cutout_list, cutout_names, dataset_dir,
                           image_directory: str, extension: str,
                           cutout_key: str, fixed_cutout_size: int, scaling: str = '',
                           max_component: int = 5,
                           verbose: bool = False, resize: int = None,
                           overwrite: bool = True, clip_image: bool = False, lower_clip: int = 0,
                           upper_clip: int = 0.01,
                           contours=[3, 5], rms_fits_file_paths=None, save_appendix=""):
    """
    takes as input paths to fits radio intensity files
    produces images in the image_directory
    lower and upper_clip are in sigma.
    """
    if not overwrite:
        print("Warning: Overwrite in convert_fits_to_images is set to False.\n",
              "This may lead to an IndexError later on. Set Overwrite to True to avoid this.")
    failed_conversion_path = os.path.join(dataset_dir,
                                          "requires_visual_inspection_nans_in_cutout.txt")
    with open(failed_conversion_path, 'w') as f:
        print('Sources with the following Source_Name require visual inspection:', file=f)
    save_succesful_extraction_index_path = os.path.join(cache_dir,
                                                        f"save_extract_index_{save_appendix}.npy")
    # if (overwrite == False) and os.path.exists(save_succesful_extraction_index_path):
    #    conversion_succeeded_index = np.load(save_succesful_extraction_index_path)
    #    print("Existing image files found. Skipping fits to png conversion.")
    #    return conversion_succeeded_index
    print("Converting fits files to image files.")
    # Convert fits to pngs
    old_image_names = image_names
    conversion_succeeded_index = []
    fail_count = 0
    for index, (angle_deg, im_fits, rms_filepath, image_name, cutout_name, cutout) in enumerate(
            zip(rotation_angles, fits_file_paths, rms_fits_file_paths, image_names, cutout_names, cutout_list)):
        image_path = os.path.join(image_directory, image_name + extension)

        succesful_image = fits_to_image(cutout, im_fits, image_path, clip_image, lower_clip, upper_clip,
                                        fixed_cutout_size,
                                        scaling=scaling, overwrite=overwrite, contours=contours,
                                        rms_filepath=rms_filepath, imsize=resize,
                                        rotation_angle_deg=angle_deg)

        if succesful_image:
            conversion_succeeded_index.append(index)
        else:
            with open(failed_conversion_path, 'a') as f:
                source_name = image_name.split('_')[0]
                print(source_name, file=f)
            fail_count += 1

    print(
        f'{fail_count} out of {len(old_image_names)} fits files failed to convert to {extension}s because of missing fits files or fits files containing NaNs.')
    print(f'Fits files converted into {extension}s and placed in \'{image_directory}\'.\n')
    np.save(save_succesful_extraction_index_path, conversion_succeeded_index)

    return conversion_succeeded_index


def get_mean_pixel_value(image_names, image_directory, extension='.jpg'):
    """
    Opens all cutouts with specified extension in directory_path
    returns the mean pixel value over all files.
    """
    assert isinstance(extension, str)
    image_paths = [os.path.join(image_directory, im_name + extension) for im_name in image_names]
    mean_list = []
    for image_path in image_paths:
        im = Image.open(image_path)
        mean_list.append(np.mean(im))
    return np.mean(mean_list)


def rescale_image(image_filepath, save_path, target_size=600, max_size=1000):
    """
    open image_filepath
    grow image such that min dimension equals target_size 
    but stop prematurely once the max dimension equals max_size
    save image to save_path
    """
    assert image_filepath.endswith('.png') or \
           image_filepath.endswith('.jpg')
    im = Image.open(image_filepath)
    np_size = np.size(im)
    min_dimension = min(np_size[0], np_size[1])
    max_dimension = max(np_size[0], np_size[1])
    scale = target_size / min_dimension
    if (scale * max_dimension > max_size):
        scale = max_size / max_dimension
    im = im.resize((int(scale * np_size[0]), int(scale * np_size[1])), Image.LANCZOS)
    im.save(save_path)


def mkdirs_safe(directory_list):
    """When given a list containing directories,
    checks if these exist, if not creates them."""
    assert isinstance(directory_list, list)
    for directory in directory_list:
        os.makedirs(directory, exist_ok=True)


def create_recursive_directories(prepend_path, current_dir, dictionary):
    """Give it a base, a current_dir to create and a dictionary of stuff yet to create.
    See create_VOC_style_directory_structure for a use case scenario."""
    mkdirs_safe([os.path.join(prepend_path, current_dir)])
    if dictionary[current_dir] == None:
        return
    else:
        for key in dictionary[current_dir].keys():
            create_recursive_directories(os.path.join(prepend_path, current_dir), key,
                                         dictionary[current_dir])


def create_VOC_style_directory_structure(root_directory, dataset_version):
    """
     We will create a directory structure identical to that of the VOC2007
     The root directory is the directory in which the directory named 'LOFARdevkit<dataset_version>'
     will be placed.
     The structure contained will be as follows:
     LOFARdevkit<dataset_version>/
     LOFAR<dataset_version>/
        |-- Annotations/
             |-- *.xml (Annotation files)
        |-- ImageSets/ (train,test,val split directory)
             |-- Main/
                 |-- test.txt (contains all imagenames without extension that should comprise the test set)
                 |-- train.txt (analoguous to the above)
                 |-- trainval.txt
                 |-- val.txt
        |-- JPGImages/
             |-- *.jpg (Image files)
        |-- DemoImages/
             |-- *.jpg (Image files)
    """
    assert isinstance(dataset_version, str)
    directories_to_make = {f'LOFARdevkit{dataset_version}': {f'LOFAR{dataset_version}':
                                                                 {'Annotations': None,
                                                                  'JPEGImages': None,
                                                                  'DemoImages': None,
                                                                  'ImageSets': {'Main': None}}}}
    create_recursive_directories(root_directory, f'LOFARdevkit{dataset_version}', directories_to_make)
    print(f'VOC style directory structure created in \'{root_directory}\'.\n')
    image_directory, demo_directory, annotations_directory, train_test_split_directory = \
        os.path.join(root_directory, f'LOFARdevkit{dataset_version}',
                     f'LOFAR{dataset_version}', 'JPEGImages'), \
        os.path.join(root_directory, f'LOFARdevkit{dataset_version}',
                     f'LOFAR{dataset_version}', 'DemoImages'), \
        os.path.join(root_directory, f'LOFARdevkit{dataset_version}',
                     f'LOFAR{dataset_version}', 'Annotations'), \
        os.path.join(root_directory, f'LOFARdevkit{dataset_version}',
                     f'LOFAR{dataset_version}', 'ImageSets', 'Main')
    return annotations_directory, image_directory, demo_directory, train_test_split_directory


def create_CLARAN_style_directory_structure(root_directory):
    """
     We will create a directory structure identical to that of the CLARAN
     The root directory is the directory in which the directory named 'RGZdevkit' will be placed.
     The structure contained will be as follows:
     RGZdevkit2017/
     RGZ2017/
        |-- Annotations/
             |-- *.xml (Annotation files)
        |-- ImageSets/ (train,test,val split directory)
             |-- Main/
                 |-- test.txt (contains all imagenames without extension that should comprise the test set)
                 |-- train.txt (analoguous to the above)
                 |-- trainval.txt
                 |-- val.txt
        |-- PNGImages/
             |-- *.png (Image files)
        |-- DemoImages/
             |-- *.jpg (Image files)
    """
    directories_to_make = {'RGZdevkit2017': {'RGZ2017':
                                                 {'Annotations': None,
                                                  'PNGImages': None,
                                                  'DemoImages': None,
                                                  'ImageSets': {'Main': None}}}}
    create_recursive_directories(root_directory, 'RGZdevkit2017', directories_to_make)
    print(f'CLARAN style directory structure created in \'{root_directory}\'.\n')
    image_directory, demo_directory, annotations_directory, train_test_split_directory = \
        os.path.join(root_directory, 'RGZdevkit2017', 'RGZ2017', 'PNGImages'), \
        os.path.join(root_directory, 'RGZdevkit2017', 'RGZ2017', 'DemoImages'), \
        os.path.join(root_directory, 'RGZdevkit2017', 'RGZ2017', 'Annotations'), \
        os.path.join(root_directory, 'RGZdevkit2017', 'RGZ2017', 'ImageSets', 'Main')
    return annotations_directory, image_directory, demo_directory, train_test_split_directory


def create_COCO_style_directory_structure(root_directory, suffix=''):
    """
     We will create a directory structure identical to that of the CLARAN
     The root directory is the directory in which the directory named 'RGZdevkit' will be placed.
     The structure contained will be as follows:
     LGZ_COCOstyle{suffix}/
        |-- Annotations/
             |-- *.json (Annotation files)
        |-- all/ (train,test,val split directory)
             |-- *.png (Image files)
        |-- train/ (train,test,val split directory)
             |-- *.png (Image files)
        |-- val/ (train,test,val split directory)
             |-- *.png (Image files)
        |-- test/ (train,test,val split directory)
             |-- *.png (Image files)
    """
    directories_to_make = {f'LGZ_COCOstyle{suffix}':
                               {'annotations': None,
                                'all': None,
                                'train': None,
                                'val': None,
                                'test': None}}
    create_recursive_directories(root_directory, f'LGZ_COCOstyle{suffix}', directories_to_make)
    print(f'COCO style directory structure created in \'{root_directory}\'.\n')
    all_directory, train_directory, val_directory, test_directory, annotations_directory = \
        os.path.join(root_directory, f'LGZ_COCOstyle{suffix}', 'all'), \
        os.path.join(root_directory, f'LGZ_COCOstyle{suffix}', 'train'), \
        os.path.join(root_directory, f'LGZ_COCOstyle{suffix}', 'val'), \
        os.path.join(root_directory, f'LGZ_COCOstyle{suffix}', 'test'), \
        os.path.join(root_directory, f'LGZ_COCOstyle{suffix}', 'annotations')
    return all_directory, train_directory, val_directory, test_directory, annotations_directory


def populate_VOC_structure_with_train_test_val_split(train_test_split_directory,
                                                     train_test_val_split, image_names, image_directory, demo_directory,
                                                     image_extension):
    """Populate folder with text files that contain imagenames without extension,
    to be used for train, test and val respectively."""
    # Create textfile paths
    test_path, train_path, trainval_path, val_path = \
        [os.path.join(train_test_split_directory, file_name + '.txt')
         for file_name in ['test', 'train', 'trainval', 'val']]
    # Append line break to image names
    image_names = np.array(list(map(lambda x: x + '\n', image_names)))

    # Shuffle and split image names
    l = np.array(range(len(image_names)))
    np.random.shuffle(l)
    tr, te, va = map(int, np.array(train_test_val_split) * len(image_names))
    train, test, val = l[:tr], l[tr:tr + te], l[tr + te:]
    trainval = np.concatenate([train, val])

    # Create and fill each textfile with sorted names
    for path, splitted_name in zip([test_path, train_path, trainval_path, val_path],
                                   [test, train, trainval, val]):
        with open(path, 'w') as text_file:
            text_file.writelines(sorted(image_names[splitted_name]))

    # Create and fill directory with test images
    test_image_source_paths = list(map(lambda x: os.path.join(image_directory,
                                                              x.replace('\n', image_extension)), image_names[test]))
    test_image_dest_paths = list(map(lambda x: os.path.join(demo_directory,
                                                            x.replace('\n', image_extension)), image_names[test]))
    # [copyfile(src, dest) for src, dest in zip(test_image_source_paths, test_image_dest_paths)]
    for src, dest in zip(test_image_source_paths, test_image_dest_paths):
        with open(src, 'rb') as fin:
            with open(dest, 'wb') as fout:
                copyfileobj(fin, fout, 128 * 1024)
    print(
        f'Train {len(train)}, test {len(test)}, val {len(val)} split made and placed in \'{train_test_split_directory}\'.')


def split_image_names_and_objects_standardized(cache_dir, cutout_list, image_names: list):
    """splits image names and objects into train/test/val split"""
    # Load sourcelist
    sourcelist_path = os.environ['LOTSS_RAW_CATALOGUE']
    sourcelist = pd.read_hdf(sourcelist_path)
    # Get fieldnames
    field_names = sorted(list(set(sourcelist.Mosaic_ID)))
    np.random.seed(42)
    np.random.shuffle(field_names)
    # split fields
    # Train, val, test split
    # (expressed in terms of their boundaries)
    field_split = [0, 38, 48, 58]
    train_fields = field_names[field_split[0]:field_split[1]]
    val_fields = field_names[field_split[1]:field_split[2]]
    test_fields = field_names[field_split[2]:field_split[3]]
    print("{len(train_fields)} pointings in training data set:")
    print(train_fields)
    print("{len(val_fields)} pointings in validation data set:")
    print(val_fields)
    print("{len(test_fields)} pointings in test data set:")
    print(test_fields)

    # Get sources in each set
    train_set_names = sourcelist[sourcelist.Mosaic_ID.isin(train_fields)].Source_Name.values
    val_set_names = sourcelist[sourcelist.Mosaic_ID.isin(val_fields)].Source_Name.values
    test_set_names = sourcelist[sourcelist.Mosaic_ID.isin(test_fields)].Source_Name.values
    print("Number of test pointings in this particular dataset creation:",
          len(list(set(sourcelist.Mosaic_ID.isin(test_fields)))))
    print("Namely:", list(set(sourcelist.Mosaic_ID.isin(test_fields))))

    # make sure we keep the rotated versions in the same split as their original
    # name_to_index_dict = {n:i for i,n in enumerate(image_names)}
    stripped_image_names = [im.split('_')[0] for im in image_names]
    splitted_index_list = []

    # Iterate over predefined train val test list
    for suffix, ijklijst in zip(['train', 'val', 'test'],
                                [list(train_set_names), list(val_set_names), list(test_set_names)]):

        new_index_list = []
        print(f'Our full {suffix} split contains {len(ijklijst)} sources.')
        print(f'(That includes all DR1 area sources, also the faint, small ones)')

        for i, (sn, n) in enumerate(zip(stripped_image_names, image_names)):
            if suffix == 'train':
                if sn in ijklijst:
                    new_index_list.append(i)
            else:
                if n.endswith('_rotated0deg') and (sn in ijklijst):
                    new_index_list.append(i)
        print(f'In this particular dataset we pick {len(new_index_list)} {suffix} sources.')
        if len(new_index_list) > 0:
            multis = sum([len(c.get_related_comp()[0]) > 0 for c in cutout_list[new_index_list]])
            print(f"The percentage multi to single comp. sources is {multis / len(new_index_list):.2%}")

        splitted_index_list.append(new_index_list)

    return splitted_index_list


def split_image_names_and_objects(split: list, image_names: list,
                                  cutout_list, multi_list, single_rot, multi_rot, random_seed: int = 42):
    """splits image names and objects into train/test/val split"""
    # make sure we keep the rotated versions in the same split as their original
    name_to_index_dict = {n: i for i, n in enumerate(image_names)}
    stripped_image_names = [im.split('_')[0] for im in image_names]
    unique_image_names = np.array(sorted(set(stripped_image_names), key=stripped_image_names.index))
    lsplit = len(split)
    l = np.array(range(len(unique_image_names)))
    np.random.seed(random_seed)
    np.random.shuffle(l)
    splitted_image_names = []
    splitted_cutout_list = []
    single_rot = [0] + single_rot
    multi_rot = [0] + multi_rot
    # calculate index borders
    split.insert(0, 0)
    l_indices = []
    l_index_borders = list(map(int, np.array(split) * len(unique_image_names)))
    # print("Split borders:", l_index_borders)
    # get a list of indices for test/train/val
    for i in range(lsplit):
        l_indices.append(l[l_index_borders[i]:l_index_borders[i + 1]])
    # apply indices to image and objects lists to get splits
    for l_ind in l_indices:
        split_names = []
        for n in unique_image_names[l_ind]:
            if n in multi_list:
                for m in multi_rot:
                    split_names.append(f'{n}_radio_DR2_rotated{m}deg')
            else:
                for s in single_rot:
                    split_names.append(f'{n}_radio_DR2_rotated{s}deg')

        split_idx = [name_to_index_dict[n] for n in split_names]

        splitted_image_names.append(image_names[split_idx])
        splitted_cutout_list.append(cutout_list[split_idx])
    assert len(flatten(splitted_image_names)) == len(image_names)

    return splitted_image_names, splitted_cutout_list


def populate_coco_structure(image_source_directory: str, image_dest_directory: list,
                            image_names: np.ndarray, image_extension: str):
    """Copy image to over train/test/val directories
    to be used for train, test and val respectively."""

    # Empty dest directories if they already exist
    # but first check that it contains only pngs

    os.makedirs(image_dest_directory, exist_ok=True)
    for f in os.listdir(image_dest_directory):
        assert f.endswith('.png'), 'Directory should only contain images.'
    for f in os.listdir(image_dest_directory):
        os.remove(os.path.join(image_dest_directory, f))

    # Copy images to dest directory

    image_source_paths = list(map(lambda x: os.path.join(image_source_directory,
                                                         x + image_extension), image_names))
    image_dest_paths = list(map(lambda x: os.path.join(image_dest_directory,
                                                       x + image_extension), image_names))
    # [copyfile(src, dest) for src, dest in zip(image_source_paths, image_dest_paths)]
    for src, dest in zip(image_source_paths, image_dest_paths):
        with open(src, 'rb') as fin:
            with open(dest, 'wb') as fout:
                copyfileobj(fin, fout, 128 * 1024)
    print(f'{len(image_source_paths)} image cutouts copied from {image_source_directory} to {image_dest_directory}.')


def populate_VOC_structure_with_train_test_val_split_claran_version(train_test_split_directory,
                                                                    train_test_split, image_names, image_directory,
                                                                    demo_directory, image_extension):
    """Populate folder with text files that contain imagenames without extension,
    to be used for train, test and val respectively."""
    # Create textfile paths
    test_path, train_path = \
        [os.path.join(train_test_split_directory, file_name + '.txt')
         for file_name in ['testD3', 'trainD3']]
    # Append line break to image names
    image_names = np.array(list(map(lambda x: x + '\n', image_names)))

    # Shuffle and split image names
    l = np.array(range(len(image_names)))
    np.random.shuffle(l)
    tr, te = map(int, np.array(train_test_split) * len(image_names))
    train, test = l[:tr], l[tr:]

    # Create and fill each textfile with sorted names
    for path, splitted_name in zip([test_path, train_path],
                                   [test, train]):
        with open(path, 'w') as text_file:
            text_file.writelines(sorted(image_names[splitted_name]))

    # Create and fill directory with test images
    test_image_source_paths = list(map(lambda x: os.path.join(image_directory,
                                                              x.replace('\n', image_extension)), image_names[test]))
    test_image_dest_paths = list(map(lambda x: os.path.join(demo_directory,
                                                            x.replace('\n', image_extension)), image_names[test]))
    # [copyfile(src, dest) for src, dest in zip(test_image_source_paths, test_image_dest_paths)]
    for src, dest in zip(test_image_source_paths, test_image_dest_paths):
        with open(src, 'rb') as fin:
            with open(dest, 'wb') as fout:
                copyfileobj(fin, fout, 128 * 1024)
    print(f'Train {len(train)}, test {len(test)} split made and placed in \'{train_test_split_directory}\'.')


def convert_corners_to_center(xmin, ymin, xmax, ymax):
    """Enter x,y coordinates of bottom left and top right coordinates,
    returns center position and width and height of the rectangle."""
    width = xmax - xmin
    height = ymax - ymin
    return xmin, ymin, width, height


def convert_center_to_corners(x, y, width, height):
    """Enter x,y coordinates of center of rectangle and its width and height
    returns bottom left and top right coordinates of the rectangle."""
    xmax = x + width
    ymax = y + height
    return x, y, xmax, ymax


def get_all_combinations_containing_first_element(idx_list):
    """Assumes to receive a list of indexes with no duplicates"""
    # Get all combinations
    if len(idx_list) > 1:
        idx_no_zero = idx_list[1:]
    else:
        idx_no_zero = idx_list
    list_of_combinations = [list(itertools.combinations(idx_no_zero, t)) for t in range(1, len(idx_list) + 1)]
    # Flatten list of lists
    combinations_no_zeros = [item for sublist in list_of_combinations for item in sublist]
    # Add first element to each tuple and just the first element itself!
    combinations = [[0]] + [[0] + list(c) for c in combinations_no_zeros]
    return combinations


def translate_combinations_to_single_bboxes(single_bboxes, combinations):
    """Receive a list of combinations. Return a single bounding box for each."""
    combined_bboxes = []
    for combination in combinations:
        relevant_bboxes = np.array(single_bboxes)[combination].T
        combined_bboxes.append((min(relevant_bboxes[0]), min(relevant_bboxes[1]),
                                max(relevant_bboxes[2]), max(relevant_bboxes[3])))
    assert len(combined_bboxes) == len(combinations)
    return combined_bboxes


def package_combination_bboxes_into_annotations(combined_bboxes):
    """Package a bbox into a detectron annotation"""
    annotations = []
    for combined_bbox in combined_bboxes:
        box = {'bbox': list(combined_bbox),
               'bbox_mode': BoxMode.XYXY_ABS,
               'category_id': 0,
               'iscrowd': 0}
        annotations.append(box)
    return annotations


def get_lofar_dicts_precomputed_bboxes(annotations, debug=False):
    boxes = [b['bbox'] for b in annotations]

    # get all combinations of the index-list of boxes
    idx_list = list(range(len(boxes)))
    if debug:
        print('num of boxes', len(boxes))
        print(boxes)
    if len(idx_list) > 12:
        # print(f'Too many bboxes in fov: {len(idx_list)}')
        # print('Skipping!')
        return None, None
    combinations = get_all_combinations_containing_first_element(idx_list)

    # Translate combinations to single boxes
    combined_bboxes = translate_combinations_to_single_bboxes(boxes, combinations)
    # combination_distribution.append(len(combined_bboxes))

    # Filter out duplicates. set! :D
    proposal_boxes = np.array(list(set(combined_bboxes)))
    if debug:
        print(
            f'number of combinations {len(combinations)} num of boxes {len(combined_bboxes)} unique {len(proposal_boxes)}')
    # unique_combination_distribution.append(len(combined_bboxes))

    # Create objectness logits 
    # should be a np array of floats
    objectness_logits = np.ones(len(proposal_boxes))

    return proposal_boxes, objectness_logits


def get_gt_bbox(annotations, cutout):
    """Given all bboxes, create a combined gt box that includes the minmax of all
    related radio components"""
    relevant_comps = [cutout.c_source] + [c for c in cutout.other_components if c.related]
    # Get bboxes of central and related components
    xmins = [c.xmin for c in relevant_comps]
    ymins = [c.ymin for c in relevant_comps]
    xmaxs = [c.xmax for c in relevant_comps]
    ymaxs = [c.ymax for c in relevant_comps]

    gt_obj = {
        "bbox": [np.min(xmins), np.min(ymins), np.max(xmaxs), np.max(ymaxs)],
        "bbox_mode": None,
        "category_id": 0,
        "iscrowd": 0
    }
    return gt_obj


def create_dict_from_images_and_annotations_coco_version(image_names, fits_filepaths,
                                                         rms_filepaths, cutout_list, extension,
                                                         image_dir: str = 'images', image_destination_dir=None,
                                                         annotations_dir='',
                                                         json_dir='', json_name='json_data.pkl', imsize=None,
                                                         debug=False, remove_unresolved=False,
                                                         precomputed_bboxes_enabled: bool = False, plot_optical=False,
                                                         save_optical=True,
                                                         training_mode=True, debug_path=None, term=None,
                                                         dr1_dr2_comparison_only=False):
    """
    :param image_dir: image directory
    :param image_names: image names
    :param image_objects: image object containing (int xmin, int ymin, int xmax, int ymax, string class_name)
    :return:
    """

    assert (len(image_names) == len(cutout_list))
    skipped_path = os.path.join(annotations_dir, f"requires_visual_inspection_{term}.txt")
    skipped_debug_path = os.path.join(annotations_dir, f"open_skipped_{term}.txt")
    with open(skipped_path, 'w') as f:
        print('Sources with the following Source_Name require visual inspection:', file=f)
    with open(skipped_debug_path, 'w') as f:
        print('Sources with the following Source_Name require visual inspection:', file=f)
    # List to store single dict for each image
    dataset_dicts = []
    depth = 0

    # Remove pngs inside prepro_plot directory
    """
    if not debug_path is None:
        prepro_plot_dir = os.path.join(debug_path,'boxes')
        for f in os.listdir(prepro_plot_dir):
            assert f.endswith('.png'), 'Directory should only contain images.'
        for f in os.listdir(prepro_plot_dir):
            os.remove(os.path.join(prepro_plot_dir,f))

    try:
        vac = pd.read_hdf('/home/rafael/data/mostertrij/data/catalogues/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2.h5')
        comp_cat = pd.read_hdf('/home/rafael/data/mostertrij/data/catalogues/LOFAR_HBA_T1_DR1_merge_ID_v1.2.comp.h5')
        comp_dict = {s: idx for s, idx in zip(comp_cat.Component_Name.values, comp_cat.Source_Name.values)}
        vac_dict = {s: idx for s, idx in zip(vac.Source_Name.values, vac.index.values)}
    except:
        print("Could not load value added cat")
        pass
    """
    if not debug_path is None:
        os.makedirs(os.path.join(debug_path, 'boxes'), exist_ok=True)

    # Iterate over all cutouts and their objects (which contain bounding boxes and class labels)
    for i, (image_name, fits_filepath, rms_filepath, cutout) in enumerate(zip(image_names,
                                                                              fits_filepaths, rms_filepaths,
                                                                              cutout_list)):

        # Get image dimensions and insert them in a python dict
        image_name = image_name + extension
        image_filename = os.path.join(image_dir, image_name)
        image_dest_filename = os.path.join(image_destination_dir, image_name)
        if not imsize is None:
            width, height = imsize, imsize
        elif extension == '.png' or extension == '.jpg':
            # width, height, depth = im.size
            width, height = Image.open(image_filename).size
        elif extension == '.npy':
            im = np.load(image_filename, mmap_mode='r')  # mmap_mode might allow faster read
            width, height, depth = np.shape(im)
        else:
            raise ValueError('Image file format must either be .png, .jpg, .jpeg or .npy')
        size_value = {'width': width, 'height': height, 'depth': depth}

        record = {}

        record["file_name"] = image_dest_filename.replace('/home/rafael', '')
        record["image_id"] = i
        record["height"] = height
        record["width"] = width
        record["focussed_comp"] = deepcopy(cutout.get_focussed_comp())
        record["related_comp"] = deepcopy(cutout.get_related_comp())
        record["unrelated_comp"] = deepcopy(cutout.get_unrelated_comp())
        record["related_unresolved"] = deepcopy(cutout.get_related_unresolved())
        record["unrelated_unresolved"] = deepcopy(cutout.get_unrelated_unresolved())
        record["wide_focus"] = deepcopy(cutout.wide_focus)

        if training_mode and save_optical:
            record["optical_sources"] = deepcopy(cutout.optical_sources)
        record["related_compnames"] = deepcopy([c.sname for c in cutout.other_components if 
            c.related])
        record["unrelated_compnames"] = deepcopy([c.sname for c in cutout.other_components if not
            c.related])

        """
        ################# Flip x-axis
        record["focussed_comp"][1] = height - record["focussed_comp"][1]
        if len(record["related_comp"])>1:
            record["related_comp"][1] = height - record["related_comp"][1]
        else:
            record["related_comp"] = [[],[]]
        if len(record["unrelated_comp"])>1:
            record["unrelated_comp"][1] = height - record["unrelated_comp"][1]
        else:
            record["unrelated_comp"] = [[],[]]
        if training_mode:
            if len(record["optical_sources"]["y"])>1:
                record["optical_sources"]["y"] = height - record["optical_sources"]["y"]
            else:
                record["related_comp"]["y"] = [[],[]]
        """

        record["RA"] = cutout.c_source.ra
        record["DEC"] = cutout.c_source.dec

        #####DEBUG-print#############################
        # if "ILTJ104706.09+534417.1_radio" in image_name:

        # Insert bounding boxes and their corresponding classes
        objs = []
        cache_list = []

        for i_s, s in enumerate([cutout.c_source] + cutout.other_components):
            # Source is unresolved, bbox not added
            if remove_unresolved and i_s > 0 and s.unresolved:
                continue
            xmin, ymin, xmax, ymax = deepcopy(s.xmin), deepcopy(s.ymin), deepcopy(s.xmax), \
                                     deepcopy(s.ymax)
            # n_comp, n_peak = s.n_comp, s.n_peak
            tup = (xmin, ymin, xmax, ymax)
            if tup in cache_list:
                "Duplicate encountered."
                continue
            cache_list.append(tup)
            # Temporary fix below
            if xmax <= xmin:
                xmin_old = xmin
                xmin = xmax
                xmin_old = xmin
                xmin = xmax
                xmax = xmin_old
            if ymax <= ymin:
                ymin_old = ymin
                ymin = ymax
                ymin_old = ymin
                ymin = ymax
                ymax = ymin_old
            # assert xmax > xmin, f"{xmax} > {xmin}"
            # assert ymax > ymin, f"{ymax} > {ymin}"
            assert isinstance(xmin, (int, float))
            assert isinstance(ymin, (int, float))
            assert isinstance(xmax, (int, float))
            assert isinstance(ymax, (int, float))
            # assert isinstance(n_comp, (int,float))
            # assert isinstance(n_peak, (int,float))

            """
            ################# Flip x-axis
            old_ymax = ymax*cutout.scale_factor
            old_ymin = ymin*cutout.scale_factor
            ymin = height - old_ymax
            ymax = height - old_ymin
            xmin, xmax = cutout.scale_factor*xmin, cutout.scale_factor*xmax
            ################# Flip x-axis
            old_ymax = ymax*cutout.scale_factor
            old_ymin = ymin*cutout.scale_factor
            ymin = height - old_ymax
            ymax = height - old_ymin
            #ymin, ymax = cutout.scale_factor*ymin, cutout.scale_factor*ymax
            xmin, xmax = cutout.scale_factor*xmin, cutout.scale_factor*xmax
            xmin, ymin, xmax, ymax, n_comp, n_peak = list(map(round,
                [xmin,ymin,xmax,ymax,n_comp,n_peak]))
            x, y, region_width, region_height = convert_corners_to_center(xmin, ymin, xmax, ymax)

            ################ COMPATIBILITY HACK: CHANGING n_comp CLASS TO STRINGS
            n_comp = str(n_comp)


            # Edit standard object for XML
            a = copy.deepcopy(default_object)
            a['name'] = str(n_comp)#+'_'+str(n_peak)
            a['n_comp'] = n_comp
            a['n_peak'] = n_peak
            a['bndbox'] = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
            annotation_list.append(('object', a))
            """

            obj = {
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_mode": None,
                # "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)

        # gt_obj =  get_gt_bbox(cutout)
        record["annotations"] = [{
            "bbox": [cutout.gt_xmin, cutout.gt_ymin, cutout.gt_xmax, cutout.gt_ymax],
            "bbox_mode": None,
            "category_id": 0,
            "iscrowd": 0
        }]

        # Add precomputed bounding boxes if enabled
        debug_prop = False
        if precomputed_bboxes_enabled:
            if debug_prop:
                print(image_name)
            proposal_boxes, objectness_logits = get_lofar_dicts_precomputed_bboxes(objs, debug=debug_prop)
            if proposal_boxes is None:
                with open(skipped_path, 'a') as f:
                    i = image_name.split('_')[0]
                    print(i, file=f)
                # with open(skipped_debug_path,'a') as f:
                # print('xdg-open', image_dest_filename,file=f)
                continue
            record["proposal_boxes"] = proposal_boxes
            record["proposal_objectness_logits"] = objectness_logits
            # record["annotations"] = [objs[0]]

            # Plot debug image
            # debug=False
            if debug:
                if not training_mode and i > 500:
                    # In inference mode we expect lots more cutouts than in training mode. Making a
                    # debug plot for all cutouts in inference mode is thus too ambitious
                    pass
                else:
                    if i == 21:
                        relus = [c.get_related_comp() for c in cutout_list]
                    if dr1_dr2_comparison_only:

                        plot_DR1_DR2_comparison(
                            image_dest_filename.replace('/train/', '/all/').replace('/test/', '/all/').replace('/val/',
                                                                                                               '/all/'),
                            debug_path,
                            [cutout.gt_xmin, cutout.gt_ymin, cutout.gt_xmax, cutout.gt_ymax],
                            proposal_boxes,
                            record["focussed_comp"],
                            record["related_comp"],
                            record["unrelated_comp"],
                            cutout, remove_unresolved, fits_filepath, rms_filepath, training_mode=training_mode,
                            plot_optical=plot_optical, dataset=term)

                    else:

                        plot_prepro(
                            image_dest_filename.replace('/train/', '/all/').replace('/test/', '/all/').replace('/val/',
                                                                                                               '/all/'),
                            debug_path,
                            [cutout.gt_xmin, cutout.gt_ymin, cutout.gt_xmax, cutout.gt_ymax],
                            proposal_boxes,
                            record["focussed_comp"],
                            record["related_comp"],
                            record["unrelated_comp"],
                            cutout, remove_unresolved, fits_filepath, rms_filepath, training_mode=training_mode,
                            plot_optical=plot_optical, dataset=term)

        else:
            if debug:
                if not training_mode and i > 500:
                    # In inference mode we expect lots more cutouts than in training mode. Making a
                    # debug plot for all cutouts in inference mode is thus too ambitious
                    pass
                else:
                    plot_prepro(
                        image_dest_filename.replace('/train/', '/all/').replace('/test/', '/all/').replace('/val/',
                                                                                                           '/all/'),
                        debug_path,
                        [cutout.gt_xmin, cutout.gt_ymin, cutout.gt_xmax, cutout.gt_ymax],
                        None,
                        record["focussed_comp"],
                        record["related_comp"],
                        record["unrelated_comp"],
                        cutout, remove_unresolved, fits_filepath, rms_filepath, training_mode=training_mode,
                        plot_optical=plot_optical, dataset=term)

        dataset_dicts.append(record)
    # Write all image dictionaries to file as one json 
    json_path = os.path.join(json_dir, json_name)
    with open(json_path, "wb") as outfile:
        pickle.dump(dataset_dicts, outfile)
        # json.dump(dataset_dicts, outfile, indent=4)
    print(f'COCO annotation file created in \'{json_dir}\'.\n')


def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:starty + cropy, startx:startx + cropx]


def plot_DR1_DR2_comparison(file_path, debug_path, gt_bbox, proposal_boxes, focus_l, rel_l, unrel_l,
                            cutout, remove_unresolved, fits_filepath, rms_filepath,
                            plot_optical=False, training_mode=True, dataset=None,
                            lower_clip=1, upper_clip=30, plot_fits=True):
    """Show debug"""

    if not file_path.endswith("rotated0deg.png"):
        return
    unlikely_names = np.load('/data2/mostertrij/data/frcnn_images/unlikely_names.npy')
    if not cutout.c_source.sname in unlikely_names:
        return
    plt.rcParams.update({'font.size': 16})

    # Find DR1 mosaic linked to the source
    dr1cat = pd.read_hdf('/data2/mostertrij/data/catalogues/LOFAR_HBA_T1_DR1_catalog_v1.0.srl.h5')
    dr1cat = dr1cat.set_index('Source_Name')
    mosaic = dr1cat.loc[cutout.c_source.sname].Mosaic_ID
    dr1_fits_filepath = f'/data2/mostertrij/data/LoTSS_DR1/{mosaic}/mosaic-blanked.fits'
    dr1_rms_filepath = f'/data2/mostertrij/data/LoTSS_DR1/{mosaic}/mosaic.rms.fits'
    assert os.path.exists(dr1_fits_filepath), dr1_fits_filepath
    assert os.path.exists(dr1_rms_filepath), dr1_rms_filepath

    # Open image 
    # im = imread(file_path)

    # Radio intensity
    c = SkyCoord(ra=cutout.c_source.ra, dec=cutout.c_source.dec, unit='deg', frame='icrs')

    ############ Plot DR1
    # Load the fits files
    field, hdr = load_fits(dr1_fits_filepath)
    ss = 200
    hdu_crop = Cutout2D(field, c, (ss, ss), wcs=WCS(hdr, naxis=2), copy=True)
    field = hdu_crop.data
    # Load the rms 
    rms, rms_hdr = load_fits(dr1_rms_filepath, dimensions_normal=False)
    rms_crop = Cutout2D(rms, c, (ss, ss), wcs=WCS(rms_hdr, naxis=2), copy=True)
    rms = rms_crop.data
    print(f"Dimensions of DR1 cutout {np.shape(field)} and rms cutout {np.shape(rms)}")
    # Clip field in sigma space 
    field = np.clip(field / rms, lower_clip, 1e9)
    # Ensures that the image color-map is within the clip values for all images
    interval = MinMaxInterval()
    # Normalize values to lie between 0 and 1 and then apply a stretch
    stretch = SqrtStretch()
    field = stretch(interval(field))
    # plot
    # Plot figure 
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': WCS(hdr, naxis=2)})
    ax1.imshow(field, origin='lower')
    ax1.set_xlabel('RA')
    ax1.set_ylabel('DEC')
    ax1.set_title(f"Human expert association \nusing LoTSS-DR1")

    """
    # Bounding box
    if training_mode:
        central_bbox_color = 'k'
        central_linestyle = 'solid'
    else:
        central_bbox_color = 'gray'
        central_linestyle = 'dashed'
    ax1.plot([gt_bbox[0],gt_bbox[2],gt_bbox[2],gt_bbox[0],gt_bbox[0]],
            np.array([gt_bbox[1],gt_bbox[1],gt_bbox[3],gt_bbox[3],gt_bbox[1]]),
            color=central_bbox_color,linestyle=central_linestyle)

    # Plot component locations
    if training_mode:
        focus_color = 'red'

    else:
        focus_color = 'lime'
    ax1.plot(focus_l[0],focus_l[1],marker='s', markersize=10,color=focus_color)
    for unresolved, x,y in zip(cutout.get_related_unresolved(), rel_l[0],rel_l[1]):
        if remove_unresolved and unresolved:
            marker='x'
        else:
            marker='.'
        ax1.plot(x,y,marker=marker,markersize=9,color='r')
    for unresolved, x,y in zip(cutout.get_unrelated_unresolved(),unrel_l[0],unrel_l[1]):
        if remove_unresolved and unresolved:
            marker='x'
        else:
            marker='.'
        ax1.plot(x,y,marker=marker,markersize=9,color='lime')
    """

    # Plot optical
    if plot_optical:
        if 'w1Mag' in cutout.optical_sources.keys():
            okey = 'w1Mag'
        else:
            # Plot legacy
            okey = 'MAG_R'
            # print("Legacy is chosen")
        for w, x, y in zip(cutout.optical_sources[okey],
                           cutout.optical_sources['x'], cutout.optical_sources['y']):
            if not np.isnan(w):
                marker = '+'
                # print("plotting legacy")
                # ax1.text(x,y,f"{w:.2f}",color='y')
                ax1.plot(x, y, marker=marker, markersize=9, color='y')
            # else:
            #    marker='x'
            # ax1.plot(x,y,marker=marker,markersize=9,color='y')

    # Plot sidepanel with just LoTSS-DR2 1-10sigma Stokes-I
    if plot_fits:
        # fits_filepath = os.path.join(image_directory, image_name + extension)
        assert fits_filepath.endswith('.fits')
        assert os.path.exists(fits_filepath)
        # Load the fits files
        field, hdr = load_fits(fits_filepath)
        # Load the rms 
        rms, rms_hdr = load_fits(rms_filepath)
        # Clip field in sigma space 
        field = np.clip(field / rms, lower_clip, 1e9)
        # Ensures that the image color-map is within the clip values for all images
        interval = MinMaxInterval()
        # Normalize values to lie between 0 and 1 and then apply a stretch
        stretch = SqrtStretch()
        field = stretch(interval(field))
        field = crop_center(field, ss, ss)
        print(f"Dimensions of DR2 cutout {np.shape(field)} and rms cutout {np.shape(rms)}")
        # plot
        ax2.imshow(field, origin='lower')
        # ax2.axes.xaxis.set_visible(False)
        # ax2.axes.yaxis.set_visible(False)
        ax2.set_xlabel('RA')
        ax2.set_ylabel(' ')
        ax2.set_title(f"Manually corrected association \nusing LoTSS-DR2")

    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)
    # ax1.set_xlim(0,200)
    # ax1.set_ylim(200,0)
    # Save and close plot
    # dest = os.path.join(debug_path,'boxes',file_path.split('/')[-1].replace('.png','.pdf'))
    if training_mode and plot_optical:
        if dataset is None:
            dest = os.path.join(debug_path, 'DR1_DR2_comparison', file_path.split('/')[-1])
        else:
            dest = os.path.join(debug_path, 'DR1_DR2_comparison', dataset, file_path.split('/')[-1])

    else:
        dest = os.path.join(debug_path, 'DR1_DR2_comparison', file_path.split('/')[-1])
    # print("Save debug plot at:", dest)
    # plt.show()
    plt.savefig(dest)
    plt.close()


def plot_prepro(file_path, debug_path, gt_bbox, proposal_boxes, focus_l, rel_l, unrel_l,
                cutout, remove_unresolved, fits_filepath, rms_filepath,
                plot_optical=False, training_mode=True, dataset=None,
                lower_clip=1, upper_clip=30, plot_fits=True):
    """Show debug"""

    if plot_optical and not file_path.endswith("rotated0deg.png"):
        return
    # Open image 
    # print(file_path)
    im = imread(file_path)

    # Plot figure 
    if plot_fits:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    else:
        f, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    # Radio intensity
    im[:, :, 1] = 255 - im[:, :, 1]
    ax1.imshow(im)
    c = SkyCoord(ra=cutout.c_source.ra, dec=cutout.c_source.dec, unit='deg', frame='icrs')
    # plt.title(c.to_string('hmsdms'), fontsize=16)
    ax1.set_title(f"RA {c.ra.value:.3f}; DEC {c.dec.value:.3f}", fontsize=16)

    if not proposal_boxes is None:
        for bbox in proposal_boxes:
            # Plot all proposal bounding boxes
            ax1.plot([bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]],
                     np.array([bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]]), linestyle='dashed', alpha=0.8)

    # Bounding box
    if training_mode:
        central_bbox_color = 'k'
        central_linestyle = 'solid'
    else:
        central_bbox_color = 'gray'
        central_linestyle = 'dashed'
    ax1.plot([gt_bbox[0], gt_bbox[2], gt_bbox[2], gt_bbox[0], gt_bbox[0]],
             np.array([gt_bbox[1], gt_bbox[1], gt_bbox[3], gt_bbox[3], gt_bbox[1]]),
             color=central_bbox_color, linestyle=central_linestyle)

    # Plot component locations
    if training_mode:
        focus_color = 'red'

    else:
        focus_color = 'lime'
    ax1.plot(focus_l[0], focus_l[1], marker='s', markersize=10, color=focus_color)
    for unresolved, x, y in zip(cutout.get_related_unresolved(), rel_l[0], rel_l[1]):
        if remove_unresolved and unresolved:
            marker = 'x'
        else:
            marker = '.'
        ax1.plot(x, y, marker=marker, markersize=9, color='r')
    for unresolved, x, y in zip(cutout.get_unrelated_unresolved(), unrel_l[0], unrel_l[1]):
        if remove_unresolved and unresolved:
            marker = 'x'
        else:
            marker = '.'
        ax1.plot(x, y, marker=marker, markersize=9, color='lime')

    # Plot optical
    if plot_optical:
        if 'w1Mag' in cutout.optical_sources.keys():
            okey = 'w1Mag'
        else:
            # Plot legacy
            okey = 'MAG_R'
            # print("Legacy is chosen")
        for w, x, y in zip(cutout.optical_sources[okey],
                           cutout.optical_sources['x'], cutout.optical_sources['y']):
            if not np.isnan(w):
                marker = '+'
                # print("plotting legacy")
                # ax1.text(x,y,f"{w:.2f}",color='y')
                ax1.plot(x, y, marker=marker, markersize=9, color='y')
            # else:
            #    marker='x'
            # ax1.plot(x,y,marker=marker,markersize=9,color='y')

    # Plot sidepanel with just LoTSS-DR2 1-10sigma Stokes-I
    if plot_fits:
        # fits_filepath = os.path.join(image_directory, image_name + extension)
        assert fits_filepath.endswith('.fits')
        assert os.path.exists(fits_filepath)
        # Load the fits files
        field, hdr = load_fits(fits_filepath)
        # Load the rms 
        rms, rms_hdr = load_fits(rms_filepath)
        # Clip field in sigma space 
        field = np.clip(field / rms, lower_clip, upper_clip)
        # Ensures that the image color-map is within the clip values for all images
        interval = MinMaxInterval()
        # Normalize values to lie between 0 and 1 and then apply a stretch
        stretch = SqrtStretch()
        field = stretch(interval(field))
        # plot
        ax2.imshow(field, origin='lower')
        ax2.axes.xaxis.set_visible(False)
        ax2.axes.yaxis.set_visible(False)
        ax2.set_title(f"LoTSS-DR2, {lower_clip} to {upper_clip} sigma, sqrt-scaling", fontsize=16)

    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)
    # ax1.set_xlim(0,200)
    # ax1.set_ylim(200,0)
    # Save and close plot
    # dest = os.path.join(debug_path,'boxes',file_path.split('/')[-1].replace('.png','.pdf'))
    if training_mode and plot_optical:
        if dataset is None:
            dest = os.path.join(debug_path, 'boxes_with_optical', file_path.split('/')[-1])
        else:
            dest = os.path.join(debug_path, 'boxes_with_optical', dataset, file_path.split('/')[-1])

    else:
        dest = os.path.join(debug_path, 'boxes', file_path.split('/')[-1])
    # print("Save debug plot at:", dest)
    # plt.show()
    plt.savefig(dest, bbox_inches='tight')
    plt.close()


def flatten_yield(l):
    '''Flatten a list or numpy array'''
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el,
                                                       (str, bytes)):
            yield from flatten_yield(el)
        else:
            yield el


def flatten(list_of_lists):
    """Flatten a list of lists even if the lists are of unequal sizes"""
    return list(chain.from_iterable(list_of_lists))


def sanity_check_visualization(debug_path, image_names, image_dir, cutout_list, extension,
                               fixed_cutout_size, suffix=''):
    """ Plot all kinds of label and box properties to inspect if weird outliers occur.
    """
    assert (len(image_names) == len(cutout_list))

    # image_objects = [(s.xmin, s.ymin, s.xmax, s.ymax, s.n_comp, s.n_peak)
    #    for cutout in cutout_list for s in [cutout.c_source] +cutout.other_sources]
    image_objects = [[(s.xmin, s.ymin, s.xmax, s.ymax)  # , s.n_comp, s.n_peak)
                      for s in [cutout.c_source] + cutout.other_sources]
                     for cutout in cutout_list]

    # property lists we want to fill (width,height,depth)
    image_widths = []
    image_heights = []
    """
    n_comps_per_annotation_object = np.array(flatten([[n_comp for 
        xmin, ymin, xmax, ymax, n_comp, n_peak in obs] 
        for obs in image_objects]))
    n_peaks_per_annotation_object = np.array(flatten([[n_peak for 
        xmin, ymin, xmax, ymax, n_comp, n_peak in obs]
        for obs in image_objects]))
    """
    box_widths_per_image = np.array([[xmax - xmin for xmin, ymin, xmax, ymax in obs]
                                     for obs in image_objects])
    box_heights_per_image = np.array([[ymax - ymin for xmin, ymin, xmax, ymax in obs]
                                      for obs in image_objects])
    box_surface_areas_per_image = np.array([[width * height for width, height in zip(widths, heights)]
                                            for widths, heights in zip(box_widths_per_image, box_heights_per_image)])
    box_surface_area_per_image = [np.sum(areas) for areas in box_surface_areas_per_image]
    objects_per_image = [len(annotation_objects) for annotation_objects in image_objects]

    # Assert that all bounding boxes fall inside the image
    for obs, width, height in zip(image_objects, image_widths, image_heights):
        for (xmin, ymin, xmax, ymax) in obs:  # , n_comp, n_peak) in obs:
            assert (
                        xmin >= 0 and xmin < width and xmax >= 0 and xmax < width and ymin >= 0 and ymin < height and ymax >= 0 and ymax < height)

            # Iterate over all images and their objects (which contain bounding boxes and class labels)
    for image_name in image_names:
        # Get image dimensions and insert them in a python dict
        image_name = image_name + extension
        image_filename = os.path.join(image_dir, image_name)
        if extension == '.png' or extension == '.jpg':
            im = Image.open(image_filename)
            # width, height, depth = im.size
            depth = 0
            width, height = im.size
        elif extension == '.npy':
            im = np.load(image_filename, mmap_mode='r')  # mmap_mode might allow faster read
            width, height, depth = np.shape(im)
        else:
            raise ValueError('Image file format must either be .png, .jpg, .jpeg or .npy')
        image_widths.append(width)
        image_heights.append(height)

    # Scatterplot bounding box widths and heights
    plt.figure(figsize=(10, 10))
    g = sns.jointplot(x=flatten(box_widths_per_image), y=flatten(box_heights_per_image),
                      height=8, joint_kws={"s": 0.5})
    ax = g.ax_joint
    ax.set_xlabel('Bounding box widths [pixels] resolution is 1.5 arcsec/pixel')
    ax.set_ylabel('Bounding box heights [pixels] resolution is 1.5 arcsec/pixel')
    plt.savefig(os.path.join(debug_path,
                             f'debug_bounding_box_dimensions{suffix}.png'), bbox_inches='tight')

    # Histogram total bounding box area per image
    plt.figure(figsize=(10, 10))
    plt.hist(box_surface_area_per_image, bins='sqrt', fill=False, histtype='step',
             linewidth='3')
    plt.xscale('log')
    plt.title('Total bounding box area per image')
    plt.xlabel('Bounding box area [pixels^2]')
    plt.ylabel('Number of objects')
    plt.savefig(os.path.join(debug_path, f'debug_bounding_box_area_per_image{suffix}.png'))

    # Print tally for a number of properties
    print_tally(debug_path, objects_per_image, 'objects per image')
    # print_tally(debug_path, n_comps_per_annotation_object, 'components per annotation object')
    # print_tally(debug_path, n_peaks_per_annotation_object, 'peaks per annotation object')

    # Scatterplot image widths and heights
    if fixed_cutout_size:
        # print("Image widths", set(image_widths))
        assert np.std(image_widths) == 0
        assert np.std(image_heights) == 0
    else:
        plt.figure(figsize=(10, 10))
        g = sns.jointplot(x=image_widths, y=image_heights, height=8, joint_kws={"s": 0.5})
        ax = g.ax_joint
        ax.set_xlabel('Image widths [pixels]')
        ax.set_ylabel('Image heights [pixels]')
        plt.title('Image dimensions')
        plt.savefig(os.path.join(debug_path, f'debug_image_dimensions{suffix}.png'))
    plt.close()


def print_tally(save_directory, a_list, description):
    """Print and plot tally of items in a_list, described by description"""
    tally = Counter(a_list)
    print(f'\nTally of the number of {description}:')
    print([f'{int(k)}: {v} {round(100 * v / len(a_list))}%' for k, v in tally.items()])

    # Plot counts 
    plt.figure(figsize=(10, 10))
    ax = sns.countplot(y=a_list, color="c");
    des = description.replace(' ', '_')
    # Plot count numbers (also as percentage of total)
    for p in ax.patches:
        percentage = '{} ({:.1f}%)'.format(p.get_width(), 100 * p.get_width() / len(a_list))
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height() / 2
        ax.annotate(percentage, (x, y))
    plt.title(f'Tally of the number of {description}')
    plt.savefig(os.path.join(save_directory, f'debug_{des}.png'))
    plt.close()


def save_obj(file_path, obj):
    with open(file_path, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_obj(file_path):
    with open(file_path, 'rb') as input:
        return pickle.load(input)


def get_bounding_boxes(output):
    """Return bounding boxes inside inference output as numpy array
    """
    assert "instances" in output
    instances = output["instances"].to(torch.device("cpu"))

    return instances.get_fields()['pred_boxes'].tensor.numpy()


def is_within(x, y, xmin, ymin, xmax, ymax):
    """Return true if x, y lies within xmin,ymin,xmax,ymax.
    False otherwise.
    """
    if xmin <= x <= xmax and ymin <= y <= ymax:
        return True
    else:
        return False


def area(bbox):
    """Return area."""
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    area = width * height
    if area < 0:
        return None
    return area


def intersect_over_union(bbox1, bbox2):
    """Return intersection over union or IoU."""
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    intersection_area = area([max(xmin1, xmin2),
                              max(ymin1, ymin2),
                              min(xmax1, xmax2),
                              min(ymax1, ymax2)])
    if intersection_area is None:
        return 0

    union_area = area(bbox1) + area(bbox1) - intersection_area
    assert intersection_area <= union_area
    return intersection_area / union_area


def sub(python_list, indexes):
    if python_list is None:
        return []
    return [python_list[i] for i in indexes]


def subs(*python_lists, indexes=[]):
    return [[python_list[i] for i in indexes]
            for python_list in python_lists]


def _check_if_central_bbox_misses_comp(n_comps, comp_scores, close_comp_scores,
                                       central_covered, focussed_comps, related_comps,
                                       unrelated_comps, central_bboxes, source_names, image_dir, output_dir,
                                       imsize,
                                       imsize_arcsec, new_related_unresolved=None, new_unrelated_unresolved=None,
                                       remove_unresolved=False, plot_misboxed=False, debug=False, cutout_list=None):
    """Check whether the predicted central box misses a number of assocatiated components
        as indicated by the ground truth"""

    # Tally for single comp
    single_comp_fail = [not central_cover for n_comp, central_cover
                        in zip(n_comps, central_covered) if n_comp == 1]
    single_comp_fail_frac = np.sum(single_comp_fail) / len(single_comp_fail)

    # Tally for multi comp
    multi_comp_binary_fail = [(central_cover and unrelated == 0 and n_comp > (related + 1)) or (not central_cover)
                              for n_comp, central_cover, related, unrelated
                              in zip(n_comps, central_covered, comp_scores, close_comp_scores) if n_comp > 1]
    multi_comp_binary_fail_frac = np.sum(multi_comp_binary_fail) / len(multi_comp_binary_fail)

    # Collect single comp sources that fail to include their gt comp
    ran = list(range(len(comp_scores)))
    fail_indices = [i for i, n_comp, central_cover
                    in zip(ran, n_comps, central_covered)
                    if not central_cover and n_comp == 1]

    if debug:
        print('single underestimation', fail_indices)
    if plot_misboxed and len(fail_indices) > 0:
        collect_misboxed(*subs(focussed_comps, related_comps, unrelated_comps, central_bboxes,
                               source_names, indexes=fail_indices), image_dir, output_dir,
                         "single_missing", imsize, imsize_arcsec, cutout_list=sub(cutout_list, fail_indices),
                         remove_unresolved=remove_unresolved,
                         new_related_unresolved=sub(new_related_unresolved, fail_indices),
                         new_unrelated_unresolved=sub(new_unrelated_unresolved, fail_indices))

    # Collect single comp sources that fail to include their gt comp
    fail_indices = [i
                    for i, n_comp, central_cover, related, unrelated
                    in zip(ran, n_comps, central_covered, comp_scores, close_comp_scores) if n_comp > 1 and \
                    ((central_cover and unrelated == 0 and n_comp > (related + 1)) or (not central_cover))]
    if debug:
        print('multi underestimation', fail_indices)
    if plot_misboxed and len(fail_indices) > 0:
        collect_misboxed(*subs(focussed_comps, related_comps, unrelated_comps, central_bboxes,
                               source_names, indexes=fail_indices), image_dir, output_dir,
                         "multi_underestimation", imsize, imsize_arcsec,
                         cutout_list=sub(cutout_list, fail_indices),
                         remove_unresolved=remove_unresolved,
                         new_related_unresolved=sub(new_related_unresolved, fail_indices),
                         new_unrelated_unresolved=sub(new_unrelated_unresolved, fail_indices))

    return single_comp_fail_frac, multi_comp_binary_fail_frac


def _check_if_central_bbox_includes_unassociated_comps(n_comps, comp_scores, close_comp_scores,
                                                       central_covered, focussed_comps, related_comps,
                                                       unrelated_comps, central_bboxes, source_names, image_dir,
                                                       output_dir,
                                                       imsize, imsize_arcsec, remove_unresolved=False,
                                                       new_related_unresolved=None, new_unrelated_unresolved=None,
                                                       plot_misboxed=False, debug=False, cutout_list=None):
    """Check whether the predicted central box includes a number of unassocatiated components
        as indicated by the ground truth"""
    # Tally for single comp
    single_comp_fail = [unrelated > 0 for n_comp, unrelated
                        in zip(n_comps, close_comp_scores) if n_comp == 1]
    single_comp_fail_frac = np.sum(single_comp_fail) / len(single_comp_fail)

    # Tally for multi comp
    multi_comp_binary_fail = [unrelated > 0 for n_comp, unrelated in
                              zip(n_comps, close_comp_scores) if n_comp > 1]
    multi_comp_binary_fail_frac = np.sum(multi_comp_binary_fail) / len(multi_comp_binary_fail)

    # Collect single comp sources that includ unassociated comps
    ran = list(range(len(close_comp_scores)))
    # fail_indices = [i for i, n_comp, total in zip(ran, n_comps, close_comp_scores)
    #        if ((n_comp == 1) and (0 != total)) ]
    # fail_indices = [i for i,c in zip(ran,single_comp_fail) if c]
    fail_indices = [i for i, n_comp, unrelated
                    in zip(ran, n_comps, close_comp_scores) if n_comp == 1 and unrelated > 0]
    if debug:
        print('single overestimation', fail_indices)
    if plot_misboxed and len(fail_indices) > 0:
        collect_misboxed(*subs(focussed_comps, related_comps, unrelated_comps, central_bboxes,
                               source_names, indexes=fail_indices), image_dir, output_dir,
                         "single_overestimation", imsize, imsize_arcsec,
                         cutout_list=sub(cutout_list, fail_indices),
                         remove_unresolved=remove_unresolved,
                         new_related_unresolved=sub(new_related_unresolved, fail_indices),
                         new_unrelated_unresolved=sub(new_unrelated_unresolved, fail_indices))

    # Collect single comp sources that fail to include their gt comp
    # fail_indices = [i for i, n_comp, total in zip(ran, n_comps, close_comp_scores)
    #        if ((n_comp > 1) and (0 != total)) ]
    # fail_indices = [i for i,c in zip(ran,multi_comp_binary_fail) if c]
    fail_indices = [i for i, n_comp, unrelated
                    in zip(ran, n_comps, close_comp_scores) if n_comp > 1 and unrelated > 0]
    if debug:
        print('multi overestimation', fail_indices)
    if plot_misboxed and len(fail_indices) > 0:
        collect_misboxed(*subs(focussed_comps, related_comps, unrelated_comps, central_bboxes,
                               source_names, indexes=fail_indices), image_dir, output_dir,
                         "multi_overestimation", imsize, imsize_arcsec,
                         cutout_list=sub(cutout_list, fail_indices),
                         remove_unresolved=remove_unresolved,
                         new_related_unresolved=sub(new_related_unresolved, fail_indices),
                         new_unrelated_unresolved=sub(new_unrelated_unresolved, fail_indices))

    return single_comp_fail_frac, multi_comp_binary_fail_frac


def collect_misboxed(focussed_comps, related_comps, unrelated_comps, central_bboxes,
                     source_names, image_dir, output_dir, fail_dir_name, imsize, imsize_arcsec,
                     cutout_list=None, debug=False,
                     remove_unresolved=False, segmentation_dir=None,
                     new_related_unresolved=None, sigma_box_fit=5, related_resolved_comps=None,
                     new_unrelated_unresolved=None, plot_optical=False):
    """Collect ground truth bounding boxes that fail to encapsulate the ground truth pybdsf
    components so that they can be inspected to improve the box-draw-process"""
    # Dirty hack to check if os.environ for showing the convex hull exists
    try:
        convex_save_path = os.environ['convex']
    except:
        convex_save_path = None

    # Make dir to collect the failed images in
    fail_dir = os.path.join(output_dir, fail_dir_name)
    os.makedirs(fail_dir, exist_ok=True)
    # Remove old directory but first check that it contains only pngs
    for f in os.listdir(fail_dir):
        assert f.endswith('.png'), 'Directory should only contain images.'
    for f in os.listdir(fail_dir):
        os.remove(os.path.join(fail_dir, f))

    # Copy debug images to this dir 
    if debug:
        print('misboxed output dir', fail_dir)
        print('image dir is:', image_dir)
        print('sourcenames len is:', len(source_names), source_names[0])

    # if code fails here the debug source name or path is probably incorrect
    # image_source_paths = [os.path.join(image_dir,source_name + f"_{imsize_arcsec}arcsec_large_radio_DR2_rotated0deg.png")
    image_source_paths = [os.path.join(image_dir, source_name + f"_radio_DR2_rotated0deg.png")
                          for source_name in source_names]
    image_dest_paths = [os.path.join(fail_dir, image_source_path.split('/')[-1])
                        for image_source_path in image_source_paths]
    image_only = False
    if image_only:

        for src, dest in zip(image_source_paths, image_dest_paths):
            with open(src, 'rb') as fin:
                with open(dest, 'wb') as fout:
                    copyfileobj(fin, fout, 128 * 1024)
    else:

        wide_focus = [c.wide_focus for c in cutout_list]
        # Iterate over all failed  items
        for i, (focus_l, rel_l, unrel_l, bbox, src, src_name, dest, w) in enumerate(zip(focussed_comps,
                                                                                        related_comps,
                                                                                        unrelated_comps, central_bboxes,
                                                                                        image_source_paths,
                                                                                        source_names,
                                                                                        image_dest_paths, wide_focus)):

            # Open source
            im = imread(src)

            if cutout_list is None:
                return

            # Plot figure 
            f, ax1 = plt.subplots(1, 1, figsize=(8, 6))
            # Radio intensity
            im[:, :, 1] = 255 - im[:, :, 1]
            ax1.imshow(im)
            ax1.plot([bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]],
                     np.array([bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]]), 'k', linewidth=3)
            # ax1.set_title('Ground truth labels')

            # Plot convex hull
            if not convex_save_path is None and not segmentation_dir is None:
                segmentation_path = os.path.join(segmentation_dir,
                                                 f"{src_name}_{imsize}_{sigma_box_fit}sigma.pkl")

                # Load segmentation map per cutout
                with open(segmentation_path, 'rb') as f:
                    segmented_cutout = pickle.load(f)

                # Given a segmentation map
                outlines = segmented_cutout.outline_segments()

                # Debug plot outlines
                if 1 == 2:
                    plt.imshow(outlines)
                    plt.show()

                # Find segmentation islands corresponding to radio components
                pixel_xs = [focus_l[0]] + w[0] + list(related_resolved_comps[i][0])
                pixel_ys = [focus_l[1]] + w[1] + list(related_resolved_comps[i][1])
                if debug:
                    print("pixel_xs", pixel_xs)
                    print("pixel_ys", pixel_ys)
                try:
                    labels = [segmented_cutout.data[int(round(y)), int(round(x))] for x, y in
                              zip(pixel_xs, pixel_ys)]
                except:
                    if debug:
                        print("no label found")
                    sdfsdf
                labels = [l for l in labels if not l == 0]
                labels = np.array(labels) - 1  # labels start at 1 as 0 is background

                # If no labels are found, enlarge the search radius
                labels2 = labels
                r = 3  # Search radius
                if labels.size != len(pixel_xs):
                    labels = list(set(flatten_yield([
                        segmented_cutout.data[int(round(y)) - r:int(round(y)) + r,
                        int(round(x)) - r:int(round(x)) + r] for x, y in \
                        zip(pixel_xs, pixel_ys)])))
                    labels = [l for l in labels if not l == 0]
                    labels = np.array(labels) - 1  # labels start at 1 as 0 is background

                # If no labels found... should not happen, these sources should have been discarded in
                # earlier preprocessing stage
                if len(labels) == 0:
                    raise Exception("No segments found using pixel_xs, pixel_ys.")

                if debug:
                    print("labels:", labels)
                # Get bounding boxes for segments
                bboxes = [segmented_cutout.segments[l].bbox for l in labels]

                # Get all non-background points from the segmentation outlines
                points = []
                for b in bboxes:
                    outline = outlines[b.iymin:b.iymax, b.ixmin:b.ixmax]
                    for i_outline_, rows in enumerate(outline):
                        for j_rows, el in enumerate(rows):
                            if el > 0:
                                points.append([b.iymin + i_outline_, b.ixmin + j_rows])
                points = np.array(points)
                if debug:
                    plt.scatter(*list(zip(*points)))
                    plt.show()

                # Create convex hull from points (removes inner points)
                if len(points) == 0:
                    print("pixel_xs", pixel_xs)
                    print("pixel_ys", pixel_ys)
                    print("labels:", labels)
                    print("points:", type(points), np.shape(points), points)
                    # plt.imshow(segmented_cutout.data)
                    # plt.show()
                    # plt.imshow(outlines)
                    # plt.show()
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 1], points[simplex, 0],
                             linestyle='solid', color='r', alpha=0.8,
                             linewidth=3)
                # Create matplotlib Path object from convexhull vertices
                # hull_path = Path( points[hull.vertices] )

            # Plot optical
            if plot_optical:

                if 'w1Mag' in cutout_list[i].optical_sources.keys():
                    okey = 'w1Mag'
                else:
                    # Plot legacy
                    okey = 'MAG_R'

                for w, x, y in zip(cutout_list[i].optical_sources[okey],
                                   cutout_list[i].optical_sources['x'], cutout_list[i].optical_sources['y']):
                    if not np.isnan(w):
                        marker = '+'
                        ax1.plot(x, y, marker=marker, markersize=9, color='y')

            # Plot component locations
            focus_color = 'red'
            ax1.plot(focus_l[0], focus_l[1], marker='s', markersize=10, color=focus_color)
            if remove_unresolved:
                # print("length of cutoutlist", len(cutout_list))
                # print("length of newrelatedunresl", len(new_related_unresolved))
                # print("i and rel", i, rel_l, cutout_list[i].get_related_comp())
                assert len(cutout_list[i].get_related_unresolved()) == len(new_related_unresolved[i])
                assert len(new_related_unresolved[i]) == len(
                    rel_l[0]), f'i {i} newrlunres {new_related_unresolved[i]}, rel_l {rel_l[0]}'
                for old_unresolved, unresolved, x, y in zip(cutout_list[i].get_related_unresolved(),
                                                            new_related_unresolved[i], rel_l[0], rel_l[1]):
                    if old_unresolved:
                        marker = 'x'
                    else:
                        marker = '.'
                    ax1.plot(x, y, marker=marker, markersize=9, color='r')
                    if unresolved != old_unresolved:
                        ax1.plot(x, y, marker='+', markersize=9, color='r')
                for old_unresolved, unresolved, x, y in zip(
                        cutout_list[i].get_unrelated_unresolved(),
                        new_unrelated_unresolved[i], unrel_l[0], unrel_l[1]):
                    if old_unresolved:
                        marker = 'x'
                    else:
                        marker = '.'
                    ax1.plot(x, y, marker=marker, markersize=9, color='lime')
                    if unresolved != old_unresolved:
                        ax1.plot(x, y, marker='+', markersize=9, color='lime')
            else:

                for x, y in zip(rel_l[0], rel_l[1]):
                    ax1.plot(x, y, marker='.', markersize=9, color='r')
                for x, y in zip(unrel_l[0], unrel_l[1]):
                    ax1.plot(x, y, marker='.', markersize=9, color='lime')

            ax1.axes.xaxis.set_visible(False)
            ax1.axes.yaxis.set_visible(False)

            """
            # Plot component locations
            ax1.plot(focus_l[0],focus_l[1],marker='s',color='r')
            for x,y in zip(rel_l[0],rel_l[1]):
                ax1.plot(x,y,marker='.',color='r')
            for x,y in zip(unrel_l[0],unrel_l[1]):
                ax1.plot(x,y,marker='.',color='lime')
            """

            # plt.show()
            plt.savefig(dest, bbox_inches='tight')
            plt.close()


def baseline(single_comps, multi_comps):
    total = single_comps + multi_comps
    if total == 0:
        print("Empty list. Cannot establish baseline.")
        return 0
    correct = single_comps / total
    print(f"Baseline assumption cat is {correct:.1%} correct")
    return correct


def our_score(single_comps, multi_comps, assoc_fail, unassoc_fail):
    fail_single = assoc_fail[0] * single_comps + unassoc_fail[0] * single_comps
    fail_multi = assoc_fail[1] * multi_comps + unassoc_fail[1] * multi_comps
    total = single_comps + multi_comps
    correct = (total - (fail_single + fail_multi)) / total
    print(f"cat is {correct:.1%} correct")
    return correct


def improv(baseline, our_score):
    print(f"{(our_score - baseline) / baseline:.2%} improvement")


def reinsert_unresolved_for_triplets(reinstate_components, unresolved_per_cutout, gt_bboxes,
                                     focussed_comps, wide_focus_comps, related_resolved_comps,
                                     source_names, segmentation_dir, imsize=200, debug=False, sigma_box_fit=5):
    """Given a list of unresolved objects, tries to reinsert them if sensible"""

    # Check if any unresolved components are inside gt bbox.
    unresolved_in_gt_bboxes = [np.any([is_within(x, y,
                                                 bbox[0], bbox[1], bbox[2], bbox[3])
                                       for unresolved, x, y in zip(unresolved_list, xs, ys) if unresolved])
                               for (xs, ys), unresolved_list, bbox
                               in zip(reinstate_components, unresolved_per_cutout, gt_bboxes)]

    # If so proceed with following actions.
    segmentation_paths = [os.path.join(segmentation_dir, f"{s}_{imsize}_{sigma_box_fit}sigma.pkl")
                          for s in source_names]
    new_unresolved_per_cutout = []
    for action_needed, segmentation_path, reinstate_comp, unresolved_list, gt_bbox, \
        focussed_comp, wide_focus, related_comp in zip(
        unresolved_in_gt_bboxes, segmentation_paths,
        reinstate_components, unresolved_per_cutout, gt_bboxes,
        focussed_comps, wide_focus_comps, related_resolved_comps):
        if action_needed:

            # Load segmentation_path map per cutout
            with open(segmentation_path, 'rb') as f:
                segmented_cutout = pickle.load(f)

            # Given a segmentation map
            outlines = segmented_cutout.outline_segments()

            # Debug plot outlines
            if debug:
                plt.imshow(outlines)
                plt.show()

            # Find segmentation islands corresponding to radio components

            pixel_xs = [focussed_comp[0]] + wide_focus[0] + related_comp[0]
            pixel_ys = [focussed_comp[1]] + wide_focus[1] + related_comp[1]
            if debug:
                print("pixel_xs", pixel_xs)
                print("pixel_ys", pixel_ys)
            try:
                labels = [segmented_cutout.data[int(round(y)), int(round(x))] for x, y in
                          zip(pixel_xs, pixel_ys)]
            except:
                if debug:
                    print("no label found")
                continue
            labels = [l for l in labels if not l == 0]
            labels = np.array(labels) - 1  # labels start at 1 as 0 is background

            # If no labels are found, enlarge the search radius
            labels2 = labels
            r = 3  # Search radius
            if labels.size != len(pixel_xs):
                labels = list(set(flatten_yield([
                    segmented_cutout.data[int(round(y)) - r:int(round(y)) + r,
                    int(round(x)) - r:int(round(x)) + r] for x, y in \
                    zip(pixel_xs, pixel_ys)])))
                labels = [l for l in labels if not l == 0]
                labels = np.array(labels) - 1  # labels start at 1 as 0 is background

            # If no labels found... should not happen, these sources should have been discarded in
            # earlier preprocessing stage
            if len(labels) == 0:
                raise Exception("No segments found using pixel_xs, pixel_ys.")

            if debug:
                print("labels:", labels)
            # Get bounding boxes for segments
            bboxes = [segmented_cutout.segments[l].bbox for l in labels]

            # Get all non-background points from the segmentation outlines
            points = []
            for b in bboxes:
                outline = outlines[b.iymin:b.iymax, b.ixmin:b.ixmax]
                for i, rows in enumerate(outline):
                    for j, el in enumerate(rows):
                        if el > 0:
                            points.append([b.iymin + i, b.ixmin + j])
            points = np.array(points)
            if debug:
                plt.scatter(*list(zip(*points)))
                plt.show()

            # Create convex hull from points (removes inner points)
            if len(points) == 0:
                print("pixel_xs", pixel_xs)
                print("pixel_ys", pixel_ys)
                print("labels:", labels)
                print("points:", type(points), np.shape(points), points)
                plt.imshow(segmented_cutout.data)
                plt.show()
                plt.imshow(outlines)
                plt.show()
            hull = ConvexHull(points)
            # Create matplotlib Path object from convexhull vertices
            hull_path = Path(points[hull.vertices])
            # This allows us to query whether a point is contained by our convex hull
            new_unresolved_list = [False for _ in range(len(unresolved_list))]
            for i, (x, y, unresolved) in enumerate(zip(reinstate_comp[0], reinstate_comp[1],
                                                       unresolved_list)):
                if unresolved:
                    should_be_reinstated = hull_path.contains_point((y, x))
                    if debug:
                        print(f"Does our gt convex hull contain point {x},{y}?")
                        print(should_be_reinstated)
                    if not should_be_reinstated:
                        new_unresolved_list[i] = True
            new_unresolved_per_cutout.append(new_unresolved_list)
        else:
            new_unresolved_per_cutout.append(unresolved_list)
    return new_unresolved_per_cutout


def _evaluate_gt_bboxes(dataset_name, debug_path, cutout_list, source_names,
                        image_dir, imsize, imsize_arcsec, remove_unresolved,
                        scale_factor=1, debug=False, flip_y=False, plot_misboxed=False,
                        segmentation_dir=None, plot_optical=False, sigma_box_fit=5):
    """ 
    Evaluate the results using our LOFAR appropriate score.

        Evaluate _predictions on the given tasks.
        Fill _results with the metrics of the tasks.

        That is: for all proposed boxes that cover the middle pixel of the input image check which
        sources from the component catalogue are inside. 
        The predicted box can fail in three different ways:
        1. No predicted box covers the focussed box
        2. The predicted central box misses a number of components
        3. The predicted central box encompasses too many components
        4. The prediction score for the predicted box is lower than other boxes that cover the middle
            pixel
        5. The prediction score is lower than x
    
    """
    assert all([c.c_source.sname == n for c, n in zip(cutout_list, source_names)])
    # Retrieve focussed, related and unrelated components
    source_names = [s for s, c in zip(source_names, cutout_list) if c.rotation_angle_deg == 0]
    cutout_list = [c for c in cutout_list if c.rotation_angle_deg == 0]
    focussed_comps = [c.get_focussed_comp() for c in cutout_list]
    related_comps = [c.get_related_comp() for c in cutout_list]
    related_resolved_comps = [[[oc.x for oc in c.other_components if oc.related and not oc.unresolved],
                               [oc.y for oc in c.other_components if oc.related and not oc.unresolved]]
                              for c in cutout_list]
    if remove_unresolved:
        related_unresolved = [c.get_related_unresolved() for c in cutout_list]
        unrelated_unresolved = [c.get_unrelated_unresolved() for c in cutout_list]
        wide_focus = [c.wide_focus for c in cutout_list]
    else:
        related_unresolved, unrelated_unresolved = None, None
    unrelated_comps = [c.get_unrelated_comp() for c in cutout_list]
    # scale_factor = cutout_list[0].scale_factor
    assert all([c.c_source.sname == n for c, n in zip(cutout_list, source_names)])

    # Count number of components in dataset
    # Retrieve number of components per central source
    n_comps = [1 + len(c[0]) if len(c[0]) > 0 else 1 for c in related_comps]
    # Get number of single and multi comp sources
    single_comps = sum([1 if n == 1 else 0 for n in n_comps])
    multi_comps = sum([1 if n > 1 else 0 for n in n_comps])
    central_covered = [True for _ in n_comps]
    print(f"We have {len(n_comps)} cutouts: {single_comps} single comp and {multi_comps} multi")

    i = 2
    if debug:
        print("#################################################################################")
        i_s = [i for i, s in enumerate(source_names) if s == 'ILTJ140638.08+552810.0']
        if len(i_s) > 0:
            i = i_s[0]
        print("debug showing source", source_names[i])
        # Check ground truth and prediction values of first item
        print("focus, related, unrelated, ncomp")
        print(focussed_comps[i])
        print(related_comps[i])
        print(unrelated_comps[i])
        if remove_unresolved:
            print("related unresolved")
            print(related_unresolved[i])
        print(n_comps[i])
        print(f"scale factor: {scale_factor}")
        # print(np.shape(focussed_comps), np.shape(related_comps),
        #        np.shape(unrelated_comps), np.shape(n_comps))

    # Get central bounding boxes only for the 0 rotation
    if flip_y:
        central_bboxes = np.array([(c.gt_xmin * scale_factor,
                                    imsize - c.gt_ymax * scale_factor,
                                    c.gt_xmax * scale_factor, imsize - c.gt_ymin * scale_factor)
                                   for c in cutout_list])
    else:
        central_bboxes = np.array([(c.gt_xmin * scale_factor, c.gt_ymin * scale_factor,
                                    c.gt_xmax * scale_factor, c.gt_ymax * scale_factor)
                                   for c in cutout_list])

    if debug:
        print("central box:", central_bboxes[i])

    # Check if other source comps fall inside predicted central box
    if remove_unresolved:
        related_unresolved = reinsert_unresolved_for_triplets(related_comps, related_unresolved,
                                                              central_bboxes, focussed_comps, wide_focus,
                                                              related_resolved_comps, source_names,
                                                              segmentation_dir,
                                                              imsize=imsize, sigma_box_fit=sigma_box_fit)
        unrelated_unresolved = reinsert_unresolved_for_triplets(unrelated_comps,
                                                                unrelated_unresolved, central_bboxes, focussed_comps,
                                                                wide_focus,
                                                                related_resolved_comps,
                                                                source_names,
                                                                segmentation_dir, imsize=imsize,
                                                                sigma_box_fit=sigma_box_fit)
        comp_scores = [np.sum([is_within(x, y,
                                         bbox[0], bbox[1], bbox[2], bbox[3]) and not unresolved
                               for unresolved, x, y in zip(unresolved_list, xs, ys)], dtype=int)
                       for (xs, ys), unresolved_list, bbox
                       in zip(related_comps, related_unresolved, central_bboxes)]
        close_comp_scores = [np.sum([is_within(x, y,
                                               bbox[0], bbox[1], bbox[2], bbox[3]) and not unresolved
                                     for unresolved, x, y in zip(unresolved_list, xs, ys)], dtype=int)
                             for (xs, ys), unresolved_list, bbox
                             in zip(unrelated_comps, unrelated_unresolved,
                                    central_bboxes)]
    else:
        comp_scores = [np.sum([is_within(x, y,
                                         bbox[0], bbox[1], bbox[2], bbox[3])
                               for x, y in zip(comps[0], comps[1])], dtype=int)
                       for comps, bbox
                       in zip(related_comps, central_bboxes)]
        close_comp_scores = [np.sum([is_within(x, y,
                                               bbox[0], bbox[1], bbox[2], bbox[3])
                                     for x, y in zip(xs, ys)], dtype=int)
                             for (xs, ys), bbox in zip(unrelated_comps,
                                                       central_bboxes)]
    if debug:
        print("comp_scores:", comp_scores[i])

    """
    def collect_misboxed(focussed_comps, related_comps, unrelated_comps, central_bboxes, 
            source_names, image_dir, output_dir, fail_dir_name, imsize, imsize_arcsec, cutout_list=None, debug=False,
            remove_unresolved=False,
            new_related_unresolved=None,
            new_unrelated_unresolved=None):
    """

    # Plot all!
    collect_misboxed(focussed_comps, related_comps, unrelated_comps, central_bboxes,
                     source_names, image_dir, debug_path,
                     "debug_all", imsize, imsize_arcsec, cutout_list=cutout_list,
                     remove_unresolved=remove_unresolved, plot_optical=plot_optical,
                     new_related_unresolved=related_unresolved,
                     new_unrelated_unresolved=unrelated_unresolved,
                     related_resolved_comps=related_resolved_comps, segmentation_dir=segmentation_dir)
    # collect_misboxed(*subs(focussed_comps, related_comps, unrelated_comps, central_bboxes,
    #     source_names, indexes=list(range(len(n_comps)))), image_dir, debug_path,
    #     "debug_all",imsize, imsize_arcsec)
    # sys.exit()
    # 1&2. "Predicted central bbox not existing or misses a number of components" can now be checked
    includes_associated_fail_fraction = _check_if_central_bbox_misses_comp(n_comps, comp_scores,
                                                                           close_comp_scores,
                                                                           central_covered, focussed_comps,
                                                                           related_comps,
                                                                           unrelated_comps, central_bboxes,
                                                                           source_names, image_dir, debug_path,
                                                                           imsize, imsize_arcsec, debug=debug,
                                                                           plot_misboxed=plot_misboxed,
                                                                           cutout_list=cutout_list,
                                                                           remove_unresolved=remove_unresolved,
                                                                           new_related_unresolved=related_unresolved,
                                                                           new_unrelated_unresolved=unrelated_unresolved)
    # 3&4. "Predicted central bbox encompasses too many or too few components" can now be checked
    includes_unassociated_fail_fraction = \
        _check_if_central_bbox_includes_unassociated_comps(n_comps, comp_scores, close_comp_scores,
                                                           central_covered, focussed_comps, related_comps,
                                                           unrelated_comps, central_bboxes, source_names, image_dir,
                                                           debug_path,
                                                           imsize, imsize_arcsec, debug=debug,
                                                           plot_misboxed=plot_misboxed, cutout_list=cutout_list,
                                                           remove_unresolved=remove_unresolved,
                                                           new_related_unresolved=related_unresolved,
                                                           new_unrelated_unresolved=unrelated_unresolved)

    # Calculate/print catalogue improvement
    base_score = baseline(single_comps, multi_comps)
    correct_cat = our_score(single_comps, multi_comps,
                            includes_associated_fail_fraction, includes_unassociated_fail_fraction)
    improv(base_score, correct_cat)

    print("Related fail fraction. Unrelated fail fraction.")
    print(includes_associated_fail_fraction, includes_unassociated_fail_fraction)
    with open(os.path.join(debug_path, f"text_{dataset_name}.txt"), 'w') as f:
        print(f"Baseline assumption cat is {base_score:.1%} correct", file=f)
        print(f"{dataset_name} cat is {correct_cat:.1%} correct", file=f)
        for i in includes_associated_fail_fraction:
            f.write(str(i) + ",")
        for i in includes_unassociated_fail_fraction:
            f.write(str(i) + ",")

    return includes_associated_fail_fraction, includes_unassociated_fail_fraction
