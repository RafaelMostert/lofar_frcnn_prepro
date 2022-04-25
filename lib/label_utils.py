import collections
import os
import pickle

import astropy.units as u
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm
from astropy.wcs import utils
from photutils import detect_sources


def flatten(l):
    '''Flatten a list or numpy array'''
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el,
                                                                   (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def get_idx_dict(comp_cat, training_mode=False):
    """Create dict that returns index of (related) objects when given sourcename"""

    if training_mode:
        if 'Source_Name' in comp_cat.keys():
            skey = 'Source_Name'
        else:
            skey = 'Parent_Source'
        source_names = list(set(comp_cat[skey].values))
        idx_dict = {s: [] for s in source_names}
        for s, idx in zip(comp_cat[skey].values, comp_cat.index.values):
            idx_dict[s].append(idx)
    else:
        idx_dict = {s: [idx] for s, idx in zip(comp_cat.Source_Name.values, comp_cat.index.values)}
    return idx_dict


def get_name_dict(comp_cat, training_mode=False):
    """Create dict that returns coords of objects when given sourcename"""

    if training_mode:
        if 'Source_Name' in comp_cat.keys():
            skey = 'Source_Name'
        else:
            skey = 'Parent_Source'
        source_names = list(set(comp_cat[skey].values))
        coord_dict = {s: [] for s in source_names}
        for s, comp_s in zip(comp_cat[skey].values, comp_cat.Component_Name.values):
            coord_dict[s].append(comp_s)
    else:
        coord_dict = {s: comp_s for s, comp_s in zip(comp_cat.Source_Name.values,
                                                     comp_cat.Component_Name.values)}
    return coord_dict


def get_coord_dict(comp_cat, training_mode=False):
    """Create dict that returns coords of objects when given sourcename"""

    if training_mode:
        source_names = list(set(comp_cat['Source_Name'].values))
        coord_dict = {s: [] for s in source_names}
        for s, ra, dec in zip(comp_cat.Source_Name.values, comp_cat.RA.values, comp_cat.DEC.values):
            coord_dict[s].append([ra, dec])
    else:
        coord_dict = {s: [[ra, dec]] for s, ra, dec in zip(comp_cat.Source_Name.values,
                                                           comp_cat.RA.values, comp_cat.DEC.values)}
    return coord_dict


def is_within(x, y, xmin, ymin, xmax, ymax):
    """Return true if x, y lies within xmin,ymin,xmax,ymax.
    False otherwise.
    """
    if xmin <= x <= xmax and ymin <= y <= ymax:
        return True
    else:
        return False


def translate_to_new_midpoint(xs, ys, old_imshape, new_imshape):
    """Given rotated points translate them to the new midpoint of the image
    Needed if the image dimensions increase after rotation"""
    return xs + (new_imshape[0] - old_imshape[0]) / 2, ys + (new_imshape[1] - old_imshape[1]) / 2


def gt_box_calc(data, cutout, source, box_scale, segmented_cutout, wcs,
                remove_unresolved, angle_deg=0, training_mode=False):
    """In this function we design the bounding box encompassing all related radio
    components within the image fov, grow the bbox to the five sigma contours
    around these components (again within the fov) and shrink the bbox to avoid as
    many unrelated components as possible without excluding related components.
    segmented_cutout is created by phot_utils
    comp_dict is a pandas dataframe
    """
    # Function used for defining the box around the given source and creating the mask
    beam_size_pix = cutout.bmaj / cutout.res[0]
    # ax,ay = zip(*source.c_hull.hull_box())
    dx, dy = data.shape

    # use segmentation map to grow the box to the 5sigma contour level
    # 1: retrieve pixel coordinates of sources that make up this value added source
    pixel_xs = [source.x] + [oc.x for oc in cutout.other_components if oc.related]
    pixel_ys = [source.y] + [oc.y for oc in cutout.other_components if oc.related]
    len_related = len(pixel_xs)

    if angle_deg == 39 and cutout.c_source.sname == 'ILTJ123527.84+531457.1':
        print("source:", source)
        print(cutout.c_source.sname)
        # plt.imshow(data)
        # plt.show()
        plt.imshow(segmented_cutout.data)
        plt.plot(pixel_xs, pixel_ys, color='white', linestyle='None', marker='.')
        plt.title([pixel_xs, pixel_ys])
        plt.show()

    # Add coordinates along PA LGZ_width line
    if len(pixel_xs) == 1:
        delta_ra_deg = source.size * np.sin(np.deg2rad(source.PA + angle_deg))
        delta_dec_deg = source.size * np.cos(np.deg2rad(source.PA + angle_deg))
        delta_x = delta_ra_deg / (cutout.res[0] * 3600)
        delta_y = delta_dec_deg / (cutout.res[1] * 3600)
        mean_x, mean_y = np.mean(pixel_xs), np.mean(pixel_ys)
        probe_hull_list = np.array([-0.3, -0.2, -0.1, 0.1, 0.2, 0.3])
        pixel_xs += list(delta_x * probe_hull_list + mean_x)
        pixel_ys += list(delta_y * probe_hull_list + mean_y)
        # If source center also save the extra probing points such that they can be used when
        # reintuting unresolved objects that fall inside the convex hull of resolved sources that
        # lie inside the best predicted bbox
        if remove_unresolved and source.center and cutout.wide_focus == [[], []]:
            cutout.wide_focus = [list(delta_x * probe_hull_list + mean_x),
                                 list(delta_y * probe_hull_list + mean_y)]

    if angle_deg == 0 and cutout.c_source.sname == 'ILTJ123527.84+531457.1':
        # plt.imshow(data)
        # plt.show()
        plt.imshow(segmented_cutout.data)
        plt.plot(pixel_xs, pixel_ys, color='white', linestyle='None', marker='.')
        plt.show()

    # Check if central pixel location is inside the image
    if not 0 < pixel_xs[0] < (cutout.size_pixels - 1) or not 0 < pixel_ys[0] < (cutout.size_pixels - 1):
        source.in_bounds = False
        return False

    xmin_tight, ymin_tight = np.min(pixel_xs[:len_related]), np.min(pixel_ys[:len_related])
    xmax_tight, ymax_tight = np.max(pixel_xs[:len_related]), np.max(pixel_ys[:len_related])

    # 3: collect segmentation labels for these xs,ys
    try:
        labels = [segmented_cutout.data[int(round(y)), int(round(x))] for x, y in
                  zip(pixel_xs, pixel_ys) if 0 <= x < cutout.size_pixels - 1 and 0 <= y < cutout.size_pixels - 1]
    except:
        # """
        print(f'\nSegment and data shapes disagree?. Shape data {data.shape}, shape segment'
              f' {segmented_cutout.data}. Source flagged!\n')
        print([[x, y] for x, y in
               zip(pixel_xs, pixel_ys)])
        # """
        source.low_sigma_flag = 1
        print("Low sigma 1!")
        return False
    labels = [l for l in labels if not l == 0]
    labels = np.array(labels) - 1  # labels start at 1 as 0 is background

    # 3.5: if no labels are found, enlarge the search radius
    labels2 = labels
    r = 3  # Search radius
    if labels.size != len(pixel_xs):
        # print(f"number of nonzero labels {labels.size}, should be {len(pixel_xs)}")
        # print("Total number of labels:",segmented_cutout.nlabels)
        # print('labels:', labels+1)
        labels = list(set(flatten([
            segmented_cutout.data[int(round(y)) - r:int(round(y)) + r, int(round(x)) - r:int(round(x)) + r] for x, y in
            zip(pixel_xs, pixel_ys) if 0 <= x < cutout.size_pixels - 1 and 0 <= y < cutout.size_pixels - 1])))
        labels = [l for l in labels if not l == 0]
        labels = np.array(labels) - 1  # labels start at 1 as 0 is background
        # print('large search radius labels:', labels+1)
        # print(f"large number of nonzero labels {labels.size}, should be {len(pixel_xs)}")
        # if set(labels2) != set(labels):
        #    print(f'Larger search-area changed the outcome:',labels2, labels)

    # 4: get min/max for these segmentationislands: use slices!
    if labels.size != 0:
        segm_labels = np.array(segmented_cutout.slices)
        xmin = np.min([xslice.start - 1 for yslice, xslice in segm_labels[labels]])
        ymin = np.min([yslice.start - 1 for yslice, xslice in segm_labels[labels]])
        xmax = np.max([xslice.stop for yslice, xslice in segm_labels[labels]])
        ymax = np.max([yslice.stop for yslice, xslice in segm_labels[labels]])
    else:
        # print("No segmentation labels found")
        if angle_deg == 39 and source.center:
            plt.imshow(segmented_cutout.data)
            plt.plot(pixel_xs, pixel_ys, color='white', linestyle='None', marker='.')
            plt.show()

        source.low_sigma_flag = 2
        # print("Low sigma 2!")
        return False
    # print('after segmentation xmin,xmax,ymin,ymax for source: ', xmin,xmax,ymin,ymax)

    # Make sure that all related coords are included in the box
    # (Needed if these coords fall outside of segmentation islands due to being too faint)
    xmin, xmax = np.min(np.concatenate([[xmin], np.floor(pixel_xs[:len_related])])), np.max(np.concatenate([[xmax],
                                                                                                            np.ceil(
                                                                                                                pixel_xs[
                                                                                                                :len_related])]))
    ymin, ymax = np.min(np.concatenate([[ymin], np.floor(pixel_ys[:len_related])])), np.max(np.concatenate([[ymax],
                                                                                                            np.ceil(
                                                                                                                pixel_ys[
                                                                                                                :len_related])]))
    # print('after including xmin,xmax,ymin,ymax for source: ', xmin,xmax,ymin,ymax)
    if angle_deg == 39 and source.sname == 'ILTJ123527.84+531457.1':
        # plt.imshow(data)
        # plt.show()
        plt.imshow(segmented_cutout.data)
        plt.plot(pixel_xs, pixel_ys, color='white', linestyle='None', marker='.')
        plt.show()

    # Increasing box size by given factor
    if box_scale != 1:
        dx = ((xmax - xmin) * box_scale - (xmax - xmin)) / 2
        xmax, xmin = xmax + dx, xmin - dx
        dy = ((ymax - ymin) * box_scale - (ymax - ymin)) / 2
        ymax, ymin = ymax + dy, ymin - dy

    # Ensuring that the bounding box is at least as large as the beam
    ####DISABLED: Flag central sources that are smaller than the beam
    if abs(xmax - xmin) < beam_size_pix:
        xmax, xmin = source.x + beam_size_pix / 2, source.x - beam_size_pix / 2
        # if source.center:
        #    source.low_sigma_flag = 3
        #    #print("Low sigma 3!")
        #    return False
    if abs(ymax - ymin) < beam_size_pix:
        ymax, ymin = source.y + beam_size_pix / 2, source.y - beam_size_pix / 2
        # if source.center:
        #    source.low_sigma_flag = 3
        #    #print("Low sigma 4!")
        #    return False

    # If components are included in the area outside the tighthest
    # rectangular bounding box that can be drawn around the now considered components
    # Shrink the bounding box down to 2/3ths of the way
    shrink_factor = 0.66

    s = 1.5
    if remove_unresolved:
        x_unrelated = [oc.x for oc in cutout.other_components
                       if not oc.related and not oc.unresolved]
        y_unrelated = [oc.y for oc in cutout.other_components
                       if not oc.related and not oc.unresolved]
    else:
        x_unrelated = [oc.x for oc in cutout.other_components if not oc.related]
        y_unrelated = [oc.y for oc in cutout.other_components if not oc.related]
    for x, y in zip(x_unrelated, y_unrelated):
        if is_within(x, y, xmin, ymin, xmax, ymax) and \
                not is_within(x, y, xmin_tight, ymin_tight, xmax_tight, ymax_tight):
            condition_cleared = False

            if abs(x - pixel_xs[0]) > abs(y - pixel_ys[0]):
                if x > xmax_tight:
                    xmax = xmax_tight + shrink_factor * (x - xmax_tight)
                    condition_cleared = True
                if x < xmin_tight:
                    xmin = xmin_tight - shrink_factor * (xmin_tight - x)
                    condition_cleared = True
                if not condition_cleared:
                    if y > ymax_tight:
                        ymax = ymax_tight + shrink_factor * (y - ymax_tight)
                        condition_cleared = True
                    if y < ymin_tight:
                        ymin = ymin_tight - shrink_factor * (ymin_tight - y)
                        condition_cleared = True

            else:
                if y > ymax_tight:
                    ymax = ymax_tight + shrink_factor * (y - ymax_tight)
                    condition_cleared = True
                if y < ymin_tight:
                    ymin = ymin_tight - shrink_factor * (ymin_tight - y)
                    condition_cleared = True
                if not condition_cleared:
                    if x > xmax_tight:
                        xmax = xmax_tight + shrink_factor * (x - xmax_tight)
                        condition_cleared = True
                    if x < xmin_tight:
                        xmin = xmin_tight - shrink_factor * (xmin_tight - x)

    # Make sure the bounding box is within bounds
    in_bounds = True
    if xmin < 0:
        xmin = min(cutout.size_pixels - 1 - beam_size_pix, max(0, xmin))
        source.in_bounds = False
    if ymin < 0:
        ymin = min(cutout.size_pixels - 1 - beam_size_pix, max(0, ymin))
        source.in_bounds = False
    if xmax > cutout.size_pixels - 1:
        xmax = max(0 + beam_size_pix, min(cutout.size_pixels - 1, xmax))
        source.in_bounds = False
    if ymax > cutout.size_pixels - 1:
        ymax = max(0 + beam_size_pix, min(cutout.size_pixels - 1, ymax))
        source.in_bounds = False
    cutout.set_gt_bbox(min(xmax, xmin), min(ymin, ymax), max(xmax, xmin), max(ymin, ymax))
    return True


def box_calc(data, cutout, source, box_scale, segmented_cutout, wcs,
             remove_unresolved, angle_deg=0, training_mode=False):
    """In this function we design the bounding box encompassing all related radio
    components within the image fov, grow the bbox to the five sigma contours
    around these components (again within the fov) and shrink the bbox to avoid as
    many unrelated components as possible without excluding related components.
    segmented_cutout is created by phot_utils
    comp_dict is a pandas dataframe
    """
    # Function used for defining the box around the given source and creating the mask
    beam_size_pix = cutout.bmaj / cutout.res[0]
    # ax,ay = zip(*source.c_hull.hull_box())
    dx, dy = data.shape

    # use segmentation map to grow the box to the 5sigma contour level
    # 1: retrieve pixel coordinates of sources that make up this value added source
    pixel_xs = [source.x]
    pixel_ys = [source.y]
    len_related = len(pixel_xs)

    if angle_deg == 39 and cutout.c_source.sname == 'ILTJ135340.85+560619.7':
        # plt.imshow(data)
        # plt.show()
        plt.imshow(segmented_cutout.data)
        plt.plot(pixel_xs, pixel_ys, color='white', linestyle='None', marker='.')
        plt.title([pixel_xs, pixel_ys])
        plt.show()

    # Add coordinates along PA LGZ_Size line
    # if training_mode and len(pixel_xs) == 1:
    # if len(pixel_xs) == 1:
    # Testing to see if all works out if we always add coordinates along PA LGZ_Size line
    if 1 == 1:  # len([1 for oc in cutout.other_components if oc.related]) > 0:
        delta_ra_deg = source.size * np.sin(np.deg2rad(source.PA + angle_deg))
        delta_dec_deg = source.size * np.cos(np.deg2rad(source.PA + angle_deg))
        delta_x = delta_ra_deg / (cutout.res[0] * 3600)
        delta_y = delta_dec_deg / (cutout.res[1] * 3600)
        mean_x, mean_y = np.mean(pixel_xs), np.mean(pixel_ys)
        probe_hull_list = np.array([-0.3, -0.2, -0.1, 0.1, 0.2, 0.3])
        pixel_xs += list(delta_x * probe_hull_list + mean_x)
        pixel_ys += list(delta_y * probe_hull_list + mean_y)
        # If source center also save the extra probing points such that they can be used when
        # reintuting unresolved objects that fall inside the convex hull of resolved sources that
        # lie inside the best predicted bbox
        if remove_unresolved and source.center and cutout.wide_focus == [[], []]:
            cutout.wide_focus = [list(delta_x * probe_hull_list + mean_x),
                                 list(delta_y * probe_hull_list + mean_y)]

    if angle_deg == 39 and cutout.c_source.sname == 'ILTJ135340.85+560619.7':
        fig, fig_ax = plt.subplots(1, 2)
        fig_ax[0].imshow(data)
        fig_ax[1].imshow(segmented_cutout.data)
        fig_ax[1].plot(pixel_xs, pixel_ys, color='white', linestyle='None', marker='.')
        plt.show()
        # sdfd

    # Check if central pixel location is inside the image
    if not 0 < pixel_xs[0] < (cutout.size_pixels - 1) or not 0 < pixel_ys[0] < (cutout.size_pixels - 1):
        source.in_bounds = False
        return False

    xmin_tight, ymin_tight = np.min(pixel_xs[:len_related]), np.min(pixel_ys[:len_related])
    xmax_tight, ymax_tight = np.max(pixel_xs[:len_related]), np.max(pixel_ys[:len_related])

    # 3: collect segmentation labels for these xs,ys
    try:
        labels = [segmented_cutout.data[int(round(y)), int(round(x))] for x, y in
                  zip(pixel_xs, pixel_ys)]
    except:
        """
        print(f'\nSegment and data shapes disagree?. Shape data {data.shape}, shape segment'
               f' {segmented_cutout.data}. Source flagged!\n')
        print([[x,y] for x,y in 
            zip(pixel_xs, pixel_ys)])
        """
        source.low_sigma_flag = 4
        return False
    labels = [l for l in labels if not l == 0]
    labels = np.array(labels) - 1  # labels start at 1 as 0 is background

    # 3.5: if no labels are found, enlarge the search radius
    labels2 = labels
    r = 3  # Search radius
    if labels.size != len(pixel_xs):
        # print(f"number of nonzero labels {labels.size}, should be {len(pixel_xs)}")
        # print("Total number of labels:",segmented_cutout.nlabels)
        # print('labels:', labels+1)
        labels = list(set(flatten([
            segmented_cutout.data[int(round(y)) - r:int(round(y)) + r, int(round(x)) - r:int(round(x)) + r] for x, y in
            zip(pixel_xs, pixel_ys)])))
        labels = [l for l in labels if not l == 0]
        labels = np.array(labels) - 1  # labels start at 1 as 0 is background
        # print('large search radius labels:', labels+1)
        # print(f"large number of nonzero labels {labels.size}, should be {len(pixel_xs)}")
        # if set(labels2) != set(labels):
        #    print(f'Larger search-area changed the outcome:',labels2, labels)

    # 4: get min/max for these segmentationislands: use slices!
    if labels.size != 0:
        segm_labels = np.array(segmented_cutout.slices)
        xmin = np.min([xslice.start - 1 for yslice, xslice in segm_labels[labels]])
        ymin = np.min([yslice.start - 1 for yslice, xslice in segm_labels[labels]])
        xmax = np.max([xslice.stop for yslice, xslice in segm_labels[labels]])
        ymax = np.max([yslice.stop for yslice, xslice in segm_labels[labels]])
    else:
        # print("No segmentation labels found")
        if angle_deg == 9 and source.center:
            plt.imshow(segmented_cutout.data)
            plt.plot(pixel_xs, pixel_ys, color='white', linestyle='None', marker='.')
            plt.show()

        source.low_sigma_flag = 5
        return False

    # Increasing box size by given factor
    if box_scale != 1:
        dx = ((xmax - xmin) * box_scale - (xmax - xmin)) / 2
        xmax, xmin = xmax + dx, xmin - dx
        dy = ((ymax - ymin) * box_scale - (ymax - ymin)) / 2
        ymax, ymin = ymax + dy, ymin - dy

    # Ensuring that the bounding box is at least as large as the beam
    #####DISABLED: Flag central sources that are smaller than the beam
    if abs(xmax - xmin) < beam_size_pix:
        xmax, xmin = source.x + beam_size_pix / 2, source.x - beam_size_pix / 2
        # if source.center:
        #    source.low_sigma_flag = 6
        #    return False
    if abs(ymax - ymin) < beam_size_pix:
        ymax, ymin = source.y + beam_size_pix / 2, source.y - beam_size_pix / 2
        # if source.center:
        #    source.low_sigma_flag = 6
        #    return False

    # If components are included in the area outside the tighthest
    # rectangular bounding box that can be drawn around the now considered components
    # Shrink the bounding box down to 2/3ths of the way
    shrink_factor = 0.66
    if remove_unresolved:
        x_other = [oc.x for oc in cutout.other_components
                   if not oc.unresolved]
        y_other = [oc.y for oc in cutout.other_components
                   if not oc.unresolved]
    else:
        x_other = [oc.x for oc in cutout.other_components]
        y_other = [oc.y for oc in cutout.other_components]
    for x, y in zip(x_other, y_other):
        if is_within(x, y, xmin, ymin, xmax, ymax) and \
                not is_within(x, y, xmin_tight, ymin_tight, xmax_tight, ymax_tight):
            condition_cleared = False

            if abs(x - pixel_xs[0]) > abs(y - pixel_ys[0]):
                if x > xmax_tight:
                    xmax = xmax_tight + shrink_factor * (x - xmax_tight)
                    condition_cleared = True
                if x < xmin_tight:
                    xmin = xmin_tight - shrink_factor * (xmin_tight - x)
                    condition_cleared = True
                if not condition_cleared:
                    if y > ymax_tight:
                        ymax = ymax_tight + shrink_factor * (y - ymax_tight)
                        condition_cleared = True
                    if y < ymin_tight:
                        ymin = ymin_tight - shrink_factor * (ymin_tight - y)
                        condition_cleared = True

            else:
                if y > ymax_tight:
                    ymax = ymax_tight + shrink_factor * (y - ymax_tight)
                    condition_cleared = True
                if y < ymin_tight:
                    ymin = ymin_tight - shrink_factor * (ymin_tight - y)
                    condition_cleared = True
                if not condition_cleared:
                    if x > xmax_tight:
                        xmax = xmax_tight + shrink_factor * (x - xmax_tight)
                        condition_cleared = True
                    if x < xmin_tight:
                        xmin = xmin_tight - shrink_factor * (xmin_tight - x)

    # Make sure the bounding box is within bounds
    in_bounds = True
    if xmin < 0:
        xmin = min(cutout.size_pixels - 1 - beam_size_pix, max(0, xmin))
        source.in_bounds = False
    if ymin < 0:
        ymin = min(cutout.size_pixels - 1 - beam_size_pix, max(0, ymin))
        source.in_bounds = False
    if xmax > cutout.size_pixels - 1:
        xmax = max(0 + beam_size_pix, min(cutout.size_pixels - 1, xmax))
        source.in_bounds = False
    if ymax > cutout.size_pixels - 1:
        ymax = max(0 + beam_size_pix, min(cutout.size_pixels - 1, ymax))
        source.in_bounds = False

    source.set_box_dimensions(min(float(xmin), float(xmax)), max(float(xmax), float(xmin)),
                              min(float(ymin), float(ymax)), max(float(ymin), float(ymax)), in_bounds)
    return True


def rotate_points(xs, ys, angle_deg: float, midpoint: tuple):
    """Given coordinates xs, ys, rotate them counterclockwise with angle in degrees around 
    a given midpoint=(x,y)"""
    angle_rad = np.radians(angle_deg)
    xs = xs - midpoint[0]
    ys = ys - midpoint[1]
    cos_rad = np.cos(angle_rad)
    sin_rad = np.sin(angle_rad)
    new_xs = (xs * cos_rad) - (ys * sin_rad)
    new_ys = (xs * sin_rad) + (ys * cos_rad)
    return new_xs + midpoint[0], new_ys + midpoint[1]


def label_maker_angle(cutout, source, box_scale, data, rms, cutout_PATH, sig5_isl, wcs,
                      angle_deg, remove_unresolved, training_mode=True, sigma_box_fit=5):
    # If unresolved but related keep source (no bounding box needed)
    if remove_unresolved and not source.center and source.related and source.unresolved:
        return True

    # Setting phot_utils parameters 
    n_pixel = sig5_isl
    # Caculating the rms and threshhold
    threshold = sigma_box_fit * rms
    segment_whole_cutout = detect_sources(data, threshold, n_pixel)
    # Calculating box
    if not data.shape == segment_whole_cutout.data.shape:
        print('Segment and data shapes disagree. Source flagged!')
        source.low_sigma_flag = 7
        return False
    if source.center:
        box_success = gt_box_calc(data, cutout, source, box_scale, segment_whole_cutout,
                                  wcs, remove_unresolved, angle_deg=angle_deg, training_mode=training_mode)
    box_success = box_calc(data, cutout, source, box_scale, segment_whole_cutout,
                           wcs, remove_unresolved, angle_deg=angle_deg, training_mode=training_mode)
    if box_success:
        return True
    return False


def label_maker(cutout, source, box_scale, data, rms, cutout_PATH, sig5_isl, wcs,
                remove_unresolved, segment_dir, training_mode=True, sigma_box_fit=5):
    # If unresolved but related keep source (no bounding box needed)
    if cutout.c_source.sname == 'ILTJ141257.00+561206.8':
        print(f"We arrived at source {cutout.c_source.sname}")
        print("rem_unr, center, related, unresolv:", remove_unresolved,
              source.center, source.related, source.unresolved)
    if remove_unresolved and not source.center and source.related and source.unresolved:
        return True

    # Setting phot_utils parameters 
    n_pixel = sig5_isl
    # Caculating the rms and threshhold
    threshold = sigma_box_fit * rms
    # print("data:", np.shape(data), np.shape(threshold), np.shape(rms))
    segment_whole_cutout = detect_sources(data, threshold, n_pixel)
    size_pix = data.shape[0]
    if segment_whole_cutout is None or not data.shape == segment_whole_cutout.data.shape:
        print('Segment and data shapes disagree. Source flagged!')
        source.low_sigma_flag = 8
        if cutout.c_source.sname == 'ILTJ141257.00+561206.8':
            print(f"Segment and data  shapes disagree for {cutout.c_source.sname}")
        return False
    # Save segment object
    segment_save_path = os.path.join(segment_dir,
                                     f"{cutout.c_source.sname}_{data.shape[0]}_{sigma_box_fit}sigma.pkl")
    if cutout.c_source.sname == 'ILTJ141257.00+561206.8':
        print(f"Segment save path, and exists {segment_save_path} {os.path.exists(segment_save_path)}")
    if remove_unresolved and not os.path.exists(segment_save_path):
        with open(segment_save_path, 'wb') as f:
            pickle.dump(segment_whole_cutout, f, protocol=4)

    # Calculating box
    if source.center:
        # print("Central source gt prior:",
        #        [cutout.gt_xmin,cutout.gt_ymin,cutout.gt_xmax,cutout.gt_ymax])
        box_success = gt_box_calc(data, cutout, source, box_scale, segment_whole_cutout,
                                  wcs, remove_unresolved, training_mode=training_mode)
        # print("Central source gt after:",
        #        [cutout.gt_xmin,cutout.gt_ymin,cutout.gt_xmax,cutout.gt_ymax])
    box_success = box_calc(data, cutout, source, box_scale, segment_whole_cutout,
                           wcs, remove_unresolved, training_mode=training_mode)
    # if source.center:
    #    print("Central source bbox after:", 
    #            [cutout.c_source.xmin, cutout.c_source.ymin,
    #                cutout.c_source.xmax, cutout.c_source.ymax])

    if not box_success:
        return source

    return True


def hull_plotter(source, axarr, edge_cases, central_source=False):
    # Create convex hull overlay
    # Extra box and hull dimensions
    xmax, xmin = source.xmax, source.xmin
    ymax, ymin = source.ymax, source.ymin
    squ_w = xmax - xmin
    squ_l = ymax - ymin
    ax, ay = source.ax, source.ay

    linestyle = '-'

    # Boundary check
    if not source.in_bounds:
        if edge_cases == False:
            return
        else:
            print(f'Warning: {source.sname} fell outside of the cut-out, will still be labeled.')

    # Assign different colors to 'main' source
    if central_source:
        edgecolor_squr = 'r'
        edgecolor_elps = 'orange'
        hull_c = 'y'
    else:
        edgecolor_squr = 'pink'
        edgecolor_elps = 'b'
        hull_c = 'cyan'

    # Convex hull 
    axarr[1].plot(ax, ay, color=hull_c, linestyle=linestyle)
    # axarr[3].plot(ax,ay,color = hull_c,linestyle=linestyle)

    # Red bounding box 
    axarr[1].add_patch(patches.Rectangle(
        (xmin, ymin), squ_w, squ_l, linewidth=1, edgecolor=edgecolor_squr, facecolor='none', linestyle=linestyle))
    axarr[2].add_patch(patches.Rectangle(
        (xmin, ymin), squ_w, squ_l, linewidth=1, edgecolor=edgecolor_squr, facecolor='none', linestyle=linestyle))
    # axarr[3].add_patch(patches.Rectangle(
    #    (xmin,ymin),squ_w,squ_l,linewidth=1,edgecolor=edgecolor_squr,facecolor='none',linestyle=linestyle))
    axarr[4].add_patch(patches.Rectangle(
        (xmin, ymin), squ_w, squ_l, linewidth=1, edgecolor=edgecolor_squr, facecolor='none', linestyle=linestyle))

    # White text: classlabels
    axarr[4].text(xmin, ymax + 2, '{}C_{}P'.format(
        int(source.n_comp), int(source.n_peak)), color='w', size=9)


def plotter(cutout, data_DR2, rms_DR2, infrared, DEBUG_PATH, edge_cases,
            low_sig, overlap_allowed, incl_difficult, field_name, save=False, save_appendix='debug',
            angle_deg=0):
    """
    Visualize the label process and show what we can/will offer our network as input.
    Consider 8 subplots, 2 rows, 4 columns.
    The first row uses DR2 radio maps if radio, 
    -The first column shows just the radio intensity map (sqrt scaled).
    -The second column shows the radio intensity map, overlayed with the Lofar Galaxy Zoo convex hull
    and optical ID ground truth annotations, plus the rectangular box drawn by us around that hull.
    -The third column shows the 5 and 4 sigma contourlines of the radio intensity map, plus all the
    stuff overlayed in the previous column.
    -The fourth column shows the infrared (WISE) intensity map overlayed with everything in the third and
    second column.
    -The fifth column shows the production of the class labels using photutils to find the number of
    components and the number of peaks
    - edge_cases is used to determine wether or not the sources on the edge are being plotted
    """
    # Initialize figure

    """
    if cutout.infrared_flag:
        print('Flagged! - Infrared not available')
        return
    elif not cutout.c_source.in_bounds:
        print('Flagged! - Central source not in bounds')
        return
    elif cutout.c_source.low_sigma_flag and not low_sig:
        print('Flagged! - Low sigma')
        return
    if not incl_difficult and cutout.difficult_flag:
        print('Flagged! - Difficult')
        return
    """

    labelsize = 6
    scaling = simple_norm(data_DR2, stretch='sqrt')
    number_of_rows = 1
    number_of_columns = 5
    f, axarr = plt.subplots(number_of_rows, number_of_columns, figsize=(15, 3))

    # Set subplot titles
    titles = ['LoTSS intensity sqrt scaled\nand optical ID (red cross)', 'LRGZ convex hull' \
                                                                         '(yellow)\nand our bounding box (red)',
              'LoTSS intensity contours\n at 3,4,5 sigma (black,grey,white)', '', 'Generated classlabels']
    for i, title in enumerate(titles):
        axarr[i].set_title(title, fontsize=7)
    for ax in axarr.flat:
        ax.set(ylabel='[pixels]')
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
        ax.set_aspect('equal', 'box')
        ax.label_outer()
        ax.invert_yaxis()
    axarr[0].set(ylabel='DR2 [pixels]')
    # Hide x labels and tick labels for top plots and y ticks for right plots.

    # First and second column: radio intensity maps
    for i in range(3):
        axarr[i].imshow(data_DR2, norm=scaling, origin='lower')
    # Third column: radio intensity contour maps
    if not data_DR2 is None and not rms_DR2 is None:
        axarr[2].contour(data_DR2 / rms_DR2, levels=[3, 4, 5], colors=['black', 'grey', 'white'], alpha=0.5)
        # axarr[3].contour(data_DR2/rms_DR2, levels=[3,4,5], colors=['black','grey','white'], alpha=0.5)

    # Fourth and fith column: segmentation maps and infrared
    ##############################
    sigma = 5.
    # Caculating the rms and threshhold
    threshold = sigma * rms_DR2
    segment_whole_cutout = detect_sources(data_DR2, threshold, 1)
    ##############################

    sum_segm = cutout.c_source.segm
    for i in range(len(cutout.other_sources)):
        # Adding segmentation map if in bounds
        if cutout.other_sources[i].in_bounds or edge_cases:
            sum_segm += cutout.other_sources[i].segm
    # axarr[4].imshow(sum_segm, origin='lower')
    axarr[4].imshow(segment_whole_cutout.data, origin='lower')
    # Load component locations
    xs, ys = cutout.focussed_comp
    # plot component locations
    axarr[4].plot(xs, ys, 'wo')

    # Showing infrared if available
    if not infrared is None:
        axarr[3].imshow(infrared, origin='lower')

    # Insert optical ID location if available    
    if hasattr(cutout.c_source, "ID_ra"):
        optical_skycoord = SkyCoord([cutout.c_source.ID_ra], [cutout.c_source.ID_dec], unit=u.deg)
        optical_id_x_position, optical_id_y_position = utils.skycoord_to_pixel(optical_skycoord, cutout.w, 0)
        for i in range(number_of_columns):
            axarr[i].plot(optical_id_x_position, optical_id_y_position, 'rx')

    hull_plotter(cutout.c_source, axarr, edge_cases, central_source=True)
    for i in range(len(cutout.other_sources)):
        # Ensures that low sigma sources and overlaping sources are excluded
        if not cutout.c_source.in_bounds or \
                (cutout.other_sources[i].low_sigma_flag > 0 and not low_sig) or \
                (cutout.other_sources[i].overlap_flag and not overlap_allowed):
            continue
        else:
            hull_plotter(cutout.other_sources[i], axarr, edge_cases)

    plt.suptitle((f'{cutout.c_source.sname}   RA {cutout.c_source.ra:.3f}; DEC {cutout.c_source.dec:.3f}'
                  f' Field: {field_name} Index: {cutout.index} Rot. {angle_deg} deg'),
                 fontsize=7)

    if save:
        plt.savefig(os.path.join(DEBUG_PATH, field_name + '_' + cutout.c_source.sname + f'_{save_appendix}.png'),
                    bbox_inches='tight')
    else:
        plt.show()
    plt.close()


# end plotter

def plotter_DR1_and_DR2(cutout, data_DR2, rms_DR2, data_DR1, rms_DR1, infrared, DEBUG_PATH, edge_cases,
                        low_sig, overlap_allowed, incl_difficult, save=False):
    """
    Visualize the label process and show what we can/will offer our network as input.
    Consider 8 subplots, 2 rows, 4 columns.
    The first row uses DR2 radio maps if radio, the second row uses radio DR1 maps.
    -The first column shows just the radio intensity map (sqrt scaled).
    -The second column shows the radio intensity map, overlayed with the Lofar Galaxy Zoo convex hull
    and optical ID ground truth annotations, plus the rectangular box drawn by us around that hull.
    -The third column shows the 5 and 4 sigma contourlines of the radio intensity map, plus all the
    stuff overlayed in the previous column.
    -The fourth column shows the infrared (WISE) intensity map overlayed with everything in the third and
    second column.
    -The fifth column shows the production of the class labels using photutils to find the number of
    components and the number of peaks
    - edge_cases is used to determine wether or not the sources on the edge are being plotted
    """
    # Initialize figure

    if cutout.infrared_flag:
        print('Flagged! - Infrared not available')
        return
    elif not cutout.c_source.in_bounds:
        print('Flagged! - Central source not in bounds')
        return
    elif cutout.c_source.low_sigma_flag > 0 and not low_sig:
        print('Flagged! - Low sigma')
        return
    if not incl_difficult and cutout.difficult_flag:
        print('Flagged! - Difficult')
        return

    labelsize = 6
    scaling = simple_norm(data_DR1, stretch='sqrt')

    number_of_rows = 2
    number_of_columns = 5

    f, axarr = plt.subplots(number_of_rows, number_of_columns, figsize=(25, 10))

    # Set subplot titles
    titles = ['LoTSS intensity sqrt scaled\nand optical ID (red cross)', 'LRGZ convex hull' \
                                                                         '(yellow)\nand our bounding box (red)',
              'LoTSS intensity contours\n5sigma (white), 4sigma (grey), 3sigma (black)', 'WISE intensity',
              'Generated classlabels']
    for i, title in zip(range(number_of_columns), titles):
        axarr[0, i].set_title(title)
    for ax in axarr.flat:
        ax.set(ylabel='[pixels]')
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
    axarr[0, 0].set(ylabel='DR1 [pixels]')
    axarr[1, 0].set(ylabel='DR2 [pixels]')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axarr.flat:
        ax.label_outer()
        ax.invert_yaxis()

    # First and second column: radio intensity maps
    for i in range(3):
        axarr[0, i].imshow(data_DR1, norm=scaling, origin='lower')
        if not data_DR2 is None:
            axarr[1, i].imshow(data_DR2, norm=scaling, origin='lower')
            # Third column: radio intensity contour maps
    if not data_DR2 is None and not rms_DR2 is None:
        axarr[1, 2].contour(data_DR2 / rms_DR2, levels=[3, 4, 5], colors=['black', 'grey', 'white'], alpha=0.5)
        axarr[1, 3].contour(data_DR2 / rms_DR2, levels=[3, 4, 5], colors=['black', 'grey', 'white'], alpha=0.5)
    if not data_DR1 is None and not rms_DR1 is None:
        axarr[0, 2].contour(data_DR1 / rms_DR1, levels=[3, 4, 5], colors=['black', 'grey', 'white'], alpha=0.5)
        axarr[0, 3].contour(data_DR1 / rms_DR1, levels=[3, 4, 5], colors=['black', 'grey', 'white'], alpha=0.5)

    # Fourth and fith column: segmentation maps and infrared
    sum_segm = cutout.c_source.segm
    for i in range(len(cutout.other_sources)):
        # Adding segmentation map if in bounds
        if cutout.other_sources[i].in_bounds or edge_cases:
            sum_segm += cutout.other_sources[i].segm
    for i in range(number_of_rows):
        axarr[i, 4].imshow(sum_segm, origin='lower')
        # Showing infrared if available
        if not infrared is None:
            axarr[i, 3].imshow(infrared, origin='lower')

    # Insert optical ID location if available    
    optical_skycoord = SkyCoord([cutout.c_source.ID_ra], [cutout.c_source.ID_dec], unit=u.deg)
    optical_id_x_position, optical_id_y_position = utils.skycoord_to_pixel(optical_skycoord, cutout.w, 0)
    for i in range(number_of_columns):
        axarr[0, i].plot(optical_id_x_position, optical_id_y_position, 'rx')
        axarr[1, i].plot(optical_id_x_position, optical_id_y_position, 'rx')

    hull_plotter(cutout.c_source, axarr, edge_cases, central_source=True)
    for i in range(len(cutout.other_sources)):
        # Ensures that low sigma sources and overlaping sources are excluded
        if not cutout.c_source.in_bounds or \
                cutout.other_sources[i].low_sigma_flag > 0 and not low_sig or \
                cutout.other_sources[i].overlap_flag and not overlap_allowed:
            continue
        else:
            hull_plotter(cutout.other_sources[i], axarr, edge_cases)

    plt.suptitle(f'{cutout.c_source.sname}   RA {cutout.c_source.ra:.3f}; DEC {cutout.c_source.dec:.3f} \n'
                 f'Index: {cutout.index} ', fontsize=7)

    if save:
        plt.savefig(os.path.join(DEBUG_PATH, field_name + '_' + cutout.c_source.sname + '_debug.png'))
    else:
        plt.show()
    plt.close()


def make_list(l, DATA_PATH, lname, edge_cases, low_sig, overlap_allowed, incl_difficult):
    # Saves list to csv
    list_columns = ['Orig_Source_Name',
                    'Source_Name', 'RA', 'DEC', 'xmin', 'xmax', 'ymin', 'ymax', 'n_comp', 'n_peak',
                    'rotation_angle_deg']
    ll = []
    list_skipped = []

    for i in range(len(l)):
        cs = l[i].c_source
        if l[i].infrared_flag:
            list_skipped.append(cs.sname)
            continue
        elif not hasattr(cs, 'in_bounds'):
            list_skipped.append(cs.sname)
            continue
        elif not cs.in_bounds:
            list_skipped.append(cs.sname)
            continue
        elif cs.low_sigma_flag > 0 and not low_sig:
            list_skipped.append(cs.sname)
            continue
        elif l[i].difficult_flag and not incl_difficult:
            list_skipped.append(cs.sname)
            continue

        ll.append(np.array([cs.sname, cs.sname, cs.ra, cs.dec, cs.xmin, cs.xmax, cs.ymin, cs.ymax, cs.n_comp, cs.n_peak,
                            l[i].rotation_angle_deg]))

        for j in range(len(l[i].other_sources)):
            osc = l[i].other_sources[j]
            if osc.in_bounds:
                if (osc.low_sigma_flag > 0 and not low_sig) or (osc.overlap_flag and not overlap_allowed):
                    continue
                else:
                    ll.append(np.array(
                        [cs.sname, osc.sname, osc.ra, osc.dec, osc.xmin, osc.xmax, osc.ymin, osc.ymax, osc.n_comp,
                         osc.n_peak]))
            else:
                print("not in bounds!")

    ll = pd.DataFrame(ll, columns=list_columns)
    ll.to_csv(path_or_buf=os.path.join(DATA_PATH, lname), sep=',', index=False)
    print(
        f'Saved list of {len(l) - len(list_skipped)} labeled sources to {DATA_PATH} with name {lname}. \n Filtered out {len(list_skipped)} sources due to: invalid radio cut-out, exceedingly large bounding boxes, bounding box overlap or signal below 5 sigma.')


def return_multi_comp_compnames(compcat, cache_path, save_appendix='', training_mode=False):
    """Counts the number and percentage of single component sources in all_image_dir.
    Returns the filenames of the multi-component sources."""
    if not training_mode: return []
    multi_save_path = os.path.join(cache_path, f"save_multi_comp_source_names_{save_appendix}.npy")
    if 'Source_Name' in compcat.keys():
        skey = 'Source_Name'
    else:
        skey = 'Parent_Source'
    if not os.path.exists(multi_save_path):
        # Load component catalogue
        comp_name_to_source_name_dict = {n: i for i, n in zip(compcat[skey].values,
                                                              compcat.Component_Name.values)}

        # Count the number of components per source name in the catalogue
        counts = pd.value_counts(compcat[skey])
        source_names = list(set(compcat[skey].values))

        # Retrieve number of components per central source
        multi_comp_source_names = [comp_name for comp_name in
                                   compcat.Component_Name.values
                                   if counts[comp_name_to_source_name_dict[comp_name]] > 1]
        np.save(multi_save_path, multi_comp_source_names)
    else:
        multi_comp_source_names = np.load(multi_save_path)

    return multi_comp_source_names
