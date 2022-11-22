""" Make quick png for all sources that are skipped 
due to missing central flux.
"""
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.visualization import MinMaxInterval, SqrtStretch

from label_library import load_fits

clip_image = True
scaling = False  # 'sqrt'
path_to_inspect = '/data2/mostertrij/data/frcnn_images/uLF300_precomputed_removed_test'
cutouts_path = '/data2/mostertrij/data/frcnn_images/cutouts'
file_with_files_to_inspect = os.path.join(path_to_inspect,
                                          'requires_visual_inspection_central_source_too_faint.txt')
source_names = pd.read_csv(file_with_files_to_inspect,
                           header=0, names=['names', 'flag'])['names'].values
flags = pd.read_csv(file_with_files_to_inspect,
                    header=0, names=['names', 'flag'])['flag'].values

flag_dict = {0: "highflux",
             1: "Component coordinates do not match up with cutout segments (in gt_box_calc)",
             2: "No segmentation labels found (in gt_box_calc)",
             3: "Bbox would be smaller than the beam (in gt_box_calc)",
             4: "Component coordinates do not match up with cutout segments (in box_calc)",
             5: "No segmentation labels found (in box_calc)",
             6: "Bbox would be smaller than the beam (in box_calc)",
             7: "Segment and cutout shape disagree (in label_maker_angle)",
             8: "Segment and cutout shape disagree (in label_maker)"}

# Output pngs here:
save_dir = os.path.join(path_to_inspect, 'debug_central_low_flux')
os.makedirs(save_dir, exist_ok=True)

count = Counter(flags)
[print(f'{v} {flag_dict[int(k)]}') for k, v in count.most_common()]

for source_name, flag in zip(source_names, flags):
    assert int(flag) > 0, "Source does not seem to be lowflux?"
    fits_filepath = os.path.join(cutouts_path, source_name + '_300arcsec_large_radio_DR2_removed.fits')
    rms_filepath = os.path.join(cutouts_path, source_name + '_300arcsec_large_radio_rms_DR2_removed.fits')
    field, hdr = load_fits(fits_filepath)
    # Load the rms 
    rms, rms_hdr = load_fits(rms_filepath)
    checkfield = field / rms
    w = int(field.shape[0] / 2)
    wc = 10
    maxval = np.max(checkfield[w - wc:w + wc, w - wc:w + wc])
    lower_clip = -1e9
    upper_clip = maxval * 3
    if clip_image:
        # Clip field in sigma space 
        field = np.clip(field / rms, lower_clip, upper_clip)
    else:
        field = field / rms
    # Ensures that the image color-map is within the clip values for all images
    interval = MinMaxInterval()
    # Normalize values to lie between 0 and 1 and then apply a stretch
    if scaling == 'sqrt':
        stretch = SqrtStretch()
        field = stretch(interval(field))

    # plt.imsave(os.path.join(save_dir,source_name+'.png'),field, cmap='viridis', origin='lower')
    c = 30
    w -= 30
    wc = wc * (field.shape[0] - 2 * c) / field.shape[0]
    field = field[c:-c, c:-c]
    plt.figure(figsize=(10, 10))
    plt.imshow(field, cmap='viridis', origin='lower')
    plt.plot(field.shape[0] / 2, field.shape[1] / 2, 'r+')
    plt.plot([w - wc, w + wc, w + wc, w - wc, w - wc], [w - wc, w - wc, w + wc, w + wc, w - wc], 'r')
    plt.title(flag_dict[int(flag)] + f" Maxsignal {maxval:.2f}")
    plt.colorbar(label='signal / rms')
    plt.ylabel('arcsec')
    plt.xlabel('arcsec')
    plt.savefig(os.path.join(save_dir, source_name + '.png'), bbox_inches='tight')
    plt.close()

print("Saved all sources that were skipped due to low central flux in:", save_dir)
