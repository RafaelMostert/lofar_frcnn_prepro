"""
This script reads a file argv[1] with plain text radio component names in it
looks for the corresponding fits cutouts in argv[2]
and transforms those fits cutouts into pngs, creating those in a user defined directory argv[3]
Example:
    python fits_to_png.py requires_visual_inspection_central_source_too_faint.txt ../cutouts png_dir
Or with component size and location:
    python fits_to_png.py \
    -i /data2/mostertrij/data/frcnn_images/uL300_precomputed_removed/requires_visual_inspection_central_source_too_faint.txt \
    -f /data2/mostertrij/data/frcnn_images/cutouts \
    -d /data2/mostertrij/data/frcnn_images_uL300_precomputed_removed/png_dir /data2/mostertrij/data/catalogues/LoTSS_DR1_corrected_cat.comp.h5
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.visualization import MinMaxInterval, SqrtStretch
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from matplotlib.patches import Ellipse

parser = argparse.ArgumentParser()

# -db DATABSE -u USERNAME -p PASSWORD -size 20
parser.add_argument("-i", "--text", type=str, help="Path to file with radio component names")
parser.add_argument("-d", "--dest", type=str, help="Path to directory to store pngs")
parser.add_argument("-f", "--fits_dir", type=str, help="Path to directory that contains fits files")
parser.add_argument("-c", "--cat_path", type=str, help="Path to component catalogue")
parser.add_argument("-u", "--upper_sigma", type=float, default=30, help="Upper sigma clip limit")
parser.add_argument("-l", "--lower_sigma", type=float, default=1, help="Lower sigma clip limit")

args = parser.parse_args()
debug = False


def debug_print(*x):
    if debug:
        print(*x)


def load_fits(fits_filepath, verbose=False):
    if verbose:
        print('Loading the following fits file:', fits_filepath)
    # Load first fits file
    with fits.open(fits_filepath) as hdulist:
        hdu = hdulist[0].data
        # Header
        hdr = hdulist[0].header
    return hdu, hdr


# Read component names
# assert len(argv) >= 4, "Requires 1) path to file with component names, 2) path to fits cutouts and 3) a path to a outputdirectory as input."
source_file_path = args.text
fits_dir = args.fits_dir
output_dir = args.dest
assert os.path.exists(source_file_path)
assert os.path.exists(fits_dir)
os.makedirs(output_dir, exist_ok=True)
plot_component_gaussian = False
if not args.cat_path is None:
    plot_component_gaussian = True
    cat_path = args.cat_path
    assert os.path.exists(cat_path)
    cat = pd.read_hdf(cat_path)
    cat = cat.set_index('Component_Name')

# Read component names file
with open(source_file_path, 'r') as f:
    lines = f.read().split('\n')
    # Skip lines that do not appear to contain a radio component name
    lines = [l for l in lines if l.startswith('ILTJ')]
print(f"We will convert {len(lines)} FITS files to pngs.")
# lines = ['ILTJ110449.95+472738.9'] + lines


# Create paths to fits files and paths to pngs
fits_paths = [os.path.join(fits_dir, l + '_300arcsec_large_radio_DR2_removed.fits') for l in lines]
rms_paths = [os.path.join(fits_dir, l + '_300arcsec_large_radio_rms_DR2_removed.fits') for l in lines]
png_paths = [os.path.join(output_dir, l + '_300arcsec_large_radio_DR2_removed.png') for l in lines]

# Loop over input and output paths and convert fits to pngs
interval = MinMaxInterval()
stretch = SqrtStretch()
for i, (name, fits_path, rms_path, png_path) in enumerate(zip(lines, fits_paths, rms_paths, png_paths)):
    if i == 0: print(f"This script will convert {fits_path} to {png_path}")
    print(f"{i + 1}/{len(lines)}")
    field, hdr = load_fits(fits_path)
    rms, _ = load_fits(rms_path)
    # Clip field to exclude noise and contrast ruining bright signal
    field = np.clip(field / rms, args.lower_sigma, args.upper_sigma)
    # Change to sqrt scaling to highlight faint signal
    field = stretch(interval(field))
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, aspect='equal')
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(field, cmap='viridis', aspect='equal', origin='lower')
    if plot_component_gaussian:
        row = cat.loc[name]
        ra, dec, min_size, maj_size, PA = row.RA, row.DEC, row.Min, row.Maj, row.PA
        w, h = np.shape(field)
        debug_print(f"plotting component location within image with dimensions {np.shape(field)}")
        debug_print("ra,dec:", ra, dec)
        skycoord = SkyCoord(ra=ra, dec=dec, unit='deg')
        x, y = skycoord_to_pixel(skycoord, WCS(hdr), 0)
        debug_print("x,y:", x, y)
        ax.plot(x, y, marker='.', color='red', alpha=0.4)
        el = Ellipse((x, y), width=min_size, height=maj_size, angle=PA, fill=False, edgecolor='red', linewidth=1)
        ax.add_artist(el)
        # Plot scalebar
        scalebar_size_arcsec = 30
        ax.plot([0.7 * w, 0.7 * w + scalebar_size_arcsec * 1.5], [0.1 * h, 0.1 * h], color='white')
        ax.text(0.7 * w + scalebar_size_arcsec * 1.5, 0.1 * h + 4, f"{scalebar_size_arcsec}\"", ha='right',
                color='white')
    plt.savefig(png_path)  # , field, cmap='viridis', origin='lower')
    plt.close()
print("Done")
