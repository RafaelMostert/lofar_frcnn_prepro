import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.modeling import models
from astropy.visualization import (PercentileInterval, SqrtStretch,
                                   ImageNormalize)
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel

flatten = lambda t: [item for sublist in t for item in sublist]


def load_fits(fits_filepath, dimensions_normal=True):
    """Load a fits file and return its header and content"""
    # Load first fits file
    hdulist = fits.open(fits_filepath)
    # Header
    hdr = hdulist[0].header
    if dimensions_normal:
        hdu = hdulist[0].data
    else:
        hdu = hdulist[0].data[0, 0]
    hdulist.close()
    return hdu, hdr


def remove_unresolved_sources_from_fits(cutout, fits_path, gauss_cat, gauss_dict, debug=False):
    """Given a path to a fits file and the corresponding cutout object,
    for all sources in the cutout object marked as unresolved we will find
    the constituent gaussian components and subtract those from the fits image.
    Finally we write the image back to the fits file."""

    # Open fits
    hdu = fits.open(fits_path)
    image = hdu[0].data
    cutout_wcs = WCS(hdu[0].header, naxis=2)

    relevant_idxs = []

    # For each unresolved source
    for unresolved_source in cutout.get_unresolved_sources():
        # Get relevant catalogue entries
        relevant_idxs.append(gauss_dict[unresolved_source.sname])
        # print("debug:", unresolved_source.sname)

    # Create gaussians
    relevant_idxs = flatten(relevant_idxs)
    gaussians = extract_gaussian_parameters_from_component_catalogue(
        gauss_cat.loc[relevant_idxs], cutout_wcs)
    # print("debug gaussians:",gaussians)
    # Subtract them from the data
    model, residual = subtract_gaussians_from_data(gaussians, image)
    hdu[0].data = residual

    # Write changes to fits file
    hdu.writeto(fits_path, overwrite=True)
    hdu.close()

    # Debug visualization
    if debug:
        norm = ImageNormalize(image, interval=PercentileInterval(99.),
                              stretch=SqrtStretch())
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(image, norm=norm)
        ax[1].imshow(residual, norm=norm)
        ax[2].imshow(model, norm=norm)
        plt.show()


def FWHM_to_sigma_for_gaussian(fwhm):
    """Given a FWHM returns the sigma of the normal distribution."""
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def extract_gaussian_parameters_from_component_catalogue(pandas_cat, wcs, arcsec_per_pixel=1.5,
                                                         PA_offset_degree=90, maj_min_in_arcsec=True,
                                                         peak_flux_is_in_mJy=True):
    # Create skycoords for the center locations of all gaussians
    c = SkyCoord(pandas_cat.RA, pandas_cat.DEC, unit='deg')

    # transform ra, decs to pixel coordinates
    if maj_min_in_arcsec:
        deg2arcsec = 1
    else:
        deg2arcsec = 3600
    if peak_flux_is_in_mJy:
        mJy2Jy = 1000
    else:
        mJy2Jy = 1
    pixel_locs = skycoord_to_pixel(c, wcs, origin=0, mode='all')
    gaussians = [models.Gaussian2D(row.Peak_flux / mJy2Jy, x, y,
                                   FWHM_to_sigma_for_gaussian(row.Maj * deg2arcsec / arcsec_per_pixel),
                                   FWHM_to_sigma_for_gaussian(row.Min * deg2arcsec / arcsec_per_pixel),
                                   theta=np.deg2rad(row.PA + PA_offset_degree))
                 for ((irow, row), x, y) in zip(pandas_cat.iterrows(), pixel_locs[0], pixel_locs[1])]
    return gaussians


def subtract_gaussians_from_data(gaussians, astropy_cutout):
    # Create indices
    yi, xi = np.indices(astropy_cutout.shape)

    model = np.zeros(astropy_cutout.shape)
    for g in gaussians:
        model += g(xi, yi)
    residual = astropy_cutout - model
    return model, residual


def find_bbox(t):
    # given a table t find the bounding box of the ellipses for the regions

    boxes = []
    # print(t.columns)
    '''
    for r in t:
        #print(r['Maj'])
        if np.isnan(r['Maj']):
            a=r['LGZ_Size']/3600.0
            b=r['LGZ_Width']/3600.0
            th=(r['LGZ_PA']+90)*np.pi/180.0
        else:
            a=r['Maj']/3600.0
            b=r['Min']/3600.0
            th=(r['PA']+90)*np.pi/180.0
    '''
    for r in t:
        if np.isnan(r['LGZ_Size']):
            a = r['Maj'] / 3600.0
            b = r['Min'] / 3600.0
            th = (r['PA'] + 90) * np.pi / 180.0
        else:
            a = r['LGZ_Size'] / 3600.0
            b = r['LGZ_Width'] / 3600.0
            th = (r['LGZ_PA'] + 90) * np.pi / 180.0

        dx = np.sqrt((a * np.cos(th)) ** 2.0 + (b * np.sin(th)) ** 2.0)
        dy = np.sqrt((a * np.sin(th)) ** 2.0 + (b * np.cos(th)) ** 2.0)
        boxes.append([r['RA'] - dx / np.cos(r['DEC'] * np.pi / 180.0),
                      r['RA'] + dx / np.cos(r['DEC'] * np.pi / 180.0),
                      r['DEC'] - dy, r['DEC'] + dy])

    boxes = np.array(boxes)
    minra = np.nanmin(boxes[:, 0])
    maxra = np.nanmax(boxes[:, 1])
    mindec = np.nanmin(boxes[:, 2])
    maxdec = np.nanmax(boxes[:, 3])

    ra = np.mean((minra, maxra))
    dec = np.mean((mindec, maxdec))
    size = 1.2 * 3600.0 * np.max((maxdec - mindec, (maxra - minra) * np.cos(dec * np.pi / 180.0)))
    return ra, dec, size


def get_mosaic_name(name):
    print('DEPRECATED FUNCTION: image_utils.get_mosaic_name(), check not necessary anymore')
    globst = os.environ['MOSAICS_PATH'] + '/' + name.rstrip() + '*'
    # print('TRYIN GO FIND THE MOSAIC PATH:')
    print(globst)
    g = glob(globst)
    print(g)
    if len(g) == 1:
        return g[0]
    elif len(g) == 0:
        raise RuntimeError('No mosaic called ' + name)
    else:
        raise RuntimeError('Mosaic name ambiguous')
