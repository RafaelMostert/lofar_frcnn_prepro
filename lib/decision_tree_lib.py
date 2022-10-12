import numpy as np
import pandas as pd
import os
import sys
try:
  from collections import Iterable
except ImportError:
  from collections.abc import Iterable
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from astropy import units as u
from astropy.wcs import WCS
import astropy.visualization as vis
from astroquery.vizier import Vizier
from scipy.spatial import cKDTree
from astropy.table import Table
import matplotlib.pyplot as plt

"""
Libary file for imaging_scripts/decision_tree.py script

Implement decision tree as in Fig. 5, The LOFAR Two-metre Sky Survey
III. First Data Release: optical/IR identifications and value-added catalogue
[Wendy Williams et al. 2018]
"""


# Load image and raw PyBDSF source catalogue (NOT gaussians, NOT component, NOT value-added)
def fits_catalogue_to_pandas(catalogue_path):
    """When given the path to a fits catalogue, returns a pandas dataframe
    that holds this catalogue."""
    # Check if path exists
    if not os.path.exists(catalogue_path):
        print('Catalogue path does not exist:', catalogue_path)
        asdasd
    return Table.read(catalogue_path).to_pandas()


# Check for artefacts
def check_for_artefacts(cat, overwrite=False, store_dir=''):
    """
    An initial selection of candidate artefacts was made
    by considering all compact bright sources (brighter than
    5 mJy and smaller than 15 00 ) and selecting their neigh-
    bours within 10 00 that are 1.5 times larger (this selects large
    sources in close proximity to compact, bright sources). Since
    such structures can in fact be real, e.g. faint lobes near a
    bright radio core, these candidate artefacts were visually
    confirmed. 733 out of 884 (83%) of such candidate sources
    were confirmed as artefacts. We note that, as a prelimi-
    nary step, this was not a complete artefact selection, e.g.
    it did not select clusters of artefacts around bright sources.
    """
    if cat is None:
        return None, None
    store_file = os.path.join(store_dir, 'artefacts_nn.npy')
    # assumptions from paper
    brightness_threshold_in_mJy = 5
    compact_bright_size_threshold_arcsec = 15
    nn_distance_arcsec = 10
    x_larger = 1.5

    # Select bright sources
    s = cat[(cat.Total_flux > brightness_threshold_in_mJy) &
            (cat.Maj < compact_bright_size_threshold_arcsec)]
    if s.empty:
        return None, cat
    # print(f"There are {len(s)} sources brighter than {brightness_threshold_in_mJy} mJy and smaller"
    #      f" than {compact_bright_size_threshold_arcsec} arcsec.")

    # Select their neighbours
    ras, decs = cat.RA, cat.DEC
    search_around_ras, search_around_decs = s.RA, s.DEC
    if overwrite or not os.path.exists(store_file):
        indices = return_nn_within_radius(ras, decs, search_around_ras, search_around_decs,
                                          nn_distance_arcsec)
        np.save(store_file, indices)
    else:
        indices = np.load(store_file)
    assert len(indices) == len(s), "If this fails, set overwrite equal to True"

    # Loop over the lists of indices
    final_indices = []
    for ind, bright_source_maj in zip(indices, s.Maj):
        if len(ind) > 1:
            temp_cat = cat.iloc[ind[1:]]
            final_indices.append(temp_cat[temp_cat.Maj > x_larger * bright_source_maj].index.values)

    if final_indices == []:
        return None, cat
    else:
        artefacts = cat.loc[list(flatten(final_indices))]
        not_artefacts = cat[~cat.index.isin(list(flatten(final_indices)))]
        return artefacts, not_artefacts


def flatten(l):
    '''Flatten a list or numpy array'''
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el,
                                                                   (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def pandas_byte_to_str(df):
    str_df = df.select_dtypes([np.object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    for col in str_df:
        df[col] = str_df[col]
    return df


# Check for large optical galaxies
def check_for_large_optical_galaxies_old(cat, MASX_size_arcsec,
                                         object_search_radius_in_arcsec=10, store_dir='', overwrite=False):
    """
    The radio emission associated with nearby galaxies that
    are extended on arcmin scales in the optical is clearly
    resolved in the LoTSS maps and can be incorrectly de-
    composed into as many as several tens of sources in the
    PyBDSF catalogue. To deal with these sources we selected
    all sources in the 2MASX catalogue larger than 60 00 and
    for each, searched for all the PyBDSF sources that are
    located (within their errors) within the ellipse defined by
    the 2MASX source parameters (using the semi-major axis,
    ‘r ext’, the K s -band axis ratio, ‘k ba’, and K s -band posi-
    tion angle, ‘k pa’). The PyBDSF sources were then auto-
    matically associated as a single physical source and identi-
    fied with the 2MASX source. We record the 2MASX source
    name as the the ID name of the LoTSS source, but take the
    co-ordinates and optical/IR photometry from the nearest
    match in the combined Pan-STARRS–AllWISE catalogue,
    with the caveat that the PanSTARRS and AllWISE pho-
    tometry is likely to be wrong for these large sources. This
    reduced the demands on visual inspection at the LGZ stage,
    and avoided the possibility of human volunteers missing out
    components of the radio emission from the galaxy in their
    classification.
    """
    if cat is None:
        return None, None
    store_file = os.path.join(store_dir, '2MASX_indices.npy')
    final_indices = []
    if overwrite or not os.path.exists(store_file):
        for i, (ra, dec) in enumerate(zip(cat.RA, cat.DEC)):
            source_name, diameter_arcsec, _ = get_2MASX_angular_diameter_arcsec(ra, dec,
                                                                                object_search_radius_in_arcsec=object_search_radius_in_arcsec,
                                                                                store_dir=store_dir,
                                                                                overwrite=False)
            if not diameter_arcsec is None:
                if diameter_arcsec > MASX_size_arcsec:
                    final_indices.append(i)
        np.save(store_file, final_indices)
    else:
        final_indices = np.load(store_file)

    if final_indices == []:
        return None, cat
    else:
        large_optical_galaxies = cat.iloc[list(flatten(final_indices))]
        not_large_optical_galaxies = cat[~cat.index.isin(list(flatten(final_indices)))]
        return large_optical_galaxies, not_large_optical_galaxies

    # Check if source is large


def check_if_source_is_large(cat, size_criterium_arcsec):
    """
    Since the size of a source is a first indication whether it is re-
    solved and possibly complex, we first considered the sources
    that are large (> 15 00 , branch A in Fig. 5). This constitutes
    around 6% of the sample. 
    """
    if cat is None:
        return None, None

    return cat[cat.Maj > size_criterium_arcsec], cat[cat.Maj <= size_criterium_arcsec]


##[Large:yes] Check if source is bright
def check_if_source_is_bright(cat, total_flux_threshold):
    """
    All large, bright sources (brighter
    than 10 mJy) were selected for visual processing in LGZ.
    Containing around 7000 sources, this constitutes around 2%
    of the PyBDSF catalogue.
    Instead of also directly processing the remaining ∼ 13k
    large, faint sources (fainter than 10 mJy – branch B) in
    LGZ, these sources were first visually sorted as (i) an
    artefact, (ii) complex structure to be processed in LGZ,
    (iii) complex structure, where the emission is clearly on
    very large scales, to be processed directly in the LGZ ‘too
    zoomed in’ post-processing step (see Section 5.2), (iv) hav-
    ing no possible match, (v) having an acceptable LR match,
    i.e. LR ID, or (vi) associated with an optically bright/large
    galaxy. It should be noted that within this category of large,
    faint radio sources, those larger than 30 00 are too large to
    have a LR estimate and so we included option (vi) to al-
    low an identification with the nearest large/bright optical
    galaxy based on the Pan-STARRS images. The ∼ 1000 such
    sources with a visually confirmed large optical galaxy match
    were then matched directly to the nearest 2MASX source,
    or in the 35 cases where there was no 2MASX source, to the
    nearest bright SDSS source. In all cases the nearest 2MASX
    or SDSS match was confirmed to be the correct match.
    Again the ID positions for these sources are taken from the
    nearest matches in the merged Pan-STARRS/AllWISE cat-
    alogue. An additional ∼ 4000 sources were included in the
    LGZ sample after this visual sorting on branch B
    """
    return cat[cat.Total_flux > total_flux_threshold], cat[cat.Total_flux <= total_flux_threshold]


##[Large:no] Check if source is isolated
def check_if_source_is_isolated(cat, nn_distance_threshold_arcsec):
    """Sources < 15 00 in size make up around 94% of the PyBDSF
    catalogue (branch C). While many of these are individual
    sources best processed using the LR method, a subset are
    components of complex sources. Visual inspection of the
    entire catalogue was impossible given the available effort,
    so we applied a series of tests to select those small sources
    most likely to be components of complex sources. We ini-
    tially considered whether the sources smaller than 15 00 have
    any nearby neighbours. Sources where the distance to the
    nearest neighbour is greater than 45 00 were considered to be
    isolated (branch D; 200k sources). A separation of 45 00
    corresponds to a linear distance of 230–330 kpc at redshifts
    of 0.35-0.7, where the bulk of the AGN population of this
    sample is located (see DR1-III) 12 . Before directly accept-
    ing the LR results for these sources, we removed those that
    were fitted by PyBDSF using multiple Gaussian compo-
    nents, or that lie in islands with other sources (i.e. with
    catalogued ‘S Code’ values of ‘M’ or ‘C’); in these cases
    (10k sources) a further decision tree was followed, taking
    into account the LR matches to the individual Gaussian
    components of the source (see Section 6.6). For the remain-
    ing small, isolated, single Gaussian-component sources (i.e.
    with catalogued ‘S Code’ values of ‘S’), we accepted the LR
    results (branch E): either the source has an acceptable LR
    match (LR ID) or it has no acceptable LR match (no ID)."""
    if cat is None:
        return None, None

    # Check for neighbours within nn_distance_threshold_arcsec
    ras, decs = cat.RA, cat.DEC
    indices = return_nn_within_radius(ras, decs, ras, decs,
                                      nn_distance_threshold_arcsec)
    final_indices = [i for i, ind in zip(np.arange(len(indices)), indices)
                     if len(ind) > 1]

    if final_indices == []:
        return None, cat
    else:
        not_isolated = cat.iloc[final_indices]
        isolated = cat[~cat.index.isin(final_indices)]
        return isolated, not_isolated

    ###[Large:no,Isolated:no] Check if source is clustered


def check_if_source_is_clustered(cat, nth_neighbour,
                                 distance_to_nth_neigbhour_arcsec, store_dir=None, overwrite=True):
    """
    Small sources that are not isolated (i.e. have at least one
    other source within 45 00 – branch F) have a higher chance of
    being a component of a complex source. For these sources
    we considered whether they are clustered to some extent,
    based on the distance to the fourth neighbouring source:
    for approximately 1100 sources this distance is less than
    45 00 (branch G). Empirically, based on visually examining
    subsamples of sources, we found that taking the fourth near-
    est neighbour maximised the number of genuinely clustered
    sources while minimising the number of unrelated sources.
    As these may be part of a larger structure or simply chance
    groups of unassociated sources that can be matched by the
    LR method, we visually sorted such ‘clustered’ sources ei-
    ther as (i) complex (to be sent to LGZ), or (ii) not complex
    (appropriate for further analysis in the decision tree), or (iii)
    as an artefact. About a quarter of the clustered (branch G)
    sources were selected for LGZ, while about another quarter
    were flagged as artefacts. The remainder were considered
    not clustered based on the visual sorting and assessed via
    branch H.
    """
    if cat is None:
        return None, None
    store_file_dis = os.path.join(store_dir, 'clustered_nn_dis.npy')
    store_file_ind = os.path.join(store_dir, 'clustered_nn_ind.npy')

    # Check for neighbours within nn_distance_threshold_arcsec
    assert len(cat) > nth_neighbour
    ras, decs = cat.RA, cat.DEC
    if overwrite or not os.path.exists(store_file_dis) or not os.path.exists(store_file_ind):
        distances_arcsec, indices = return_nth_nn(ras, decs, nth_neighbour)
        np.save(store_file_dis, distances_arcsec)
        np.save(store_file_ind, indices)
    else:
        distances_arcsec = np.load(store_file_dis)
        indices = np.load(store_file_ind)

    final_indices = [i for i, distance in zip(np.arange(len(cat)), distances_arcsec)
                     if distance < distance_to_nth_neigbhour_arcsec]

    if final_indices == []:
        return None, cat
    else:
        clustered = cat.iloc[final_indices]
        not_clustered = cat[~cat.index.isin(final_indices)]
        return clustered, not_clustered


###[Large:yes,Bright:yes] Create cutout for LGZ
def create_LGZ_cutouts(cat, image_path, nn_search_radius_arcsec=180, nn_search_attempts=10,
                       cutout_max_size_arcsec=300, cutout_min_size_arcsec=60, arcsec_per_pixel=1.5,
                       store_cutout_dir=None, overwrite=False, cutout_name=''):
    if cat is None:
        return None
    print(f"-- Making LGZ cutouts for these {len(cat)} sources.")
    # For each source
    ras, decs = cat.RA, cat.DEC
    new_ras, new_decs = np.zeros(len(cat)), np.zeros(len(cat))
    sizes = []
    for i, (i_source, source) in enumerate(cat.iterrows()):

        # Repeat nn_search_attempt times
        old_ra, old_dec = source.RA, source.DEC
        for j in range(nn_search_attempts):

            # Gather all sources within nn_search_radius_arcsec
            indices = return_nn_within_radius(ras, decs, old_ra, old_dec,
                                              nn_search_radius_arcsec)[0]

            # Set new center based on their mean RA and DEC
            new_ra = np.mean(cat.iloc[indices].RA.values)
            new_dec = np.mean(cat.iloc[indices].DEC.values)
            if old_ra == new_ra and old_dec == new_dec:
                new_ras[i] = new_ra
                new_decs[i] = new_dec
                break
            else:
                old_ra = new_ra
                old_dec = new_dec

        # Find bounding box for the sources 
        ra, dec, size_arcsec = find_bbox(cat.iloc[indices])

        # Ensuring that the cutout stays within certain limits
        if np.isnan(size_arcsec):
            ra, dec = source.RA, source.DEC
            size_arcsec = 60

        if size_arcsec < 60:
            size_arcsec = 60

        if size_arcsec > 300.0:
            # revert just to original
            ra, dec = source.RA, source.DEC
            size_arcsec = 300.0
            ra, dec, size_arcsec = find_bbox([source])

        new_ras[i] = ra
        new_decs[i] = dec
        sizes.append(size_arcsec)

    # return fits cutout
    cutouts = make_cutouts(image_path, new_ras, new_decs, sizes, sizes, arcsec_per_pixel)

    # save fits cutout
    n_sources_with_nans = 0
    print()
    for i, cutout in enumerate(cutouts):

        sys.stdout.write('\r')
        print(i)

        # Do not plot sources that contain nans
        image = cutout.data
        if np.isnan(image).any():
            n_sources_with_nans += 1
            continue

        file_path = os.path.join(store_cutout_dir, f'{cutout_name}{i:06d}.png')
        if overwrite or not os.path.exists(file_path):
            image[image < 0] = 0
            plt.figure(figsize=(15, 15))
            # transform = vis.SqrtStretch()

            # Create an ImageNormalize object
            stretch = vis.simple_norm(image, 'sqrt')

            # image = transform(cutout.data)
            plt.imshow(image, aspect='equal', norm=stretch, interpolation="nearest", origin='lower')
            plt.savefig(file_path)
            plt.close()

    if n_sources_with_nans > 0:
        print(f"-- skipping {n_sources_with_nans} sources as they contain NaNs.")


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


def make_cutouts(image_path, ras, decs, widths_arcsec, heights_arcsec,

                 arcsec_per_pixel, dimensions_normal=True):
    hdu, hdr = load_fits(image_path, dimensions_normal=dimensions_normal)

    skycoords = SkyCoord(ras, decs, unit='degree')

    widths_pixel = np.array(widths_arcsec) / arcsec_per_pixel
    heights_pixel = np.array(heights_arcsec) / arcsec_per_pixel

    # Create cutout for each skycoord
    cutouts = []
    for i, (w, h, co) in enumerate(zip(widths_pixel, heights_pixel, skycoords)):
        # Extract cutout
        hdu_crop = Cutout2D(hdu, co, (w, h), wcs=WCS(hdr, naxis=2), copy=True)
        cutouts.append(hdu_crop)
    return cutouts


def find_bbox2(cat):
    '''Find bounding box for a set of sources
    Parameters
    ----------
    cat : pandas dataframe, expects sources with RA and DEC in degree
            and Maj and Min in arcsec
    Returns
    -------
    ra,dec : int (degree)
    size : int (arcsec)
    '''
    ras, decs = cat.RA, cat.DEC

    return ra, dec, size_arcsec


def find_bbox(t):
    # given a cat t find the bounding box of the ellipses for the regions

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
    for i_r, r in t.iterrows():
        a = r['Maj'] / 3600.0
        b = r['Min'] / 3600.0
        th = (r['PA'] + 90) * np.pi / 180.0

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
    size = 3600.0 * np.max((maxdec - mindec, (maxra - minra) * np.cos(dec * np.pi / 180.0)))
    return ra, dec, size


###[Large:yes,Bright:no] Visual sorting
def end_of_tree(full_cat, cat, message):
    if cat is None or len(cat) == 0:
        print(f"\n{message}:\nNo sources")
    else:
        print(f"\n{message}:\n{len(cat)} sources ({len(cat) / len(full_cat) * 100:.2g}%)")
        # print(f"\n{len(cat)} sources ({len(cat)/len(full_cat)*100:.2g}%) end up {message}")


def slice_of_tree(full_cat, cat, message):
    if cat is None or len(cat) == 0:
        print(f"-- {message}: No sources.")
    else:
        print(f"-- {message}: {len(cat)} sources ({len(cat) / len(full_cat) * 100:.2g}%)")


def make_kdtree(ras, decs):
    '''This makes a `scipy.spatial.CKDTree` on (`ra`, `decl`).
    Parameters
    ----------
    ras,decs : array-like
        The right ascension and declination coordinate pairs in decimal degrees.
    Returns
    -------
    `scipy.spatial.CKDTree`
        The cKDTRee object generated by this function is returned and can be
        used to run various spatial queries.
    '''

    cosdec = np.cos(np.radians(decs))
    sindec = np.sin(np.radians(decs))
    cosra = np.cos(np.radians(ras))
    sinra = np.sin(np.radians(ras))
    xyz = np.column_stack((cosra * cosdec, sinra * cosdec, sindec))

    # generate the kdtree
    kdt = cKDTree(xyz, copy_data=True)

    return xyz, kdt


def return_nn_within_radius(ras, decs, search_around_ras, search_around_decs, radius_in_arcsec):
    '''Makes a `scipy.spatial.CKDTree` on (`ras`, `decs`)
    and return nearest neighbours within the given radius.
    Parameters
    ----------
    ras,decs : array-like
        The right ascension and declination coordinate pairs in decimal degrees.
    radius_in_arcsec : int
        search radisu for nns in arcsec
    Returns
    -------
    indexes of nearest neighbours within the given radius in arcsec
    '''
    # Create kdtree
    xyz, ra_dec_tree = make_kdtree(ras, decs)

    cosdec = np.cos(np.radians(search_around_decs))
    sindec = np.sin(np.radians(search_around_decs))
    cosra = np.cos(np.radians(search_around_ras))
    sinra = np.sin(np.radians(search_around_ras))
    search_xyz = np.column_stack((cosra * cosdec, sinra * cosdec, sindec))

    # Query kdtree for nearest neighbour distances
    radius_in_rad = np.deg2rad(radius_in_arcsec / 3600)
    kd_out = ra_dec_tree.query_ball_point(search_xyz, 2 * np.sin(radius_in_rad / 2))

    return kd_out


def return_nth_nn(ras, decs, nth_nearest_neighbour):
    '''Makes a `scipy.spatial.CKDTree` on (`ras`, `decs`)
    and return the nth nearest neighbour and the distance to it in arcsec.
    Parameters
    ----------
    ras,decs : array-like
        The right ascension and declination coordinate pairs in decimal degrees.
    nth_nearest_neighbour : int
        nth nearest neighbour to look for
    Returns
    -------
    Nearest neighbour distance in arcsec
    Nearest neighbour indice as bonus
    '''

    # Create kdtree
    xyz, ra_dec_tree = make_kdtree(ras, decs)

    # Query kdtree for nearest neighbour distances
    # Note that as the first neighbour will always be the source itself
    # hence the +1
    kd_out = ra_dec_tree.query(xyz, k=nth_nearest_neighbour + 1)

    nn_distances_rad = kd_out[0][:, nth_nearest_neighbour]
    nn_distances_arcsec = np.rad2deg(nn_distances_rad) * 3600
    return nn_distances_arcsec, kd_out[1][:, nth_nearest_neighbour]


def return_nn_distance_in_arcsec(ras, decs, subset_indices=None):
    '''Makes a `scipy.spatial.CKDTree` on (`ras`, `decs`)
    and return nearest neighbour distances in degrees.
    Assuming small angles, we make use of the approximation:
    angle = 2arcsin(a/2) ~= a
    For the LoTSS cat, the errors introduced with this assumption are of the order 1e-6 arcsec.
    Parameters
    ----------
    ras,decs : array-like
        The right ascension and declination coordinate pairs in decimal degrees.
    subset_indices : array-like
        Indices of the subset for which you want nn distances to the rest of the RAs and DECs
    Returns
    -------
    Nearest neighbour distances in arcsec
    Nearest neighbour indices as bonus
    '''
    # Create kdtree
    xyz, ra_dec_tree = make_kdtree(ras, decs)

    # Query kdtree for nearest neighbour distances
    if subset_indices is None:
        kd_out = ra_dec_tree.query(xyz, k=2)
    else:
        kd_out = ra_dec_tree.query(xyz[tuple(subset_indices)], k=2)

    nn_distances_rad = kd_out[0][:, 1]
    nn_distances_arcsec = np.rad2deg(nn_distances_rad) * 3600
    return nn_distances_arcsec, kd_out[1][:, 1]


def get_2MASX_angular_diameter_arcsec(ra, dec, object_search_radius_in_arcsec=10, store_dir='', overwrite=False):
    """Use Vizier to query 2MASX.
        ra and dec are expected to be in degree"""

    assert isinstance(ra, float)
    assert isinstance(dec, float)

    store_path = os.path.join(store_dir, f'ra{ra}dec{dec}searchradius{object_search_radius_in_arcsec}.npy')
    if overwrite or not os.path.exists(store_path):

        v = Vizier(columns=["*", "r.K20e"], catalog="2MASX")
        result = v.query_region(SkyCoord(ra=ra, dec=dec,
                                         unit=(u.deg, u.deg), frame='fk5'),
                                radius=object_search_radius_in_arcsec * u.arcsec,
                                catalog="2MASX")
        if len(result) > 0:
            for k in result.keys():
                rr = result[k].to_pandas()
                source_name = f"2MASX J{rr['_2MASX'][0].decode('utf-8')}"
                diameter_arcsec = rr['r.K20e'].values[0]
                verbose_message = (f"2MASX J{source_name} reports an angular diameter of "
                                   f"{diameter_arcsec:.2g} arcsec.")
                # print(diameter_arcsec, type(diameter_arcsec))
                # if not isinstance(diameter_arcsec, float):
                #    diameter_arcsec = None
        else:
            verbose_message = "Not present in 2MASX catalogue."
            source_name, diameter_arcsec = None, None
        np.save(store_path, (source_name, diameter_arcsec, verbose_message))
    else:
        source_name, diameter_arcsec, verbose_message = np.load(store_path)
        if not diameter_arcsec is None:
            diameter_arcsec = float(diameter_arcsec)

    return source_name, diameter_arcsec, verbose_message


# Get 2MASX cat
def check_for_large_optical_galaxies(cat, MASX_size_arcsec,
                                     store_dir='', overwrite=False, verbose=False):
    """
    The radio emission associated with nearby galaxies that
    are extended on arcmin scales in the optical is clearly
    resolved in the LoTSS maps and can be incorrectly de-
    composed into as many as several tens of sources in the
    PyBDSF catalogue. To deal with these sources we selected
    all sources in the 2MASX catalogue larger than 60 00 and
    for each, searched for all the PyBDSF sources that are
    located (within their errors) within the ellipse defined by
    the 2MASX source parameters (using the semi-major axis,
    ‘r ext’, the K s -band axis ratio, ‘k ba’, and K s -band posi-
    tion angle, ‘k pa’). The PyBDSF sources were then auto-
    matically associated as a single physical source and identi-
    fied with the 2MASX source. We record the 2MASX source
    name as the the ID name of the LoTSS source, but take the
    co-ordinates and optical/IR photometry from the nearest
    match in the combined Pan-STARRS–AllWISE catalogue,
    with the caveat that the PanSTARRS and AllWISE pho-
    tometry is likely to be wrong for these large sources. This
    reduced the demands on visual inspection at the LGZ stage,
    and avoided the possibility of human volunteers missing out
    components of the radio emission from the galaxy in their
    classification.
    """
    masx_path = os.path.join(store_dir, 'masx.csv')
    if overwrite or not os.path.exists(masx_path):
        v = Vizier(columns=['RAJ2000', 'DEJ2000', "2MASX", "Kpa", "r.ext", "Kb/a"],
                   column_filters={"r.ext": f">{MASX_size_arcsec}"}, catalog="2MASX")
        v.ROW_LIMIT = -1
        cat_2MASX = v.get_catalogs('VII/233')[0]
        cat_2MASX = cat_2MASX.to_pandas()
        cat_2MASX.to_csv(masx_path)
    else:
        cat_2MASX = pd.read_csv(masx_path)

    if verbose:
        print(f'2masx cat ({len(cat_2MASX)} entries) loaded')

    # Create kdtree from cat
    _, ra_dec_tree = make_kdtree(cat.RA, cat.DEC)

    # Transform 2masx coordinates to xyz
    masx_ras, masx_decs = cat_2MASX['RAJ2000'], cat_2MASX['DEJ2000']
    cosdec = np.cos(np.radians(masx_decs))
    sindec = np.sin(np.radians(masx_decs))
    cosra = np.cos(np.radians(masx_ras))
    sinra = np.sin(np.radians(masx_ras))
    search_xyzs = np.column_stack((cosra * cosdec, sinra * cosdec, sindec))

    # turn search radii into radians
    radii_in_rad = np.radians(cat_2MASX['r.ext'] / 3600)
    # turn search radii into cartesian space
    radii_cartesian = 2 * np.sin(radii_in_rad / 2)

    # Query kdtree for nearest neighbour distances
    kd_outs = [ra_dec_tree.query_ball_point(search_xyz, radius_cartesian) for search_xyz, radius_cartesian
               in zip(search_xyzs, radii_cartesian)]
    if verbose:
        print('Searched around all kdtree')

    # return cats
    matching_indices = list(set(flatten(kd_outs)))
    if matching_indices == []:
        return None, cat
    else:
        large_optical_galaxies = cat.iloc[matching_indices]
        not_large_optical_galaxies = cat[~cat.index.isin(matching_indices)]
        return large_optical_galaxies, not_large_optical_galaxies

        ## Loop over kdtree results to match radio cat to 2masx
    # index_name_pairs = [[i,name[2:-1]] for i, name in zip(kd_outs,cat_2MASX['_2MASX'])if not not i]
    # identifier = ['' for i in range(len(cat))]
    # for i, name in index_name_pairs:
    #    for j in i:
    #        identifier[j] = '2MASX '+name
    # tally = sum([1 for i in identifier if not i == ''])
    # print(tally)
    #
    # print(f'Created column with 2masx names ({tally}) if present for each cat entry:',time.time()-start)
    # return identifier
