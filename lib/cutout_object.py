import sys

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import utils
from astropy.wcs.utils import skycoord_to_pixel

sys.path.insert(0, '/data1/MRP1/CLARAN/lofar_frcnn_tools/')
from lib.shape_maker import Make_Shape


class source(object):
    '''
    Class object containing all relevant information of a source in a cut-out.
    dr is a pandas datarow of the source in question taken from the component catalogue
    dr_vac is a pandas datarow of the source in question taken from the value added catalogue

    '''
    __slots__ = ('ra', 'dec', 'low_sigma_flag', 'overlap_flag', 'center', 'sname',
                 'ID_flag', 'size', 'width', 'PA', 'va_sname', 'related', 'unresolved',
                 'x', 'y', 'x0', 'y0', 'vax0', 'vay0', 'c_hull', 'xmin', 'xmax', 'ymin', 'ymax',
                 'in_bounds')

    def __init__(self, dr, va_sname=None, center=False, x=None, y=None, related=False,
                 unresolved=False):

        self.ra = dr['RA']
        self.dec = dr['DEC']
        self.low_sigma_flag = 0
        self.overlap_flag = False
        self.center = center
        self.in_bounds = True
        self.unresolved = unresolved
        self.x = x
        self.y = y
        self.related = related
        self.xmin = 0.0
        self.xmax = 1.0
        self.ymin = 0.0
        self.ymax = 1.0

        if 'Component_Name' in dr.keys():
            self.sname = dr['Component_Name']
            try:
                self.ID_flag = dr['ID_flag']
            except:
                pass
        else:
            self.sname = dr['Source_Name']

        if 'LGZ_Size' in dr.keys() and np.isnan(dr['LGZ_Size']):
            self.size = dr['LGZ_Size']
            self.width = dr['LGZ_Width']
            self.PA = dr['LGZ_PA']
            self.ID_ra = dr['ID_ra']
            self.ID_dec = dr['ID_dec']
        else:
            self.size = dr['Maj']
            self.width = dr['Min']
            self.PA = dr['PA']

        if not va_sname is None:
            self.va_sname = va_sname
        else:
            self.va_sname = None

    def set_xy(self, xy, idx):
        self.x = xy[0][idx]
        self.y = xy[1][idx]

    def set_vax0y0(self, x0, y0):
        # Set central pixel coordinates of the image
        self.vax0 = x0
        self.vay0 = y0

    def set_x0y0(self, x0, y0):
        # Set central pixel coordinates of the image
        self.x0 = x0
        self.y0 = y0

    def set_convex_hull(self, comp_cat, idx_dict, training_mode=False):
        # Save convex hull of source coponents
        if training_mode:
            cl = comp_cat.loc[idx_dict[self.va_sname]]
        else:
            cl = comp_cat.loc[idx_dict[self.sname]]
        self.c_hull = Make_Shape(cl)

    """
    def set_convex_hull(self,cc, training_mode=False):
        # Save convex hull of source coponents
        if training_mode:
            cl = cc[(cc['Source_Name'] == self.va_sname)]
        else:
            cl = cc[(cc['Source_Name'] == self.sname)]
        self.c_hull = Make_Shape(cl)
    """

    def set_box_dimensions(self, xmin, xmax, ymin, ymax, in_bounds):
        # Set pixel coordinates of box corners and save hull coordinates
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.in_bounds = in_bounds

    """
    def set_mask(self,data):
        # Saves mask of bounding box
        self.mask = np.ones(data.shape,dtype=int)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if (j >= self.xmax) or (j <= self.xmin) or (i >= self.ymax) or (i <= self.ymin):
                    self.mask[i][j] = 0 
    #end set_mask()
    """

    def set_comps_peaks(self, segm, peaks):
        self.n_comp = np.amax(segm)
        self.n_peak = len(peaks)

        # Temporary constraints
        if self.n_peak == 0:
            self.n_peak = 1
        if self.n_comp > 3:
            self.n_comp = 3
        if self.n_peak > 5:
            self.n_peak = 5
        if self.n_peak < self.n_comp:
            self.n_peak = self.n_comp

    # end set_comps_peaks()

    def toggle_low_sigma(self):
        self.low_sigma_flag = True
    # end toggle_low_sigma()


class prediction(object):
    '''
    Class object containing all relevant information of a predicted source in a cut-out.
    '''

    def __init__(self, l, label):
        self.score = l[1]
        self.xmin = np.min(np.array(l[2:4]).astype(float))
        self.ymin = np.max(np.array(l[2:4]).astype(float))
        self.xmax = np.min(np.array(l[4:6]).astype(float))
        self.ymax = np.max(np.array(l[4:6]).astype(float))
        self.label = label[1:]


class cutout(object):
    '''
    Class object containing information necessary to make and label a cut-out centered around a given source (c_source). 
    Center of the cutout is given by cra and cdec (in degrees).
    This center coincides with the center of the unassociated center source to get a similar
    centering in train as in predict mode.
    '''
    __slots__ = ('c_source', 'mosaic_ID_cat', 'size_pixels', 'index', 'RA', 'DEC',
                 'other_components', 'crop_offset', 'wide_focus',
                 'infrared_flag', 'difficult_flag', 'rotation_angle_deg', 'scale_factor',
                 'lofarname_DR2', 'size_arcsec', 'min_ra', 'max_ra', 'min_dec', 'max_dec', 'bmaj',
                 'res', 'focussed_comp', 'related_comp', 'unrelated_comp',
                 'unrelated_names', 'optical_sources', 'gt_xmin', 'gt_ymin', 'gt_xmax', 'gt_ymax')

    def __init__(self, dr, index, cra, cdec, va_sname=None, mosaic_id=None, ):
        self.c_source = source(dr, va_sname=va_sname, center=True)
        if mosaic_id is None:
            self.mosaic_ID_cat = dr['Mosaic_ID'].decode()
        else:
            self.mosaic_ID_cat = mosaic_id
        self.index = index
        self.RA = cra
        self.DEC = cdec
        self.other_components = []
        self.infrared_flag = False
        self.difficult_flag = False
        self.rotation_angle_deg = 0
        self.scale_factor = 1
        self.wide_focus = [[], []]
        self.gt_xmin, self.gt_ymin, self.gt_xmax, self.gt_ymax = 0.0, 0.0, 0.0, 0.0

    # end __init__()

    def set_gt_bbox(self, xmin, ymin, xmax, ymax):
        self.gt_xmin, self.gt_ymin, self.gt_xmax, self.gt_ymax = xmin, ymin, xmax, ymax

    def set_optical_sources(self, optical_pixel_locations, optical_cat,
                            final_pixel_size, current_pixel_size, DR2=False):

        if DR2 and optical_pixel_locations[0] == []:
            print(f"No optical sources found in DR2 optical catalogue for {self.c_source.sname}")
            optical_source_list = {'x': [], 'y': [],
                                   'MAG_R': [], 'MAG_W1': [], 'MAG_W2': []}
            self.optical_sources = optical_source_list
            return
        # print(f"optical sources found")

        # scale and translate to adjust for crop 
        optical_x = optical_pixel_locations[0] - self.crop_offset
        optical_y = optical_pixel_locations[1] - self.crop_offset
        # Flip ys to compensate for RA going from + to -
        # remove components outside of fov
        optical_y = self.size_pixels - optical_y
        # Filter out optical sources that fall outside the image
        indices_inside = [t for t, (a, b) in enumerate(zip(optical_x, optical_y))
                          if (0 < a < (self.size_pixels - 1)) and (0 < b < (self.size_pixels - 1))]
        if DR2:
            optical_source_list = {'x': optical_x[indices_inside], 'y': optical_y[indices_inside],
                                   'MAG_R': optical_cat['MAG_R'].values[indices_inside],
                                   'MAG_W1': optical_cat['MAG_W1'].values[indices_inside],
                                   'MAG_W2': optical_cat['MAG_W2'].values[indices_inside]}
        else:
            optical_source_list = {'x': optical_x[indices_inside], 'y': optical_y[indices_inside],
                                   'w1Mag': optical_cat['w1Mag'].values[indices_inside],
                                   'iFApFlux': optical_cat['iFApFlux'].values[indices_inside]}

        self.optical_sources = optical_source_list

    def set_scale_factor(self, final_pixel_size, current_pixel_size):
        assert final_pixel_size <= current_pixel_size
        self.scale_factor = final_pixel_size / current_pixel_size
        self.crop_offset = (current_pixel_size - final_pixel_size) / 2

    def set_rotation_angle(self, rot_angle_deg):
        self.rotation_angle_deg = rot_angle_deg

    def set_lofarname_DR1(self, lofarname_DR1):
        self.lofarname_DR1 = lofarname_DR1
        setattr(self, 'lofarname_DR1', lofarname_DR1)

    def set_lofarname_DR2(self, lofarname_DR2):
        self.lofarname_DR2 = lofarname_DR2

    def set_wisename(self, wisename):
        self.wisename = wisename

    def set_bounds_degrees(self, min_ra, max_ra, min_dec, max_dec):
        self.min_ra, self.max_ra = min_ra, max_ra
        self.min_dec, self.max_dec = min_dec, max_dec

    def set_size(self, size):
        self.size_arcsec = size

    def update_pixel_locs(self, new_focus, new_related, new_unrelated):
        """Update radio component x and y pixellocations after rotation"""
        # update focussed
        self.c_source.x, self.c_source.y = new_focus[0], new_focus[1]
        r = 0
        u = 0
        for i, c in enumerate(self.other_components):
            if c.related:
                self.other_components[i].set_xy(new_related, r)
                r += 1
            else:
                self.other_components[i].set_xy(new_unrelated, u)
                u += 1

    def get_focussed_comp(self):
        return np.array([self.c_source.x, self.c_source.y])

    def get_related_comp(self):
        x_related = [oc.x for oc in self.other_components if oc.related]
        y_related = [oc.y for oc in self.other_components if oc.related]
        return np.array([x_related, y_related])

    def get_unresolved_sources(self):
        return np.array([oc for oc in self.other_components if oc.unresolved])

    def get_related_unresolved(self):
        return np.array([oc.unresolved for oc in self.other_components if oc.related])

    def get_unrelated_unresolved(self):
        return np.array([oc.unresolved for oc in self.other_components if not oc.related])

    def get_unrelated_comp(self):
        x_unrelated = [oc.x for oc in self.other_components if not oc.related]
        y_unrelated = [oc.y for oc in self.other_components if not oc.related]
        return np.array([x_unrelated, y_unrelated])

    def set_beam(self, wcs, hdu):
        self.res = utils.proj_plane_pixel_scales(wcs)
        self.bmaj = hdu[0].header['BMAJ']
        self.size_pixels = int(round(self.size_arcsec / hdu[0].header['CDELT2']))

    def toggle_infrared_flag(self):
        print('Missing infrared flagged')
        self.infrared_flag = True

    def toggle_difficult_flag(self):
        self.difficult_flag = True

    # def adjust_pixel_locations(self, locs, )

    def save_other_components(self, wcs, idx_cat, compcat_subset, unresolved_dict,
                              training_mode=True, remove_unresolved=False):

        # Get all pixel coordinates through skycoord_to_pixel
        focussed_skycoords = SkyCoord([[self.c_source.ra, self.c_source.dec]], unit='deg')
        focussed_pixels = skycoord_to_pixel(focussed_skycoords, wcs)
        # scale and translate to adjust for crop and flip x-axis to compensate for negative to
        # positive right ascension
        focussed_xs, focussed_ys = np.array(focussed_pixels) - self.crop_offset
        # Flip ys to compensate for RA going from + to -
        focussed_ys = self.size_pixels - focussed_ys
        self.c_source.set_xy([focussed_xs, focussed_ys], 0)
        if len(compcat_subset) > 0:
            other_skycoords = SkyCoord(list(zip(compcat_subset.RA.values,
                                                compcat_subset.DEC.values)), unit='deg')
            other_pixels = skycoord_to_pixel(other_skycoords, wcs)
            xs, ys = other_pixels[0] - self.crop_offset, other_pixels[1] - self.crop_offset

            # Flip ys to compensate for RA going from + to -
            ys = self.size_pixels - ys

            # Determine related components
            if training_mode:
                related = compcat_subset.index.isin(idx_cat[self.c_source.va_sname])

                # if self.c_source.sname != self.c_source.va_sname:
                #    print("Check if we see related sources")
                #    print("Our focussed comp is:", self.c_source.sname)
                #    print("Parent source is:", self.c_source.va_sname)
                #    print("Idx_cat gives:", idx_cat[self.c_source.va_sname])
                #    print("related gives:", related , '\n')

            else:
                related = [False for _ in range(len(xs))]

            # Add all to component list 
            if 'Source_Name' in compcat_subset.keys():
                source_name_key = True
            else:
                source_name_key = False
            existing_components = [s.sname for s in self.other_components]
            if training_mode:
                for s, x, y, r in zip(compcat_subset.itertuples(), xs, ys, related):
                    if not s.Component_Name in existing_components:
                        un = False
                        if remove_unresolved:
                            try:
                                un = unresolved_dict[s.Component_Name] and s.Maj / s.Min < 1.5 \
                                     and s.Maj < 9
                            except:
                                # In this case the component name does not exist in the raw PyBDSF
                                # source catalogue, but only exists in the component catalogue as
                                # a deblended source
                                # print('Component name missing in unresolved cat:', s.Component_Name)
                                # print(f'Its maj/min ratio is {s.Maj/s.Min} (threshold <1.5) and its Maj axis {s.Maj} (our threshold 9)')
                                pass

                        if source_name_key:
                            va_name = s.Source_Name
                        else:
                            va_name = s.Parent_Source
                        self.other_components.append(source(s._asdict(), va_sname=va_name,
                                                            x=x, y=y, related=r,
                                                            unresolved=un))
            else:
                for s, x, y, r in zip(compcat_subset.itertuples(), xs, ys, related):
                    if not s.Source_Name in existing_components:
                        un = False
                        if remove_unresolved:
                            try:
                                un = unresolved_dict[s.Source_Name] and s.Maj / s.Min < 1.5 \
                                     and s.Maj < 9
                                # print('Component name found in unresolved dict:', s.Source_Name, un)
                            except:
                                # In this case the component name does not exist in the raw PyBDSF
                                # source catalogue, but only exists in the component catalogue as
                                # a deblended source
                                # print('Component name missing in unresolved cat:', s.Source_Name)
                                pass
                        self.other_components.append(source(s._asdict(), x=x, y=y,
                                                            related=r, unresolved=un))

    def calculate_pixel_centers(self, w):
        self.res = utils.proj_plane_pixel_scales(w)
        ras = np.zeros(len(self.other_sources) + 1)
        decs = np.zeros(len(self.other_sources) + 1)
        ras[0] = self.c_source.ra
        decs[0] = self.c_source.dec
        for i in range(len(self.other_sources)):
            ras[i + 1] = self.other_sources[i].ra
            decs[i + 1] = self.other_sources[i].dec
        sc = SkyCoord(ras, decs, unit=u.deg)
        x0s, y0s = utils.skycoord_to_pixel(sc, w, 0)
        self.c_source.set_x0y0(x0s[0], y0s[0])
        for i in range(len(self.other_sources)):
            self.other_sources[i].set_x0y0(x0s[i + 1], y0s[i + 1])
        if not self.c_source.va_ra is None:
            varas = np.zeros(len(self.other_sources) + 1)
            vadecs = np.zeros(len(self.other_sources) + 1)
            varas[0] = self.c_source.va_ra
            vadecs[0] = self.c_source.va_dec
            for i in range(len(self.other_sources)):
                varas[i + 1] = self.other_sources[i].va_ra
                vadecs[i + 1] = self.other_sources[i].va_dec
            vasc = SkyCoord(varas, vadecs, unit=u.deg)
            vax0s, vay0s = utils.skycoord_to_pixel(vasc, w, 0)
            self.c_source.set_vax0y0(vax0s[0], vay0s[0])
            for i in range(len(self.other_sources)):
                self.other_sources[i].set_vax0y0(vax0s[i + 1], vay0s[i + 1])

    def remove_components(self, indexes):
        keep_indexes = [i for i in range(len(self.other_components)) if not i in indexes]
        self.other_components = list(np.array(self.other_components)[keep_indexes])

    def box_overlap(self):
        # Function used to check for overlap between the different sources
        def overlap_check(source, other_source):

            if source.xmin >= other_source.xmin and \
                    source.xmin <= other_source.xmax and \
                    source.ymin >= other_source.ymin and \
                    source.ymin <= other_source.ymax or \
                    source.xmax >= other_source.xmin and \
                    source.xmax <= other_source.xmax and \
                    source.ymax >= other_source.ymin and \
                    source.ymax <= other_source.ymax or \
                    source.xmin >= other_source.xmin and \
                    source.xmin <= other_source.xmax and \
                    source.ymax >= other_source.ymin and \
                    source.ymax <= other_source.ymax or \
                    source.xmax >= other_source.xmin and \
                    source.xmax <= other_source.xmax and \
                    source.ymin >= other_source.ymin and \
                    source.ymin <= other_source.ymax:
                return True
            else:
                return False

        for i in range(len(self.other_sources)):
            source = self.other_sources[i]
            if not source.in_bounds:
                continue  # Skip this source if its bounding box is not in the cut-out
            if overlap_check(source, self.c_source):  # Checking with central source
                self.other_sources[i].overlap_flag = True
                self.toggle_difficult_flag()
                print('Overlapping (central) source found!')
            else:  # Checking with all other sources
                for j in range(len(self.other_sources)):
                    if i != j and self.other_sources[j].in_bounds and overlap_check(source, self.other_sources[j]):
                        self.other_sources[i].overlap_flag = True
                        self.toggle_difficult_flag()
                        print('Overlapping source found!')

    def add_prediction(self, l, label):
        if 'predictions' not in self.__dict__:
            self.predictions = []
        self.predictions.append(prediction(l, label))

    def print_all(self):
        print('----------------------')
        print(f'Central source name: {self.c_source.sname}')
        print('Other sources:')
        for i in range(len(self.other_sources)):
            print(self.other_sources[i].sname)
        print(f'RA-DEC: {self.c_source.ra},{self.c_source.dec}')
        print(f'DR1 name: {self.lofarname_DR1}')
        print(f'DR2 name: {self.lofarname_DR2}')
        print(f'Wise name: {self.wisename}')
        print('----------------------')
