import numpy as np
from shapely.geometry import Polygon
from shapely.ops import cascaded_union


def ellipse(x0, y0, a, b, pa, n=200):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    st = np.sin(theta)
    ct = np.cos(theta)
    pa = np.deg2rad(pa + 90)
    sa = np.sin(pa)
    ca = np.cos(pa)
    p = np.empty((n, 2))
    p[:, 0] = x0 + a * ca * ct - b * sa * st
    p[:, 1] = y0 + a * sa * ct + b * ca * st
    return Polygon(p)


class Make_Shape(object):
    '''
	Before being altered slightly the code for this class was taken from the process_lgz.py code written by Martin Hardcastle. 

    Basic idea taken from remove_lgz_sources.py -- maybe should be merged with this one day
    but the FITS keywords are different.
    '''

    def __init__(self, clist):
        '''
        clist: a list of components that form part of the source, with RA, DEC, DC_Maj...
        '''
        ra = np.mean(clist['RA'])
        dec = np.mean(clist['DEC'])

        ellist = []
        for i in range(len(clist)):
            # Import the RA and DEC of the components
            n_ra, n_dec = clist.iloc[i]['RA'], clist.iloc[i]['DEC']
            # Calculate the central coordinates of the ellipses relative to the mean
            x = 3600 * np.cos(dec * np.pi / 180.0) * (ra - n_ra)
            # x=3600*np.cos(dec*np.pi/180.0)*(n_ra-ra)
            y = 3600 * (n_dec - dec)
            # Form the ellipses with relative coordinates
            newp = ellipse(x, y, clist.iloc[i]['DC_Maj'] + 0.1, clist.iloc[i]['DC_Min'] + 0.1, clist.iloc[i]['PA'])
            ellist.append(newp)

        self.cp = cascaded_union(ellist)
        self.ra = ra
        self.dec = dec
        self.h = self.cp.convex_hull
        a = np.asarray(self.h.exterior.coords)

        # for i,e in enumerate(ellist):
        #    if i==0:
        #        a=np.asarray(e.exterior.coords)
        #    else:
        #        a=np.append(a,e.exterior.coords,axis=0)
        """
        mdist2=0
        bestcoords=None

        # Here the coordinates of the two points that are furthest appart from each other are selected
        for r in a:
            dist2=(a[:,0]-r[0])**2.0+(a[:,1]-r[1])**2.0
            idist=np.argmax(dist2)
            mdist=dist2[idist]
            if mdist>mdist2:
                mdist2=mdist
                bestcoords=(r,a[idist])
        self.mdist2=mdist2
        self.bestcoords=bestcoords
        """
        self.a = a
        self.best_a = 0

    def length(self):
        # Returns distance between the two 'best coordinates'
        return np.sqrt(self.mdist2)

    # end Make_shape.length

    def pa(self):
        # Returns the north to east angle of the convex_hull main axis
        p1, p2 = self.bestcoords
        dp = p2 - p1
        angle = (180 * np.arctan2(dp[1], dp[0]) / np.pi) - 90
        if angle < -180:
            angle += 360
        if angle < 0:
            angle += 180
        return angle

    # end Make_shape.length

    def width(self):
        # Returns the maximum distance from the length vector to the outer edge of the hull
        p1, p2 = self.bestcoords
        print(p1, p2)
        d = np.cross(p2 - p1, self.a - p1) / self.length()
        print(self.a.shape, d.shape)
        self.best_a = self.a[np.argmax(d)]
        return 2 * np.max(d)

    # end Make_shape.width

    def geo_center(self):
        # Returns the geometric center of the convex hull
        w, pa = self.width(), self.pa()
        dx, dy = w / 2. * np.cos((np.pi / 180) * (pa)), w / 2. * np.sin((np.pi / 180) * (pa))
        print('dx,dy:', dx, dy)
        ax, ay = self.best_a
        print('best_a', self.best_a)
        # print self.a
        return self.ra + ((ax + dx) / (3600 * np.cos(self.dec * np.pi / 180.0))), self.dec + ((ay + dy) / 3600)

    # end Make_shape.geo_center

    def hull_box(self):
        # Returns the x,y coordinates of the box enclosing the convexl hull
        # print self.a
        # return self.ra+((self.a[:,0])/(3600*np.cos(self.dec*np.pi/180.0))),self.dec+((self.a[:,1])/3600)
        return self.a
    # end Make_shape.hull_box
