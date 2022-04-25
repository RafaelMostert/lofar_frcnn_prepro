# Import massive fits table, drop all columns except the ones we need
# Export to hdf5 using pandas

from astropy.io import fits

# dat = Table.read('dr2_combined.fits', format='fits')
hdu_obj = fits.open('/net/beerze/data2/wwilliams/projects/lofar_surveys/DR2/opt/dr2_combined.fits', memmap=True)
print(hdu_obj.info())
print()
print(hdu_obj[0].header)
print([k for k in hdu_obj[1].header.keys() if not k.startswith('COMMENT')])
print(hdu_obj[1].columns)
print("first row:", hdu_obj[1].data[0])
hdu_obj.close()
"""
#dat = Table.read('/net/beerze/data2/wwilliams/projects/lofar_surveys/DR2/opt/dr2_combined.fits', format='fits')
good_cols = ['ra', 'dec','iFApFlux', 'w1Mag']
cols = list(dat.columns)
for col in cols:
    print(col)
bad_cols = [c for c in cols if not c in good_cols]
del dat[bad_cols]
dat.info
dat[:5]
df = dat.to_pandas()
df.to_hdf('dr2_combined.h5')
"""
