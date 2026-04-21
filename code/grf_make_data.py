'''
GRF and sampling
'''

import sys
sys.path.append("../")
from config import *

from interpmodules import helpers
import joblib
import nibabel as nb
import numpy as np
import pyvista as pv


#-------------GENERATE GRFS----------#

# convert gifti image to pyvista mesh
path = neuromaps.datasets.fetch_fslr(density = '4k')['sphere'].L
gii = nb.load(path)

pvdata = helpers.gii_to_polydata(
    (gii.darrays[0].data, gii.darrays[1].data)
    )

# instantiate GRFs
make_ranges = np.arange(25,105,5) # 25 to 100 in steps of 5

for ran in make_ranges:
    instance = helpers.make_random_maps(ran=ran, mesh=pvdata, seed=321)
    pvdata[str(ran)] = instance

joblib.dump(pvdata, projdir + '/data/fsLR-4k_sphere_grf.pkl')



#-------------GENERATE SAMPLES----------#

# EVEN samples
make_percents = np.arange(5,55,5) # 5 to 50 in steps of 5

for p in make_percents:
    sampling = helpers.get_traning_set(mesh=pvdata, sphere=pvdata, pct_split=p, n_iters=1, parcellation_name=None, method='fibonacci', sphere_foci=None, seed=321)
    labels = ("known_coords", "known_verts", "unknown_coords", "unknown_verts")
    joblib.dump((labels, sampling), projdir + f'/data/sampling/fsLR-4k_sphere_samp-fibonacci_pct-{str(p)}.pkl')

# RANDOM samples
for p in test_percentages:
    sampling = helpers.get_traning_set(mesh=pvdata, sphere=pvdata, pct_split=p, n_iters=100, parcellation_name=None, method='random', sphere_foci=None, seed=321)
    labels = ("known_coords", "known_verts", "unknown_coords", "unknown_verts")
    joblib.dump((labels, sampling), projdir + f'/data/sampling/fsLR-4k_sphere_samp-random_pct-{str(p)}_iters-100.pkl')
 