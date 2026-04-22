'''
Preparing microarray data
'''

import sys
sys.path.append("../")

from config import *

import abagen
import os
import pandas as pd
import pyvista as pv



receptor_names = ['5HT1A', '5HT1B', '5HT2A', '5HT4', '5HTT']
tracer_names = ['cumi101', 'az10419369', 'cimbi36', 'sb207145', 'dasb']
gene_symbols = ['HTR1A', 'HTR1B', 'HTR2A', 'HTR4', 'SLC6A4']

mappings = pd.DataFrame([receptor_names, tracer_names, gene_symbols], index = ['receptor', 'tracer', 'gene']).T
mappings.to_csv(projdir + '/data/beliveau2017_serotonergic_names_mappings.csv')



os.environ['ABAGEN_DATA'] = projdir + "/data/ahba/"

donor_ids = ['10021', '12876', '14380', '15496', '15697', '9861']
cic_ids = ['H0351.1009', 'H0351.1012', 'H0351.1015', 'H0351.1016', 'H0351.2001', 'H0351.2002']

map_to_cic = {'10021': 'H0351.2002', \
              '12876': 'H0351.1009', \
              '14380': 'H0351.1012', \
              '15496': 'H0351.1015', \
              '15697': 'H0351.1016', \
              '9861': 'H0351.2001'}


#------------Update probe coordinates--------#
microarray = abagen.fetch_microarray(donors = donor_ids)

datadir = os.environ.get('ABAGEN_DATA') + '/microarray/'

for donor_id in donor_ids:
    _src = datadir + f"/normalized_microarray_donor{donor_id}/"
    os.rename(_src + "SampleAnnot.csv", _src + "SampleAnnot_original.csv")


# all probes included
for donor_id in donor_ids:
    df0 = pd.read_csv(datadir + f'/normalized_microarray_donor{donor_id}/SampleAnnot_original.csv')
    df1 = pd.read_csv(projdir + f'/data/yee_transformed-points/{map_to_cic[donor_id]}_SampleAnnot_RAS_mni_nonlin.csv')

    df0['mni_x'] = df1['mni_nlin_x']
    df0['mni_y'] = df1['mni_nlin_y']
    df0['mni_z'] = df1['mni_nlin_z']

    df0.to_csv(datadir + f'/normalized_microarray_donor{donor_id}/SampleAnnot.csv')

e, m = abagen.get_samples_in_mask(mask=None, \
                                  corrected_mni=False, \
                                  lr_mirror = None, \
                                  ibf_threshold=0.1)

joblib.dump((e, m), projdir + '/data/abagen_get_samples_in_mask_ibf-01_coords-cic.pkl')

# initiate a new dataframe to compile the expressions and then save it

serotonergic = pd.DataFrame([None])

for gene_symbol in gene_symbols:

    if serotonergic.empty: serotonergic = e[gene_symbol]
    else: serotonergic = pd.concat([ serotonergic, e[gene_symbol] ], axis = 1)

serotonergic = serotonergic.drop(0, axis = 0)
serotonergic = serotonergic.drop(0, axis = 1)

joblib.dump((serotonergic,m), projdir + '/data/abagen_ibf-01_serotonergic_gene_expression.pkl')


# put the data in point set format
microarray = pv.PolyData(m.to_numpy())
for gene_symbol in gene_symbols:
    microarray['mrna_' + gene_symbol] = e[gene_symbol]

# getting volumetric covariates in the same point sets
import joblib
import nibabel as nb
from nibabel.affines import apply_affine
import numpy as np
import numpy.linalg as npl
import os
from scipy.ndimage import map_coordinates

reregistered_pet_dir = '/path/to/outputs/from/volume/coregistration/'

fnames = os.listdir(reregistered_pet_dir)

for fname in fnames:
    PET = nb.load(reregistered_pet_dir + fname)

    sourcetag = fname.split('_')[0].split('-')[1]
    desctag = fname.split('_')[1].split('-')[1]

    maptag = f'pet_{sourcetag}_{desctag}'

    points = microarray.points
    img_data = PET.get_fdata()
    inverse_aff = npl.inv(PET.affine)
    ijk_points = apply_affine(inverse_aff, points)

    vals = np.zeros(microarray.n_points)
    for pi in range(points.shape[0]):
        
        vox_value = map_coordinates(img_data, ijk_points[pi,:].reshape(-1,1), order=1)
        vals[pi] = vox_value[0]
        
    microarray[maptag] = vals

joblib.dump(microarray, projdir + '/data/icbm09c_cic-volume_microarray-pet.pkl')