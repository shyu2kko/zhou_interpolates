'''
cross-validation (5 fold)
'''
import sys
sys.path.append('../')
from config import *
from interpmodules import deterministic, geospatial, helpers, metrics
import joblib
import pandas as pd
import wrappers
import pandas as pd


# load tracer - gene mappings
mappings = pd.read_csv(projdir + '/data/beliveau2017_serotonergic_names_mappings.csv')

space = "fsLR-4k"
strategy = "all"
percentage = "100"
datasource = 'microarray-pet'

# load gene-receptor-tracer mappings
mappings = pd.read_csv(projdir + '/data/beliveau2017_serotonergic_names_mappings.csv')

# load microarray points data
src_pvdata = joblib.load(projdir + f'/data/icbm09c_cic-volume_microarray-pet.pkl')

# load dsk centroids
sphere_dsk33 = joblib.load( projdir + '/data/fsLR-4k_sphere_parc-dsk33.pkl')
ctx_dsk33 = joblib.load(projdir + '/data/fsLR-4k_midthickness_parc-dsk33.pkl')

# re-register them to icbm09
mni_cents = fslr_to_mni_coords_transform(coords = ctx_dsk33[2], inverse=True)

rank_scores_plotdata = pd.DataFrame()

#for gi, gene in enumerate(mappings['gene']):
gene = 'HTR1A'
maptag = f"mrna_{gene}"
cov = wrappers.generate_covariates(curr_map=maptag, src_polydata=src_pvdata, \
                         trg_polydata=None, nspins=0, datasource=datasource)
all_run_rank_scores = pd.DataFrame(columns = range(mni_cents.shape[0]) )

for j in range(mni_cents.shape[0]):
    center = mni_cents[j,:]

    M = {}
    run_metric_scores = pd.DataFrame()

    for approach in approaches:
    
        if approach in ['idw', 'knn', 'rbf']:
            _,_,_,mdf_concat = deterministic._find_param_best_cv(method=approach, \
                                                                 pvdata=src_pvdata, \
                                                                    known_coords=src_pvdata.points, \
                                                                   full_map=src_pvdata[maptag], \
                                                                 distdep = True, center = center,\
                                                                 nfolds=5, seed=321)
            M[approach] = mdf_concat
            
        if approach in ['swr', 'krige', 'regkrige']:
            
            mdf_concat = geospatial._iterate_chunks(method=approach, pvdata=src_pvdata, covariates=cov, \
                full_map=src_pvdata[maptag], center = center, distdep = True, nfolds=5, seed=321)
            
            M[approach] = mdf_concat
        
        run_metric_scores[approach] = np.nanmean(M[approach], axis =1)

    joblib.dump(M, projdir + f'/results/distdep_cv/microarray/use_case_microarray_{gene}_dsk-parc{j}_scores.pkl')
    
    print(run_metric_scores)

    run_rank_scores = metrics.accuracy_rank(run_metric_scores.T)
    run_rank_scores = np.nanmean(run_rank_scores, axis = 1)
    all_run_rank_scores[j] = run_rank_scores

joblib.dump(all_run_rank_scores, projdir + f'/results/distdep_cvmicroarray/use_case_microarray_{gene}_distdep_cvbrain.pkl')
    
#rank_scores_plotdata[gene] = all_run_rank_scores.mean(axis=1)
