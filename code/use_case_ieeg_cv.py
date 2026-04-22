import sys
sys.path.append("../")

from config import *
from figs_config import *
from interpmodules import deterministic, geospatial, helpers, metrics
import joblib
import pandas as pd
import wrappers

# load dsk centroids
ctx_dsk33 = joblib.load(projdir + '/data/fsLR-4k_midthickness_parc-dsk33.pkl')

# re-register them to icbm09
mni_cents = fslr_to_mni_coords_transform(coords = ctx_dsk33[2], inverse=True)

rank_scores_plotdata = pd.DataFrame()

for fband in fbands:

    maptag = f"{fband}_no_outlier"
    datasource = f"ieeg-{fband}-pet"
    src_pvdata = joblib.load(projdir + f'/data/icbm09c_left-volume_ieeg-{fband}-pet.pkl')
    cov = wrappers.generate_covariates(curr_map=f"{fband}_zscored_no_outlier", src_polydata=src_pvdata, \
                             trg_polydata=None, nspins=0, datasource=datasource)
    
    all_run_rank_scores = pd.DataFrame(columns = range(mni_cents.shape[0]) )
                                       
    for j in range(mni_cents.shape[0]):
        center = mni_cents[j,:]
        
        M = {}
        run_metric_scores = pd.DataFrame()
        run_rank_scores = None
        for approach in approaches:
         
            if approach in ['idw', 'knn', 'rbf']:
                _,_,_,mdf_concat = deterministic._find_param_best_cv(method=approach, \
                                                                     pvdata=src_pvdata, \
                                                                        known_coords=src_pvdata.points, \
                                                                       full_map=src_pvdata[f"{fband}_zscored_no_outlier"], \
                                                                     distdep = True, center = center,\
                                                                     nfolds=5, seed=321)
                M[approach] = mdf_concat
    
            if approach in ['swr', 'krige', 'regkrige']:
                mdf_concat = geospatial._iterate_chunks(method=approach, pvdata=src_pvdata, covariates=cov, \
            full_map=src_pvdata[f"{fband}_zscored_no_outlier"], center = center, distdep = True, nfolds=5, seed=321)
                M[approach] = mdf_concat
            
            else: continue
            joblib.dump(M, projdir + f'/results/distdep_cv/ieeg/use_case_ieeg_{fband}_dsk-parc{j}_scores.pkl')
            
            run_metric_scores[approach] = M[approach].mean(axis=1)
        
            run_rank_scores = metrics.accuracy_rank(run_metric_scores)
            run_rank_scores = np.mean(run_rank_scores, axis = 1)
            
            all_run_rank_scores[j] = run_rank_scores

    joblib.dump(all_run_rank_scores, projdir + f'/results/distdep_cv/ieeg/use_case_ieeg_{fband}_dsk33_cvbrain.pkl')