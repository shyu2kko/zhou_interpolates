from config import *
import wrappers

import argparse as argp

from joblib import Parallel, delayed, parallel_backend

parser = argp.ArgumentParser()
parser.add_argument("--geometry", '-g', type=StopIteration)
parser.add_argument("--percentage", '-pc', type=int)
parser.add_argument("--strategy", '-s', type=str)
parser.add_argument("--niterations", '-n', type=int)
parser.add_argument("--maptag", '-m', type=str)

args = parser.parse_args()

print(args._get_kwargs())

if args.strategy == 'fibonacci': niterations = 1
else: niterations = int(args.niterations)

polydata = fetch_polydata(geometry=args.geometry, space = 'fsLR-4k', datasource = 'grf')
desc,all_sampling = fetch_sample(**dict(args._get_kwargs()))

print(desc)
print(polydata.array_names)

Yk0 = wrappers.generate_covariates(polydata[args.maptag], polydata, 3000, 'grf')


set_up_analysis = {"datasource": 'grf', \
                   "all_sampling": all_sampling, \
                    "Yk0": Yk0, \
                    "reduced_mesh": polydata, \
                    "maptag": args.maptag, \
                    "outdir": outdir, \
                    "tags": dict(args._get_kwargs()), \
                    "verbose": 1}

with parallel_backend('multiprocessing', n_jobs=50):
    Parallel()(
        delayed(wrappers.run_one_iteration)(
            **set_up_analysis,
            curr_iter=iter_id
        )
        for iter_id in range(niterations)
    )