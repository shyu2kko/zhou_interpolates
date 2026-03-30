from config import *
import wrappers

import argparse as argp

parser = argp.ArgumentParser()
parser.add_argument("--geometry", '-g', type=str)
parser.add_argument("--datasource", '-d', type=str)
parser.add_argument("--percentage", '-pc', type=int)
parser.add_argument("--strategy", '-s', type=str)
parser.add_argument("--niterations", '-n', type=int)
parser.add_argument("--maptag", '-m', type=str)

args = parser.parse_args()

print(args._get_kwargs())

if args.strategy == 'fibonacci': niterations = 1
else: niterations = int(args.niterations)

space='fsLR-4k'

no_ground_truth=False

if args.geometry in ['left-volume', 'cx-volume', 'cic-volume', 'volume']: 
    space = 'icbm09c'
    src_polydata = fetch_polydata(geometry=args.geometry, space=space, datasource = args.datasource)
    if args.datasource == "microarray-pet": 
        trg_polydata = fetch_polydata(geometry='reduced-midthickness', space='fsLR-4k', datasource = 'pet')
        benchmark_polydata = fetch_polydata(geometry='reduced-midthickness', space='fsLR-4k', datasource = 'beliveau')
    elif args.datasource in [ f"ieeg-{fband}-pet" for fband in fbands ]:
        trg_polydata = fetch_polydata(geometry='reduced-midthickness', space='fsLR-4k', datasource = 'pet')
        benchmark_polydata = fetch_polydata(geometry='reduced-midthickness', space='fsLR-4k', datasource = 'neuromaps')
    all_sampling = None
    no_ground_truth=True
    Yk0, yk1 = wrappers.generate_covariates(curr_map=args.maptag, src_polydata=src_polydata, trg_polydata=trg_polydata, nspins=3000, datasource=args.datasource)

if args.geometry not in ['left-volume', 'cx-volume', 'cic-volume', 'volume']: 
    desc,all_sampling = fetch_sample(**dict(args._get_kwargs()))
    src_polydata = fetch_polydata(geometry=args.geometry, space=space, datasource = args.datasource)
    trg_polydata = src_polydata
    benchmark_polydata = src_polydata

    print('building covariates')
    Yk0, yk1 = wrappers.generate_covariates(curr_map=args.maptag, src_polydata=src_polydata, trg_polydata=trg_polydata, nspins=3000, datasource=args.datasource)
        


print(src_polydata.array_names)



set_up_analysis = {"datasource": args.datasource, \
                   "all_sampling": all_sampling, \
                    "Yk0": Yk0, \
                    "yk1": yk1, \
                    "src_polydata": src_polydata, \
                    "trg_polydata": trg_polydata, \
                    "benchmark_polydata": benchmark_polydata, \
                    "maptag": args.maptag, \
                    "outdir": outdir, \
                    "tags": dict(args._get_kwargs()), \
                    "usecase": no_ground_truth, \
                    "verbose": 1}


for iter_id in range(niterations):
    wrappers.run_one_iteration(
            **set_up_analysis,
            curr_iter=iter_id)
