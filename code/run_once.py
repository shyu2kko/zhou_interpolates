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

polydata = fetch_polydata(geometry=args.geometry, space = 'fsLR-4k', datasource = args.datasource)
desc,all_sampling = fetch_sample(**dict(args._get_kwargs()))

print(desc)
print(polydata.array_names)

print('building covariates')
Yk0 = wrappers.generate_covariates(args.maptag, polydata, 3000, args.datasource)


set_up_analysis = {"datasource": args.datasource, \
                   "all_sampling": all_sampling, \
                    "Yk0": Yk0, \
                    "reduced_mesh": polydata, \
                    "maptag": args.maptag, \
                    "outdir": outdir, \
                    "tags": dict(args._get_kwargs()), \
                    "verbose": 1}

for iter_id in range(niterations):
    wrappers.run_one_iteration(
                **set_up_analysis,
                curr_iter=iter_id)