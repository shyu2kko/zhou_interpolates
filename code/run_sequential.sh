#!/bin/bash

# submit run_once.py serially and manually define input data

cd /path/to/project/directory/code/

#:::::::::::::::::Running GRF experiments:::::::::::::::::::::::::::::::::#
source batch_run_config 'grf'

for p in ${percentages[@]}; do
    for m in ${maptags[@]}; do

        python run_once.py -g 'sphere' -d 'grf' -p ${p} -m ${m} -n 1 -s 'fibonacci'
        python run_once.py -g 'sphere' -d 'grf' -p ${p} -m ${m} -n 100 -s 'random'
    
    done
done

#:::::::::::::::Running surface map experiments::::::::::::#
source batch_run_config 'neuromaps'

for p in ${percentages[@]}; do
    for m in ${maptags[@]}; do

        python run_once.py -g 'reduced-midthickness' -d 'neuromaps' -p ${p} -m ${m} -n 1 -s 'fibonacci'

        python run_once.py -g 'reduced-midthickness' -d 'neuromaps' -p ${p} -m ${m} -n 100 -s 'dsk33'

        #python run_once.py -g 'reduced-midthickness' -d 'neuromaps' -p ${p} -m ${m} -n 100 -s 'lobes'
    
    done
done


#::::::::::::::Running use case with iEEG alpha::::::::::::::::::#
source batch_run_config 'ieeg'

for m in ${maptags[@]}; do

    python run_once.py -g 'left-volume' -d ieeg-${m}-pet -p 100 -m ${m}_zscored_no_outlier -n 1 -s 'all'

done


#::::::::::Running use case with HTR1A::::::::::::::::::::::::::::#
source batch_run_config 'microarray'

for m in ${maptags[@]}; do

    python run_once.py -g 'cic-volume' -d 'microarray-pet' -p 100 -m ${m} -n 1 -s 'all'

done
