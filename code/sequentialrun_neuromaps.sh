#!/bin/bash

# submit run_once.py serially
# it will take roughly 3 days for all combinations
# should not be worse for neuromaps

cd /home/yzhou/gitrepo/interpolate/code/

percentages=(2 5 10 25 50)
maptags=('abagen_genepc1' \
'hcps1200_megalpha' \
'hcps1200_megbeta' \
'hcps1200_megdelta' \
'hcps1200_meggamma1' \
'hcps1200_meggamma2' \
'hcps1200_megtheta' \
'hcps1200_megtimescale' \
'hcps1200_myelinmap' \
'hcps1200_thickness' \
'hill2010_devexp' \
'hill2010_evoexp' \
'margulies2016_fcgradient01' \
'mueller2013_intersubjvar' \
'raichle_cbf' \
'raichle_cbv' \
'raichle_cmr02' \
'raichle_cmrglc' \
'sydnor2021_SAaxis' \
'xu2020_FChomology' \
'xu2020_evoexp')

for p in ${percentages[@]}; do
    for m in ${maptags[@]}; do

        python run_once.py -g 'reduced-midthickness' -d 'neuromaps' -p ${p} -m ${m} -n 1 -s 'fibonacci'

        python run_once.py -g 'reduced-midthickness' -d 'neuromaps' -p ${p} -m ${m} -n 100 -s 'dsk33'

        python run_once.py -g 'reduced-midthickness' -d 'neuromaps' -p ${p} -m ${m} -n 100 -s 'lobes'
    
    done
done