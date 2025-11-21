#!/bin/bash

# submit run_once.py serially
# it will take roughly 3 days for all combinations
# should not be worse for neuromaps

cd /home/yzhou/gitrepo/interpolate/code/

percentages=(2 5 10 25 50)
maptags=('25' '50' '75' '100')

for p in ${percentages[@]}; do
    for m in ${maptags[@]}; do

        python run_once.py -g 'sphere' -p ${p} -m ${m} -n 100 -s 'random'
    
    done
done