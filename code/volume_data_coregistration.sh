#!/bin/bash


wdir=/path/to/working/directory/


export ANTSPATH=/path/to/ants/bin/


#:::::::::::::::::::::::::::::Nonlinear Transform from MNI to fsLR:::::::::::::::::::::::::#
# target
tpl_mni=$wdir/mni_icbm152_t1_tal_nlin_sym_09c.nii
# source
tpl_fslr=$wdir/icbm_avg_152_t1_tal_nlin_symmetric_VI.nii


# get nonlinear transform
$ANTSPATH/antsRegistrationSyNQuick.sh -d 3 \
    -m $tpl_fslr \
    -f $tpl_mni \
    -o $wdir/xfm/nlin6asym_to_2009csym \
    -n 32


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
# resample volumetric annotations from Neuromaps
# to the same ICBM2009C MNI template

IFS=$' ' read -d '' -r -a maps < $wdir/image_paths # array of path strings

tpl=$wdir/mni_icbm152_t1_tal_nlin_sym_09c.nii

for map in ${maps[@]}; do

    # get nonlinear transform
    brain_in=$map

    map_basename0="${map##*/}"
    map_basename="${map_basename0%.*.*}"

    $ANTSPATH/antsRegistrationSyNQuick.sh -d 3 \
        -m $brain_in \
        -f $tpl \
        -o $wdir/xfm/${map_basename}_to_cic \
        -n 32

    $ANTSPATH/antsApplyTransforms -d 3 \
        -i $brain_in \
        -r $tpl \
        -o $wdir/out/"${map_basename}_reregistered.nii" \
        -t $wdir/xfm/"${map_basename}_to_cic1Warp.nii.gz" \
        -t $wdir/xfm/"${map_basename}_to_cic0GenericAffine.mat" \
        --interpolation "Linear" \
        -v 1
    done