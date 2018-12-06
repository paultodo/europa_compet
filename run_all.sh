#!/bin/bash

# ARGUMENTS
# ARG    CONTENT                                    DEFAULT
# $1 [input file path ]                             /home/c4c-user/data/input/
# $2 [output file path]                             /home/c4c-user/data/output/
# $3 [code root path, where we unzip to]            /home/c4c-user/data/code/

# Load parameters HORIZON and NUMBEROFSTEPS
start_time=`date +%s`

source $1/taskParameters.ini

# Clean cache and output_dir
rm -f $3/cache/*
rm -f $2/*

# Run all steps
for step in $(seq 0 ${NUMBEROFSTEPS-1})
do
    bash $3/predict.sh $step $1 $2 $3
    set -e
    ls $2/Y$step.h5
    set +e
done



# score
python3 $3/compute_score.py --input_dir $1 --output_dir $2

end_time=`date +%s`
runtime=$((end_time-start_time))
echo $runtime" seconds"
echo $runtime" seconds" >> $2/score.txt
