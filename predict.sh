#!/bin/bash
# ARG    CONTENT                                    DEFAULT
# $1   which evaluation step (0,1,2...)
# $2 [input file path ]                             /home/c4c-user/data/input/
# $3 [output file path]                             /home/c4c-user/data/output/
# $4 [code root path, where we unzip to]            /home/c4c-user/data/code/

source $2/taskParameters.ini

PYTHONPATH=$(realpath $4) python3 $4/main.py \
    --eval_step $1 \
    --input_dir $2 \
    --output_dir $3 \
    --code_dir $4 \
    --use_flask 1 \
    --gpu_count 1

# Keep this, this if for time stamping your submission
makeready.sh $3/Y$1
