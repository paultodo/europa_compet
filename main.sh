#!/bin/bash

# ARGUMENTS
# ARG    CONTENT                                    DEFAULT
# $1 [input file path ]                             /home/c4c-user/data/input/
# $2 [output file path]                             /home/c4c-user/data/output/
# $3 [code root path, where we unzip to]            /home/c4c-user/data/code/

EVAL_STEP=$1
INPUT_DIR=$(realpath "$2")
OUTPUT_DIR=$(realpath "$3")
CODE_DIR=$(realpath "$4")
GPU_COUNT=$5
PORT=$6

PORT=4130

ALIVE_COUNTER=100
while [  $ALIVE_COUNTER -gt 0 ]; do
    echo The counter is $ALIVE_COUNTER
    ALIVE_OUTPUT=$(curl -fs http://127.0.0.1:$PORT/alive)
    
    if [[ $ALIVE_OUTPUT == "alive" ]]; then
        break
    fi
    
    let ALIVE_COUNTER=ALIVE_COUNTER-1 
    sleep .05
done

if [ $ALIVE_COUNTER -eq 0 ]; then
    exit 1
fi

echo "ALIVE OUTPUT $ALIVE_OUTPUT"
echo "ALIVE OUTPUT $ALIVE_COUNTER"

echo -d "{\"eval_step\": $EVAL_STEP, \"input_dir\": \"$INPUT_DIR\", \"output_dir\" : \"$OUTPUT_DIR\"}" -H "Content-Type: application/json" -X POST http://127.0.0.1:$PORT/prediction

PREDICTION_OUTPUT=$(curl -d "{\"eval_step\": $EVAL_STEP, \"input_dir\": \"$INPUT_DIR\", \"output_dir\" : \"$OUTPUT_DIR\"}" -H "Content-Type: application/json" -X POST http://127.0.0.1:$PORT/prediction)

echo $PREDICTION_OUTPUT


COLLECT_COUNTER=100
while [ $COLLECT_COUNTER -gt 0 ]; do
    echo The counter is $COLLECT_COUNTER
    COLLECT_OUTPUT=$(curl http://127.0.0.1:$PORT/collect)
    echo $COLLECT_OUTPUT
    
    if [[ $COLLECT_OUTPUT == "finished" ]]; then
        break
    fi
    
    let COLLECT_COUNTER=COLLECT_COUNTER-1 
    sleep .05
done

if [ $COLLECT_COUNTER -eq 0 ]; then
    exit 1
fi