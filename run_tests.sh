# Parameters:
# $1 input_dir
# $2 model_dir
# $3 code_dir
if [ -z ${3+x} ]; then
    echo "ERROR: Please set all params: run_tests.sh input_dir output_dir code_dir"
    exit 1
fi

INPUT_DIR=$(realpath "$1")
OUTPUT_DIR=$(realpath "$2")
CODE_DIR=$(realpath "$3")
rm -f "$CODE_DIR/tests_scores_log.txt"
echo
echo
echo "Outputing the tests scores to $CODE_DIR/tests_scores_log.txt"
echo
echo
echo "To follow how the full prediction tests advance, check $INPUT_DIR/cache/flask_log.txt or $OUTPUT_DIR"
echo
echo
py.test-3 --input_dir="$INPUT_DIR" --output_dir="$OUTPUT_DIR" --code_dir="$CODE_DIR" -s -v "$CODE_DIR/tests/"
