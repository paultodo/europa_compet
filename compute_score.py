import argparse
import os
from src.utils import get_taskParameters, compute_score


# Default values based on starting kit v1.01
parser = argparse.ArgumentParser()
root_dir = os.getcwd()
parser.add_argument("--input_dir", help="Input data path, must contain \
                    taskParameters.ini, train/ and adapt/",
                    type=str, default=os.path.join(root_dir, 'sample_data'))
parser.add_argument("--output_dir", help="Output data path, will store prediction",
                    type=str, default=os.path.join(root_dir, 'result'))
args = parser.parse_args()


if __name__ == '__main__':
    taskParameters = get_taskParameters(args.input_dir)
    score = compute_score(os.path.join(args.input_dir, 'adapt'),
                          args.output_dir,
                          **taskParameters
                          )
    result = 'Final MSE: {}\n'.format(score.mean())
    result += 'Final RMSE: {}\n'.format(score.mean()**0.5)
    print(result)
    with open(os.path.join(args.output_dir, 'score.txt'), 'w') as f:
        f.write(result)
