import numpy as np

import argparse
import os
import os.path
import subprocess
from src.utils import get_taskParameters
from glob import glob


# Default values based on starting kit v1.01
parser = argparse.ArgumentParser()
root_dir = os.getcwd()
parser.add_argument("--input_dir", help="Input data path, must contain \
                    taskParameters.ini, train/ and adapt/",
                    type=str, default=os.path.join(root_dir, 'sample_data'))
parser.add_argument("--output_dir", help="Output data path, will store prediction",
                    type=str, default=os.path.join(root_dir, 'result'))
parser.add_argument("--code_dir", help="Code path",
                    type=str, default=os.path.join(root_dir, 'code'))
parser.add_argument("--use_flask", help="Use flask server for faster prediction",
                    type=int, default=0)
parser.add_argument("--gpu_count", help="Default count of GPU",
                    type=int, default=1)
args = parser.parse_args()


if __name__ == '__main__':
    taskParameters = get_taskParameters(args.input_dir)
    # Clean cache and results
    completed = subprocess.run([
        'rm',
        '-rf',
        os.path.join(args.code_dir, 'cache')
    ])
    for path in glob(os.path.join(args.output_dir, '*.h5')):
        os.remove(path)

    os.makedirs(os.path.join(args.code_dir, 'cache'), exist_ok=True)

    for i in range(taskParameters['NUMBEROFSTEPS']):
        completed = subprocess.run([
            'python3',
            os.path.join(args.code_dir, 'main.py'),
            '--eval_step', str(i),
            '--input_dir', args.input_dir,
            '--output_dir', args.output_dir,
            '--code_dir', args.code_dir,
            '--use_flask', str(args.use_flask),
            '--gpu_count', str(args.gpu_count),
        ])

    subprocess.run('curl -X GET localhost:4130/shutdown', shell=True)
