"""
Prepare model for prediction.
If files in model_dir is sufficient, do nothing
Otherwise decompress weights or retrain model
"""
import argparse
import os
import subprocess


def prepare_model(configs, batch_size=4500, nb_epoch=18, test_run=False):
    cmd = [
        'python3',
        os.path.join(configs['code_dir'], 'prepare_model.py'),
        '--input_dir', configs['input_dir'],
        '--model_dir', configs['model_dir'],
        '--horizon', str(configs['HORIZON']),
        '--batch_size', str(batch_size),
        '--nb_epoch', str(nb_epoch),
    ]
    if test_run:
        cmd.append('--test_run')
    subprocess.run(cmd)


def does_model_exists(model_dir):
    """Output True if model exists"""
    model_fn_isfile = os.path.isfile(os.path.join(model_dir, 'model_fn.pkl'))
    model_params_isfile = os.path.isfile(os.path.join(model_dir, 'model_params.pkl'))
    model_global_isfile = os.path.isfile(os.path.join(model_dir, 'global_params.pkl'))
    model_weights_isfile = os.path.isfile(os.path.join(model_dir, 'model_weights.h5'))
    # model_weights_xz_isfile = os.path.isfile(os.path.join(model_dir, 'model_weights.h5.xz'))

    return (model_fn_isfile and
            model_params_isfile and
            model_global_isfile and
            model_weights_isfile)
            # (model_weights_isfile or model_weights_xz_isfile))


def decompress_model_if_necessary(model_dir):
    """Decompress the weights of the model if they are compressed (format is xz)"""
    model_weights_xz_file = os.path.join(model_dir, 'model_weights.h5.xz')
    model_weights_xz_isfile = os.path.isfile(model_weights_xz_file)
    if model_weights_xz_isfile:
        subprocess.run([
            'xz',
            '-d', '-k',
            model_weights_xz_file
        ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    root_dir = '/home/c4c-user/data'
    parser.add_argument("--input_dir", help="Input data path, must contain \
                        taskParameters.ini, train/ and adapt/",
                        type=str, default=os.path.join(root_dir, 'sample_data'))
    parser.add_argument("--model_dir", help="Path to store trained model",
                        type=str, default=os.path.join(root_dir, 'result'))
    parser.add_argument("--horizon", help="Prediction horizon",
                        type=int, default=12)
    parser.add_argument("--batch_size", help="Batch size if training",
                        type=int, default=4500)
    parser.add_argument("--nb_epoch", help="Nb epoch if training",
                        type=int, default=18)
    parser.add_argument("--test_run", help="Run a mock epoch for testing purpose",
                        action='store_true')
    parser.add_argument("--gpu_count", help="Default count of GPU",
                        type=int, default=8)
    args = parser.parse_args()

    """Try to load model at given dir, if fail train new model"""
    if not does_model_exists(args.model_dir):
        if os.path.isfile(os.path.join(args.model_dir, 'model_weights.h5.xz')):
            decompress_model_if_necessary(args.model_dir)
        else:
            print('Cannot load model. Training model.')
            from src.train_model import train_model
            if args.test_run:
                train_model(model_dir=args.model_dir,
                            horizon=args.horizon,
                            input_dir=args.input_dir,
                            batch_size=1000,
                            time_step_per_epoch=1,
                            nb_epoch=1,
                            )
            else:
                train_model(model_dir=args.model_dir,
                            horizon=args.horizon,
                            input_dir=args.input_dir,
                            batch_size=args.batch_size,
                            time_step_per_epoch=args.batch_size,
                            nb_epoch=args.nb_epoch,
                            test_run=args.test_run
                            )
            assert does_model_exists(args.model_dir)
