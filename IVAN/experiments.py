from nnverify.common.dataset import Dataset
from nnverify.common import Domain
from parse_args import parse_args

#FEAT: os.environ carries --dtype to the analysis subprocess as IVAN_DTYPE.
import os
import subprocess
# Changes for GPU
import sys




def create_test(args):
    if args.program == 'deeppoly':
        program = Domain.DEEPPOLY 
    elif args.program == 'zono':
        program = Domain.DEEPZ
    elif args.program == 'ibp':
        program = Domain.BOX
    elif args.program == 'polyzono':
        program = Domain.POLYZ
    else:
        raise ValueError("Invalid program name. Choose from 'deeppoly', 'zono', 'ibp', or 'polyz'.")
    
    if args.dataset == 'mnist':
        dataset = Dataset.MNIST
    elif args.dataset == 'cifar10':
        dataset = Dataset.CIFAR10
    #FEAT: the cf-harness's CIFAR-100 configs and VNN-COMP specs (--dataset vnnlib:<benchmark folder>).
    elif args.dataset == 'cifar100':
        dataset = Dataset.CIFAR100
    elif args.dataset.startswith('vnnlib:'):
        dataset = Dataset.VNNLIB
    else:
        raise ValueError("Invalid dataset name. Choose from 'mnist', 'cifar10', 'cifar100' or 'vnnlib:<benchmark folder>'.")
    vnnlib = args.dataset[len('vnnlib:'):] if dataset == Dataset.VNNLIB else None
    
    network = args.network+'.onnx'
    res = f'''
from unittest import TestCase
from nnverify.common.dataset import Dataset
from nnverify.common import Domain
from nnverify.analyzer import Analyzer
import nnverify.config as config

class NewTest(TestCase):
    def new_test(self):
        # Changes for GPU
        args = config.Args(net="{network}", domain={program}, dataset={dataset}, eps={args.eps}, count={args.batch_size}, run_batch=True,
                           warmup={args.warmup}, repeat={args.repeat}, normalization="{args.normalization}",
                           num_images={args.num_images}, results_file={args.results_file!r}, vnnlib={vnnlib!r})
        Analyzer(args).run_analyzer()
'''

    file_path = 'nnverify/tests/experiment.py'

    with open(file_path, 'w') as f:
        f.write(res)
    
def run_test():
    print(f'Experiment:')
    print(f'Network: {args.network}')
    print(f'Batch size: {args.batch_size}')
    print(f'Program: {args.program}')
    print(f'Epsilon: {args.eps}')
    print(f'Dataset: {args.dataset}')
    print()
    # Changes for GPU
    # Use the interpreter running this script, not whatever python3 is first on PATH
    cmd = [sys.executable, '-m', 'unittest', '-v', 'nnverify.tests.experiment.NewTest.new_test']
    #FEAT: the analysis runs in this subprocess, so --dtype reaches it as IVAN_DTYPE (read by nnverify/config.py).
    #FEAT: --profile_comm reaches it the same way, as IVAN_PROFILE_COMM.
    result = subprocess.run(cmd, capture_output=True, text=True,
                            env=dict(os.environ, IVAN_DTYPE=args.dtype, IVAN_PROFILE_COMM=str(int(args.profile_comm)),
                                     IVAN_BATCH_TIMEOUT=str(args.batch_timeout),
                                     IVAN_WARMUP_TIMEOUT=str(args.warmup_timeout)))
    print(result.stdout)
    print(result.stderr)
    # Changes for GPU
    return result.returncode

args = parse_args()
create_test(args)
# Changes for GPU
sys.exit(run_test())