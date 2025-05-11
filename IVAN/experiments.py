from nnverify.common.dataset import Dataset
from nnverify.common import Domain
from parse_args import parse_args

import subprocess




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
    else:
        raise ValueError("Invalid dataset name. Choose from 'mnist' or 'cifar10'.")
    
    network = args.network+'.onnx'
    res = f'''
from unittest import TestCase
from nnverify.common.dataset import Dataset
from nnverify.common import Domain
from nnverify.analyzer import Analyzer
import nnverify.config as config

class NewTest(TestCase):
    def new_test(self):
        args = config.Args(net="{network}", domain={program}, dataset={dataset}, eps={args.eps}, count={args.batch_size}, run_batch=True)
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
    cmd = 'python3 -m unittest -v nnverify.tests.experiment.NewTest.new_test'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    # print(result.stderr)

args = parse_args()
create_test(args)
run_test()