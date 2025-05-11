import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="ConstraintFlow experiments")

    parser.add_argument('--single_experiment', dest='single_experiment', action='store_true', help='Enable single experiment mode')
    parser.add_argument('--multiple_experiments', dest='single_experiment', action='store_false', help='Disable single experiment mode')
    parser.set_defaults(single_experiment=True)

    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--eps', type=float, default=0.01, help='Epsilon')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset')

    parser.add_argument('--network', type=str, default='mnist_relu_3_50', help='Network name')
    parser.add_argument('--network_format', type=str, default='onnx', help='Network format')
    parser.add_argument('--network_type', type=str, default='fcn', help='Network type')

    parser.add_argument('--program', type=str, default='deeppoly', help='AutoLirpa program')
    
    parser.add_argument('--num_runs', type=int, default=1, help='Run the experiment n times for average times')
    parser.add_argument('--exp_name', type=str, default='eran')

    parser.add_argument('--print_intermediate_results', dest='print_intermediate_results', action='store_true', help='Print intermediate results')
    parser.set_defaults(print_intermediate_results=False)

    parser.add_argument('--record_time', dest='record_time', action='store_true', help='Log function times')
    parser.set_defaults(record_time=False)

    parser.add_argument('--logs', type=str, default='stdout', help='Logs file')


    args = parser.parse_args()
    return args