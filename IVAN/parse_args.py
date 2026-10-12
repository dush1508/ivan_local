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

    # Changes for GPU
    # Harness mode (experiments/compare_ivan.py): constraintflow-gpu's warmup/repeat protocol
    parser.add_argument('--warmup', type=int, default=0, help='Untimed runs on the next batches of images (harness mode)')
    parser.add_argument('--repeat', type=int, default=None,
                        help='Timed runs on the first batch_size images; setting it selects harness mode')
    parser.add_argument('--num_images', type=int, default=0,
                        help='Harness mode: verify this many images, batch_size at a time, one timed run per batch; '
                             'warmup j uses the images after them. Needs --results_file and --repeat (any value).')
    parser.add_argument('--results_file', type=str, default=None,
                        help='With --num_images: one JSON line per warmup and timed batch, written outside the timer')
    parser.add_argument('--normalization', choices=['ivan', 'cf'], default='ivan',
                        help="Input normalization: IVAN's own (MNIST mean 0/std 1) or constraintflow-gpu's "
                             "(MNIST 0.1307/0.3081); CIFAR is the same in both")
    #FEAT: --dtype float64 runs the analysis in fp64, the precision oracle for bound mismatches.
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32',
                        help='float64 computes weights, input box and every bound in fp64')
    #FEAT: --profile_comm records CPU-GPU communication per timed batch (nnverify/comm_profile.py).
    parser.add_argument('--profile_comm', action='store_true',
                        help='With --num_images: profile each timed batch with torch.profiler and add syncs, '
                             'alloc_syncs, gpu_idle_pct and sync_calls to its results line; its time goes to '
                             'profiled_time_s instead of time_s. Warmups are not profiled.')


    args = parser.parse_args()
    return args