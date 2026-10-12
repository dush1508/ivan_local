# Changes for GPU
import contextlib
import gc
import io
import os
import sys
import torch

import nnverify.attack
import nnverify.domains
import nnverify.util as util
import nnverify.specs.spec as specs
import time
import nnverify.bnb.bnb as bnb

from nnverify import config
from nnverify.common import Status
from nnverify.specs.out_spec import OutSpecType
from nnverify.common.result import Result, Results
from nnverify.proof_transfer.template import TemplateStore
from nnverify.domains import build_transformer, get_domain_transformer

def get_precision(lb):
    verified = (lb >= 0).all(dim=1)
    precision = verified.sum() / verified.shape[0]
    return precision

class Analyzer:
    def __init__(self, args, net=None, template_store=None):
        """
        @param args: configuration arguments for the analyzer such as the network, domain, dataset, attack, count, dataset,
            epsilon and split
        """
        self.args = args
        self.net = net
        self.template_store = template_store
        self.timeout = args.timeout
        self.device = config.DEVICE
        self.transformer = None
        self.init_time = None

        if self.net is None:
            self.net = util.get_net(self.args.net, self.args.dataset)
        if self.template_store is None:
            self.template_store = TemplateStore()

    def analyze(self, prop):
        self.update_transformer(prop)
        tree_size = 1

        # Check if classified correctly
        if True or nnverify.attack.check_adversarial(prop.input, self.net, prop):
            status = self.analyze_no_split()
            return Status.MISS_CLASSIFIED, tree_size

        # Check Adv Example with an Attack
        if self.args.attack is not None:
            adv = self.args.attack.search_adversarial(self.net, prop, self.args)
            if nnverify.attack.check_adversarial(adv, self.net, prop):
                return Status.ADV_EXAMPLE, tree_size

        if self.args.split is None:
            status = self.analyze_no_split()
        elif self.args.split is None:
            status = self.analyze_no_split_adv_ex(prop)
        else:
            bnb_analyzer = bnb.BnB(self.net, self.transformer, prop, self.args, self.template_store)
            if self.args.parallel:
                bnb_analyzer.run_parallel()
            else:
                bnb_analyzer.run()

            status = bnb_analyzer.global_status
            tree_size = bnb_analyzer.tree_size
        return status, tree_size

    def update_transformer(self, prop):
        if self.transformer is not None and 'update_input' in dir(self.transformer) \
                and prop.out_constr.constr_type == OutSpecType.LOCAL_ROBUST:
            self.transformer.update_input(prop)
        else:
            self.transformer = get_domain_transformer(self.args, self.net, prop, complete=False)

    def analyze_no_split_adv_ex(self, prop):
        # TODO: handle feasibility
        lb, _, adv_ex = self.transformer.compute_lb()
        status = Status.UNKNOWN
        if torch.all(lb >= 0):
            status = Status.VERIFIED
        elif adv_ex is not None:
            status = Status.ADV_EXAMPLE
        return status

    def analyze_no_split(self):
        lb, ub = self.transformer.bound()
        # ub = self.transformer.compute_ub()
        status = Status.UNKNOWN
        if torch.all(lb >= 0):
            status = Status.VERIFIED
        print('LB: ', lb)
        precision = get_precision(lb)
        print('Precision: ', precision)
        print('UB: ', 0)
        return status

    def run_analyzer(self):
        """
        Prints the output of verification - count of verified, unverified and the cases for which the adversarial example
            was found
        """
        # Changes for GPU
        if self.args.repeat is not None:
            return self.run_harness()
        print('Running on the network: ', self.args.net)
        print('Number of verification instances: ', self.args.count)
        print('Running in Batch: ', self.args.run_batch)
        print('Timeout of verification: ', self.args.timeout)
        print('Using %s abstract domain' % self.args.domain)

        props, inputs, net = specs.get_specs(self.args.dataset, spec_type=self.args.spec_type, count=self.args.count, run_batch=self.args.run_batch, 
                                        eps=self.args.eps, net=self.net)
        
        self.net = net
        # Changes for GPU
        if str(self.device).startswith('cuda'):
            # Untimed pass so CUDA context / cuBLAS / cuDNN init is not counted in the timed run
            warmup_start = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                get_domain_transformer(self.args, self.net, props[0].get_input_clause(0), complete=False).bound()
            torch.cuda.synchronize()
            print('Warmup time: ', time.time() - warmup_start)
        start_time = time.time()
        print('Start time: ', start_time)
        results = self.analyze_domain(props)
        print('Total time: ', time.time())
        results.compute_stats()
        print('Results: ', results.output_count)
        print('Average time:', results.avg_time)
        print('Total time:!!!!!!!!!!!!!!', time.time() - start_time)
        return results

    # Changes for GPU
    def run_harness(self):
        """constraintflow-gpu's measurement protocol (its cli.py `run`), for experiments/compare_ivan.py.

        `warmup` untimed runs on the next batches of test images, then `repeat` timed runs on the first
        `count` images. Like cli.py's run(), a run parses the network and builds the normalized input
        boxes and the output spec (y_true - y_j, j != y_true) before running the domain and concretizing.
        Peak memory is torch.cuda.max_memory_allocated() after reset_peak_memory_stats(), as in cli.py.
        Prints the harness's `Warmup run` / `Run` / `Lower bounds` / `Upper bounds` lines.
        """
        args = self.args
        bs, warmup, repeat = args.count, args.warmup, args.repeat
        # Each run loads its own network; the copy __init__ loaded would sit in every timed run's peak.
        self.net = None
        is_cuda = str(self.device).startswith('cuda')
        if is_cuda:
            # Same cap as constraintflow-gpu's cli.py (_configure_cuda_cpu_threads): with all cores,
            # each small CPU op waits at an OpenMP barrier whenever another process holds a core.
            torch.set_num_threads(min(torch.get_num_threads(), int(os.environ.get('CF_CPU_THREADS', '8'))))
        disable_tf32 = os.environ.get('CF_DISABLE_TF32', '0') not in ('', '0')
        if disable_tf32:
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        print(f'Network: {args.net}')
        print(f'Domain: {args.domain}, device: {self.device}, normalization: {args.normalization}')
        print(f'TF32: {"disabled (CF_DISABLE_TF32)" if disable_tf32 else "PyTorch default"}')
        #FEAT: --dtype float64 (precision oracle): domains build their tensors in the default dtype.
        if config.DTYPE == 'float64':
            torch.set_default_dtype(torch.float64)
        print(f'dtype: {torch.get_default_dtype()}')

        num_images = getattr(args, 'num_images', 0)
        inputs, labels = next(iter(util.prepare_data(
            args.dataset, batch_size=num_images + warmup * bs if num_images else bs * (warmup + 1))))
        #FEAT: ToTensor() yields fp32 images; follow the run's dtype so the input box is built in it.
        inputs = inputs.to(torch.get_default_dtype())

        def run_once(offset):
            net = util.get_net(args.net, args.dataset)
            props, net = specs.get_linf_spec(inputs[offset:offset + bs], labels[offset:offset + bs], args.eps,
                                             args.dataset, True, net, normalization=args.normalization)
            transformer = get_domain_transformer(args, net, props[0].get_input_clause(0), complete=False)
            return transformer.bound()

        def sync():
            if is_cuda:
                torch.cuda.synchronize()

        if num_images:
            return self._run_image_batches(run_once, sync, bs, num_images, warmup, is_cuda)

        for i in range(warmup):
            start = time.perf_counter()
            run_once((i + 1) * bs)
            sync()
            print(f'Warmup run {i + 1}/{warmup}: {time.perf_counter() - start:.6f} s', flush=True)
            gc.collect()
            # Not needed for the per-run peak (max_memory_allocated counts live tensors only),
            # as in constraintflow-gpu's cli.py
            # if is_cuda:
            #     torch.cuda.empty_cache()

        # Freeze the startup heap so the between-run gc.collect() only scans per-run garbage.
        gc.collect()
        gc.freeze()

        mem_label = 'Peak GPU memory' if is_cuda else 'Peak CPU memory'
        times, peaks = [], []
        for i in range(repeat):
            sync()
            resident = 0
            if is_cuda:
                torch.cuda.reset_peak_memory_stats()
                resident = torch.cuda.memory_allocated()
            start = time.perf_counter()
            lb, ub = run_once(0)
            sync()
            run_time = time.perf_counter() - start
            if is_cuda:
                peak = torch.cuda.max_memory_allocated()
            else:
                import resource
                peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            times.append(run_time)
            peaks.append(peak)
            resident_note = f', Resident GPU memory: {resident} bytes' if is_cuda else ''
            print(f'Run {i + 1}/{repeat}: {run_time:.6f} s, {mem_label}: {peak} bytes{resident_note}', flush=True)
            lb, ub = lb.detach().cpu(), ub.detach().cpu()
            gc.collect()
            # if is_cuda:
            #     torch.cuda.empty_cache()

        # Full float32 precision and no '...' elision, so the harness can parse the exact bounds.
        torch.set_printoptions(precision=8, sci_mode=True, threshold=sys.maxsize)
        print(f'Lower bounds: {lb}')
        print(f'Upper bounds: {ub}')
        torch.set_printoptions(profile='default')
        print(f'Total time: {sum(times):.6f} seconds')
        print(f'{mem_label}: {max(peaks)} bytes')

    # Changes for GPU
    def _run_image_batches(self, run_once, sync, bs, num_images, warmup, is_cuda):
        """run_harness with num_images: warmups on the images after the evaluated ones, then one
        timed run per batch, each writing one JSON line after its timer stops. No error handling:
        a failing batch kills the process, and the batches already written stay in the file."""
        import json
        #FEAT: --profile_comm profiles each timed region.
        from nnverify.comm_profile import CommProfile

        def write(f, record):
            f.write(json.dumps(record) + '\n')
            f.flush()

        with open(self.args.results_file, 'w') as out:
            for j in range(warmup):
                lo = num_images + j * bs
                start = time.perf_counter()
                run_once(lo)
                sync()
                warmup_time = time.perf_counter() - start
                print(f'Warmup run {j + 1}/{warmup}: {warmup_time:.6f} s', flush=True)
                write(out, {'kind': 'warmup', 'index': j, 'images': [lo, lo + bs], 'time_s': warmup_time})
                gc.collect()

            # Freeze the startup heap so the between-run gc.collect() only scans per-run garbage.
            gc.collect()
            gc.freeze()

            n_batches = num_images // bs
            for i in range(n_batches):
                lo = i * bs
                print(f'=== batch {i} start ===', flush=True)
                #FEAT: profiler on around the timer; comm.region() is exactly the timed region.
                comm = CommProfile(config.PROFILE_COMM)
                with comm:
                    sync()
                    resident = None
                    if is_cuda:
                        torch.cuda.reset_peak_memory_stats()
                        resident = torch.cuda.memory_allocated()
                    start = time.perf_counter()
                    with comm.region():
                        lb, ub = run_once(lo)
                        sync()
                    run_time = time.perf_counter() - start
                # Per-run peaks exist only on the GPU; a CPU process's peak RSS is lifetime-wide.
                peak = torch.cuda.max_memory_allocated() if is_cuda else None
                write(out, {'kind': 'batch', 'batch': i, 'images': [lo, lo + bs], comm.time_key: run_time,
                            'peak_bytes': peak, 'resident_bytes': resident,
                            'lower': lb.detach().cpu().tolist(), 'upper': ub.detach().cpu().tolist(),
                            **comm.metrics()})
                comm.save_trace(self.args.results_file, i)
                print(f'Run {i + 1}/{n_batches}: {run_time:.6f} s, Peak memory: {peak} bytes, '
                      f'Resident memory: {resident} bytes', flush=True)
                print(f'=== batch {i} end ===', flush=True)
                del lb, ub
                gc.collect()

    # There are multiple clauses in the inout specification
    # Property should hold on all the input clauses
    @staticmethod
    def extract_status(cl_status):
        for status in cl_status:
            if status != Status.VERIFIED:
                return status
        return Status.VERIFIED

    def analyze_domain(self, props):
        results = Results(self.args)
        t1 = time.time()
        print('Time: ', t1)
        for i in range(len(props)):
            print("************************** Proof %d *****************************" % (i+1))
            num_clauses = props[i].get_input_clause_count()
            clause_ver_status = []
            ver_start_time = time.time()

            for j in range(num_clauses):
                cl_status, tree_size = self.analyze(props[i].get_input_clause(j))
                clause_ver_status.append(cl_status)

            status = self.extract_status(clause_ver_status)
            print(status)
            ver_time = time.time() - ver_start_time
            results.add_result(Result(ver_time, status, tree_size=tree_size))
            t2 = time.time()
            print('Time: ', t2)


        return results

