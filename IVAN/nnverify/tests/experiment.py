
from unittest import TestCase
from nnverify.common.dataset import Dataset
from nnverify.common import Domain
from nnverify.analyzer import Analyzer
import nnverify.config as config

class NewTest(TestCase):
    def new_test(self):
        # Changes for GPU
        args = config.Args(net="mnist_relu_3_50.onnx", domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.004, count=100, run_batch=True,
                           warmup=0, repeat=1, normalization="cf",
                           num_images=200, results_file='/tmp/claude-1307560/-home-db50-cf-harness/3d66755e-420b-4c1b-8788-d2c8a9cceb4b/scratchpad/ivan_smoke.jsonl')
        Analyzer(args).run_analyzer()
