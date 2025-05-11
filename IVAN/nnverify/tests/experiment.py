
from unittest import TestCase
from nnverify.common.dataset import Dataset
from nnverify.common import Domain
from nnverify.analyzer import Analyzer
import nnverify.config as config

class NewTest(TestCase):
    def new_test(self):
        args = config.Args(net="mnist_relu_3_50.onnx", domain=Domain.DEEPPOLY, dataset=Dataset.MNIST, eps=0.01, count=1, run_batch=True)
        Analyzer(args).run_analyzer()
