from unittest import TestCase
from nnverify.common.dataset import Dataset
from nnverify.common import Domain
# from nnverify.bnb import Split
from nnverify.analyzer import Analyzer
import nnverify.config as config
import ssl

# This is a hack to avoid SSL related errors while getting CIFAR10 data
from nnverify.specs.input_spec import InputSpecType

ssl._create_default_https_context = ssl._create_unverified_context

def test_aval():
    args = config.Args(net='nnverify/nets/mnist_relu_3_50.onnx', domain=Domain.DEEPZ, dataset=Dataset.MNIST, eps=10.1)
    Analyzer(args).run_analyzer()

test_aval()