from enum import Enum


class Dataset(Enum):
    CIFAR10 = 1
    MNIST = 2
    ACAS = 3
    OVAL_CIFAR = 4
    #FEAT: inputs of the cf-harness configs IVAN had no loader for (CIFAR-100 ResNets, VNN-COMP boxes).
    CIFAR100 = 5
    VNNLIB = 6