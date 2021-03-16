#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import copy
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.kernel.fix import Normalization
from typing import Tuple, List
from graphdot.microkernel import (
    TensorProduct,
    SquareExponential as sExp,
    KroneckerDelta as kDelta,
    Convolution as kConv,
)
from graphdot.microprobability import (
    Additive as Additive_p,
    Constant,
    UniformProbability,
    AssignProbability
)
from .graph.hashgraph import Graph


class PreCalcKernel:
    def __init__(self, X, K, theta):
        self.X = X
        self.K = K
        self.theta_ = theta
        self.exptheta = np.exp(self.theta_)

    def __call__(self, X, Y=None, eval_gradient=False, *args, **kwargs):
        X_idx = np.searchsorted(self.X, X).ravel()
        Y_idx = np.searchsorted(self.X, Y).ravel() if Y is not None else X_idx
        if eval_gradient:
            return self.K[X_idx][:, Y_idx], \
                   np.zeros((len(X_idx), len(Y_idx), 1))
        else:
            return self.K[X_idx][:, Y_idx]

    def diag(self, X, eval_gradient=False):
        X_idx = np.searchsorted(self.X, X).ravel()
        if eval_gradient:
            return np.diag(self.K)[X_idx], np.zeros((len(X_idx), 1))
        else:
            return np.diag(self.K)[X_idx]

    @property
    def hyperparameters(self):
        return ()

    @property
    def theta(self):
        return np.log(self.exptheta)

    @theta.setter
    def theta(self, value):
        self.exptheta = np.exp(value)

    @property
    def n_dims(self):
        return len(self.theta)

    @property
    def bounds(self):
        theta = self.theta.reshape(-1, 1)
        return np.c_[theta, theta]

    @property
    def requires_vector_input(self):
        return False

    def clone_with_theta(self, theta):
        clone = copy.deepcopy(self)
        clone.theta = theta
        return clone

    def get_params(self, deep=False):
        return dict(
            X=self.X,
            K=self.K,
            theta=self.theta_
        )


def get_kernels() -> Tuple[MarginalizedGraphKernel, Normalization]:
    k = 0.90
    k_bounds = (0.1, 1.0)
    # s_bounds = (0.1, 10.0)
    knode = TensorProduct(
        AtomicNumber=kDelta(0.75, k_bounds),
        AtomicNumber_list_1=kConv(kDelta(k, k_bounds)),
        AtomicNumber_list_2=kConv(kDelta(k, k_bounds)),
        AtomicNumber_list_3=kConv(kDelta(k, k_bounds)),
        AtomicNumber_list_4=kConv(kDelta(k, k_bounds)),
        AtomicNumber_count_1=kDelta(k, k_bounds),
        AtomicNumber_count_2=kDelta(k, k_bounds),
        MorganHash=kDelta(k, k_bounds),
        Ring_count=kDelta(k, k_bounds),
        RingSize_list=kConv(kDelta(k, k_bounds)),
        Hcount=kDelta(k, k_bounds),
        Chiral=kDelta(k, k_bounds),
    )
    kedge = TensorProduct(
        Order=kDelta(k, k_bounds),
        Stereo=kDelta(k, k_bounds),
        RingStereo=kDelta(k, k_bounds),
        Conjugated=kDelta(k, k_bounds)
    )
    start_probability=Additive_p(
        Concentration=Constant(1.0)
        # Concentration=AssignProbability(1.0)
    )
    kernel = MarginalizedGraphKernel(
        node_kernel=knode,
        edge_kernel=kedge,
        q=0.01,
        q_bounds=(1e-4, 1.0),
        p=start_probability
    )
    return kernel, Normalization(kernel)


def get_preCakc_kernel(graphs: List[Graph], smiles: List[str]):
    _, kernel = get_kernels()
    idx = np.argsort(smiles)
    smiles = np.asarray(smiles)[idx]
    graphs = np.asarray(graphs)[idx]
    K = kernel(graphs)
    return PreCalcKernel(smiles, K, kernel.theta)
