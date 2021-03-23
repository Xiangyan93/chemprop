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
from .graph.graph import Graph
from graphdot.util.pretty_tuple import pretty_tuple


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


class Norm(Normalization):
    @property
    def requires_vector_input(self):
        return False

    def get_params(self, deep=False):
        return dict(
            kernel=self.kernel,
        )

    @property
    def n_dims(self):
        """Returns the number of non-fixed hyperparameters of the kernel."""
        return self.theta.shape[0]


class NormalizationMolSize(Norm):
    def __init__(self, kernel, s=100.0, s_bounds=(1e2, 1e3)):
        super().__init__(kernel)
        self.s = s
        self.s_bounds = s_bounds

    def __diag(self, K, l, r, K_gradient=None):
        l_lr = np.repeat(l, len(r)).reshape(len(l), len(r))
        r_rl = np.repeat(r, len(l)).reshape(len(r), len(l))
        se = np.exp(-((1 / l_lr ** 2 - 1 / r_rl.T ** 2) / self.s) ** 2)
        K = np.einsum("i,ij,j,ij->ij", l, K, r, se)
        if K_gradient is None:
            return K
        else:
            K_gradient = np.einsum("ijk,i,j,ij->ijk", K_gradient, l, r, se)
            if self.s_bounds == "fixed":
                return K, K_gradient
            else:
                dK_s = 2 * (l_lr - r_rl.T) ** 2 / self.s ** 3 * K
                dK_s = dK_s.reshape(len(l), len(r), 1)
                print(K.max(), K.min())
                print(dK_s.max(), dK_s.min())
                print(self.theta)
                return K, np.concatenate([K_gradient, dK_s], axis=2)
            # return K, np.einsum("ijk,i,j,ij->ijk", K_gradient, l, r, se)

    def __call__(self, X, Y=None, eval_gradient=False, **options):
        """Normalized outcome of
        :py:`self.kernel(X, Y, eval_gradient, **options)`.

        Parameters
        ----------
        Inherits that of the graph kernel object.

        Returns
        -------
        Inherits that of the graph kernel object.
        """
        if eval_gradient is True:
            R, dR = self.kernel(X, Y, eval_gradient=True, **options)
            if Y is None:
                ldiag = rdiag = R.diagonal()
            else:
                ldiag, ldDiag = self.kernel.diag(X, True, **options)
                rdiag, rdDiag = self.kernel.diag(Y, True, **options)
            ldiag_inv = 1 / ldiag
            rdiag_inv = 1 / rdiag
            ldiag_rsqrt = np.sqrt(ldiag_inv)
            rdiag_rsqrt = np.sqrt(rdiag_inv)
            return self.__diag(R, ldiag_rsqrt, rdiag_rsqrt, dR)
        else:
            R = self.kernel(X, Y, **options)
            if Y is None:
                ldiag = rdiag = R.diagonal()
            else:
                ldiag = self.kernel.diag(X, **options)
                rdiag = self.kernel.diag(Y, **options)
            ldiag_inv = 1 / ldiag
            rdiag_inv = 1 / rdiag
            ldiag_rsqrt = np.sqrt(ldiag_inv)
            rdiag_rsqrt = np.sqrt(rdiag_inv)
            # K = ldiag_rsqrt[:, None] * R * rdiag_rsqrt[None, :]
            return self.__diag(R, ldiag_rsqrt, rdiag_rsqrt)

    @property
    def n_dims(self):
        if self.s_bounds == "fixed":
            return len(self.kernel.theta)
        else:
            return len(self.kernel.theta) + 1

    @property
    def hyperparameters(self):
        if self.s_bounds == "fixed":
            return self.kernel.hyperparameters
        else:
            return pretty_tuple(
                'MarginalizedGraphKernel',
                ['starting_probability', 'stopping_probability', 'node_kernel',
                 'edge_kernel', 'normalize_size']
            )(self.kernel.p.theta,
              self.kernel.q,
              self.kernel.node_kernel.theta,
              self.kernel.edge_kernel.theta,
              self.s)

    @property
    def hyperparameter_bounds(self):
        if self.s_bounds == "fixed":
            return self.kernel.hyperparameter_bounds
        else:
            return pretty_tuple(
                'GraphKernelHyperparameterBounds',
                ['starting_probability', 'stopping_probability', 'node_kernel',
                 'edge_kernel', 'normalize_size']
            )(self.kernel.p.bounds,
              self.kernel.q_bounds,
              self.kernel.node_kernel.bounds,
              self.kernel.edge_kernel.bounds,
              self.s_bounds)

    @property
    def theta(self):
        if self.s_bounds == "fixed":
            return self.kernel.theta
        else:
            return np.r_[self.kernel.theta, self.s]

    @theta.setter
    def theta(self, value):
        if self.s_bounds == "fixed":
            self.kernel.theta = value
        else:
            self.kernel.theta = value[:-1]
            self.s = value[-1]

    @property
    def bounds(self):
        if self.s_bounds == "fixed":
            return self.kernel.bounds
        else:
            return np.r_[self.kernel.bounds, np.reshape(self.s_bounds, (1, 2))]

    def clone_with_theta(self, theta):
        clone = copy.deepcopy(self)
        clone.theta = theta
        return clone

    def get_params(self, deep=False):
        return dict(
            kernel=self.kernel,
            s=self.s,
            s_bounds=self.s_bounds,
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
    return kernel, NormalizationMolSize(kernel, 10000)


def get_preCakc_kernel(graphs: List[Graph], smiles: List[str]):
    _, kernel = get_kernels()
    idx = np.argsort(smiles)
    smiles = np.asarray(smiles)[idx]
    graphs = np.asarray(graphs)[idx]
    K = kernel(graphs)
    return PreCalcKernel(smiles, K, kernel.theta)
