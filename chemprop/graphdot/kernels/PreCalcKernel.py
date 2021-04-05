#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import copy
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
from chemprop.graphdot.kernels.BaseKernelConfig import BaseKernelConfig


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
