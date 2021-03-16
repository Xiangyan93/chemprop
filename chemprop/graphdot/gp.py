#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
from typing import Dict, List, Literal
import math
import numpy as np
from graphdot.model.gaussian_process.gpr import GaussianProcessRegressor
from graphdot.model.gaussian_process.nystrom import *
from sklearn.gaussian_process._gpc import GaussianProcessClassifier as GPC
from chemprop.data import get_class_sizes, get_data, MoleculeDataLoader, MoleculeDataset, set_cache_graph, split_data
from chemprop.graphdot.graph.hashgraph import Graph
from chemprop.graphdot.kernel import get_kernels, PreCalcKernel


def _predict(predict, X, return_std=False, return_cov=False, memory_save=True,
            n_memory_save=10000):
    if return_cov or not memory_save:
        return predict(X, return_std=return_std, return_cov=return_cov)
    else:
        N = len(X)
        y_mean = np.array([])
        y_std = np.array([])
        for i in range(math.ceil(N / n_memory_save)):
            X_ = X[i * n_memory_save:(i + 1) * n_memory_save]
            if return_std:
                [y_mean_, y_std_] = predict(
                    X_, return_std=return_std, return_cov=return_cov)
                y_std = np.r_[y_std, y_std_]
            else:
                y_mean_ = predict(
                    X_, return_std=return_std, return_cov=return_cov)
            y_mean = np.r_[y_mean, y_mean_]
        if return_std:
            return y_mean, y_std
        else:
            return y_mean


class GPR(GaussianProcessRegressor):
    def predict_(self, Z, return_std=False, return_cov=False):
        if not hasattr(self, 'Kinv'):
            raise RuntimeError('Model not trained.')
        Ks = self._gramian(Z, self.X)
        ymean = (Ks @ self.Ky) * self.y_std + self.y_mean
        if return_std is True:
            Kss = self._gramian(Z, diag=True)
            Kss.flat[::len(Kss) + 1] -= self.alpha
            std = np.sqrt(
                np.maximum(0, Kss - (Ks @ (self.Kinv @ Ks.T)).diagonal())
            )
            return (ymean, std)
        elif return_cov is True:
            Kss = self._gramian(Z)
            Kss.flat[::len(Kss) + 1] -= self.alpha
            cov = np.maximum(0, Kss - Ks @ (self.Kinv @ Ks.T))
            return (ymean, cov)
        else:
            return ymean

    def predict(self, X, return_std=False, return_cov=False):
        return _predict(self.predict_, X, return_std=return_std,
                        return_cov=return_cov)

    @classmethod
    def load_cls(cls, f_model, kernel):
        store_dict = pickle.load(open(f_model, 'rb'))
        kernel = kernel.clone_with_theta(store_dict.pop('theta'))
        model = cls(kernel)
        model.__dict__.update(**store_dict)
        return model

    """sklearn GPR parameters"""
    @property
    def kernel_(self):
        return self.kernel

    @property
    def X_train_(self):
        return self._X

    @X_train_.setter
    def X_train_(self, value):
        self._X = value


def add_gp_results(train_data: MoleculeDataset,
                   val_data: MoleculeDataset,
                   test_data: MoleculeDataset,
                   dataset_type: Literal['classification', 'regression'],
                   kernel: PreCalcKernel
                   ):
    """
    X_train = list(map(Graph.from_rdkit, train_data.mols(flatten=True)))
    X_val = list(map(Graph.from_rdkit, val_data.mols(flatten=True)))
    X_test = list(map(Graph.from_rdkit, test_data.mols(flatten=True)))
    """
    X_train = train_data.smiles(flatten=True)
    X_val = val_data.smiles(flatten=True)
    X_test = test_data.smiles(flatten=True)
    y_train = np.asarray(train_data.targets())
    if y_train.shape[1] == 1:
        y_train = y_train.ravel()
    if dataset_type == 'classification':
        pass
    else:
        train_data.set_gp_predict(kernel(X_train))
        val_data.set_gp_predict(kernel(X_val, X_train))
        test_data.set_gp_predict(kernel(X_test, X_train))
        """
        from sklearn.metrics import mean_absolute_error
        gpr = GPR(kernel=kernel, optimizer=None, alpha=0.01, normalize_y=True)
        gpr.fit(X_train, y_train)
        # y_pred, y_std = gpr.predict(X_train, return_std=True)
        n = 50
        y_pred, y_std = gpr.predict_loocv(X_train, y_train, return_std=True)
        if y_pred.ndim == 1:
            y_pred = np.concatenate([y_pred.reshape(len(y_pred), 1)]*n, axis=1)
            y_std = np.concatenate([y_std.reshape(len(y_std), 1)]*n, axis=1)
        train_data.set_gp_predict(y_pred)
        train_data.set_gp_uncertainty(y_std)
        y_pred, y_std = gpr.predict(X_val, return_std=True)
        if y_pred.ndim == 1:
            y_pred = np.concatenate([y_pred.reshape(len(y_pred), 1)]*n, axis=1)
            y_std = np.concatenate([y_std.reshape(len(y_std), 1)]*n, axis=1)
        val_data.set_gp_predict(y_pred)
        val_data.set_gp_uncertainty(y_std)
        y_pred, y_std = gpr.predict(X_test, return_std=True)
        if y_pred.ndim == 1:
            y_pred = np.concatenate([y_pred.reshape(len(y_pred), 1)]*n, axis=1)
            y_std = np.concatenate([y_std.reshape(len(y_std), 1)]*n, axis=1)
        test_data.set_gp_predict(y_pred)
        test_data.set_gp_uncertainty(y_std)
        """
