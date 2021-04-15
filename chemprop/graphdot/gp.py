#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
from typing import Dict, List, Literal
import copy
import math
import numpy as np
from graphdot.model.gaussian_process.gpr import GaussianProcessRegressor
from graphdot.model.gaussian_process.nystrom import *
from sklearn.gaussian_process._gpc import GaussianProcessClassifier
from sklearn.svm import SVC
from chemprop.data import get_class_sizes, get_data, MoleculeDataLoader, MoleculeDataset, set_cache_graph, split_data
from chemprop.graphdot.graph.graph import Graph
from chemprop.graphdot.kernels import PreCalcKernel


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


class SVMClassifier(SVC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._SVC = SVC(*args, **kwargs)

    @property
    def kernel_(self):
        return self._SVC.kernel

    @staticmethod
    def _remove_nan_X_y(X, y):
        if None in y:
            idx = np.where(y!=None)[0]
        else:
            idx = ~np.isnan(y)
        return np.asarray(X)[idx], y[idx].astype(int)

    def fit(self, X, y, sample_weight=None):
        self.SVCs = []
        if y.ndim == 1:
            X_, y_ = self._remove_nan_X_y(X, y)
            super().fit(X_, y_, sample_weight)
        else:
            for i in range(y.shape[1]):
                SVC = copy.deepcopy(self._SVC)
                X_, y_ = self._remove_nan_X_y(X, y[:, i])
                SVC.fit(X_, y_, sample_weight)
                self.SVCs.append(SVC)

    def predict(self, X):
        if self.SVCs:
            y_mean = []
            for SVC in self.SVCs:
                y_mean.append(SVC.predict(X))
            return np.concatenate(y_mean).reshape(len(y_mean), len(X)).T
        else:
            return super().predict(X)

    def predict_proba(self, X):
        if self.SVCs:
            y_mean = []
            for SVC in self.SVCs:
                y_mean.append(SVC.predict_proba(X)[:, 1])
            return np.concatenate(y_mean).reshape(len(y_mean), len(X)).T
        else:
            return super().predict_proba(X)[:, 1]


class GPC(GaussianProcessClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._GPC = GaussianProcessClassifier(*args, **kwargs)

    @property
    def kernel_(self):
        return self._GPC.kernel

    @staticmethod
    def _remove_nan_X_y(X, y):
        if None in y:
            idx = np.where(y!=None)[0]
        else:
            idx = ~np.isnan(y)
        return np.asarray(X)[idx].reshape(-1, 1), y[idx].astype(int)

    def fit(self, X, y):
        self.GPCs = []
        if y.ndim == 1:
            X_, y_ = self._remove_nan_X_y(X, y)
            super().fit(X_, y_)
        else:
            for i in range(y.shape[1]):
                GPC = copy.deepcopy(self._GPC)
                X_, y_ = self._remove_nan_X_y(X, y[:, i])
                GPC.fit(X_, y_)
                self.GPCs.append(GPC)

    def predict(self, X):
        if self.GPCs:
            y_mean = []
            for GPC in self.GPCs:
                y_mean.append(GPC.predict(X))
            return np.concatenate(y_mean).reshape(len(y_mean), len(X)).T
        else:
            return super().predict(X)

    def predict_proba(self, X):
        if self.GPCs:
            y_mean = []
            for GPC in self.GPCs:
                print(GPC.predict_proba(X))
                y_mean.append(GPC.predict_proba(X)[:, 1])
            # print(y_mean)
            return np.concatenate(y_mean).reshape(len(y_mean), len(X)).T
        else:
            return super().predict_proba(X)[:, 1]


def add_gp_results(args,
                   train_data: MoleculeDataset,
                   val_data: MoleculeDataset,
                   test_data: MoleculeDataset,
                   kernel: PreCalcKernel,
                   gp_type: List[Literal['predict', 'predict_u', 'kernel']]
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

    if 'kernel' in gp_type:
        train_data.set_K(kernel(X_train))
        val_data.set_K(kernel(X_val, X_train))
        test_data.set_K(kernel(X_test, X_train))
    elif args.dataset_type == 'classification':
        n = 1
        # svc = SVMClassifier(kernel=kernel, C=args.C_, probability=True)
        model = GPC(kernel=kernel, optimizer=None, n_jobs=args.num_workers)
        model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_train)
        if y_pred.ndim == 1:
            y_pred = np.concatenate([y_pred.reshape(len(y_pred), 1)]*n, axis=1)
        train_data.set_gp_predict(y_pred)

        y_pred = model.predict_proba(X_val)
        if y_pred.ndim == 1:
            y_pred = np.concatenate([y_pred.reshape(len(y_pred), 1)]*n, axis=1)
        val_data.set_gp_predict(y_pred)

        y_pred = model.predict_proba(X_test)
        if y_pred.ndim == 1:
            y_pred = np.concatenate([y_pred.reshape(len(y_pred), 1)]*n, axis=1)
        test_data.set_gp_predict(y_pred)
    else:
        n = 1
        gpr = GPR(kernel=kernel, optimizer=None, alpha=args.alpha_, normalize_y=True)
        gpr.fit(X_train, y_train)
        # y_pred, y_std = gpr.predict(X_train, return_std=True)
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
