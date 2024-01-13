#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
from tqdm import trange
from logging import Logger
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from chemprop.data import get_class_sizes, get_data, MoleculeDataLoader, MoleculeDataset, set_cache_graph, split_data, \
    get_task_names
from chemprop.utils import build_optimizer, build_lr_scheduler, load_checkpoint, makedirs, \
    save_checkpoint, save_smiles_splits, load_frzn_model, multitask_mean
from chemprop.nn_utils import param_count, param_count_all
from chemprop.models import MoleculeModel
from chemprop.constants import MODEL_FILE_NAME
from chemprop.train.loss_functions import get_loss_func
from chemprop.train import train
from chemprop.train.predict import predict
from chemprop.args import TrainArgs


class MPNN:
    def __init__(self,
                 save_dir: str, data_path: str,
                 dataset_type: Literal['regression', 'classification', 'multiclass', 'spectra'],
                 loss_function: Literal['mse', 'bounded_mse', 'binary_cross_entropy', 'cross_entropy', 'mcc', 'sid',
                                        'wasserstein', 'mve', 'evidential', 'dirichlet'],
                 smiles_columns: List[str] = None, target_columns: List[str] = None,
                 multiclass_num_classes: int = 3,
                 features_generator=None,
                 no_features_scaling: bool = False,
                 features_only: bool = False,
                 features_size: int = 0,
                 epochs: int = 30,
                 depth: int = 3,
                 hidden_size: int = 300,
                 ffn_num_layers: int = 2,
                 ffn_hidden_size: int = None,
                 dropout: float = 0.0,
                 batch_size: int = 50,
                 ensemble_size: int = 1,
                 number_of_molecules: int = 1,
                 mpn_shared: bool = False,
                 atom_messages: bool = False,
                 undirected: bool = False,
                 n_jobs: int = 8,
                 class_balance: bool = False,
                 checkpoint_dir: str = None,
                 checkpoint_frzn: str = None,
                 frzn_ffn_layers: int = 0,
                 freeze_first_only: bool = False,
                 seed: int = 0,
                 logger: Logger = None,
                 ):
        args = TrainArgs()
        args.save_dir = save_dir
        args.data_path = data_path
        args.dataset_type = dataset_type
        args.loss_function = loss_function
        args.smiles_columns = smiles_columns
        args.target_columns = target_columns
        args.multiclass_num_classes = multiclass_num_classes
        args.features_generator = features_generator
        args.no_features_scaling = no_features_scaling
        args.features_only = features_only
        args.features_size = features_size
        args.epochs = epochs
        args.depth = depth
        args.hidden_size = hidden_size
        args.ffn_num_layers = ffn_num_layers
        args.ffn_hidden_size = ffn_hidden_size
        args.dropout = dropout
        args.batch_size = batch_size
        args.ensemble_size = ensemble_size
        args.number_of_molecules = number_of_molecules
        args.mpn_shared = mpn_shared
        args.atom_messages = atom_messages
        args.undirected = undirected
        args.num_workers = n_jobs
        args.class_balance = class_balance
        args.checkpoint_dir = checkpoint_dir
        args.checkpoint_frzn = checkpoint_frzn
        args.frzn_ffn_layers = frzn_ffn_layers
        args.freeze_first_only = freeze_first_only
        args.seed = seed
        args.process_args()
        args.task_names = get_task_names(path=args.data_path, smiles_columns=args.smiles_columns,
                                         target_columns=args.target_columns, ignore_columns=args.ignore_columns)
        args._parsed = True
        self.args = args
        self.features_scaler = None
        self.logger = logger

    def fit(self, train_data):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        args = self.args
        args.train_data_size = len(train_data)
        logger = self.logger
        if logger is not None:
            debug, info = logger.debug, logger.info
        else:
            debug = info = str

        # Set pytorch seed for random initial weights
        torch.manual_seed(args.pytorch_seed)

        if args.dataset_type == 'classification':
            train_class_sizes = get_class_sizes(train_data, proportion=False)
            args.train_class_sizes = train_class_sizes

        if args.features_scaling:
            self.features_scaler = train_data.normalize_features(
                replace_nan_token=0)

        atom_descriptor_scaler = None
        bond_feature_scaler = None

        args.train_data_size = len(train_data)

        # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (
        # regression only)
        if args.dataset_type == 'regression':
            debug('Fitting scaler')
            scaler = train_data.normalize_targets()
            args.spectra_phase_mask = None
        else:
            args.spectra_phase_mask = None
            scaler = None

        # Get loss function
        loss_func = get_loss_func(args)
        """
        # Automatically determine whether to cache
        if len(train_data) <= args.cache_cutoff:
            set_cache_graph(True)
            num_workers = 0
        else:
            set_cache_graph(False)
            num_workers = args.num_workers
        """
        train_data_loader = MoleculeDataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            class_balance=args.class_balance,
            shuffle=True,
            seed=args.seed
        )

        if args.class_balance:
            debug(
                f'With class_balance, effective train size = {train_data_loader.iter_size:,}')

        self.models = []
        for model_idx in range(args.ensemble_size):
            # Tensorboard writer
            save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
            makedirs(save_dir)
            # try:
            #    writer = SummaryWriter(log_dir=save_dir)
            # except:
            #    writer = SummaryWriter(logdir=save_dir)
            writer = None
            # Load/build model
            if args.checkpoint_paths is not None:
                debug(
                    f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
                model = load_checkpoint(
                    args.checkpoint_paths[model_idx], logger=logger)
            else:
                debug(f'Building model {model_idx}')
                model = MoleculeModel(args)

            # Optionally, overwrite weights:
            if args.checkpoint_frzn is not None:
                debug(
                    f'Loading and freezing parameters from {args.checkpoint_frzn}.')
                model = load_frzn_model(
                    model=model, path=args.checkpoint_frzn, current_args=args, logger=logger)

            debug(model)

            if args.checkpoint_frzn is not None:
                debug(
                    f'Number of unfrozen parameters = {param_count(model):,}')
                debug(
                    f'Total number of parameters = {param_count_all(model):,}')
            else:
                debug(f'Number of parameters = {param_count_all(model):,}')

            if args.cuda:
                print('Moving model to cuda')
            model = model.to(args.device)

            # Ensure that model is saved in correct location for evaluation if 0 epochs
            # save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, scaler,
            #                 features_scaler, atom_descriptor_scaler, bond_feature_scaler, None)

            # Optimizers
            optimizer = build_optimizer(model, args)

            # Learning rate schedulers
            scheduler = build_lr_scheduler(optimizer, args)

            n_iter = 0
            for epoch in trange(args.epochs):
                debug(f'Epoch {epoch}')
                n_iter = train(
                    model=model,
                    data_loader=train_data_loader,
                    loss_func=loss_func,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    n_iter=n_iter,
                    logger=logger,
                    writer=writer
                )
                if isinstance(scheduler, ExponentialLR):
                    scheduler.step()
            self.models.append(model)
            self.scaler = scaler
            # save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, scaler, features_scaler,
            #                 atom_descriptor_scaler, bond_feature_scaler, None)

    def predict_uncertainty(self, pred_data):
        args = self.args
        if args.features_scaling:
            pred_data.normalize_features(self.features_scaler)
        pred_data_loader = MoleculeDataLoader(
            dataset=pred_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        sum_test_preds = 0.
        for model in self.models:
            preds = predict(
                model=model,
                data_loader=pred_data_loader,
                scaler=self.scaler,
                return_unc_parameters=True
            )
            sum_test_preds += np.array(preds)
        preds = sum_test_preds / len(self.models)
        if self.args.dataset_type == 'classification':
            preds = np.asarray(preds)
            preds = np.concatenate([preds, 1-preds], axis=1)
            return 0.25 - np.var(preds, axis=1)
        elif self.args.dataset_type == 'multiclass':
            return 0.25 - np.var(preds, axis=1)
        elif self.args.dataset_type == 'regression':
            if self.model.loss_function == "mve":
                preds, var = preds
                return np.array(var).ravel()
            elif self.args.loss_function == 'evidential':
                preds, lambdas, alphas, betas = preds
                return (np.array(betas) / (np.array(lambdas) * (np.array(alphas) - 1))).ravel()
            else:
                raise ValueError
        else:
            raise ValueError

    def predict_value(self, pred_data):
        args = self.args
        if args.features_scaling:
            pred_data.normalize_features(self.features_scaler)
        pred_data_loader = MoleculeDataLoader(
            dataset=pred_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        sum_test_preds = 0.
        for model in self.models:
            preds = predict(
                model=model,
                data_loader=pred_data_loader,
                scaler=self.scaler,
                return_unc_parameters=True
            )
            sum_test_preds += np.array(preds)
        preds = sum_test_preds / len(self.models)
        if self.args.dataset_type in ['classification', 'multiclass']:
            return np.asarray(preds).ravel()
        elif self.args.dataset_type == 'regression':
            return np.asarray(preds).ravel()
        else:
            raise ValueError()
