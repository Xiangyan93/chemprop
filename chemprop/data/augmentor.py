#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import List
import random
import copy
from chemprop.features.featurization import MolGraph


class BaseAugmentor(ABC):
    @abstractmethod
    def __call__(self, molgraph: MolGraph, seed: int = 0):
        pass
