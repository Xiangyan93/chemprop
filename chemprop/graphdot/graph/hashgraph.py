#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
CWD = os.path.dirname(os.path.abspath(__file__))
import pandas as pd
from graphdot import Graph as G
from graphdot.graph._from_networkx import _from_networkx
import networkx as nx
from rxntools.reaction import *
from chemprop.graphdot.graph.from_rdkit import _from_rdkit, rdkit_config


class Graph(G):
    @classmethod
    def from_smiles(self, smiles, _rdkit_config=rdkit_config()):
        mol = Chem.MolFromSmiles(smiles)
        g = self.from_rdkit(mol, _rdkit_config)
        return g

    @classmethod
    def from_rdkit(cls, mol, _rdkit_config=rdkit_config()):
        _rdkit_config.preprocess(mol)
        g = _from_rdkit(cls, mol, _rdkit_config)
        # g = g.permute(rcm(g))
        return g
