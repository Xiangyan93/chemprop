# Uncertainty Qualification for Molecular Property Prediction
This repository contains tools to evaluate uncertainty qualification (UQ) methods for molecular property prediction. This repository was forked from Chemprop. The original repository, with additional documentation, can be found [here](https://github.com/chemprop/chemprop).

## Reproducing Results
The raw evaluations used to produce the paper's results can be found in `uncertainty_evaluation/evaluations.csv`. These values can be recalculated and visualized by following the steps outlined below.

### Installing Dependencies
1. Install Miniconda from [https://conda.io/miniconda.html](https://conda.io/miniconda.html)
2. `cd /path/to/chemprop`
3. `conda env create -f environment.yml`
4. `conda activate chemprop_uncertainty` (or `source activate chemprop_uncertainty` for older versions of conda)

### Prepare Data
1. `tar xvzf data.tar.gz`
2. `python uncertainty_evaluation/generate_logp.py` (optional, regenerates logp.csv)

### Run Experiments
1. `python uncertainty_evaluation/populate_build.py`
2. `bash uncertainty_evaluation/populate.sh`

### Evaluate and Plot Experiments
1. `cd uncertainty_evaluation`
2. `jupyter notebook`
3. Open `Analysis.ipynb` and run all cells. If imports fail, make sure you're using the right conda environment.
