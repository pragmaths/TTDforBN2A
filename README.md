# TTDforBN2A

Code accompanying the manuscript *"Tensor Train Decomposition for BN2A Structure Learning"*.

This repository contains MATLAB scripts that implement and validate a Tensor Train (TT) based pipeline for learning the structure of BN2A models, following the structure learning algorithm of Gu (2026). The TT representation enables Gu's rank tests to be performed via core contractions, without materializing the joint response tensor explicitly.

## Contents

- `tt_svd_full.m`, `tt_full.m`, `tt_eval.m`: core TT-SVD utilities.
- `bn2a_population_tensor.m`, `generate_bn2a_responses.m`: BN2A model construction and sampling.
- `qmatrix_identification_full_gu_v2.m`: implementation of Gu (2026) Propositions 1 and 2 over TT cores.
- `full_q_comparison.m`: main experiment at J=15, K=3.
- `sample_size_barrier_ttsvd.m`: sample-size dependence of the pipeline.
- `gu_toy_example_verify.m`: verification on Gu's J=5, K=2 toy example.
- `regression_J15.m`, `bisection_J16.m`, `bisection_J18.m`: diagnostic scripts.

## Requirements

MATLAB R2020a or later. No external toolboxes required.

## Usage

Each script is self-contained. Run, for example, `full_q_comparison.m` from the MATLAB prompt to reproduce the main result at J=15, K=3.

## Citation

If you use this code, please cite the accompanying paper.
