# Triple-Variability-Source MEP IO Model
## Introduction

This repository contains the implementation and analysis of a novel statistical model for analyzing motor-evoked potential (MEP) input-output (IO) curves in transcranial magnetic stimulation (TMS) studies.

## Key Features

1. **Triple-Variability-Source Model**: Separates three distinct sources of variability in MEP responses:
   - Stimulation-side variability ($v_\textrm{x}$)
   - Response-side multiplicative variability ($v_\textrm{y}$)
   - Additive background noise ($v_\textrm{add}$)

2. **Improved Recruitment Curve Modeling**: Uses a logarithmic logistic function without a lower plateau, allowing for the detection of responses below the noise floor.

3. **Superior Goodness-of-Fit**: Demonstrates improved fit compared to previous dual-variability-source models, as evidenced by lower Bayesian Information Criterion (BIC) scores.

4. **Enhanced Physiological Variability Estimation**: Accurately estimates physiological variability by separating it from technical noise.

5. **Broad Applicability**: Potential applications in clinical and experimental neuroscience for analysing brain stimulation IO curves with reduced risk of spurious results and statistical bias.

## Model Code
This folder contains the triple-variability-source model class and its cross-validation sample code.
1. `Logistic3P3VCurveModel_GEV.m` is the model class.
2. `simulateDatabase.m` is a function to generate simulated data points according to the given model parameters.
3. `Sample_Model_Validation_Opt.m` runs the optimisation based on the simulated dataset.
4. `Figure_Model_Validation.m` is the code to plot the cross-validation results, including a representative example and its Probability-Probability (P-P) plot.
5. `results_v1.mat` saves the cross-validation results.
6. `corr_PP.mat` saves the values of the correlation coefficient ($R$) of P-P plots for each cross-validation iteration. 

One can use `Logistic3P3VCurveModel_GEV.m` to run optimisation on their datasets. Please ensure that you have enough background noise measurements for the model optimisation.


## Significance

This model provides a robust framework for extracting quantitative information about both the expected recruitment and variability sources in MEP responses. It offers a valuable tool for studying the physiology of MEPs, excitation, and neuromodulation.


## Publication
[1] Ma K, Liu S, Qin M, Goetz SM. Extraction of three mechanistically different variability and noise sources in the trial-to-trial variability of brain stimulation. IEEE Transactions on Neural Systems and Rehabilitation Engineering. 2024 Dec 25.
