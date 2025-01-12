#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 17:03:48 2024

@author: mualla
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_chi_square = pd.read_csv('./results/v2_to_pe_model/dataset_1/estimated_v2_to_pe_no_bin_params.csv')
data_chi_square_2 = pd.read_csv('./results/pe_to_v2_model/dataset_1/estimated_negative_binned_params_new.csv')

data_chi_square = data_chi_square[['v', 'a', 'ter', 'a2', 'ter2',  'pe_slope', 'pe_intercept' ]]
data_chi_square_2 = data_chi_square_2[['v', 'a', 'ter', 'a2', 'ter2',  'v2_slope', 'v2_intercept' ]]

# Concatenate the two DataFrames
combined_data = pd.concat([data_chi_square, data_chi_square_2], axis=1)

# Calculate the correlation matrix
correlation_matrix = combined_data.corr()
lower = correlation_matrix.iloc[7:, :7]

mask = np.triu(np.ones_like(lower, dtype=bool), k=1)

labels_model_1 = ['Drift Rate', 'Boundary', 'NDT', 'Confidence Boundary', 'Confidence NDT', 'Pe Slope', 'Pe Intercept']
labels_model_2 = ['Drift Rate', 'Boundary', 'NDT', 'Confidence Boundary', 'Confidence NDT', 'V2 Slope', 'V2 Intercept']

axis_labels = ['BayesFlow', 'Quantile Optimization']

# Plot the heatmap for the entire 7x7 matrix
plt.figure(figsize=(13, 10))
sns.heatmap(lower, annot=True, cmap='flare', fmt='.2f', linewidths=.5, mask = mask, 
            xticklabels=labels_model_1, yticklabels=labels_model_2, annot_kws={"fontsize": 14})

plt.xticks(rotation=35, fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel(axis_labels[1],  fontsize=18)
plt.ylabel(axis_labels[0],  fontsize=18)

plt.tight_layout()
plt.savefig('./results/correlation_matrix_quantile_optim.pdf')
plt.show()
