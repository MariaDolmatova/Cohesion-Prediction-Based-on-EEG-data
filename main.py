import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import torch
import torch.nn as nn
from plotly.subplots import make_subplots
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from src.models.cnn import fold_cross_validation_120dataset, train_model_early_stopping
from src.models.pca import pca
from src.models.svm import multi_datasets, train_svm
from torch.utils.data import DataLoader, Subset, TensorDataset
from src.utils.random_seed import set_random_seed
from src.utils.reshape_datasets import reshape_input_eeg
from src.utils.labels_preparation import process_labels
from src.utils.visuals import plot_binarisation_choice, label_distribution, plot_heatmap, plot_grid_search_results, plot_f1_scores, plot_pca, plot_cross_validation_results
from logger import get_logger


# Binarise labels
out_cohesion = process_labels("Averaged Cohesion scores.csv", "labels.csv")

# Binary threshold visual
plot_binarisation_choice(out_cohesion)

# Pie chart label distribution
label_distribution()

# Reshape all datasets
reshape_input_eeg("correlations_array.csv", "reshaped_correlations.csv", has_part=False)
reshape_input_eeg("correlations_array5.csv", "reshaped_correlations5.csv", has_part=True)
reshape_input_eeg("correlations_array10.csv", "reshaped_correlations10.csv", has_part=True)
reshape_input_eeg("correlations_array60.csv", "reshaped_correlations60.csv", has_part=True)
reshape_input_eeg("correlations_array120.csv", "reshaped_correlations120.csv", has_part=True)

################################
# Perform SVM
best_model, best_params, best_score, results_df = train_svm("reshaped_correlations120.csv", "labels.csv")
results_df.head()

# Visuals for labels heatmap + best F1 score for SVM
heatmap_data = results_df.pivot_table(index="param_svc__kernel", columns="param_svc__gamma", values="mean_test_f1")

# Plot heatmap
plot_heatmap(heatmap_data)
plot_grid_search_results(results_df)

################################
# Perform multiple datasets SVM
datasets = [
    ("reshaped_correlations.csv", "labels.csv"),
    ("reshaped_correlations10.csv", "labels.csv"),
    ("reshaped_correlations120.csv", "labels.csv"),
    ("reshaped_correlations5.csv", "labels.csv"),
    ("reshaped_correlations60.csv", "labels.csv"),
]

results_df = multi_datasets(datasets)
results_df  # noqa: B018

# Visuals for nultiple dataset SVM
dataset_scores = results_df.groupby("Dataset", as_index=False)["Best F1 Score"].mean()
plot_f1_scores(dataset_scores)


################################
# Perform PCA

pca("reshaped_correlations.csv", 15)
pca("reshaped_correlations120.csv", 15)

#plot for pca
plot_pca

################################
# CNN 
train_model_early_stopping(model, train_loader, val_loader, optimizer, criterion, epochs, patience, min_delta)
fold_cross_validation_120dataset(model_class, data_X, data_Y, config_path, n_splits)

# Visuals for CNN

plot_cross_validation_results(fold, train_losses, val_losses, train_accs, val_accs)

""""WHAT HAPPENED HERE??????"""

    fold_val_losses.append(val_losses[-1])
    fold_val_accs.append(val_accs[-1])
    fold_val_f1s.append(val_f1s[-1])

mean_loss = np.mean(fold_val_losses)  # For calculating the average of all folds
mean_acc = np.mean(fold_val_accs)
mean_f1 = np.mean(fold_val_f1s)

logger.info(f"Average loss: {mean_loss:.4f}")
logger.info(f"Average accuracy: {mean_acc:.4f}")
logger.info(f"Average F1 score: {mean_f1:.4f}")
