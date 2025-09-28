import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score, mean_squared_log_error, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from tabpfn import TabPFNClassifier, TabPFNRegressor
from xgboost import XGBClassifier, XGBRegressor
from tabpfn_extensions import interpretability
import shapiq 
import torch
import matplotlib.pyplot as plt
from sklearn.utils import Bunch
import shap
from sklearn.preprocessing import StandardScaler
from tabpfn_extensions.unsupervised import unsupervised
from imblearn.over_sampling import SMOTE
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import precision_recall_curve, average_precision_score, auc, roc_curve
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pickle 
from MLstatkit.stats import Delong_test
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

def spiegelhalter_z(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    numerator = np.sum(y_true - y_prob)
    denominator = np.sqrt(np.sum(y_prob * (1 - y_prob)))
    return numerator / denominator if denominator != 0 else np.nan

def load_custom_dataset(filepath, outcome_variable, normalize=True):
    df = pd.read_csv(filepath)
    # Define the predictor and outcome variables
    predictor_variables = ['Variable1', 'Variable2', 'Variable3']
    # adm_yr = 'AdmYR'
    # Drop rows where the outcome variable is NA
    df = df.dropna(subset=[outcome_variable])
    # Separate predictors and outcome
    X = df[predictor_variables]
    y = df[outcome_variable]
    # adm_yr_column = df[adm_yr]
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=predictor_variables)
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name=outcome_variable)
    # Normalize the predictor variables if requested
    if normalize:
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X)
    else:
        X_processed = X.values
    # Concatenate the predictors with the adm_yr column
    # X_processed = np.hstack((X_processed, adm_yr_column.values.reshape(-1, 1)))
    # Create sklearn style dataset
    dataset = Bunch(data=X_processed,
                    target=y.to_numpy(),
                    feature_names=predictor_variables,
                    target_names=[outcome_variable])
    return dataset

# SHAP functions, adapted from the TabPFN documentation
def plot_shap_feature(shap_values_, feature_name, n_plots=1):
    import shap
    inds = shap.utils.potential_interactions(
        shap_values_[:, feature_name], shap_values_
    )
    # make plots colored by each of the top three possible interacting features
    for i in range(n_plots):
        shap.plots.scatter(
            shap_values_[:, feature_name],
            color=shap_values_[:, inds[i]],
            show=False,
        )
        plt.title(
            f"SHAP value plot for {shap_values_.feature_names[feature_name]} with a color coding representing the value of {shap_values_.feature_names[inds[i]]}"
        )
        plt.tight_layout()

def plot_shap(shap_values: np.ndarray, rank: int = 1, figsize=(10, 6)):
    import shap
    if len(shap_values.shape) == 3:
        print("Computing shap values for the second class (index 1).")
        shap_values = shap_values[:, :, 1]
    plt.figure(figsize=figsize)
    shap.plots.bar(shap_values=shap_values, show=False)
    plt.title("Aggregate feature importances across the test examples")
    plt.tight_layout()  
    plt.show()
    plt.figure(figsize=figsize)
    shap.summary_plot(shap_values=shap_values, show=False)
    plt.title(
        "Feature importances for each feature for each test example (a dot is one feature for one example)"
    )
    plt.tight_layout()  
    plt.show()
    most_important = shap_values.abs.mean(0).values.argsort()[-rank]
    print(
        f'Now we analyze the strongest feature interactions of the most important feature, namely the feature "{most_important}".'
    )
    if len(shap_values) > 1:
        plot_shap_feature(shap_values, most_important)

def get_sens_spec(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    else:
        sensitivity = specificity = np.nan
    return sensitivity, specificity

def bootstrap_ci(y_true, y_pred, metric_func, n_bootstrap=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    stats = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        try:
            stat = metric_func(y_true[idx], y_pred[idx])
        except Exception:
            stat = np.nan
        stats.append(stat)
    stats = np.array(stats)
    lower = np.nanpercentile(stats, 2.5)
    upper = np.nanpercentile(stats, 97.5)
    return lower, upper

def bootstrap_ci_proba(y_true, y_proba, metric_func, n_bootstrap=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    stats = []
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        try:
            stat = metric_func(y_true[idx], y_proba[idx])
        except Exception:
            stat = np.nan
        stats.append(stat)
    stats = np.array(stats)
    lower = np.nanpercentile(stats, 2.5)
    upper = np.nanpercentile(stats, 97.5)
    return lower, upper

def balance_proba(proba):
    # proba: shape (n_samples, 2)
    balanced = proba / class_freq
    balanced /= balanced.sum(axis=1, keepdims=True)
    return balanced

def flexible_shap_analysis(
    dataset_path,
    outcome_variable,
    predictor_variables,  
    model_type='tabpfn',
    n_top_features=10,
    top_feat_plot_title='Mean SHAP Values for Top Features',
    n_shap_plots=5,
    impute=True,
    save_prefix=None
):
    # Load data
    dataset = load_custom_dataset(dataset_path, outcome_variable=outcome_variable, normalize=False)
    X = dataset.data
    y = dataset.target
    feature_names = dataset.feature_names
    # Split the dataset based on the value of 'AdmYR' - replace with your values as needed
    train_indices = X[:,-1] != 2019
    test_indices = X[:,-1] == 2019
    X_train = X[train_indices, :-1]
    y_train = y[train_indices]
    X_test = X[test_indices, :-1]
    y_test = y[test_indices]
    # Impute missing values
    imputer = KNNImputer(n_neighbors=5)
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    clf = TabPFNClassifier(balance_probabilities=True)
    clf.fit(X_train_imputed, y_train)
    feature_names_nlist = np.array(feature_names)
    shap_values = interpretability.shap.get_shap_values(
        estimator=clf,
        test_x=X_test_imputed[:50],
        attribute_names=np.array(feature_names),
        algorithm="permutation",
    )
    mean_shap_values = shap_values.abs.mean(0).values
    mean_shap_values = mean_shap_values[:, 1]
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean SHAP Value': mean_shap_values
    }).sort_values(by='Mean SHAP Value', ascending=False)
    top_features = shap_df['Feature'].head(n_top_features).tolist()
    # Plot mean SHAP values as a bar plot
    plt.figure(figsize=(10, 8))
    plt.barh(shap_df['Feature'].head(n_top_features), shap_df['Mean SHAP Value'].head(n_top_features), color='skyblue')
    plt.xlabel('Mean SHAP Value')
    plt.ylabel('Feature')
    plt.title(f'{top_feat_plot_title}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f'{save_prefix}_mean_shap_values_bar_plot_{model_type}.png')
    plt.show()
    # Plot individual SHAP plots for top predictors
    for i, feature in enumerate(top_features[:n_shap_plots]):
        feature_idx = feature_names.index(feature)
        if model_type == 'tabpfn':
            plot_shap(shap_values, rank=i+1)
            if save_prefix:
                plt.savefig(f'{save_prefix}_shap_summary_plot_rank{i+1}_tabpfn.png')
    # Optionally save DataFrame
    if save_prefix:
        shap_df.to_csv(f'{save_prefix}_shap_df_{model_type}.csv', index=False)
    return shap_df, top_features