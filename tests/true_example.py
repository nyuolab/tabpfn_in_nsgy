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

# Load data
filtered_df = pd.read_csv("data.csv")

# Define variables
predictor_variables = ['Variable1', 'Variable2', 'Variable3']
outcome_variables = ['outcome1', 'outcome2']
# admyr = ['AdmYR'] - include only if interested in splitting by year
filtered_df = filtered_df[predictor_variables + outcome_variables]
# filtered_df = filtered_df[predictor_variables + outcome_variables + admyr]

def load_custom_dataset(filepath, outcome_variable, normalize=True):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(filepath)
    # Define the predictor and outcome variables
    predictor_variables = ['Variable1', 'Variable2', 'Variable3']
    # adm_yr = 'AdmYR'
    # Drop rows where the outcome variable is NA
    df = df.dropna(subset=[outcome_variable])
    # Separate the predictors and the outcome
    X = df[predictor_variables]
    y = df[outcome_variable]
    # adm_yr_column = df[adm_yr]
    # Ensure X and y are pandas DataFrames or Series before calling to_numpy
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
    # Create a Bunch object to mimic the structure of sklearn datasets
    dataset = Bunch(data=X_processed,
                    target=y.to_numpy(),
                    feature_names=predictor_variables,
                    target_names=[outcome_variable])
    return dataset

## SHAP functions, adapted from the TabPFN documentation
def plot_shap_feature(shap_values_, feature_name, n_plots=1):
    import shap
    # we can use shap.approximate_interactions to guess which features
    # may interact with age
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
    plt.tight_layout()  # Adjust layout to prevent y-axis from being cut off
    plt.show()
    plt.figure(figsize=figsize)
    shap.summary_plot(shap_values=shap_values, show=False)
    plt.title(
        "Feature importances for each feature for each test example (a dot is one feature for one example)"
    )
    plt.tight_layout()  # Adjust layout to prevent y-axis from being cut off
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
    # Split the dataset based on the value of 'AdmYR'
    train_indices = X[:,-1] != 2019
    test_indices = X[:,-1] == 2019
    X_train = X[train_indices, :-1]
    y_train = y[train_indices]
    X_test = X[test_indices, :-1]
    y_test = y[test_indices]
    # Impute missing values for SVM, RandomForest, and DecisionTree
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
    # Individual SHAP plots for top predictors
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

# Example usage:
shap_df, top_features = flexible_shap_analysis(
    dataset_path="data.csv",
    outcome_variable='outcome1',
    predictor_variables = ['Variable1', 'Variable2', 'Variable3'],
    model_type='tabpfn',
    n_top_features=10,
    top_feat_plot_title='Placeholder Title',
    n_shap_plots=5,
    save_prefix='/path/to/save/results'
)

def run_classification_experiment(
    filepath,
    outcome_variable,
    n_splits=10,
    test_size=0.2,
    n_bootstrap=1000,
    plot_roc_auc=False,
    plot_auroc_vs_samples=False,
    plot_title=None,
    plot_filename=None,
    balance_probabilities=False
):
    dataset = load_custom_dataset(filepath, outcome_variable, normalize=False)
    X = dataset.data
    y = dataset.target
    feature_names = dataset.feature_names
    metrics = {
        'ROC AUC':      (roc_auc_score, True),
        'Average Precision': (average_precision_score, True),
        'F1 Score':     (lambda y, yhat: f1_score(y, yhat, average='binary'), False),
        'Sensitivity':  (lambda y, yhat: recall_score(y, yhat, average='binary'), False),
        'Specificity':  (lambda y, yhat: get_sens_spec(y, yhat)[1], False),
        "Youden's J": (youdens_j, False),
        'Accuracy':     (accuracy_score, False),
    }
    models = ['XGB', 'TabPFN', 'SVM', 'RandomForest', 'DecisionTree']
    results = {model: {metric: [] for metric in metrics} for model in models}
    roc_curves = {'XGB': [], 'TabPFN': []}
    for split in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=split
        
        imputer = KNNImputer(n_neighbors=5)
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        clf_xgb = XGBClassifier()
        clf_xgb.fit(X_train_imputed, y_train)
        clf_tabpfn = TabPFNClassifier(balance_probabilities=True)
        clf_tabpfn.fit(X_train_imputed, y_train)
        clf_svm = SVC(probability=True, random_state=42)
        clf_svm.fit(X_train_imputed, y_train)
        clf_rf = RandomForestClassifier(random_state=42)
        clf_rf.fit(X_train_imputed, y_train)
        clf_dt = DecisionTreeClassifier(random_state=42)
        clf_dt.fit(X_train_imputed, y_train)
        # Compute class frequencies for balancing
        if balance_probabilities:
            unique, counts = np.unique(y_train, return_counts=True)
            class_freq = np.zeros(2)
            for u, c in zip(unique, counts):
                class_freq[int(u)] = c
            class_freq = class_freq / class_freq.sum()
            class_freq[class_freq == 0] = 1e-8
            def balance_proba(proba):
                balanced = proba / class_freq
                balanced /= balanced.sum(axis=1, keepdims=True)
                return balanced
        else:
            balance_proba = lambda x: x
        proba_xgb = balance_proba(clf_xgb.predict_proba(X_test_imputed))[:, 1]
        proba_tabpfn = clf_tabpfn.predict_proba(X_test_imputed)[:, 1]
        proba_svm = balance_proba(clf_svm.predict_proba(X_test_imputed))[:, 1]
        proba_rf = balance_proba(clf_rf.predict_proba(X_test_imputed))[:, 1]
        proba_dt = balance_proba(clf_dt.predict_proba(X_test_imputed))[:, 1]
        proba_dict = {
            'TabPFN': proba_tabpfn,
            'XGB': proba_xgb,
            'SVM': proba_svm,
            'RandomForest': proba_rf,
            'DecisionTree': proba_dt
        }
        pred_dict = {}
        thresholds_dict = {}
        for model, proba in proba_dict.items():
            fpr, tpr, thresholds = roc_curve(y_test, proba)
            youden_j = tpr - fpr
            best_idx = np.argmax(youden_j)
            best_threshold = thresholds[best_idx]
            thresholds_dict[model] = best_threshold
            pred_dict[model] = (proba >= best_threshold).astype(int)
        # Store ROC curves for plotting
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, proba_xgb)
        fpr_tabpfn, tpr_tabpfn, _ = roc_curve(y_test, proba_tabpfn)
        roc_curves['XGB'].append((fpr_xgb, tpr_xgb))
        roc_curves['TabPFN'].append((fpr_tabpfn, tpr_tabpfn))
        # Store metrics
        for metric_name, (metric_func, uses_proba) in metrics.items():
            if uses_proba:
                results['TabPFN'][metric_name].append(metric_func(y_test, proba_tabpfn))
                results['XGB'][metric_name].append(metric_func(y_test, proba_xgb))
                results['SVM'][metric_name].append(metric_func(y_test, proba_svm))
                results['RandomForest'][metric_name].append(metric_func(y_test, proba_rf))
                results['DecisionTree'][metric_name].append(metric_func(y_test, proba_dt))
            else:
                results['TabPFN'][metric_name].append(metric_func(y_test, pred_dict['TabPFN']))
                results['XGB'][metric_name].append(metric_func(y_test, pred_dict['XGB']))
                results['SVM'][metric_name].append(metric_func(y_test, pred_dict['SVM']))
                results['RandomForest'][metric_name].append(metric_func(y_test, pred_dict['RandomForest']))
                results['DecisionTree'][metric_name].append(metric_func(y_test, pred_dict['DecisionTree']))
    # Aggregate results
    agg_results = []
    pval_results = []
    for metric_name in metrics:
        row_mean = {'Metric': metric_name + ' Mean'}
        row_std = {'Metric': metric_name + ' Std'}
        for model in models:
            vals = results[model][metric_name]
            row_mean[model] = np.mean(vals)
            row_std[model] = np.std(vals)
        agg_results.extend([row_mean, row_std])
        # Paired p-value TabPFN vs XGB
        tabpfn_vals = results['TabPFN'][metric_name]
        xgb_vals = results['XGB'][metric_name]
        try:
            stat, pval = wilcoxon(tabpfn_vals, xgb_vals)
        except Exception:
            stat, pval = ttest_rel(tabpfn_vals, xgb_vals)
        pval_results.append({'Metric': metric_name + ' TabPFN vs XGB p-value', 'p-value': pval})
    df_results = pd.DataFrame(agg_results)
    df_pvals = pd.DataFrame(pval_results)
    # ROC AUC plot for XGB and TabPFN
    roc_auc_plot = None
    if plot_roc_auc:
        plt.figure(figsize=(8, 6))
        display_names = {'XGB': 'Extreme Gradient Boosting', 'TabPFN': 'Tabular Foundation Model'}
        # Plot mean ROC curve for each model
        for model, color in zip(['XGB', 'TabPFN'], ['b', 'g']):
            mean_fpr = np.linspace(0, 1, 100)
            tprs = []
            for fpr, tpr in roc_curves[model]:
                tprs.append(np.interp(mean_fpr, fpr, tpr))
            mean_tpr = np.mean(tprs, axis=0)
            std_tpr = np.std(tprs, axis=0)
            mean_auc = np.mean(results[model]['ROC AUC'])
            plt.plot(mean_fpr, mean_tpr, color=color, label=f"{display_names[model]} (AUROC={mean_auc:.3f})")
            plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color=color, alpha=0.2)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(plot_title or 'ROC AUC Curve')
        pval_row = df_pvals[df_pvals['Metric'] == 'ROC AUC TabPFN vs XGB p-value']
        if not pval_row.empty:
            pval = pval_row['p-value'].values[0]
            plt.legend(title=f"p={pval:.4f}", loc = 'lower right')
        else:
            plt.legend()
        if plot_filename:
            plt.savefig(plot_filename)
        roc_auc_plot = plt.gcf()
        plt.close()
    # --- AUROC vs. Training Sample Size Plot with 95% CI ---
    auroc_sample_sizes_plot = None
    if plot_auroc_vs_samples:
        sample_fracs = [i / 10.0 for i in range(1, 11)]  # 0.1 to 1.0 in steps of 0.1
        n_repeats = 10
        auroc_results = {'XGB': [], 'TabPFN': [], 'XGB_CI': [], 'TabPFN_CI': [], 'n_samples': []}
        rng = np.random.RandomState(42)
        train_indices = X[:, -1] != 2019
        test_indices = X[:, -1] == 2019
        X_full_train = X[train_indices, :-1]
        y_full_train = y[train_indices]
        X_test_final = X[test_indices, :-1]
        y_test_final = y[test_indices]
        imputer = KNNImputer(n_neighbors=5)
        X_test_final_imputed = imputer.fit_transform(X_test_final)
        for frac in sample_fracs:
            auroc_xgb_list = []
            auroc_tabpfn_list = []
            n_samples = int(len(X_full_train) * frac)
            for repeat in range(n_repeats):
                idx = rng.choice(len(X_full_train), n_samples, replace=False)
                X_sub = X_full_train[idx]
                y_sub = y_full_train[idx]
                X_sub_imputed = imputer.fit_transform(X_sub)
                # XGBoost
                clf_xgb_sub = XGBClassifier()
                clf_xgb_sub.fit(X_sub_imputed, y_sub)
                proba_xgb = clf_xgb_sub.predict_proba(X_test_final_imputed)[:, 1]
                auroc_xgb_list.append(roc_auc_score(y_test_final, proba_xgb))
                # TabPFN
                clf_tabpfn_sub = TabPFNClassifier(balance_probabilities=True)
                clf_tabpfn_sub.fit(X_sub_imputed, y_sub)
                proba_tabpfn = clf_tabpfn_sub.predict_proba(X_test_final_imputed)[:, 1]
                auroc_tabpfn_list.append(roc_auc_score(y_test_final, proba_tabpfn))
        # Mean and 95% CI for each classifier
        mean_xgb = np.mean(auroc_xgb_list)
        mean_tabpfn = np.mean(auroc_tabpfn_list)
        ci_xgb = np.percentile(auroc_xgb_list, [2.5, 97.5])
        ci_tabpfn = np.percentile(auroc_tabpfn_list, [2.5, 97.5])
        auroc_results['XGB'].append(mean_xgb)
        auroc_results['TabPFN'].append(mean_tabpfn)
        auroc_results['XGB_CI'].append(ci_xgb)
        auroc_results['TabPFN_CI'].append(ci_tabpfn)
        auroc_results['n_samples'].append(n_samples)
        # Convert CIs to arrays for plotting
        xgb_ci = np.array(auroc_results['XGB_CI'])
        tabpfn_ci = np.array(auroc_results['TabPFN_CI'])
        plt.figure(figsize=(8, 6))
        plt.plot(auroc_results['n_samples'], auroc_results['XGB'], marker='o', label='XGBoost', color='b')
        plt.plot(auroc_results['n_samples'], auroc_results['TabPFN'], marker='o', label='TabPFN', color='g')
        # Add shaded 95% CI
        plt.fill_between(auroc_results['n_samples'], xgb_ci[:, 0], xgb_ci[:, 1], color='b', alpha=0.2)
        plt.fill_between(auroc_results['n_samples'], tabpfn_ci[:, 0], tabpfn_ci[:, 1], color='g', alpha=0.2)
        plt.xlabel('Number of Training Samples')
        plt.ylabel('Mean AUROC')
        plt.ylim(0.5, 1.0)
        if plot_title:
            plt.title(plot_title)
        plt.legend()
        plt.tight_layout()
        auroc_sample_sizes_plot = plt.gcf()
        if plot_filename:
            plt.savefig(plot_filename.replace('.png', '_auroc_vs_samples.png'))
        plt.close()
    return df_results, df_pvals, roc_auc_plot, auroc_sample_sizes_plot


results_df, pvals_df, roc_auc_plot, auroc_sample_sizes_plot = run_classification_experiment(
    filepath="data.csv",
    outcome_variable='outcome1',
    n_splits=10,
    plot_roc_auc=True,
    plot_auroc_vs_samples=False,
    plot_title="Title",
    plot_filename="/path/to/save/plot.png",
    balance_probabilities=True
)
results_df.to_csv("results.csv", index=False)
pvals_df.to_csv("pvalues.csv", index=False)
# roc_auc_plot.savefig("xgb_tabpfn_roc_auc.png")
# auroc_sample_sizes_plot.savefig("auroc_vs_samples.png")
