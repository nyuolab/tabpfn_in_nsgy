import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score, mean_squared_log_error, f1_score, RocCurveDisplay
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
from sklearn.impute import IterativeImputer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import precision_recall_curve, average_precision_score, auc, matthews_corrcoef

# Initialize a dataframe to store the results
results_df = pd.DataFrame(columns=['Task', 'Metric', 'XGB', 'TabPFN'])

def load_custom_dataset(filepath, outcome_variable):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(filepath)
    
    # Define the predictor and outcome variables
    predictor_variables = ['Age', 'CPT', 'SEX', 'RACE_NEW', 'ETHNICITY_HISPANIC', 'INOUT', 'TRANST', 'ANESTHES', 'SURGSPEC', 'ELECTSURG', 'HEIGHT', 'WEIGHT', 'DIABETES', 'SMOKE', 'DYSPNEA', 'FNSTATUS2', 'VENTILAT', 'HXCOPD', 'ASCITES', 'HXCHF', 'HYPERMED', 'RENAFAIL', 'DIALYSIS', 'DISCANCR', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDDIS', 'TRANSFUS', 'ASACLAS', 'PRSODM', 'PRBUN', 'PRCREAT', 'PRALBUM', 'PRBILI', 'PRSGOT', 'PRALKPH', 'PRWBC', 'PRHCT', 'PRPLATE', 'PRPTT', 'PRINR', 'PRPT']
    outcome_variable = outcome_variable
    
    # Drop rows where the outcome variable is NA
    df = df.dropna(subset=[outcome_variable])
    
    # Separate the predictors and the outcome
    X = df[predictor_variables]
    y = df[outcome_variable]
    
    # Ensure X and y are pandas DataFrames or Series before calling to_numpy
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=predictor_variables)
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name=outcome_variable)
    
    # Normalize the predictor variables
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Create a Bunch object to mimic the structure of sklearn datasets
    dataset = Bunch(data=X_normalized,
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
        print("Computing shap values for the first class (index 1).")
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

def plot_precision_recall_curve(recall_xgb, precision_xgb, recall_tabpfn, precision_tabpfn, plot_title, save_filename):
    plt.figure(figsize=(8, 6))
    plt.plot(recall_xgb, precision_xgb, label='XGBClassifier')
    plt.plot(recall_tabpfn, precision_tabpfn, label='TabPFNClassifier')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(plot_title)
    plt.legend()
    plt.savefig(save_filename)
    plt.show()


def plot_f1_scores_XGBvTabPFN(X_train, y_train, X_test, y_test, title, filename):
    sample_sizes = np.linspace(0.1, 1, 10)
    sample_counts = [int(round(s * len(X_train))) for s in sample_sizes]
    
    f1_scores_xgb = []
    f1_scores_tabpfn = []
    
    for s_fraction, s_count in zip(sample_sizes, sample_counts):
        if s_fraction == 1.0:
            X_train_sample, y_train_sample = X_train, y_train
        else:
            X_train_sample, _, y_train_sample, _ = train_test_split(
                X_train, y_train, train_size=s_fraction, random_state=42
            )
        
        # Train XGBClassifier
        clf_xgb = XGBClassifier()
        clf_xgb.fit(X_train_sample, y_train_sample)
        prediction_probabilities_xgb = clf_xgb.predict_proba(X_test)
        f1_xgb = f1_score(y_test, clf_xgb.predict(X_test), average='binary')
        f1_scores_xgb.append(f1_xgb)
        
        # Train TabPFNClassifier
        clf = TabPFNClassifier()
        clf.fit(X_train_sample, y_train_sample)
        prediction_probabilities = clf.predict_proba(X_test)
        f1_tabpfn = f1_score(y_test, clf.predict(X_test), average='binary')
        f1_scores_tabpfn.append(f1_tabpfn)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sample_counts, f1_scores_xgb, label='XGBClassifier')
    plt.plot(sample_counts, f1_scores_tabpfn, label='TabPFNClassifier')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('F1 Score')
    plt.title(title)
    plt.legend()
    plt.show()
    plt.savefig(filename)

def plot_roc_auc_scores_XGBvTabPFN(X_train, y_train, X_test, y_test, title, filename):
    sample_sizes = np.linspace(0.1, 1, 10)
    sample_counts = [int(round(s * len(X_train))) for s in sample_sizes]
    
    roc_auc_scores_xgb = []
    roc_auc_scores_tabpfn = []
    
    for s_fraction, s_count in zip(sample_sizes, sample_counts):
        if s_fraction == 1.0:
            X_train_sample, y_train_sample = X_train, y_train
        else:
            X_train_sample, _, y_train_sample, _ = train_test_split(
                X_train, y_train, train_size=s_fraction, random_state=42
            )
        
        # Train XGBClassifier
        clf_xgb = XGBClassifier()
        clf_xgb.fit(X_train_sample, y_train_sample)
        prediction_probabilities_xgb = clf_xgb.predict_proba(X_test)
        roc_auc_xgb = roc_auc_score(y_test, prediction_probabilities_xgb[:, 1])
        roc_auc_scores_xgb.append(roc_auc_xgb)
        
        # Train TabPFNClassifier
        clf = TabPFNClassifier()
        clf.fit(X_train_sample, y_train_sample)
        prediction_probabilities = clf.predict_proba(X_test)
        roc_auc_tabpfn = roc_auc_score(y_test, prediction_probabilities[:, 1])
        roc_auc_scores_tabpfn.append(roc_auc_tabpfn)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sample_counts, roc_auc_scores_xgb, label='XGBClassifier')
    plt.plot(sample_counts, roc_auc_scores_tabpfn, label='TabPFNClassifier')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('ROC AUC')
    plt.title(title)
    plt.legend()
    plt.show()
    plt.savefig(filename)

### Binary Classification Tasks
dataset = load_custom_dataset('/path/to/input.csv', outcome_variable='outcome_variable_name')

# Access the data, target, feature names, and target names
X = dataset.data
y = dataset.target
feature_names = dataset.feature_names
target_names = dataset.target_names

print("Feature names:", feature_names)
print("Target names:", target_names)
print("Data shape:", X.shape)
print("Target shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform random undersampling - perform only if the dataset is considerably imbalanced.
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

# Initialize and train XGBClassifier
clf_xgb = XGBClassifier()
clf_xgb.fit(X_train_rus, y_train_rus)
# clf_xgb.fit(X_train, y_train) - use if not using random undersampling

# Predict probabilities with XGBClassifier
prediction_probabilities_xgb = clf_xgb.predict_proba(X_test)
roc_auc_xgb = roc_auc_score(y_test, prediction_probabilities_xgb[:, 1])
accuracy_xgb = accuracy_score(y_test, clf_xgb.predict(X_test))
f1_score_xgb = f1_score(y_test, clf_xgb.predict(X_test), average='binary')
ap_score_xgb = average_precision_score(y_test, clf_xgb.predict(X_test))
mcc_score_xgb = matthews_corrcoef(y_test, clf_xgb.predict(X_test))

# Initialize and train TabPFNClassifier
clf = TabPFNClassifier()
clf.fit(X_train_rus, y_train_rus)
# clf.fit(X_train, y_train) - use if not using random undersampling

# Predict probabilities with TabPFNClassifier
prediction_probabilities = clf.predict_proba(X_test)
roc_auc_tabpfn = roc_auc_score(y_test, prediction_probabilities[:, 1])
accuracy_tabpfn = accuracy_score(y_test, clf.predict(X_test))
f1_score_tabpfn = f1_score(y_test, clf.predict(X_test), average='binary')
ap_score_tabpfn = average_precision_score(y_test, clf.predict(X_test))
mcc_score_tabpfn = matthews_corrcoef(y_test, clf.predict(X_test))

# Save results to dataframe
results_df = pd.concat([results_df, pd.DataFrame([{'Task': 'binary_outcome_variable', 'Metric': 'ROC AUC', 'XGB': roc_auc_xgb, 'TabPFN': roc_auc_tabpfn}])], ignore_index=True)
results_df = pd.concat([results_df, pd.DataFrame([{'Task': 'binary_outcome_variable', 'Metric': 'Accuracy', 'XGB': accuracy_xgb, 'TabPFN': accuracy_tabpfn}])], ignore_index=True)
results_df = pd.concat([results_df, pd.DataFrame([{'Task': 'binary_outcome_variable', 'Metric': 'F1 Score', 'XGB': f1_score_xgb, 'TabPFN': f1_score_tabpfn}])], ignore_index=True)
results_df = pd.concat([results_df, pd.DataFrame([{'Task': 'binary_outcome_variable', 'Metric': 'Average Precision', 'XGB': ap_score_xgb, 'TabPFN': ap_score_tabpfn}])], ignore_index=True)
results_df = pd.concat([results_df, pd.DataFrame([{'Task': 'binary_outcome_variable', 'Metric': 'MCC', 'XGB': mcc_score_xgb, 'TabPFN': mcc_score_tabpfn}])], ignore_index=True)

# Plot AUC ROC curves for XGBoost and TabPFN
roc_display_xgb = RocCurveDisplay.from_estimator(
    clf_xgb, X_test, y_test, name="XGBClassifier", color="blue"
)
roc_display_tabpfn = RocCurveDisplay.from_estimator(
    clf, X_test, y_test, name="TabPFNClassifier", color="orange"
)
plt.figure(figsize=(8, 6))
roc_display_xgb.plot(ax=plt.gca())
roc_display_tabpfn.plot(ax=plt.gca())
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("AUROC - TabPFN vs. XGB")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('/path/to/output/roc_curve.png')

# Calculate precision-recall values
precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, clf_xgb.predict_proba(X_test)[:, 1])
precision_tabpfn, recall_tabpfn, _ = precision_recall_curve(y_test, clf.predict_proba(X_test)[:, 1])

# Create displays
disp_xgb = PrecisionRecallDisplay(precision=precision_xgb, recall=recall_xgb, 
                                  estimator_name=f"XGBClassifier (AP={ap_score_xgb:.2f})")
disp_tabpfn = PrecisionRecallDisplay(precision=precision_tabpfn, recall=recall_tabpfn, 
                                     estimator_name=f"TabPFNClassifier (AP={ap_score_tabpfn:.2f})")

# Plot XGBoost PR curve
fig, ax = plt.subplots(figsize=(8,6))
disp_xgb.plot(ax=ax)
# Add TabPFN PR curve on the same axes
disp_tabpfn.plot(ax=ax)

# Plot random chance level (positive class rate)
pos_rate = np.mean(y_test)
ax.plot([0, 1], [pos_rate, pos_rate], linestyle="--", color="gray", 
        label=f"Random Chance (AP={pos_rate:.2f})")

ax.set_title("Precision-Recall Curve")
ax.legend()
plt.tight_layout()
plt.show()
plt.savefig('/path/to/output/PR_curve.png')
plt.close()

# Get SHAP values for XGBClassifier
# Initialize and train XGBClassifier
clf_xgb = XGBClassifier()
clf_xgb.fit(X_train_rus, y_train_rus)
# clf_xgb.fit(X_train, y_train) - use if not using random undersampling

# Get SHAP values for XGBClassifier
explainer = shap.TreeExplainer(clf_xgb)
shap_values_xgb = explainer(X_test[:50])
shap_values_xgb.feature_names = feature_names

fig = plot_shap(shap_values_xgb, rank=1)
plt.savefig('/path/to/output/xgb_rus_rank1.png')
# fig = plot_shap(shap_values_xgb, rank=2) - use if you want to plot the second rank, can continue with rank 3, 4, etc.
# plt.savefig('path/to/output/xgb_rus_rank2.png')

mean_shap_values_xgb = shap_values_xgb.abs.mean(0).values
shap_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean SHAP Value': mean_shap_values_xgb
})

# Sort the DataFrame by the mean SHAP values in descending order
shap_df = shap_df.sort_values(by='Mean SHAP Value', ascending=False)

# Plot the mean SHAP values as a bar graph
plt.figure(figsize=(10, 8))
plt.barh(shap_df['Feature'], shap_df['Mean SHAP Value'], color='skyblue')
plt.xlabel('Mean Absolute SHAP Value', fontsize = 15)
plt.ylabel('Feature', fontsize = 15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Mean SHAP Values - XGB', fontsize = 15)
plt.gca().invert_yaxis()  # Invert y-axis to have the highest values at the top
plt.tight_layout()
plt.show()
plt.savefig('/path/to/output/mean_shap_values_bar_plot_xgb_rus.png')
shap_df.to_csv('/path/to/output/shap_df_xgb_rus.csv')

# Plot XGBClassifier and TabPFNClassifier ROC AUC scores across varying sample sizes
plot_roc_auc_scores_XGBvTabPFN(X_train_rus, y_train_rus, X_test, y_test, title = 'ROC AUC - XGB vs. TabPFN', filename='/path/to/output/roc_auc_scores_XGBvTabPFN_rus.png')

plot_f1_scores_XGBvTabPFN(X_train_rus, y_train_rus, X_test, y_test, title = 'F1 Score - XGB vs. TabPFN', filename='/path/to/output/f1_scores_XGBvTabPFN_rus.png')

# SHAP for TabPFNClassifier
clf = TabPFNClassifier()
clf.fit(X_train_rus, y_train_rus)
# clf.fit(X_train, y_train) - use if not using random undersampling
feature_names_nlist = np.array(feature_names)

shap_values = interpretability.shap.get_shap_values(
    estimator=clf,
    test_x=X_test[:50],
    attribute_names=feature_names_nlist,
    algorithm="permutation",
)

fig = plot_shap(shap_values, rank=1)
plt.savefig('/path/to/output/rank1_tabpfn_rus.png')
plt.close()
# fig = plot_shap(shap_values, rank=2) - use if you want to plot the second rank, can continue with rank 3, 4, etc.
# plt.savefig('/path/to/output/shap_summary_plot_rank2_tabpfn_rus.png')
# plt.close()

mean_shap_values = shap_values.abs.mean(0).values
mean_shap_values = mean_shap_values[:, 0]
shap_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean SHAP Value': mean_shap_values
})

# Sort the DataFrame by the mean SHAP values in descending order - note this only takes the top 10 features by mean absolute SHAP value
shap_df = shap_df.sort_values(by='Mean SHAP Value', ascending=False)
top10_df = shap_df.head(10)

# Plot the mean SHAP values as a bar graph
plt.figure(figsize=(10, 8))
plt.barh(top10_df['Feature'], top10_df['Mean SHAP Value'], color='skyblue')
# plt.barh(shap_df['Feature'], shap_df['Mean SHAP Value'], color='skyblue')
plt.xlabel('Mean Absolute SHAP Value', fontsize = 15)
plt.ylabel('Feature', fontsize = 15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Mean SHAP Values - TabPFN', fontsize = 15)
plt.gca().invert_yaxis()  # Invert y-axis to have the highest values at the top
plt.tight_layout()
plt.show()
plt.savefig('/path/to/output/mean_shap_values_bar_plot_tabpfn_rus_top10.png')
shap_df.to_csv('/path/to/output/shap_df_tabpfn_rus.csv')

### Continuous Regression Task
dataset = load_custom_dataset('/path/to/input.csv', outcome_variable = 'outcome_variable_name')
X = dataset.data
y = dataset.target
feature_names = dataset.feature_names
target_names = dataset.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost regressor
clf_xgb = XGBRegressor()
clf_xgb.fit(X_train, y_train)

# Predict values with XGBRegressor
predictions_xgb = clf_xgb.predict(X_test)

# Evaluate the model
mse_xgb = mean_squared_error(y_test, predictions_xgb)
r2_xgb = r2_score(y_test, predictions_xgb)

# Create a scatter plot of predictions vs. true values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions_xgb, color='lightblue', alpha=0.6, edgecolor='k')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**-0.2, 10**2.2) # Adjust limits for log scale
plt.ylim(10**-0.2, 10**2.2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('XGBoost Prediction')
# Get current axes and determine limits for y=x line
ax = plt.gca()
xlims = ax.get_xlim()
ylims = ax.get_ylim()
min_lim = min(xlims[0], ylims[0])
max_lim = max(xlims[1], ylims[1])
plt.plot([min_lim, max_lim], [min_lim, max_lim], 'r--', label='Exact')
plt.legend()
plt.tight_layout() 
plt.show()
# Add text box for MSE and R2 in the top left corner of the plot
plt.text(0.05, 0.95, f'MSE: {mse_xgb:.2f}\nR²: {r2_xgb:.2f}', 
         transform=plt.gca().transAxes, 
         fontsize=12, 
         verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.savefig('/path/to/output/xgb_loglog.png')
plt.close()

# Initialize and train TabPFNRegressor
regressor = TabPFNRegressor(ignore_pretraining_limits=True)  
regressor.fit(X_train, y_train)

# Predict on the test set
predictions = regressor.predict(X_test)

# Evaluate the model
mse_tabpfn = mean_squared_error(y_test, predictions)
r2_tabpfn = r2_score(y_test, predictions)

# Create a scatter plot of predictions vs. true values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, color='orange', alpha=0.6, edgecolor='k')
# Switch to log-log scale
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**-0.2, 10**2.2)
plt.ylim(10**-0.2, 10**2.2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('TabPFN Prediction')
# Get current axes and determine limits for y=x line
ax = plt.gca()
xlims = ax.get_xlim()
ylims = ax.get_ylim()
min_lim = min(xlims[0], ylims[0])
max_lim = max(xlims[1], ylims[1])
plt.plot([min_lim, max_lim], [min_lim, max_lim], 'r--', label='Exact')
plt.legend()
plt.tight_layout()  # Prevents label cutoff
plt.show()
# Add text box for MSE and R2 in the top left corner of the plot
plt.text(0.05, 0.95, f'MSE: {mse_tabpfn:.2f}\nR²: {r2_tabpfn:.2f}', 
         transform=plt.gca().transAxes, 
         fontsize=12, 
         verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.savefig('/path/to/output/tabpfn_loglog.png')
plt.close()

# Save results to dataframe and CSV file
results_df = pd.concat([results_df, pd.DataFrame([{'Task': 'outcome_variable', 'Metric': 'MSE', 'XGB': mse_xgb, 'TabPFN': mse_tabpfn}])], ignore_index=True)
results_df = pd.concat([results_df, pd.DataFrame([{'Task': 'outcome_variable', 'Metric': 'R²', 'XGB': r2_xgb, 'TabPFN': r2_tabpfn}])], ignore_index=True)
results_df.to_csv('/path/to/output/output.csv', index=False)
