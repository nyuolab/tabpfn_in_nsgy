from ml_utils import load_custom_dataset, get_sens_spec, spiegelhalter_z, plot_shap, plot_shap_feature, boostrap_ci, bootstrap_ci_proba, balance_proba, flexible_shap_analysis

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

# Run classification experiment to compare TabPFN with XGBoost and other classifiers
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
        # Paired p-value for TabPFN vs XGB
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
    # Optionally plot AUROC by training sample size - only useful if trying to see effect of sample size 
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


# Example usage:
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
