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
