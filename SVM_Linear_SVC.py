import pandas as pd
import numpy as np
import argparse
import multiprocessing
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def scale_data(X_subset, categorical_cols, scaler=None): # Function for Scaling Non-Categorical Data
    fit_n_transform = False

    if scaler is None:
        scaler = StandardScaler()
        fit_n_transform = True
    
    X_subset_non_categorical = X_subset.drop(categorical_cols, 
                                             axis=1)
    X_subset_categorical = X_subset[categorical_cols]

    if fit_n_transform:
        X_subset_non_categorical_scaled = pd.DataFrame(scaler.fit_transform(X_subset_non_categorical), 
                                                       columns=X_subset_non_categorical.columns)
    else:
        X_subset_non_categorical_scaled = pd.DataFrame(scaler.transform(X_subset_non_categorical), 
                                                       columns=X_subset_non_categorical.columns)

    X_subset_scaled = pd.concat([X_subset_non_categorical_scaled.reset_index(drop=True), X_subset_categorical.reset_index(drop=True)], 
                                axis=1)

    return X_subset_scaled, scaler


def data_preparation(df, test_size=0.2, random_state=42):
    # Drop all non-important columns

    non_important_cols = ['Artist', 'Song Title', 'Hit', 'Spotify ID', 'Spotify Song Title', 'Spotify Primary Artist', 'Video Title of Audio', 'Release Year']
    X = df.drop(non_important_cols, 
                axis=1)
    y = df['Hit']

    # One-Hot Encode Categorical Variables

    X_categorical = pd.get_dummies(data=X[['Spotify Key']], 
                                   prefix="Spotify Key", 
                                   prefix_sep=" ", 
                                   dtype=int, 
                                   columns=["Spotify Key"])

    X_non_categorical = X.drop('Spotify Key', 
                               axis=1)
    X = pd.concat([X_non_categorical, X_categorical], 
                  axis=1)

    # Split Data

    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=test_size, 
                                                        stratify=y, 
                                                        random_state=random_state)

    # Scale Data

    X_train, data_scaler = scale_data(X_train, X_categorical.columns)
    X_test, _ = scale_data(X_test, 
                           X_categorical.columns, 
                           scaler=data_scaler)
    
    return X_train, X_test, y_train, y_test, X_categorical.columns


def data_preparation_k_fold(df, n_folds=5):
    # Drop all non-important columns

    non_important_cols = ['Artist', 'Song Title', 'Hit', 'Spotify ID', 'Spotify Song Title', 'Spotify Primary Artist', 'Video Title of Audio', 'Release Year']
    X = df.drop(non_important_cols, 
                axis=1)
    y = df['Hit']

    # One-Hot Encode Categorical Variables

    X_categorical = pd.get_dummies(data=X[['Spotify Key']], 
                                   prefix="Spotify Key",  
                                   prefix_sep=" ", 
                                   dtype=int, 
                                   columns=["Spotify Key"])
    X_non_categorical = X.drop('Spotify Key', 
                               axis=1)
    X = pd.concat([X_non_categorical, X_categorical], 
                  axis=1)

    # Split and Scale Data

    X_train_x = []
    y_train_x = []
    X_test_x = []
    y_test_x = []

    skf = StratifiedKFold(n_splits=n_folds)

    for (train_index, test_index) in skf.split(X, y):
        current_X_train = X.loc[train_index].reset_index(drop=True)
        current_y_train = y.loc[train_index].reset_index(drop=True)
        current_X_test = X.loc[test_index].reset_index(drop=True)
        current_y_test = y.loc[test_index].reset_index(drop=True)

        current_X_train, data_scaler = scale_data(current_X_train, 
                                                  X_categorical.columns)
        current_X_test, _ = scale_data(current_X_test, 
                                       X_categorical.columns, 
                                       scaler=data_scaler)

        X_train_x.append(current_X_train)
        y_train_x.append(current_y_train)
        X_test_x.append(current_X_test)
        y_test_x.append(current_y_test)
    
    return X_train_x, X_test_x, y_train_x,  y_test_x, X_categorical.columns


def identify_highly_correlated_features(X_train, threshold=0.9): # Function for removing Highly Correlated Features
    correlation_matrix = X_train.corr()
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

    return [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]


def identify_low_variance_features(X_train, categorical_cols, threshold=0.5): # Function for removing Low Variance Non-Categorical Features
    X_train_non_categorical = X_train.drop(categorical_cols, 
                                           axis=1)
    feature_filter = VarianceThreshold(threshold)
    X_train_non_categorical_filtered = feature_filter.fit_transform(X_train_non_categorical)

    return X_train_non_categorical.columns[list(~np.array(feature_filter.get_support()))]


def perform_grid_search(X_train, y_train, X_test, y_test, random_state=42): # Function for performing Grid Search with Linear SVC
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'max_iter': [1000, 2500, 5000],
        'tol': [1e-4, 1e-3, 1e-2],
        'loss': ['hinge', 'squared_hinge']
    }
    
    model = LinearSVC(random_state=random_state,
                      class_weight='balanced')

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='balanced_accuracy',
        cv=3,
        verbose=4,
        n_jobs=multiprocessing.cpu_count()-1
    )

    grid_search.fit(X_train, 
                    y_train)    
    best_svm = grid_search.best_estimator_

    y_pred_train = best_svm.predict(X_train)
    y_pred_test = best_svm.predict(X_test)

    print(f"Best SVM Accuracy on Training Set: {accuracy_score(y_train, y_pred_train)}\nBest SVM Accuracy on CV: {grid_search.best_score_}\nBest SVM Accuracy on Test Set: {accuracy_score(y_test, y_pred_test)}")
    print(f"Best Parameters: {grid_search.best_params_}")


def perform_randomized_search(X_train, y_train, X_test, y_test, random_state=42): # Function for performing Randomized Search with Linear SVC
    param_grid = {
        'C': np.linspace(0.01, 10, 50),
        'penalty': ['l1', 'l2'],
        'max_iter': np.arrange(500, 2500, 100),
        'tol': np.linspace(1e-3, 1e-1, 20),
        'loss': ['hinge', 'squared_hinge']
    }

    model = LinearSVC(random_state=random_state,
                      class_weight='balanced')

    randomized_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        scoring='balanced_accuracy',
        n_iter=100,
        cv=3,
        verbose=4,
        n_jobs=multiprocessing.cpu_count()-1
    )

    randomized_search.fit(X_train, 
                          y_train)
    best_svm = randomized_search.best_estimator_

    y_pred_train = best_svm.predict(X_train)
    y_pred_test = best_svm.predict(X_test)

    print(f"Best SVM Accuracy on Training Set: {accuracy_score(y_train, y_pred_train)}\nBest SVM Accuracy on CV: {randomized_search.best_score_}\nBest SVM Accuracy on Test Set: {accuracy_score(y_test, y_pred_test)}")
    print(f"Best Parameters: {randomized_search.best_params_}")


def objective(trial, X_train, y_train, random_state=42): # Objective Function for performing Hyper Parameter Search Optuna with Linear SVC
    params = {
        'class_weight':'balanced',
        'random_state': random_state,
        'C': trial.suggest_float('C', 1e-2, 1e2),
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
        'max_iter': trial.suggest_int('max_iter', 500, 5000),
        'tol': trial.suggest_float('tol', 1e-3, 1e-1),
        'loss': trial.suggest_categorical('loss', ['hinge', 'squared_hinge']),
        'dual': trial.suggest_categorical('dual', ['auto']),
    }
    
    if params['penalty'] == 'l1':
        params['dual'] = False
        params['loss'] = 'squared_hinge'

    model = LinearSVC(**params)
    scores = cross_val_score(model, 
                             X_train, 
                             y_train, 
                             cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
                             scoring='f1_weighted') # balanced_accuracy, f1, f1_weighted, recall
    
    return np.mean(scores)


def perform_optuna_search(X_train, y_train, random_state=42): # Function for performing Hyper Parameter Search Optuna
    study = optuna.create_study(direction='maximize') 
    study.optimize(lambda trial: objective(trial, X_train, y_train, random_state), 
                   n_trials=50)
    
    print(f"Best Optuna Trial: {study.best_trial}")
    print(f"Best Optuna Parameters: {study.best_params}")
    print(f"Best Optuna Trial: {study.best_value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, help="Enter a Run Name")
    parser.add_argument('--cross_validation', action='store_true', default=False, required=False, help="Pass 'cross_validation' to run Cross Validation Evaluation")
    parser.add_argument('--subset_ratios', nargs='+', default=[0.8, 0.2], required=False, help="Choose a Train/Test Ratio. Default is 0.8/0.2")
    parser.add_argument('--random_state', type=int, default=42, required=False, help="Choose a Random State. Default is 42")
    parser.add_argument('--hyper_parameter_search', default='none', choices=['grid_search', 'randomized_search', 'optuna', 'none'], required=False, help="Choose between 'grid_search', 'randomized_search', 'optuna', or 'none'")
    parser.add_argument('--class_imbalance_handling', default='weighting', choices=['only_weighting', 'oversample', 'undersample', 'resample'], required=False, help="Choose between 'oversample', 'undersample', or 'only_weighting'")
    args = parser.parse_args()

    # Load in Data

    data_df = pd.read_csv("Final Datasets after Further Filtering\Reordered_Preprocessed_Labeled_Songs_per_Hot_100_with_Spotify_Features_and_Audio_Features_52896.csv")

    # Defining different Data Splits

    data_splits = {'Full': data_df,
                   '1960s': data_df[(data_df['Release Year'] >= 1960) & (data_df['Release Year'] < 1970)].reset_index(drop=True),
                   '1970s': data_df[(data_df['Release Year'] >= 1970) & (data_df['Release Year'] < 1980)].reset_index(drop=True),
                   '1980s': data_df[(data_df['Release Year'] >= 1980) & (data_df['Release Year'] < 1990)].reset_index(drop=True),
                   '1990s': data_df[(data_df['Release Year'] >= 1990) & (data_df['Release Year'] < 2000)].reset_index(drop=True),
                   '2000s': data_df[(data_df['Release Year'] >= 2000) & (data_df['Release Year'] < 2010)].reset_index(drop=True),
                   '2010s': data_df[(data_df['Release Year'] >= 2010) & (data_df['Release Year'] < 2020)].reset_index(drop=True),
                   '2020s': data_df[data_df['Release Year'] >= 2020].reset_index(drop=True)}
    
    ds_accuracies_training = []
    ds_accuracies_test = []
    ds_precisions = []
    ds_recalls = []
    ds_f1s = []
    
    # Loop over all Data Splits

    for ds_name, ds in data_splits.items():
        # Code for Imbalance Handling: Resample

        if args.class_imbalance_handling == 'resample':
            ds_positives = ds[ds['Hit'] == 1].sample(frac=1, ignore_index=True, random_state=args.random_state)
            ds_negatives = ds[ds['Hit'] == 0].sample(frac=1, ignore_index=True, random_state=args.random_state)

            if len(ds_positives) >= len(ds_negatives):
                ds = pd.concat([ds_positives[:len(ds_negatives)], ds_negatives]).sample(frac=1, ignore_index=True, random_state=args.random_state)
            else:
                ds = pd.concat([ds_positives, ds_negatives[:len(ds_positives)]]).sample(frac=1, ignore_index=True, random_state=args.random_state)

        if not args.cross_validation: # Code for Regular Run
            X_train, X_test, y_train, y_test, categorical_cols = data_preparation(ds, 
                                                                                  test_size=float(args.subset_ratios[1]), 
                                                                                  random_state=args.random_state)

            # Remove Features with a Very High Correlation

            highly_correlated_features = identify_highly_correlated_features(X_train)

            X_train = X_train.drop(columns=highly_correlated_features)
            X_test = X_test.drop(columns=highly_correlated_features)

            # Remove Feature with a Very Low Variance

            low_variance_features = identify_low_variance_features(X_train, 
                                                                   categorical_cols)

            X_train = X_train.drop(columns=low_variance_features)
            X_test = X_test.drop(columns=low_variance_features)

            oversampler = SMOTE(random_state=args.random_state, 
                                sampling_strategy=1)
            undersampler = RandomUnderSampler(random_state=args.random_state, 
                                              sampling_strategy=1)
            
            # Code for Imbalance Handling: Oversample

            if args.class_imbalance_handling == 'oversample':
                X_train, y_train = oversampler.fit_resample(X_train, 
                                                            y_train)

            # Code for Imbalance Handling: Undersample

            if args.class_imbalance_handling == 'undersample':
                X_train, y_train = undersampler.fit_resample(X_train, 
                                                             y_train)

            # Code for Hyper Parameter Search: Grid Search

            if args.hyper_parameter_search == 'grid_search':
                perform_grid_search(X_train, 
                                    y_train, 
                                    X_test, 
                                    y_test, 
                                    random_state=args.random_state)
                
            # Code for Hyper Parameter Search: Randomized Search

            if args.hyper_parameter_search == 'randomized_search':
                perform_randomized_search(X_train, 
                                          y_train, 
                                          X_test, 
                                          y_test, 
                                          random_state=args.random_state)

            # Code for Hyper Parameter Search: Optuna

            if args.hyper_parameter_search == 'optuna':
                perform_optuna_search(X_train, 
                                      y_train, 
                                      random_state=args.random_state)

            # Train model

            # svm = LinearSVC(random_state=args.random_state, # After search on Balanced Accuracy with regular CV
            #                 class_weight='balanced',
            #                 C=50.43355830722913,
            #                 penalty='l2',
            #                 max_iter=4685,
            #                 tol=0.008591948409902099,
            #                 loss='squared_hinge',
            #                 dual='auto')

            # svm = LinearSVC(random_state=args.random_state, # After search on F1 with regular CV
            #                 class_weight='balanced',
            #                 C=42.38314585579962,
            #                 penalty='l1',
            #                 max_iter=1839,
            #                 tol=0.0024833156356524784,
            #                 loss='squared_hinge',
            #                 dual='auto')
            
            # svm = LinearSVC(random_state=args.random_state, # After search on Recall with regular CV
            #                 class_weight='balanced',
            #                 C=61.15566004690584,
            #                 penalty='l1',
            #                 max_iter=4079,
            #                 tol=0.007003605145501782,
            #                 loss='squared_hinge',
            #                 dual='auto')  

            svm = LinearSVC(random_state=args.random_state, # After search on F1 Weighted with regular CV
                            class_weight='balanced',
                            C=0.15475743561983535,
                            penalty='l2',
                            max_iter=4682,
                            tol=0.0010381137674891905,
                            loss='hinge',
                            dual='auto')               
            
            svm.fit(X_train, 
                    y_train)
            y_pred_training = svm.predict(X_train)
            y_pred_test = svm.predict(X_test)

            # Regular Evaluation

            accuracy_training = accuracy_score(y_train, y_pred_training)
            accuracy_test = accuracy_score(y_test, y_pred_test)
            precision = precision_score(y_test, y_pred_test)
            recall = recall_score(y_test, y_pred_test)
            f1 = f1_score(y_test, y_pred_test)
            cm = confusion_matrix(y_test, y_pred_test)

            print(f"\nRegular Evaluation for Datasplit {ds_name}:\n")
            print(f"- Accuracy on Training Set: {accuracy_training}")
            print(f"- Accuracy: {accuracy_test}")
            print(f"- Precision: {precision}")
            print(f"- Recall: {recall}")
            print(f"- F1 Score: {f1}")
            print(f"- Confusion Matrix:\n{cm}\n\n")

            ds_accuracies_training.append(accuracy_training)
            ds_accuracies_test.append(accuracy_test)
            ds_precisions.append(precision)
            ds_recalls.append(recall)
            ds_f1s.append(f1)
        else: # Code for Cross-Validation Evaluation
            X_train_x, X_test_x, y_train_x, y_test_x, current_categorical_cols = data_preparation_k_fold(ds, 
                                                                                                         n_folds=10)
            
            cross_validation_accuracies_train = []
            cross_validation_accuracies_test = []
            cross_validation_precisions = []
            cross_validation_recalls = []
            cross_validation_f1_scores = []
            cm_over_folds = np.zeros((2, 2), dtype=int)

            # Loop over all Cross-Validation Splits

            for i in range(len(X_train_x)):
                current_X_train = X_train_x[i]
                current_X_test = X_test_x[i]
                current_y_train = y_train_x[i]
                current_y_test = y_test_x[i]

                # Remove Features with a Very High Correlation

                current_highly_correlated_features = identify_highly_correlated_features(current_X_train)

                current_X_train = current_X_train.drop(columns=current_highly_correlated_features)
                current_X_test = current_X_test.drop(columns=current_highly_correlated_features)

                # Remove Feature with a Very Low Variance

                current_low_variance_features = identify_low_variance_features(current_X_train, 
                                                                               current_categorical_cols)

                current_X_train = current_X_train.drop(columns=current_low_variance_features)
                current_X_test = current_X_test.drop(columns=current_low_variance_features)

                current_oversampler = SMOTE(random_state=args.random_state, 
                                            sampling_strategy=1)
                current_undersampler = RandomUnderSampler(random_state=args.random_state,
                                                        sampling_strategy=1)
                
                # Code for Imbalance Handling: Oversample
                
                if args.class_imbalance_handling == 'oversample':
                    current_X_train, current_y_train = current_oversampler.fit_resample(current_X_train, 
                                                                                        current_y_train)
                    
                # Code for Imbalance Handling: Undersample

                if args.class_imbalance_handling == 'undersample':
                    current_X_train, current_y_train = current_undersampler.fit_resample(current_X_train, 
                                                                                        current_y_train)

                # Train model

                # svm = LinearSVC(random_state=args.random_state, # After search on Balanced Accuracy
                #                 class_weight='balanced',
                #                 C=50.43355830722913,
                #                 penalty='l2',
                #                 max_iter=4685,
                #                 tol=0.008591948409902099,
                #                 loss='squared_hinge',
                #                 dual='auto')
                
                # svm = LinearSVC(random_state=args.random_state, # After search on F1
                #                 class_weight='balanced',
                #                 C=42.38314585579962,
                #                 penalty='l1',
                #                 max_iter=1839,
                #                 tol=0.0024833156356524784,
                #                 loss='squared_hinge',
                #                 dual='auto')
                
                # svm = LinearSVC(random_state=args.random_state, # After search on Recall
                #         class_weight='balanced',
                #         C=61.15566004690584,
                #         penalty='l1',
                #         max_iter=4079,
                #         tol=0.007003605145501782,
                #         loss='squared_hinge',
                #         dual='auto')

                svm = LinearSVC(random_state=args.random_state, # After search on F1 Weighted
                                class_weight='balanced',
                                C=0.15475743561983535,
                                penalty='l2',
                                max_iter=4682,
                                tol=0.0010381137674891905,
                                loss='hinge',
                                dual='auto')
                
                svm.fit(current_X_train, 
                        current_y_train)
                current_y_pred_train = svm.predict(current_X_train)
                current_y_pred_test = svm.predict(current_X_test)

                # Single Split Evaluation

                cross_validation_accuracies_train.append(accuracy_score(current_y_train, current_y_pred_train))
                cross_validation_accuracies_test.append(accuracy_score(current_y_test, current_y_pred_test))
                cross_validation_precisions.append(precision_score(current_y_test, current_y_pred_test))
                cross_validation_recalls.append(recall_score(current_y_test, current_y_pred_test))
                cross_validation_f1_scores.append(f1_score(current_y_test, current_y_pred_test))
                cm_over_folds += confusion_matrix(current_y_test, current_y_pred_test)
            
            # Cross-Validation Evaluation
            
            print(f"\nCross-Validation Evaluation over all {len(cross_validation_accuracies_test)} Folds for Datasplit {ds_name}:\n")
            print(f"- Mean Accuracy on Training Set: {np.array(cross_validation_accuracies_train).mean()}")
            print(f"- Mean Accuracy: {np.array(cross_validation_accuracies_test).mean()}")
            print(f"- Mean Precision: {np.array(cross_validation_precisions).mean()}")
            print(f"- Mean Recall: {np.array(cross_validation_recalls).mean()}")
            print(f"- Mean F1 Score: {np.array(cross_validation_f1_scores).mean()}")
            print(f"- Confusion Matrix:\n{cm_over_folds}\n\n")

            ds_accuracies_training.append(np.array(cross_validation_accuracies_train).mean())
            ds_accuracies_test.append(np.array(cross_validation_accuracies_test).mean())
            ds_precisions.append(np.array(cross_validation_precisions).mean())
            ds_recalls.append(np.array(cross_validation_recalls).mean())
            ds_f1s.append(np.array(cross_validation_f1_scores).mean())

            # Save Confusion Matrix of Data Split to File

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_over_folds, annot=True, 
                        fmt='d', 
                        cmap='Blues', 
                        xticklabels=['No Hit', 'Hit'], 
                        yticklabels=['No Hit', 'Hit'])
            plt.title(f"Confusion Matrix of SVM (Linear SVC) over All {len(cross_validation_accuracies_test)} Folds with {ds_name} Dataset")
            plt.savefig(f"Confusion Matrices/SVM/Confusion_Matrix_{ds_name}.png")

    # Save Results to File
    
    if not args.cross_validation:
        pd.DataFrame({"Data Split": data_splits.keys(), 
                      "Training Accuracy": ds_accuracies_training,
                      "Test Accuracy": ds_accuracies_test,
                      "Precision": ds_precisions,
                      "Recall": ds_recalls,
                      "F1": ds_f1s}).to_csv(f"Regular Evaluation Results/SVM/Results_{args.run}.csv", 
                                            index=False)
    else:
        pd.DataFrame({"Data Split": data_splits.keys(), 
                      "Training Accuracy": ds_accuracies_training,
                      "Test Accuracy": ds_accuracies_test,
                      "Precision": ds_precisions,
                      "Recall": ds_recalls,
                      "F1": ds_f1s}).to_csv(f"10-Fold Cross-Validation Results/SVM/Results_{args.run}.csv", 
                                            index=False)


if __name__ == "__main__":
    main()    
