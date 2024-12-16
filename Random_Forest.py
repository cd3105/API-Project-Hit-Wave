import pandas as pd
import numpy as np
import argparse
import multiprocessing
import optuna
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def scale_data(X_subset, categorical_cols, scaler=None):
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

    non_numeric_cols = ['Artist', 'Song Title', 'Hit', 'Spotify ID', 'Spotify Song Title', 'Spotify Primary Artist', 'Video Title of Audio']
    X = df.drop(non_numeric_cols, 
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


def data_preparation_k_fold(df, test_size=0.2, n_folds=5, random_state=42):
    # Drop all non-important columns

    non_numeric_cols = ['Artist', 'Song Title', 'Hit', 'Spotify ID', 'Spotify Song Title', 'Spotify Primary Artist', 'Video Title of Audio']
    X = df.drop(non_numeric_cols, 
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

    sss = StratifiedShuffleSplit(n_splits=n_folds, 
                                 test_size=test_size, 
                                 random_state=random_state)

    for (train_index, test_index) in sss.split(X, y):
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
    
    return X_train_x, X_test_x, y_train_x,  y_test_x


def identify_highly_correlated_features(X_train, threshold=0.9):
    correlation_matrix = X_train.corr()
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

    return [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]


def identify_low_variance_features(X_train, categorical_cols, threshold=0.5):
    X_train_non_categorical = X_train.drop(categorical_cols, 
                                           axis=1)
    feature_filter = VarianceThreshold(threshold)
    X_train_non_categorical_filtered = feature_filter.fit_transform(X_train_non_categorical)

    return X_train_non_categorical.columns[list(~np.array(feature_filter.get_support()))]


def perform_grid_search(X_train, y_train, X_test, y_test, sample_class_weights=None, random_state=42):
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [None, 10, 20, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 5, 10],
        'max_features': [None, 'sqrt', 'log2'],
        'bootstrap': [False, True]
    }
    
    model = RandomForestClassifier(random_state=random_state, 
                                   criterion='log_loss')

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='balanced_accuracy',
        cv=3,
        verbose=1,
        n_jobs=multiprocessing.cpu_count()-1
    )

    grid_search.fit(X_train, 
                    y_train, 
                    sample_weight=sample_class_weights)    
    best_rf = grid_search.best_estimator_

    y_pred_train = best_rf.predict(X_train)
    y_pred_test = best_rf.predict(X_test)

    print(f"Best RF Accuracy on Training Set: {accuracy_score(y_train, y_pred_train)}\nBest RF Accuracy on CV: {grid_search.best_score_}\nBest RF Accuracy on Test Set: {accuracy_score(y_test, y_pred_test)}")
    print(f"Best Parameters: {grid_search.best_params_}")


def perform_randomized_search(X_train, y_train, X_test, y_test, sample_class_weights=None, random_state=42):
    param_grid = {
        'n_estimators': np.arange(10, 100, 11),
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': np.arange(2, 11),
        'min_samples_leaf': np.arange(1, 11),
        'max_features': [None, 'sqrt', 'log2'],
        'bootstrap': [False, True]
    }

    model = RandomForestClassifier(random_state=random_state, 
                                   criterion='log_loss')

    randomized_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        scoring='balanced_accuracy',
        n_iter=100,
        cv=3,
        verbose=1,
        n_jobs=multiprocessing.cpu_count()-1
    )

    randomized_search.fit(X_train, 
                          y_train, 
                          sample_weight=sample_class_weights)
    best_rf = randomized_search.best_estimator_

    y_pred_train = best_rf.predict(X_train)
    y_pred_test = best_rf.predict(X_test)

    print(f"Best RF Accuracy on Training Set: {accuracy_score(y_train, y_pred_train)}\nBest RF Accuracy on CV: {randomized_search.best_score_}\nBest RF Accuracy on Test Set: {accuracy_score(y_test, y_pred_test)}")
    print(f"Best Parameters: {randomized_search.best_params_}")

def objective(trial, X_train, y_train, sample_class_weights, random_state=42):
    params = {
        'random_state': random_state,
        'criterion': 'log_loss', 
        'n_estimators': trial.suggest_int('n_estimators', 10, 100),       
        'max_depth': trial.suggest_categorical('max_depth', [None, 10, 20, 30, 40, 50]),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 11), 
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 11), 
        'max_features': trial.suggest_categorical('max_features', [None, 'sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [False, True])
    }
    model = RandomForestClassifier(**params)
    scores = cross_val_score(model, 
                             X_train, 
                             y_train, 
                             cv=3, 
                             fit_params={"sample_weight": sample_class_weights})
    
    return np.mean(scores)

def perform_optuna_search(X_train, y_train, sample_class_weights=None, random_state=42):
    study = optuna.create_study(direction='maximize') 
    study.optimize(lambda trial: objective(trial, X_train, y_train, sample_class_weights, random_state), 
                   n_trials=100)
    
    print(f"Best Optuna Trial: {study.best_trial}")
    print(f"Best Optuna Parameters: {study.best_params}")
    print(f"Best Optuna Trial: {study.best_value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cross_validation', action='store_true', default=False, required=False, help="Pass 'cross_validation' to run Cross Validation Evaluation")
    parser.add_argument('--subset_ratios', nargs='+', default=[0.8, 0.2], required=False, help="Choose a Train/Test Ratio. Default is 0.8/0.2")
    parser.add_argument('--random_state', type=int, default=42, required=False, help="Choose a Random State. Default is 42")
    parser.add_argument('--hyper_parameter_search', default='none', choices=['grid_search', 'randomized_search', 'optuna', 'none'], required=False, help="Choose between 'grid_search', 'randomized_search', 'optuna', or 'none'. Default is 'none'.")
    parser.add_argument('--class_imbalance_handling', default='only_weighting', choices=['oversample', 'undersample', 'resample', 'only_weighting'], required=False, help="Choose between 'oversample', or 'undersample', or 'only_weighting'. Default is 'only_weighting'.")
    args = parser.parse_args()

    # Load in Data

    data_df = pd.read_csv("Final Datasets after Further Filtering\Reordered_Preprocessed_Labeled_Songs_per_Hot_100_with_Spotify_Features_and_Audio_Features_53260.csv")
    
    if args.class_imbalance_handling == 'resample':
        data_df_positives = data_df[data_df['Hit'] == 1].sample(frac=1, ignore_index=True, random_state=args.random_state)
        data_df_negatives = data_df[data_df['Hit'] == 0].sample(frac=1, ignore_index=True, random_state=args.random_state)

        if len(data_df_positives) >= len(data_df_negatives):
            data_df = pd.concat([data_df_positives[:len(data_df_negatives)], data_df_negatives]).sample(frac=1, ignore_index=True, random_state=args.random_state)
        else:
            data_df = pd.concat([data_df_positives, data_df_negatives[:len(data_df_positives)]]).sample(frac=1, ignore_index=True, random_state=args.random_state)
    
    X_train, X_test, y_train, y_test, categorical_cols = data_preparation(data_df, 
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
    
    if args.class_imbalance_handling == 'oversample':
        X_train, y_train = oversampler.fit_resample(X_train, 
                                                    y_train)

    if args.class_imbalance_handling == 'undersample':
        X_train, y_train = undersampler.fit_resample(X_train, 
                                                     y_train)
    
    sample_class_weights = compute_sample_weight(class_weight='balanced', 
                                                 y=y_train)

    if args.hyper_parameter_search == 'grid_search':
        perform_grid_search(X_train, 
                            y_train, 
                            X_test, 
                            y_test,
                            sample_class_weights=sample_class_weights,
                            random_state=args.random_state)

    if args.hyper_parameter_search == 'randomized_search':
        perform_randomized_search(X_train, 
                                  y_train, 
                                  X_test, 
                                  y_test, 
                                  sample_class_weights=sample_class_weights,
                                  random_state=args.random_state)

    if args.hyper_parameter_search == 'optuna':
        perform_optuna_search(X_train, 
                              y_train,
                              sample_class_weights=sample_class_weights,
                              random_state=args.random_state)

    # Train model

    rf = RandomForestClassifier(random_state=args.random_state, 
                                 criterion='gini',
                                 class_weight='balanced',
                                 max_features='sqrt',
                                 min_samples_leaf=9,
                                 min_samples_split=4,
                                 max_depth=10, # Best according to search is 40 but leads to overfitting
                                 n_estimators=200, # Best according to search is 87 but adjusted due to max_depth adjustment
                                 bootstrap=True)
    
    rf.fit(X_train, 
            y_train)
    y_pred_training = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    # Evaluation

    print("\nRegular Evaluation:\n")
    print(f"- Accuracy on Training Set: {accuracy_score(y_train, y_pred_training)}")
    print(f"- Accuracy: {accuracy_score(y_test, y_pred_test)}")
    print(f"- Precision: {precision_score(y_test, y_pred_test)}")
    print(f"- Recall: {recall_score(y_test, y_pred_test)}")
    print(f"- F1 Score: {f1_score(y_test, y_pred_test)}\n\n")

    if args.cross_validation:
        X_train_x, X_test_x, y_train_x, y_test_x = data_preparation_k_fold(data_df, 
                                                                           test_size=float(args.subset_ratios[1]), 
                                                                           random_state=args.random_state)
        
        cross_validation_accuracies_train = []
        cross_validation_accuracies_test = []
        cross_validation_precisions = []
        cross_validation_recalls = []
        cross_validation_f1_scores = []

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
                                                                           categorical_cols)

            current_X_train = current_X_train.drop(columns=current_low_variance_features)
            current_X_test = current_X_test.drop(columns=current_low_variance_features)

            current_oversampler = SMOTE(random_state=args.random_state, 
                                        sampling_strategy=1)
            current_undersampler = RandomUnderSampler(random_state=args.random_state,
                                                      sampling_strategy=1)
            
            if args.class_imbalance_handling == 'oversample':
                current_X_train, current_y_train = current_oversampler.fit_resample(current_X_train, 
                                                                                    current_y_train)

            if args.class_imbalance_handling == 'undersample':
                current_X_train, current_y_train = current_undersampler.fit_resample(current_X_train, 
                                                                                     current_y_train)

            # Train model

            rf = RandomForestClassifier(random_state=args.random_state, 
                                        criterion='gini',
                                        class_weight='balanced',
                                        max_features='sqrt',
                                        min_samples_leaf=9,
                                        min_samples_split=4,
                                        max_depth=10, 
                                        n_estimators=200,
                                        bootstrap=True)
    
            rf.fit(current_X_train, 
                    current_y_train)
            current_y_pred_train = rf.predict(current_X_train)
            current_y_pred_test = rf.predict(current_X_test)

            # Evaluation

            cross_validation_accuracies_train.append(accuracy_score(current_y_train, current_y_pred_train))
            cross_validation_accuracies_test.append(accuracy_score(current_y_test, current_y_pred_test))
            cross_validation_precisions.append(precision_score(current_y_test, current_y_pred_test))
            cross_validation_recalls.append(recall_score(current_y_test, current_y_pred_test))
            cross_validation_f1_scores.append(f1_score(current_y_test, current_y_pred_test))

            print(f"\nCross-Validation Evaluation of Fold {i}:\n")
            print(f"- Accuracy on Training Set: {cross_validation_accuracies_train[i]}")
            print(f"- Accuracy: {cross_validation_accuracies_test[i]}")
            print(f"- Precision: {cross_validation_precisions[i]}")
            print(f"- Recall: {cross_validation_recalls[i]}")
            print(f"- F1 Score: {cross_validation_f1_scores[i]}\n\n")
        
        print(f"\nCross-Validation Evaluation overall Folds:\n")
        print(f"- Mean Accuracy on Training Set: {np.array(cross_validation_accuracies_train).mean()}")
        print(f"- Mean Accuracy: {np.array(cross_validation_accuracies_test).mean()}")
        print(f"- Mean Precision: {np.array(cross_validation_precisions).mean()}")
        print(f"- Mean Recall: {np.array(cross_validation_recalls).mean()}")
        print(f"- Mean F1 Score: {np.array(cross_validation_f1_scores).mean()}\n\n")


if __name__ == "__main__":
    main()    
