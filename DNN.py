import pandas as pd
import numpy as np
import argparse
import optuna
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, SGD
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold, train_test_split
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
    
    return X_train, X_test, y_train, y_test


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
    
    return X_train_x, X_test_x, y_train_x,  y_test_x


def objective(trial, X_train, y_train, class_weights): # Objective Function for performing Hyper Parameter Search Optuna with DNN
    n_layers = trial.suggest_int('n_layers', 2, 4)

    n_units_1 = trial.suggest_categorical('n_units_1', [16,32,64,128,256,512])
    n_units_2 = trial.suggest_categorical('n_units_2', [16,32,64,128,256,512])
    n_units_3 = trial.suggest_categorical('n_units_3', [16,32,64,128,256,512])
    n_units_4 = trial.suggest_categorical('n_units_4', [16,32,64,128,256,512])
    units = [n_units_1, n_units_2, n_units_3, n_units_4]

    dropout_1 = trial.suggest_float('dropout_1', 0.0, 0.5)
    dropout_2 = trial.suggest_float('dropout_2', 0.0, 0.5)
    dropout_3 = trial.suggest_float('dropout_3', 0.0, 0.5)
    dropout_4 = trial.suggest_float('dropout_4', 0.0, 0.5)
    dropouts = [dropout_1, dropout_2, dropout_3, dropout_4]

    lr = trial.suggest_float('lr', 0.00001, 0.01)
    optimizer = trial.suggest_categorical('optimizer', [Adam, RMSprop, SGD, Adagrad])
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128, 256])
    n_epochs = trial.suggest_int('n_epochs', 10, 50)
    early_stopping_patience = trial.suggest_int('early_stopping_patience', 3, 6)
    early_stopping_start = trial.suggest_int('early_stopping_start', 5, 10)

    early_stopping = EarlyStopping(patience=early_stopping_patience,
                                   start_from_epoch=early_stopping_start,
                                   restore_best_weights=True)

    skf = StratifiedKFold(n_splits=3)
    accuracy_of_all_folds = []
    precision_of_all_folds = []
    recall_of_all_folds = []

    for (train_index, val_index) in skf.split(X_train, y_train):
        current_X_train, current_X_val = X_train[train_index], X_train[val_index]
        current_y_train, current_y_val = y_train[train_index], y_train[val_index]

        model = Sequential()
        model.add(Input(shape=(X_train.shape[1],)))

        for i in range(n_layers):
            model.add(Dense(units[i], activation='relu'))
            model.add(Dropout(dropouts[i]))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=optimizer(learning_rate=lr),
                    loss='binary_crossentropy',
                    metrics=['accuracy', Precision(), Recall()])

        with tf.device('/GPU:0'):
            _ = model.fit(current_X_train,
                          current_y_train,
                          validation_data=(current_X_val, current_y_val),
                          epochs=n_epochs,
                          batch_size=batch_size,
                          verbose=1,
                          class_weight=class_weights,
                          callbacks=[early_stopping])
        
        current_evaluation = model.evaluate(current_X_val, 
                                            current_y_val,
                                            batch_size=batch_size,
                                            sample_weight=np.array([class_weights[l] for l in current_y_val]))
        accuracy_of_all_folds.append(current_evaluation[1])
        precision_of_all_folds.append(current_evaluation[2])
        recall_of_all_folds.append(current_evaluation[3])

    return np.mean(np.array(accuracy_of_all_folds)) # np.mean(np.array(recall_of_all_folds)) # (2 * (np.mean(np.array(precision_of_all_folds)) * np.mean(np.array(recall_of_all_folds)))) / (np.mean(np.array(precision_of_all_folds)) + np.mean(np.array(recall_of_all_folds))) # np.mean(np.array(accuracy_of_all_folds))


def perform_optuna_search(X_train, y_train, class_weights=None): # Function for performing Hyper Parameter Search Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, class_weights),
                   n_trials=100)

    print(f"Best Optuna Trial: {study.best_trial}")
    print(f"Best Optuna Parameters: {study.best_params}")
    print(f"Best Optuna Trial: {study.best_value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, help="Enter a Run Name")
    parser.add_argument('--cross_validation', action='store_true', default=False, required=False, help="Pass 'cross_validation' to run Cross Validation Evaluation")
    parser.add_argument('--subset_ratios', nargs='+', default=[0.8, 0.2], required=False, help="Choose a Train/Test Ratio. Default is 0.8/0.2")
    parser.add_argument('--random_state', type=int, default=42, required=False, help="Choose a Random State. Default is 42")
    parser.add_argument('--hyper_parameter_search', default='none', choices=['grid_search', 'randomized_search', 'optuna', 'none'], required=False, help="Choose between 'optuna', or 'none'. Default is 'none'.")
    parser.add_argument('--class_imbalance_handling', default='only_weighting', choices=['oversample', 'undersample', 'resample', 'only_weighting'], required=False, help="Choose between 'oversample', or 'undersample', or 'only_weighting'. Default is 'only_weighting'.")
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
            data_df_positives = data_df[data_df['Hit'] == 1].sample(frac=1, ignore_index=True, random_state=args.random_state)
            data_df_negatives = data_df[data_df['Hit'] == 0].sample(frac=1, ignore_index=True, random_state=args.random_state)

            if len(data_df_positives) >= len(data_df_negatives):
                data_df = pd.concat([data_df_positives[:len(data_df_negatives)], data_df_negatives]).sample(frac=1, ignore_index=True, random_state=args.random_state)
            else:
                data_df = pd.concat([data_df_positives, data_df_negatives[:len(data_df_positives)]]).sample(frac=1, ignore_index=True, random_state=args.random_state)
    
        if not args.cross_validation: # Code for Regular Run
            X_train, X_test, y_train, y_test = data_preparation(ds, 
                                                                test_size=float(args.subset_ratios[1]), 
                                                                random_state=args.random_state)

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
            
            # Determine Class Weights

            class_weights = compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(y_train),
                                                 y=y_train)

            class_weights = {class_label: weight for class_label, weight in zip(np.unique(y_train), class_weights)}

            # Convert DFs to Numpy Arrays

            X_train = X_train.values
            y_train = y_train.values
            X_test = X_test.values
            y_test = y_test.values

            # Code for Hyper Parameter Search: Optuna

            if args.hyper_parameter_search == 'optuna':
                perform_optuna_search(X_train,
                                      y_train,
                                      class_weights=class_weights)

            # Train model

            early_stopping = EarlyStopping(patience=6, # Model after Hyper Parameter Search on Accuracy
                                           start_from_epoch=10,
                                           restore_best_weights=True)

            model = Sequential([ 
                Input(shape=(X_train.shape[1],)),
                Dense(512, activation='relu'),
                Dropout(0.4265699166569474),
                Dense(512, activation='relu'),
                Dropout(0.21574232933043488),
                Dense(64, activation='relu'),
                Dropout(0.24849188827329852),
                Dense(256, activation='relu'),
                Dropout(0.08084929509328602),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.0004883871919152422),
                          loss='binary_crossentropy',
                          metrics=['accuracy', Precision(), Recall()])

            with tf.device('/GPU:0'):
                _ = model.fit(X_train,
                              y_train,
                              validation_data=(X_test, y_test),
                              epochs=30,
                              batch_size=128,
                              verbose=0,
                              class_weight=class_weights,
                              callbacks=[early_stopping])

            y_pred_training = (model.predict(X_train, batch_size=128) > 0.5).astype(int)
            y_pred_test = (model.predict(X_test, batch_size=128) > 0.5).astype(int)

            early_stopping = EarlyStopping(patience=6, # Model after Hyper Parameter Search on F1
                                           start_from_epoch=10,
                                           restore_best_weights=True)

            model = Sequential([ 
                Input(shape=(X_train.shape[1],)),
                Dense(512, activation='relu'),
                Dropout(0.2604980282150817),
                Dense(256, activation='relu'),
                Dropout(0.3269106698468404),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer=SGD(learning_rate=0.006036608075718107),
                          loss='binary_crossentropy',
                          metrics=['accuracy', Precision(), Recall()])

            with tf.device('/GPU:0'):
                _ = model.fit(X_train,
                              y_train,
                              validation_data=(X_test, y_test),
                              epochs=48,
                              batch_size=16,
                              verbose=0,
                              class_weight=class_weights,
                              callbacks=[early_stopping])

            y_pred_training = (model.predict(X_train, batch_size=16) > 0.5).astype(int)
            y_pred_test = (model.predict(X_test, batch_size=16) > 0.5).astype(int)

            early_stopping = EarlyStopping(patience=6, # Model after Hyper Parameter Search on Recall
                                           start_from_epoch=8,
                                           restore_best_weights=True)

            model = Sequential([ 
                Input(shape=(X_train.shape[1],)),
                Dense(32, activation='relu'),
                Dropout(0.045428120505839664),
                Dense(256, activation='relu'),
                Dropout(0.1952683906769386),
                Dense(16, activation='relu'),
                Dropout(0.47852128656474174),
                Dense(256, activation='relu'),
                Dropout(0.05415266705456989),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer=SGD(learning_rate=0.009061854198865267),
                          loss='binary_crossentropy',
                          metrics=['accuracy', Precision(), Recall()])

            with tf.device('/GPU:0'):
                _ = model.fit(X_train,
                              y_train,
                              validation_data=(X_test, y_test),
                              epochs=12,
                              batch_size=16,
                              verbose=0,
                              class_weight=class_weights,
                              callbacks=[early_stopping])

            y_pred_training = (model.predict(X_train, batch_size=16) > 0.5).astype(int)
            y_pred_test = (model.predict(X_test, batch_size=16) > 0.5).astype(int)

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
            X_train_x, X_test_x, y_train_x, y_test_x = data_preparation_k_fold(ds, 
                                                                               n_folds=10)
            
            cross_validation_accuracies_train = []
            cross_validation_accuracies_test = []
            cross_validation_precisions = []
            cross_validation_recalls = []
            cross_validation_f1_scores = []
            cm_over_folds = np.zeros((2, 2), dtype=int)

            for i in range(len(X_train_x)):
                current_X_train = X_train_x[i]
                current_X_test = X_test_x[i]
                current_y_train = y_train_x[i]
                current_y_test = y_test_x[i]

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
                
                # Determine Class Weights

                current_class_weights = compute_class_weight(class_weight='balanced',
                                                             classes=np.unique(current_y_train),
                                                             y=current_y_train)

                current_class_weights = {class_label: weight for class_label, weight in zip(np.unique(current_y_train), current_class_weights)}

                # Convert DFs to Numpy Arrays

                current_X_train = current_X_train.values
                current_y_train = current_y_train.values
                current_X_test = current_X_test.values
                current_y_test = current_y_test.values

                # Train model

                # current_early_stopping = EarlyStopping(patience=6, # Model after Hyper Parameter Search on Accuracy
                #                                        start_from_epoch=10,
                #                                        restore_best_weights=True)

                # model = Sequential([ 
                #     Input(shape=(current_X_train.shape[1],)),
                #     Dense(512, activation='relu'),
                #     Dropout(0.4265699166569474),
                #     Dense(512, activation='relu'),
                #     Dropout(0.21574232933043488),
                #     Dense(64, activation='relu'),
                #     Dropout(0.24849188827329852),
                #     Dense(256, activation='relu'),
                #     Dropout(0.08084929509328602),
                #     Dense(1, activation='sigmoid')
                # ])
                
                # model.compile(optimizer=Adam(learning_rate=0.0004883871919152422),
                #               loss='binary_crossentropy',
                #               metrics=['accuracy', Precision(), Recall()])

                # with tf.device('/GPU:0'):
                #     _ = model.fit(current_X_train,
                #                   current_y_train,
                #                   validation_data=(current_X_test, current_y_test),
                #                   epochs=30,
                #                   batch_size=128,
                #                   verbose=0,
                #                   class_weight=current_class_weights,
                #                   callbacks=[current_early_stopping])
                
                # current_y_pred_train = (model.predict(current_X_train, batch_size=128) > 0.5).astype(int)
                # current_y_pred_test = (model.predict(current_X_test, batch_size=128) > 0.5).astype(int)

                current_early_stopping = EarlyStopping(patience=6, # Model after Hyper Parameter Search on F1
                                                       start_from_epoch=10,
                                                       restore_best_weights=True)

                model = Sequential([ 
                    Input(shape=(current_X_train.shape[1],)),
                    Dense(512, activation='relu'),
                    Dropout(0.2604980282150817),
                    Dense(256, activation='relu'),
                    Dropout(0.3269106698468404),
                    Dense(1, activation='sigmoid')
                ])
                
                model.compile(optimizer=SGD(learning_rate=0.006036608075718107),
                              loss='binary_crossentropy',
                              metrics=['accuracy', Precision(), Recall()])

                with tf.device('/GPU:0'):
                    _ = model.fit(current_X_train,
                                  current_y_train,
                                  validation_data=(current_X_test, current_y_test),
                                  epochs=48,
                                  batch_size=16,
                                  verbose=0,
                                  class_weight=current_class_weights,
                                  callbacks=[current_early_stopping])

                current_y_pred_train = (model.predict(current_X_train, batch_size=16) > 0.5).astype(int)
                current_y_pred_test = (model.predict(current_X_test, batch_size=16) > 0.5).astype(int)

                current_early_stopping = EarlyStopping(patience=6, # Model after Hyper Parameter Search on Recall
                                                       start_from_epoch=8,
                                                       restore_best_weights=True)

                # model = Sequential([ 
                #     Input(shape=(current_X_train.shape[1],)),
                #     Dense(32, activation='relu'),
                #     Dropout(0.045428120505839664),
                #     Dense(256, activation='relu'),
                #     Dropout(0.1952683906769386),
                #     Dense(16, activation='relu'),
                #     Dropout(0.47852128656474174),
                #     Dense(256, activation='relu'),
                #     Dropout(0.05415266705456989),
                #     Dense(1, activation='sigmoid')
                # ])
                
                # model.compile(optimizer=SGD(learning_rate=0.009061854198865267),
                #               loss='binary_crossentropy',
                #               metrics=['accuracy', Precision(), Recall()])

                # with tf.device('/GPU:0'):
                #     _ = model.fit(current_X_train,
                #                   current_y_train,
                #                   validation_data=(current_X_test, current_y_test),
                #                   epochs=12,
                #                   batch_size=16,
                #                   verbose=0,
                #                   class_weight=current_class_weights,
                #                   callbacks=[current_early_stopping])

                # current_y_pred_train = (model.predict(current_X_train, batch_size=16) > 0.5).astype(int)
                # current_y_pred_test = (model.predict(current_X_test, batch_size=16) > 0.5).astype(int)

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
            plt.title(f"Confusion Matrix of Deep Neural Network over All {len(cross_validation_accuracies_test)} Folds with {ds_name} Dataset")
            plt.savefig(f"Confusion Matrices/DNN/Confusion_Matrix_{ds_name}.png")
    
    # Save Results to File
    
    if not args.cross_validation:
        pd.DataFrame({"Data Split": data_splits.keys(), 
                      "Training Accuracy": ds_accuracies_training,
                      "Test Accuracy": ds_accuracies_test,
                      "Precision": ds_precisions,
                      "Recall": ds_recalls,
                      "F1": ds_f1s}).to_csv(f"Regular Evaluation Results/DNN/Results_{args.run}.csv", 
                                            index=False)
    else:
        pd.DataFrame({"Data Split": data_splits.keys(), 
                      "Training Accuracy": ds_accuracies_training,
                      "Test Accuracy": ds_accuracies_test,
                      "Precision": ds_precisions,
                      "Recall": ds_recalls,
                      "F1": ds_f1s}).to_csv(f"10-Fold Cross-Validation Results/DNN/Results_{args.run}.csv", 
                                            index=False)


if __name__ == "__main__":
    main()    
