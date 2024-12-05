import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

data = pd.read_csv("Final Datasets after Filtering/Reordered_Preprocessed_Labeled_Songs_per_Hot_100_with_Spotify_Features_and_Audio_Features_12291.csv")
data_cols = list(data.columns)

#Feature selection

#1. drop all non-numerical features
non_numeric_cols = ['Artist', 'Song Title', 'Hit', 'Spotify ID', 'Spotify Song Title', 'Spotify Primary Artist', 'Video Title of Audio']
X = data.drop(non_numeric_cols, axis=1)
y = data['Hit']

#1a. split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#2. remove features with a very high correlation
correlation_matrix = X_train.corr()

upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.90)]
X_train = X_train.drop(columns=to_drop)
X_test = X_test.drop(columns=to_drop)

#3. train random forest model with feature importance

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

top_features = feature_importance.head(20)['feature'].tolist()

X_train_top_features = X_train[top_features]
X_test_top_features = X_test[top_features]

#3a. param grid search

# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'learning_rate': [0.05, 0.1, 0.2],
#     'max_depth': [3, 5, 7],
#     'min_child_weight': [1, 3, 5],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0]
# }
#
# grid_search = GridSearchCV(
#     estimator=XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
#     param_grid=param_grid,
#     scoring='accuracy',
#     cv=3,
#     verbose=1,
#     n_jobs=-1
# )
#
# grid_search.fit(X_train_top_features, y_train)
# best_xgb = grid_search.best_estimator_
#
# y_pred = best_xgb.predict(X_test_top_features)
# print("Tuned Accuracy:", accuracy_score(y_test, y_pred))
# print("Best Parameters:", grid_search.best_params_)

#4. retrain model on top features

xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', colsample_bytree=1.0, learning_rate=0.05, max_depth=7, min_child_weight=1, n_estimators=200, subsample=0.8)
xgb.fit(X_train_top_features, y_train)
y_pred = xgb.predict(X_test_top_features)

#5. evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

