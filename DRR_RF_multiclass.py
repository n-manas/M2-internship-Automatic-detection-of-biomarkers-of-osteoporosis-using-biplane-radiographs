import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
import seaborn as sns
import argparse
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import multilabel_confusion_matrix
import pickle
import logging
import traceback
#import tensorflow as tf

# Import features
root_path = "/mnt/SSD/nmanas/"
df = pd.read_excel(root_path + 'In vitro_DRR_features_no_id.xlsx')
df.head()
param = df.columns

# Get ground truth
df_truth = pd.read_excel(root_path + "In vitro_DRR_features_ground truth.xlsx")


# make X and y
X = df.copy()
y = df_truth[['Target']]

# Standard scaler exceptions
exceptions = ["Vertebrae_L1","Vertebrae_L2","Vertebrae_L3","Vertebrae_L4","Sexe"]
cols = X.columns.difference(exceptions).values

numeric_features = X.columns.difference(exceptions).values
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)], remainder="passthrough")

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(random_state=0))])

# Parameters to optimize:
# min_sample_split: = 26 according to test
# Recommended default values are the square root of the total number of features for
# classification problems, and 1/3 the total number for regression problems.
#print("min sample split should be: ", np.sqrt(X.columns.shape[0]))

model_params = {
    'classifier__n_estimators': [40,41,42,43,44,45,46,47,48,49,50],
    'classifier__min_samples_split': [1,2,3,4,5,6,7,8,9,10],
    'classifier__max_leaf_nodes': [None,2,3,4,5,6,7,8,9,10],
    'classifier__min_samples_leaf': [40,41,42,43,44,45,46,47,48,49,50]
}

#Scores used in GridSearch
scoring = {'precision_score': "precision_weighted", "accuracy_score": make_scorer(accuracy_score), "balanced_accuracy_score": "balanced_accuracy", "recall_score": "recall_weighted", "f1_score": "f1_weighted"}

#Score loop for filenames
scores = {"r2_score", "accuracy_score", "balanced_accuracy_score", "recall_score", "f1_score"}


for s in scores:
    print("Fitting model")
    grid_forest = GridSearchCV(pipe, model_params, cv=10, verbose=0, scoring=scoring,return_train_score=True,refit=s)
    grid_forest = grid_forest.fit(X, y.values.ravel())

    print("Best Validation Score: {}".format(grid_forest.best_score_))
    print("Best params: {}".format(grid_forest.best_params_))
    results = pd.DataFrame(grid_forest.cv_results_)

    # save the model to disk
    filename = root_path + 'DRR_RF_' + s + '_multiclass.sav'
    pickle.dump(grid_forest, open(filename, 'wb'))
    excel_path = root_path + "DRR_RF_" + s + "_results_multiclass.xlsx"
    results.to_excel(excel_path)
