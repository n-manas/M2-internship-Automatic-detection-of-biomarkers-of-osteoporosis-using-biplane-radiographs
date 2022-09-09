import os
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import multilabel_confusion_matrix
from imblearn.pipeline import make_pipeline
import pickle

# Import features
root_path = "/mnt/SSD/nmanas/"
df = pd.read_excel(root_path + 'In vitro_DRR_features_no_id.xlsx')
df2 = pd.read_excel(root_path + 'In vitro_DRR_all_data.xlsx')
df.head()
param = df.columns

# make X and y
X = df.copy()

# Choose variable for regressor
#y = df2[['densite_trabeculaire_mg_cm3']]
#y = df2[['Resistance_anterieure']]
y = df2[['Resistance_uniaxiale']]

# Standard scaler exceptions
exceptions = ["Vertebrae_L1", "Vertebrae_L2",
    "Vertebrae_L3", "Vertebrae_L4", "Sexe"]
cols = X.columns.difference(exceptions).values

numeric_features = X.columns.difference(exceptions).values
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)], remainder="passthrough")

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
pipe = Pipeline(steps=[('preprocessor', preprocessor),
                       ('classifier', RandomForestRegressor(random_state=0))])

# Parameters to optimize:
# min_sample_split: = 26 according to test
# Recommended default values are the square root of the total number of features for
# classification problems, and 1/3 the total number for regression problems.
#print("min sample split should be: ", np.sqrt(X.columns.shape[0]))


model_params = {
    'classifier__n_estimators': [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
    'classifier__min_samples_split': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'classifier__max_leaf_nodes': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'classifier__min_samples_leaf': [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
}

#Scores used in GridSearch
scoring = {'precision_score': "precision_weighted", "accuracy_score": make_scorer(accuracy_score), "balanced_accuracy_score": "balanced_accuracy", "recall_score": "recall_weighted", "f1_score": "f1_weighted"}

#Score loop for filenames
scores = {"r2_score", "accuracy_score", "balanced_accuracy_score", "recall_score", "f1_score"}


for s in scores:
    print("Fitting model")
    grid_forest = GridSearchCV(
        pipe, model_params, cv=10, verbose=0, scoring=scoring, return_train_score=True, refit=s)
    grid_forest = grid_forest.fit(X, y.values.ravel())

    print("Best Validation Score: {}".format(grid_forest.best_score_))            
    print("Best params: {}".format(grid_forest.best_params_))
    results = pd.DataFrame(grid_forest.cv_results_)
    # save the model to disk (select correct filename
    #filename = root_path + 'DRR_RF_reg_test_' + s + '_d_trabec.sav'
    #filename = root_path + 'DRR_RF_reg_test_' + s + '_r_anterieure.sav'
    filename = root_path + 'DRR_RF_reg_' + s + '_r_uniaxiale.sav'
    pickle.dump(grid_forest, open(filename, 'wb'))
    #excel_path = root_path + "DRR_RF_reg_test_" + s + "_results_d_trabec.xlsx"
    #excel_path = root_path + "DRR_RF_reg_test_" + s + "_results_r_anterieure.xlsx"
    excel_path = root_path + "DRR_RF_reg_" + s + "_results_r_uniaxiale.xlsx"
    results.to_excel(excel_path)
