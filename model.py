# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:23:54 2020

@author: ShabnamS
"""

# Importing the libraries
import numpy as np
import pandas as pd
# import re
# import time
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
# from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
import seaborn as sns
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.metrics import make_scorer

from preprocess import *


def read_dataset(filename):   
    df = pd.read_stata(filename,convert_categoricals=False)
    sr = pd.io.stata.StataReader(filename)
    labels = sr.value_labels()
    sr.close()
    #df['IDSEQ'] = 'NAMCS2016_' + df.index.astype(str).map(lambda x: f'{x:0>6}') 
    selected_vars = ["VMONTH", "VDAYR", "AGE", "SEX", "RACER", "PAYTYPER",
                     "USETOBAC",'SENBEFOR',"ETOHAB", "ALZHD", "ARTHRTIS",
                     "ASTHMA", "ADD", "AUTISM", "CANCER", "CEBVD", "CKD",
                     "COPD", "CHF", "CAD", "DEPRN", "DIABTYP2", "ESRD", "HEPB",
                     "HEPC", "HPE", "HIV", "HYPLIPID", "HTN", "OBESITY",
                     "OSA", "OSTPRSIS", "SUBSTAB", "REGIONOFF", "MSA"]    
    df = df[selected_vars]
    df = remap_paytyper(df)
    df = remap_usetobac(df)
    df = remap_injury(df)
    df = remap_vmonth(df)
    df = remap_vdayr(df)
    df = remap_racer(df)
    df = remap_regionoff(df)
    df = df[df['AGE'].astype(int) >= 18]
    df = remap_ager(df)
    df['SEX'] = df['SEX'].map( {1:1, 2:0})
    df['SENBEFOR'] = df['SENBEFOR'].map( {1:1, 2:0})
    df['MSA'] = df['MSA'].map( {1:1, 2:0})

    return df, labels

def convert_categorical_to_dummies(df):
    categorical_features = ["AGE", "VMONTH", "VDAYR", "RACER", "PAYTYPER",
                            "USETOBAC", "REGIONOFF"] 
    
    df = pd.get_dummies(df, columns = categorical_features, prefix =categorical_features, dtype=np.int64)
    return df


namcs2016, labels = read_dataset('../namcs_predict_depression/namcs2016-stata.dta')
namcs2016['Total_CD'] = namcs2016.drop('DEPRN',axis=1).loc[:, "ALZHD":"SUBSTAB"].sum(axis=1)
#namcs2015, labels = read_dataset('namcs2015-stata.dta')
print(namcs2016['DEPRN'].mean())
namcs2016 = convert_categorical_to_dummies(namcs2016)

#from custom_transformers import PassthroughTransformer
from sklearn.pipeline import FeatureUnion
def set_features_target(df):
    # Importing the dataset
    X = df.loc[:, df.columns != 'DEPRN']
    y = df.loc[:, df.columns == 'DEPRN'].values.ravel()    
    return X, y

def scale_numeric_column(X):
    #print(X.columns)
    numeric_features = ['Total_CD']
    # categorical_features = X.columns.difference(['Total_CD']).to_list()
    # print(categorical_features)
    numeric_transformer = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='median')),
      ('scaler', StandardScaler())])
    
    other_features = X.columns.difference(['Total_CD']).to_list()
    other_transformer = 'passthrough'

    feature_names = numeric_features + other_features
    
    preprocessor = ColumnTransformer(
      transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('other', other_transformer, other_features),
                    #('cat', OneHotEncoder(handle_unknown='error'), categorical_features),
                    ], remainder = 'passthrough' )

    clf = Pipeline(steps=[('preprocessor', preprocessor)])
    X = clf.fit_transform(X)
    #preprocessor.get_feature_names()
    #print(preprocessor.named_transformers_['other'].get_feature_names(input_features=other_features))
    # feature_names = (clf.named_steps['preprocessor']
    #                   .named_transformers_['other']
    #                   .get_feature_names(input_features=other_features))

    # print("Output feature names")
    # print(feature_names)
    # print("Number of output feature names")
    # print(len(feature_names))
    return X, feature_names

def build_test_train_set(df):
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                        random_state = 5,
                                                        shuffle=True,
                                                        stratify=y)
    print(np.mean(y_train))
    print(np.mean(y_test))
    return X, y, X_train, X_test, y_train, y_test

# for name,step in pipeline.named_steps.items():
#     if hasattr(step, 'get_feature_names'):
#         print(step.get_feature_names())


namcs2016.dtypes
X, y = set_features_target(namcs2016)
print(X.shape, y.shape)
X, feature_names = scale_numeric_column(X)
print(X.shape, y.shape)
print(X)
print(feature_names)
X, y, X_train, X_test, y_train, y_test = build_test_train_set(namcs2016)

print(X[:, 2].sum(), namcs2016['AGE_1'].sum())

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

rfbase = RandomForestClassifier(n_jobs = 3)
cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=1)

param_grid = {
    'n_estimators': [500],
    'max_features': [0.2, 0.25],
    'bootstrap'   : [True], #[False, True],
    "criterion"   : ["entropy"], #["gini", "entropy"],
   # 'class_weight' : [{0:0.5, 1: 1}],# [{0:x, 1: 1} for x in np.arange(0,1,0.1)] ,              
    'max_depth'   : [9, 10, 11]
}


custom_scorer =  {'acc'       : 'balanced_accuracy',
                  'roc_auc'   : 'roc_auc',
                  'log_loss'  : 'neg_log_loss'}
rf_fit = GridSearchCV(estimator=rfbase, param_grid=param_grid, cv = cv,
                      scoring = custom_scorer,
                    refit='roc_auc', return_train_score=True)

grid_result = rf_fit.fit(X_train, y_train)
# report the best configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# report all configurations
final_model = grid_result.best_estimator_
best_result = grid_result.best_score_
print(best_result)

pd.DataFrame(grid_result.cv_results_)[['mean_test_log_loss',
                                       'mean_test_roc_auc',
                                   'param_n_estimators',
                                   'param_max_features']]

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    plt.figure(figsize=(4,4))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")

    
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')

y_scores = (final_model.predict_proba(X_test)[:, 1])# >= 0.15)

p, r, thresholds = precision_recall_curve(y_test, y_scores)
plot_precision_recall_vs_threshold(p, r, thresholds)

def plot_roc_curve(fpr, tpr):
    """
    The ROC curve, modified from 
    Hands-On Machine learning with Scikit-Learn and TensorFlow; p.91
    """
    plt.figure(figsize=(4,4))
    plt.title('ROC Curve')
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0, 1.01, 0.2), rotation=0)
    plt.xlabel("False Positive Rate", size=13)
    plt.ylabel("True Positive Rate (Recall)", size=13)
    plt.legend(loc='best')

fpr, tpr, auc_thresholds = roc_curve(y_test, y_scores)
print(auc(fpr, tpr)) # AUC of ROC
plot_roc_curve(fpr, tpr)

def plot_cm(y_test, y_scores, label=None):
        plt.figure(figsize=[4,4])
        cm = confusion_matrix(y_test, y_scores)
        plt.subplot(111)
        ax = sns.heatmap(cm, annot=True, cmap='summer_r', cbar=False, 
                    annot_kws={"size": 14}, fmt='g')
        cmlabels = ['True Negatives', 'False Positives',
                  'False Negatives', 'True Positives']
        for i,t in enumerate(ax.texts):
            t.set_text(t.get_text() + "\n" + cmlabels[i])
        plt.title('Confusion Matrix', size=15)
        plt.xlabel('Predicted Values', size=13)
        plt.ylabel('True Values', size=13)

plot_cm(y_test, final_model.predict(X_test), label=None)
def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]

def precision_recall_threshold(p, r, thresholds, t=0.5):
    """
    plots the precision recall curve and shows the current value for each
    by identifying the classifier's threshold (t).
    """
    
    # generate new class predictions based on the adjusted_classes
    # function above and view the resulting confusion matrix.
    y_pred_adj = adjusted_classes(y_scores, t)
    print(pd.DataFrame(confusion_matrix(y_test, y_pred_adj),
                       columns=['pred_neg', 'pred_pos'], 
                       index=['neg', 'pos']))
    close_default_clf = np.argmin(np.abs(thresholds - t))
    print(r[close_default_clf], p[close_default_clf])
    close_default_clf = np.argmin(np.abs(thresholds - t))


for i in np.arange(0.1,0.8,0.1):  
  precision_recall_threshold(p, r, thresholds, i)
  
gmeans = np.sqrt(tpr * (1-fpr))
ix = np.argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))


classifier = RandomForestClassifier(n_estimators = 500,
                                    max_depth = 11,
                                    max_features = 0.2,
                                    bootstrap = True,
                                    criterion = 'entropy',
                                    random_state = 0)
classifier.fit(X_train, y_train)
import pickle
with open('model/deprn_model_rforest.pkl', 'wb') as file:
    pickle.dump(classifier, file)

