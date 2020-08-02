# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:56:15 2020

@author: ShabnamS
"""

import flask
from flask import Flask

import pickle
import numpy as np
import pandas as pd


import pickle
import numpy as np
import pandas as pd

def create_x_test(int_features, comorbidity_cols):
    from pandas.api.types import CategoricalDtype
    cols = ['AGE', "SEX", "RACER", "PAYTYPER",
            'VDAYR', 'VMONTH', 'SENBEFOR', 'REGIONOFF', 'MSA',
            'USETOBAC', 'ETOHAB', 'SUBSTAB',
            'ADD', 'ALZHD', 'ARTHRTIS', 'ASTHMA', 'AUTISM',
            'CAD', 'CANCER', 'CEBVD', 'CHF', 'CKD', 'COPD',
            'DIABTYP2','ESRD','HEPB', 'HEPC', 'HIV', 'HPE', 'HTN','HYPLIPID',
            'OBESITY', 'OSA', 'OSTPRSIS']
    input_variables = pd.DataFrame([int_features],
                                columns=cols)
    input_variables['AGE'] = input_variables['AGE'].astype(CategoricalDtype(categories=['1','2','3','4']))
    input_variables['RACER'] = input_variables['RACER'].astype(CategoricalDtype(categories=['White','Black','Other']))
    input_variables['PAYTYPER'] = input_variables['PAYTYPER'].astype(CategoricalDtype(categories=['Private','Medicare','Medicaid','Other']))
    input_variables['VMONTH'] = input_variables['VMONTH'].astype(CategoricalDtype(categories=['Spring','Summer','Autumn','Winter']))
    input_variables['VDAYR'] = input_variables['VDAYR'].astype(CategoricalDtype(categories=['Saturday','Sunday','Weekday']))
    input_variables['REGIONOFF'] = input_variables['REGIONOFF'].astype(CategoricalDtype(categories=['Northeast','Midwest','South','West']))
    input_variables['USETOBAC'] = input_variables['USETOBAC'].astype(CategoricalDtype(categories=['Not_current','Current']))

    categorical_features = ["AGE", "VMONTH", "VDAYR", "RACER", "PAYTYPER",
                                "USETOBAC", "REGIONOFF"] 
        
    input_variables = pd.get_dummies(input_variables, columns = categorical_features, prefix =categorical_features, dtype=np.int64)
    input_variables = input_variables.astype(int)
    cols2 = ['Total_CD', 'ADD', 'AGE_1', 'AGE_2', 'AGE_3', 'AGE_4',
            'ALZHD', 'ARTHRTIS', 'ASTHMA', 'AUTISM',
            'CAD', 'CANCER', 'CEBVD', 'CHF', 'CKD', 'COPD',
            'DIABTYP2', 'ESRD', 'ETOHAB', 'HEPB', 'HEPC', 'HIV', 'HPE', 'HTN', 'HYPLIPID',
            'MSA', 'OBESITY', 'OSA', 'OSTPRSIS',
            'PAYTYPER_Medicaid', 'PAYTYPER_Medicare', 'PAYTYPER_Other', 'PAYTYPER_Private',
            'RACER_Black', 'RACER_Other', 'RACER_White',
            'REGIONOFF_Midwest', 'REGIONOFF_Northeast', 'REGIONOFF_South', 'REGIONOFF_West', 
            'SENBEFOR', 'SEX', 'SUBSTAB', 'USETOBAC_Current', 'USETOBAC_Not_current',
            'VDAYR_Saturday','VDAYR_Sunday', 'VDAYR_Weekday',
            'VMONTH_Autumn', 'VMONTH_Spring', 'VMONTH_Summer', 'VMONTH_Winter']
    df = pd.DataFrame(np.nan, index=np.arange(1), columns=cols2)
    df = df.combine_first(input_variables)
    df = df.fillna(0).astype(int)
    df['Total_CD'] = df.loc[:, comorbidity_cols].sum(axis=1)
    df=df.reindex(columns=cols2)
    X_test = df.values #.ravel()
    return X_test

# with open('model/deprn_model_rforest.pkl', 'rb') as f:
#     model = pickle.load(f)
# y_scores = (model.predict_proba(X_test)[:, 1] >= 0.15)
# print(y_scores)

demo_cols = ['AGE', "SEX", "RACER", "PAYTYPER",
            'VDAYR', 'VMONTH', 'SENBEFOR', 'REGIONOFF', 'MSA',
            'USETOBAC', 'ETOHAB', 'SUBSTAB']
comorbidity_cols = ['ADD', 'ALZHD', 'ARTHRTIS', 'ASTHMA', 'AUTISM', 'CAD', 'CANCER',
'CEBVD', 'CHF', 'CKD', 'COPD', 'DIABTYP2', 'ESRD', 'HEPB', 'HEPC', 
'HIV', 'HPE', 'HTN', 'HYPLIPID', 'OBESITY', 'OSA', 'OSTPRSIS']
# Use pickle to load in the pre-trained model.
with open(f'model/deprn_model_rforest.pkl', 'rb') as f:
    model = pickle.load(f)


app = flask.Flask(__name__, template_folder='templates')
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        int_features = []
        for i in range(len(demo_cols)):
            feature = flask.request.form.get(demo_cols[i]) 
            int_features.append(feature)
        print(int_features)
        comorbidities = [0 for i in range(len(comorbidity_cols))]
        for i in range(len(comorbidity_cols)):
            if flask.request.form.get(comorbidity_cols[i]) == 'on':
                # checkbox is checked
                comorbidities[i] = 1
            else:
                # checkbox is not checked
                comorbidities[i] = 0
        print(comorbidities)
        print(len(comorbidities))
        int_features.extend(comorbidities)  
        print(len(int_features))
        X_test = create_x_test(int_features, comorbidity_cols)
        y_scores = (model.predict_proba(X_test)[:, 1])
        print(y_scores)
        prediction = np.where(y_scores > 0.2, 'You most pobably have depression, please consult your health care provider',
                                                 'You do not have depression')[0]
        prediction
        # prediction = model.predict(input_variables)[0]
        return flask.render_template('main.html',
                                     result=prediction,
                                     )






if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug = True)

