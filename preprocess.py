# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 17:21:27 2020

@author: ShabnamS
"""
import numpy as np
import pandas as pd



def remap_paytyper(data):
    """Remap column
    This function takes in a dataframe column of values and map the values 
    according to a dictionary
    
    Args:
    data(object): dataframe of user and books
    dict_ratings(dictionary): dictionary of rating labels ex: {'col1':{1:'A',2:'B'}}
    
    Returns:
    Pandas DataFrmae(object): dataframe column remapped
    """

    dict_labels = {'PAYTYPER':{
        1: 'Private', #'Private insurance'
        2: 'Medicare', #'Medicare'
        3: 'Medicaid', #'Medicaid'
        4: 'Other', #'Other'
        5: 'Other', #'Self-pay'
        6: 'Other', #'Other'
        7: 'Other', #'Other'
        -8: 'Other', #'Missing'
        -9: 'Other'}}  #'Missing'        
    for field,values in dict_labels.items():
        print("Remapping column %s"%field)
        data.replace({field:values},inplace=True)    
    #data['PAYTYPER'] = data['PAYTYPER'].astype('category')
    print(data['PAYTYPER'].dtype)
    print("Completed")
    return data

def remap_usetobac(data):
    """Remap column
    This function takes in a dataframe column of values and map the values 
    according to a dictionary
    
    Args:
    data(object): dataframe of user and books
    dict_ratings(dictionary): dictionary of rating labels ex: {'col1':{1:'A',2:'B'}}
    
    Returns:
    Pandas DataFrmae(object): dataframe column remapped
    """

    dict_labels = {'USETOBAC':{
        1: 'Not_current', #'Not current'
        2: 'Current', #'Current'
        -8: 'Not_current', #'Other/Missing'
        -9: 'Not_current'}}  #'Other/Missing'
        
    for field,values in dict_labels.items():
        print("Remapping column %s"%field)
        data.replace({field:values},inplace=True)
        print("Completed")
    return data

def remap_injury(data):
    """Remap column
    This function takes in a dataframe column of values and map the values 
    according to a dictionary
    
    Args:
    data(object): dataframe of user and books
    dict_ratings(dictionary): dictionary of rating labels ex: {'col1':{1:'A',2:'B'}}
    
    Returns:
    Pandas DataFrmae(object): dataframe column remapped
    """

    dict_labels = {'INJURY':{
        0: '0', #'No'
        1: '1', #'Yes'
        2: '0', #'Questionable'
        -8: '', #'Other/Missing'
        -9: ''}}  #'Other/Missing'
        
    for field,values in dict_labels.items():
        print("Remapping column %s"%field)
        data.replace({field:values},inplace=True)
        print("Completed")
    return data



def remap_vmonth(data):
    """Remap column
    This function takes in a dataframe column of values and map the values 
    according to a dictionary
    
    Args:
    data(object): dataframe of user and books
    dict_ratings(dictionary): dictionary of rating labels ex: {'col1':{1:'A',2:'B'}}
    
    Returns:
    Pandas DataFrmae(object): dataframe column remapped
    """
    
    bins = [0, 3, 6, 9, 12, np.inf]
    names = ['Winter',
             'Spring',
             'Summer', 
             'Autumn', 
             'Winter2']    
    #print(data['VMONTH'].value_counts())   
    data['VMONTH'] = pd.cut(data['VMONTH'], bins, labels=names, right=False)
    data['VMONTH'] = data['VMONTH'].astype(str).str.replace('Winter2','Winter')
    #print(data['VMONTH'].value_counts())
    return data

def remap_vdayr(data):
    """Remap column
    This function takes in a dataframe column of values and map the values 
    according to a dictionary
    
    Args:
    data(object): dataframe of user and books
    dict_ratings(dictionary): dictionary of rating labels ex: {'col1':{1:'A',2:'B'}}
    
    Returns:
    Pandas DataFrmae(object): dataframe column remapped
    """
    
    bins = [0, 2, 7, 8]
    names = ['Sunday',   # Weekend
             'Weekday', # Weekday
             'Saturday']    
    #print(data['VDAYR'].value_counts())       
    data['VDAYR'] = pd.cut(data['VDAYR'], bins, labels=names, right=False)
    #data['WEEKDAY'] = data['VDAYR']
    #data = data.drop(['VDAYR'], axis=1)
    #print(data['VDAYR'].value_counts())
    return data

def remap_ager(data):
    """Remap column
    This function takes in a dataframe column of values and map the values 
    according to a dictionary
    
    Args:
    data(object): dataframe of user and books
    dict_ratings(dictionary): dictionary of rating labels ex: {'col1':{1:'A',2:'B'}}
    
    Returns:
    Pandas DataFrmae(object): dataframe column remapped
    """
    
    bins = [18, 30, 45, 65, np.inf]
    #print(data['AGE'])
    names = ['1',
             '2',
             '3', 
             '4']    
    print(data['AGE'].value_counts())   
    data['AGE'] = pd.cut(data['AGE'], bins, labels=names, right=False)
    print(data['AGE'].value_counts())
    return data


def remap_racer(data):
    """Remap column
    This function takes in a dataframe column of values and map the values 
    according to a dictionary
    
    Args:
    data(object): dataframe of user and books
    dict_ratings(dictionary): dictionary of rating labels ex: {'col1':{1:'A',2:'B'}}
    
    Returns:
    Pandas DataFrmae(object): dataframe column remapped
    """
    
    bins = [0, 2, 3, 4]
    names = ['White',
             'Black',
             'Other']    
    #print(data['RACER'].value_counts())       
    data['RACER'] = pd.cut(data['RACER'], bins, labels=names, right=False)
    data['RACER'] = data['RACER'].astype(str)
    #print(data['RACER'].value_counts())
    return data



def remap_regionoff(data):
    """Remap column
    This function takes in a dataframe column of values and map the values 
    according to a dictionary
    
    Args:
    data(object): dataframe of user and books
    dict_ratings(dictionary): dictionary of rating labels ex: {'col1':{1:'A',2:'B'}}
    
    Returns:
    Pandas DataFrmae(object): dataframe column remapped
    """
    
    dict_labels = {'REGIONOFF':{
        1: 'Northeast', 
        2: 'Midwest', 
        3: 'South', 
        4: 'West'}}  
        
    for field,values in dict_labels.items():
        #print("Remapping column %s"%field)
        data.replace({field:values},inplace=True)
        #print("Completed")
    data['REGIONOFF'] = data['REGIONOFF'].astype(str)
    return data