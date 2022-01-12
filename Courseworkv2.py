# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 19:52:47 2022

@author: Max
"""

from pandas import read_csv
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import random
import time
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, SequentialFeatureSelector
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import Adam


def load_data():
    #import data 
    test_filename = 'C:/Users/Max/Documents/Max/Birkbeck/2nd Year/Applied Machine Learning/Coursework/test_imperson_without4n7_balanced_data.csv'
    test_df = read_csv(test_filename, header = None)
    X_test = test_df.iloc[1:, 0:152].reset_index(drop=True)
    Y_test = test_df.iloc[1:, 152:].reset_index(drop=True)
    
    
    #import training data
    train_filename = 'C:/Users/Max/Documents/Max/Birkbeck/2nd Year/Applied Machine Learning/Coursework/train_imperson_without4n7_balanced_data.csv'
    train_df = read_csv(train_filename, header =None)
    X_train = train_df.iloc[1:, 0:152].reset_index(drop=True)
    Y_train = train_df.iloc[1:, 152:].reset_index(drop=True)
    
    return X_train, X_test, Y_train, Y_test

def normalise(X_train, X_test):
    cols = X_train.columns
    norm = MinMaxScaler()
    X_train = norm.fit_transform(X_train)
    X_test = norm.transform(X_test)
    
    return pd.DataFrame(X_train, columns = cols), pd.DataFrame(X_test, columns = cols)

def standardise(X_train, X_test):
    cols = X_train.columns
    stand = StandardScaler()
    X_train = stand.fit_transform(X_train)
    X_test = stand.transform(X_test)
    
    return pd.DataFrame(X_train, columns = cols), pd.DataFrame(X_test, columns = cols)

def sae(X_train, X_test, code_size = 2, loss = 'binary_crossentropy',
        epochs = 5, optimizer = 'adam'):

    ##Function to implement a stacked autoencoder to
    ## generate features from training dataset
  
    
    input_size = X_train.shape[1] 
    hidden_1_size = input_size//2
    code_size = code_size
    
    stacked_encoder = keras.models.Sequential([
        keras.layers.Dense(input_size, activation = 'relu'),
        keras.layers.Dense(hidden_1_size, activation = 'relu'),
        keras.layers.Dense(code_size, activation = 'relu', activity_regularizer=(l1(10e-6)))])
    
    stacked_decoder = keras.models.Sequential([
        keras.layers.Dense(hidden_1_size, activation = 'relu', input_shape = [code_size]),
        keras.layers.Dense(input_size, activation = 'sigmoid')])
    
    stacked_ae = keras.models.Sequential([stacked_encoder, stacked_decoder])
    stacked_ae.compile(loss=loss,
                       optimizer = optimizer)
    stacked_ae.fit(X_train, X_train, epochs = epochs)
    
    sae_train = pd.DataFrame(stacked_encoder.predict(X_train)).add_prefix('sae_')
    sae_test = pd.DataFrame(stacked_encoder.predict(X_test)).add_prefix('sae_')
    
    X_train = pd.concat((pd.DataFrame(X_train), sae_train), axis = 1)
    X_test = pd.concat((pd.DataFrame(X_test), sae_test), axis = 1)
    
    return X_train, X_test

def k_cluster(X_train, X_test, n_clusters = 12, normalise = True):
    ##Kmeans clustering to generate normalised extra feature  
    kmeans = KMeans(n_clusters = n_clusters, random_state=(7))
    kmeans_train = kmeans.fit(X_train)
    kmeans_test = kmeans.predict(X_test)
    
    cluster_train = pd.DataFrame(kmeans_train.labels_)
    cluster_test = pd.DataFrame(kmeans_test)
    
    if normalise == True:
        X_train['cluster'] = cluster_train/n_clusters
        X_test['cluster'] = cluster_test/n_clusters
    else:
        X_train['cluster'] = cluster_train
        X_test['cluster'] = cluster_test
        
    return X_train, X_test
     
def pca(X_train, X_test, var_explained = .97):
    ##PCA 
    pca_model = PCA(var_explained)
    pca_train_features = pca_model.fit_transform(X_train)
    pca_test_features = pca_model.transform(X_test)
    
    pca_train_features = pd.DataFrame(pca_train_features).add_prefix('pca_')
    pca_test_features = pd.DataFrame(pca_test_features).add_prefix('pca_')
    
    X_train = pd.concat((X_train, pca_train_features), axis = 1)
    X_test = pd.concat((X_test, pca_test_features), axis = 1)
    
    return X_train, X_test

def select_features( X_train, X_test, Y_train, num_features = 50):
    ## Use ExtraTreesclassifier and SelectKBest socring methods to 
    # grade each feature. Allocate score to each methods returned 
    # features in descending order and choose best n number of features

    ##ExtraTreesClassifier to evaluate importance of features  
    model = ExtraTreesClassifier(random_state=7)
    extra_trees_class = model.fit(X_train, Y_train)
    features_normalised = np.std([trees.feature_importances_ for trees
                                  in extra_trees_class.estimators_], axis = 0)
    
    features_normalised = pd.DataFrame(features_normalised, index = X_train.columns)
    ExT_feat_sort = features_normalised.sort_values(by=[0], ascending = False)
    
    ##SelectKBest on training data
    select_best = SelectKBest(k='all')
    best_model = select_best.fit(X_train, Y_train)
    sk_scores = pd.DataFrame(best_model.scores_, index = X_train.columns)
    SKBest_feat_sort = sk_scores.sort_values(by=[0], ascending = False)
    
    scores = pd.DataFrame(0, index = [0], columns=X_train.columns)
    fe_scores = [SKBest_feat_sort, ExT_feat_sort]
    
    for score in fe_scores:
        for i in range(0, len(score[0])):
            feat_name = score.iloc[i].name
            scores.loc[0, feat_name] += i
    scores = scores.sort_values(by=0, axis = 1)
    
    features = scores.iloc[:, 0:num_features]
    
    #Select them from training and test sets
    X_train = X_train.loc[:, features.columns]
    X_test = X_test.loc[:, features.columns]
    
    return X_train, X_test, features


seed_value = 7           
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

overview = []


# =============================================================================
# Logistic Regression model
X_train, X_test, Y_train, Y_test = load_data()

#Cross val and training error 
log_model = LogisticRegression()
kfold = KFold(n_splits=5)
cross_val = cross_val_score(log_model, X_train, Y_train, cv=kfold)
cross_val_mean = cross_val.mean()
print('Log Regression training accuracy:', cross_val_mean)

#Test scores
log_model_fit = log_model.fit(X_train, Y_train)
log_preds = log_model_fit.predict(X_test)
class_report = classification_report(Y_test, log_preds, output_dict=True)
confusion_mat = confusion_matrix(Y_test, log_preds)
print('Logistic Regression testing results',confusion_mat,
      classification_report(Y_test, log_preds))

X_train, X_test = normalise(X_train, X_test)

X_train, X_test = sae(X_train, X_test, code_size = 30)

X_train, X_test = k_cluster(X_train, X_test, n_clusters = 5, normalise = False)

X_train, X_test, features = select_features(X_train, X_test, Y_train, num_features = 5)


#Cross val and training error 
log_model = LogisticRegression()
kfold = KFold(n_splits=5)
cross_val = cross_val_score(log_model, X_train, Y_train, cv=kfold)
cross_val_mean = cross_val.mean()
print('Log Regression training accuracy:', cross_val_mean)

#Test scores
log_model_fit = log_model.fit(X_train, Y_train)
log_preds = log_model_fit.predict(X_test)
class_report = classification_report(Y_test, log_preds, output_dict=True)
confusion_mat = confusion_matrix(Y_test, log_preds)
print('Logistic Regression testing results',confusion_mat,
      classification_report(Y_test, log_preds))

overview.append(['logistic reg', cross_val_mean, class_report['accuracy']])
# =============================================================================


# =============================================================================
# Support vector classifier
X_train, X_test, Y_train, Y_test = load_data()

X_train, X_test = normalise(X_train, X_test)

X_train, X_test = sae(X_train, X_test, code_size = 5)

#X_train, X_test = k_cluster(X_train, X_test, n_clusters = 5, normalise = False)

X_train, X_test, features = select_features(X_train, X_test, Y_train, num_features = 5)


#Cross val and training error 
svc_model_1 = SVC(random_state=(7))
kfold = KFold(n_splits=3)
cross_val = cross_val_score(svc_model_1, X_train, Y_train, cv=kfold)
cross_val_mean = cross_val.mean()
print('Support vector training accuracy:', cross_val_mean)

#Test scores
svc_model_1 = svc_model_1.fit(X_train, Y_train)
svc_1_preds = svc_model_1.predict(X_test)
class_report = classification_report(Y_test, svc_1_preds, output_dict=True)
confusion_mat = confusion_matrix(Y_test, svc_1_preds)
print('Support vector testing results',confusion_mat,
      classification_report(Y_test, svc_1_preds))

# =============================================================================


# =============================================================================
# Random Forrest 
X_train, X_test, Y_train, Y_test = load_data()

#Cross val and training error 
rf_model_1 = RandomForestClassifier(random_state=(7))
kfold = KFold(n_splits=5)
cross_val = cross_val_score(rf_model_1, X_train, Y_train, cv=kfold)
cross_val_mean = cross_val.mean()
print('Random Forrest training accuracy:', cross_val_mean)

#Test scores
rf_model_1_fit = rf_model_1.fit(X_train, Y_train)
rf_1_preds = rf_model_1_fit.predict(X_test)
class_report = classification_report(Y_test, rf_1_preds, output_dict=True)
confusion_mat = confusion_matrix(Y_test, rf_1_preds)
print('Random Forrest testing results',confusion_mat,
      classification_report(Y_test, rf_1_preds))

X_train, X_test = normalise(X_train, X_test)

X_train, X_test = sae(X_train, X_test, code_size = 30, epochs = 5)

X_train, X_test, features = select_features(X_train, X_test, Y_train, num_features = 5)

rf_model_2 = RandomForestClassifier(random_state=(7))
kfold = KFold(n_splits=5)
cross_val = cross_val_score(rf_model_2, X_train, Y_train, cv=kfold)
cross_val_mean = cross_val.mean()
print('Random Forrest training accuracy:', cross_val_mean)

rf_model_2 = rf_model_2.fit(X_train, Y_train)
rf_preds_2 = rf_model_2.predict(X_test)
class_report = classification_report(Y_test, rf_preds_2)
confusion_mat = confusion_matrix(Y_test, rf_preds_2)
print(class_report)
print(confusion_mat)

# # =============================================================================

# =============================================================================
# Multi layer perceptron 
X_train, X_test, Y_train, Y_test = load_data()

X_train, X_test = normalise(X_train, X_test)

X_train, X_test = sae(X_train, X_test, code_size = 3, epochs = 10)


X_train, X_test, features = select_features(X_train, X_test, Y_train, num_features = 5)


mlp_model = MLPClassifier(hidden_layer_sizes =(2,1),
                                            activation = 'tanh',
                                            solver='adam',
                                            random_state=(7))

params = {'activation': ('relu', 'tanh'), 
          'alpha':(0.0001, 0.001, 0.01),
          'batch_size': (200, 500, 2000)}

clf = GridSearchCV(mlp_model, params, cv = 4)
grid = clf.fit(X_train, Y_train)

#Cross val and training error 
kfold = KFold(n_splits=5)
cross_val = cross_val_score(mlp_model, X_train, Y_train, cv=kfold)
cross_val_mean = cross_val.mean()
print('MLP training accuracy:', cross_val_mean)

#Test scores
mlp_model_fit = mlp_model.fit(X_train, Y_train)
mlp_preds = mlp_model_fit.predict(X_test)
class_report = classification_report(Y_test, mlp_preds, output_dict=True)
confusion_mat = confusion_matrix(Y_test, mlp_preds)
print('MLP testing results',confusion_mat,
      classification_report(Y_test, mlp_preds))

# =============================================================================














