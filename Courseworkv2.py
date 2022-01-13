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
from sklearn.decomposition import PCA, FastICA
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

seed_value = 7           
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)
tf.keras.backend.set_floatx('float64')


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

def delete_zero_var(X_train, X_test):
    ##Delete any features that have a variance of 0.
    list_to_remove = []
    for i in range(0, X_train.shape[1]):
        if (X_train.var()[X_train.columns[i]] == 0) == True:
            list_to_remove.append(i)
            
    new_X_train = X_train.drop(X_train.columns[list_to_remove], axis = 1)
    new_X_test = X_test.drop(X_test.columns[list_to_remove], axis = 1)
    
    return new_X_train, new_X_test

def correlation_remove(X_train, X_test):
    ##Deletes any features that have a correlation of 1 with another 
    ## feature, implying they are identical to each other. 
    correlations = X_train.corr()
    list_of_cols = set()
    
    increment = 0
    
    for i in range(0, correlations.shape[0]):
        for j in range(increment, correlations.shape[0]):
            if correlations.index[j] == correlations.columns[i]:
                pass
            else:
                if correlations.values[j, i] == 1.0:
                    list_of_cols.add(correlations.index[j])
            
        increment += 1
                    
    X_train = X_train.drop(X_train[list_of_cols], axis = 1)
    X_test = X_test.drop(X_test[list_of_cols], axis = 1)
            
    return X_train, X_test

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

def sae_deep(X_train, X_test, code_size = 2, loss = 'binary_crossentropy',
        epochs = 5, optimizer = 'adam'):

    ##Function to implement a stacked autoencoder to
    ## generate features from training dataset
    
    X_train_clean, X_test_clean = delete_zero_var(X_train, X_test)
    X_train_clean, X_test_clean = correlation_remove(X_train_clean, X_test_clean)
    
    input_size = X_train_clean.shape[1] 
    neuron_diff = (input_size-code_size)/3
    hidden_1_size = input_size-neuron_diff
    hidden_2_size = hidden_1_size-neuron_diff
    
    
    code_size = code_size
    
    stacked_encoder = keras.models.Sequential([
        keras.layers.Dense(input_size, activation = 'relu'),
        keras.layers.Dense(hidden_1_size, activation = 'relu'),
        keras.layers.Dense(hidden_2_size, activation = 'relu'),
        keras.layers.Dense(code_size, activation = 'relu', activity_regularizer=(l1(10e-6)))])
    
    stacked_decoder = keras.models.Sequential([
        keras.layers.Dense(hidden_2_size, activation = 'relu', input_shape = [code_size]),
        keras.layers.Dense(hidden_1_size, activation = 'relu'),
        keras.layers.Dense(input_size, activation = 'sigmoid')])
    
    stacked_ae = keras.models.Sequential([stacked_encoder, stacked_decoder])
    stacked_ae.compile(loss=loss,
                       optimizer = optimizer)
    stacked_ae.fit(X_train_clean, X_train_clean, epochs = epochs)
    
    sae_train = pd.DataFrame(stacked_encoder.predict(X_train_clean)).add_prefix('sae_')
    sae_test = pd.DataFrame(stacked_encoder.predict(X_test_clean)).add_prefix('sae_')
    
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

def cross_val(model, X_train, Y_train, n_splits = 5):
  kfold = KFold(n_splits=n_splits)
  c_val = cross_val_score(model, X_train, Y_train, cv=kfold)
  return c_val.mean()

def test_score(model, X_test, Y_test):
  model_preds = model.predict(X_test)
  class_report = classification_report(Y_test, model_preds, output_dict= True)
  confusion_mat = confusion_matrix(Y_test, model_preds)
  return class_report, confusion_mat


# =============================================================================
# Investigation into best feature generation techniques
X_train, X_test, Y_train, Y_test = load_data()

X_train_norm, X_test_norm = normalise(X_train, X_test)

X_train_stand, X_test_stand = standardise(X_train, X_test)

X_train_pca, X_test_pca = pca(X_train_stand, X_test_stand, var_explained=.9)

X_train_sae, X_test_sae = sae(X_train_norm, X_test_norm, code_size=30,
                              epochs = 10)

X_train_clus, X_test_clus = k_cluster(X_train_sae.iloc[:,-30],
                                      X_test_sae.iloc[:,-30], 
                                      n_clusters=5)

ica = FastICA(n_components=4).fit(X_train_stand)
X_train_ica = pd.concat((X_train_stand, pd.DataFrame(ica.transform(X_train_stand)).add_prefix('ica_')), axis = 1)
X_test_ica = pd.concat((X_test_stand, pd.DataFrame(ica.transform(X_test_stand)).add_prefix('ica_')), axis = 1)

names = [['pca', X_train_pca, X_test_pca],
        ['sae', X_train_sae, X_test_sae], 
        ['kmeans sae', X_train_clus, X_test_clus],
        ['ica', X_train_ica, X_test_ica]]

feat_gen = []

for name in names:
  selectkbest = SelectKBest(k=5).fit(name[1], Y_train)
  cols = selectkbest.get_support(indices = True)
  x_train = name[1].iloc[:,cols]
  x_test = name[2].iloc[:, cols]
  features = x_train.columns
  log_model = LogisticRegression().fit(x_train, Y_train)
  cv = cross_val(log_model, x_train, Y_train, n_splits = 4)
  class_report, confusion_mat = test_score(log_model, x_test, Y_test)
  feat_gen.append([name[0], features.values, cv, class_report['accuracy'], confusion_mat[0][1]])

# =============================================================================

# =============================================================================
# Generate features for training 

X_train, X_test, Y_train, Y_test = load_data()

X_train_norm, X_test_norm = normalise(X_train, X_test)

X_train_sae, X_test_sae = sae(X_train_norm, X_test_norm, code_size =30,
                              epochs = 10)


overview = []
# =============================================================================


# =============================================================================
# Logistic Regression model

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

overview.append(['baseline logistic reg', 'all', cross_val_mean, class_report['accuracy'], confusion_mat[0][1]])


X_train_log, X_test_log, features = select_features(X_train_sae, X_test_sae, Y_train, num_features = 5)


#Cross val and training error 
cross_val = cross_val_score(log_model, X_train_log, Y_train, cv=kfold)
cross_val_mean = cross_val.mean()
print('Log Regression training accuracy:', cross_val_mean)

#Test scores
log_model_fit = log_model.fit(X_train_log, Y_train)
log_preds = log_model_fit.predict(X_test_log)
class_report = classification_report(Y_test, log_preds, output_dict=True)
confusion_mat = confusion_matrix(Y_test, log_preds)
print('Logistic Regression testing results',confusion_mat,
      classification_report(Y_test, log_preds))

overview.append(['logistic reg w sae + k', features.columns.values, cross_val_mean, class_report['accuracy'], confusion_mat[0][1]])
# =============================================================================


# =============================================================================
# Support vector classifier

X_train_svc, X_test_svc, features = select_features(X_train_sae, X_test_sae, Y_train, num_features = 5)


#Cross val and training error 
svc_model_1 = SVC(random_state=(7))
kfold = KFold(n_splits=3)
cross_val = cross_val_score(svc_model_1, X_train_svc, Y_train, cv=kfold)
cross_val_mean = cross_val.mean()
print('Support vector training accuracy:', cross_val_mean)

#Test scores
svc_model_1 = svc_model_1.fit(X_train_svc, Y_train)
svc_1_preds = svc_model_1.predict(X_test_svc)
class_report = classification_report(Y_test, svc_1_preds, output_dict=True)
confusion_mat = confusion_matrix(Y_test, svc_1_preds)
print('Support vector testing results',confusion_mat,
      classification_report(Y_test, svc_1_preds))

overview.append(['svc with sae', features.columns.values, cross_val_mean, class_report['accuracy'], confusion_mat[0][1]])

# =============================================================================


# =============================================================================
# Random Forrest 

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

overview.append(['baseline rf', features.columns.values, cross_val_mean, class_report['accuracy'], confusion_mat[0][1]])


X_train_rf, X_test_rf, features = select_features(X_train_sae, X_test_sae, Y_train, num_features = 5)

rf_model_2 = RandomForestClassifier(random_state=(7))
kfold = KFold(n_splits=5)
cross_val = cross_val_score(rf_model_2, X_train_rf, Y_train, cv=kfold)
cross_val_mean = cross_val.mean()
print('Random Forrest training accuracy:', cross_val_mean)

rf_model_2 = rf_model_2.fit(X_train_rf, Y_train)
rf_preds_2 = rf_model_2.predict(X_test_rf)
class_report = classification_report(Y_test, rf_preds_2, output_dict=True)
confusion_mat = confusion_matrix(Y_test, rf_preds_2)

overview.append(['rf w sae', features.columns.values, cross_val_mean, class_report['accuracy'], confusion_mat[0][1]])


# # =============================================================================

# =============================================================================
# Multi layer perceptron 

X_train_mlp, X_test_mlp, features = select_features(X_train_sae, X_test_sae, Y_train, num_features = 5)


mlp_model = MLPClassifier(hidden_layer_sizes =(2,1),
                                            activation = 'tanh',
                                            solver='adam',
                                            random_state=(7))

# params = {'activation': ('relu', 'tanh'), 
#           'alpha':(0.0001, 0.001, 0.01),
#           'batch_size': (200, 500, 2000)}

# clf = GridSearchCV(mlp_model, params, cv = 4)
# grid = clf.fit(X_train, Y_train)

#Cross val and training error 
kfold = KFold(n_splits=5)
cross_val = cross_val_score(mlp_model, X_train_mlp, Y_train, cv=kfold)
cross_val_mean = cross_val.mean()
print('MLP training accuracy:', cross_val_mean)

#Test scores
mlp_model_fit = mlp_model.fit(X_train_mlp, Y_train)
mlp_preds = mlp_model_fit.predict(X_test_mlp)
class_report = classification_report(Y_test, mlp_preds, output_dict=True)
confusion_mat = confusion_matrix(Y_test, mlp_preds)
print('MLP testing results',confusion_mat,
      classification_report(Y_test, mlp_preds))

overview.append(['mlp w sae', features.columns.values, cross_val_mean, class_report['accuracy'], confusion_mat[0][1]])


# =============================================================================

print(pd.DataFrame(overview, columns = ['name', 'features', 'train_cv_acc', 'test_acc', 'fp']))


# =============================================================================
# SVC model refinement 

X_train_sae_d, X_test_sae_d = sae_deep(X_train_norm, X_test_norm, 
                                        code_size = 30,
                                        epochs = 10)
X_train_svc, X_test_svc, features = select_features(X_train_sae_d, 
                                                    X_test_sae_d,
                                                    Y_train,
                                                    num_features = 5)

params = {'C': (0.1, 1, 10), 
          'kernel':('poly', 'rbf', 'sigmoid'),
          'degree': (3,4,5),
          'gamma': ('scale', 'auto')}


#Cross val and training error 
svc_model_1 = SVC(random_state=(7))
kfold = KFold(n_splits=3)
clf = GridSearchCV(svc_model_1, params, cv = kfold)
grid = clf.fit(X_train_svc, Y_train)

# cross_val = cross_val_score(svc_model_1, X_train_svc, Y_train, cv=kfold)
# cross_val_mean = cross_val.mean()
# print('Support vector training accuracy:', cross_val_mean)

# #Test scores
# svc_model_1 = svc_model_1.fit(X_train_svc, Y_train)
# svc_1_preds = svc_model_1.predict(X_test_svc)
# class_report = classification_report(Y_test, svc_1_preds, output_dict=True)
# confusion_mat = confusion_matrix(Y_test, svc_1_preds)
# print('Support vector testing results',confusion_mat,
#       classification_report(Y_test, svc_1_preds))

# overview.append(['svc with sae select 7', features.columns.values, cross_val_mean, class_report['accuracy'], confusion_mat[0][1]])

print(pd.DataFrame(overview, columns = ['name', 'features', 'train_cv_acc', 'test_acc', 'fp']))

# =============================================================================









