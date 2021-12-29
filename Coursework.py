# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 19:45:28 2021

@author: Max
"""
from pandas import read_csv
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cluster import KMeans
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import Adam



# =============================================================================
# Prepare data
def null_values(data):
    print(data.isnull().values.any()) # Check to see if there are any NaN values


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

def delete_zero_var(X_train, X_test):
    ##Delete any features that have a variance of 0.
    list_to_remove = []
    for i in range(0, X_train.shape[1]):
        if (X_train.var()[X_train.columns[i]] == 0) == True:
            list_to_remove.append(i)
            
    new_X_train = X_train.drop(X_train.columns[list_to_remove], axis = 1)
    new_X_test = X_test.drop(X_test.columns[list_to_remove], axis = 1)
    
    return new_X_train, new_X_test
            
          
# ==========================================================================


# =============================================================================
# Visualisations
def heat_map(data, dimensions = 2):
    plt.imshow(data, cmap = 'hot')
    plt.show()
    
def plot_data(data):
    for i in range(0, data.shape[1]):
        series = data.iloc[:, i]
        plt.figure(i)
        series.plot()
        plt.title(data.columns[i])

def plot_hist(data):
    for i in range(0, data.shape[1]):
        plt.figure(i)
        plt.hist(data.iloc[:,i])
        
# =============================================================================

#import test data 
test_filename = 'test_imperson_without4n7_balanced_data.csv'
test_df = read_csv(test_filename, header = None)
X_test = test_df.iloc[1:, 0:152]
Y_test = test_df.iloc[1:, 152:]


#import training data
train_filename = 'train_imperson_without4n7_balanced_data.csv'
train_df = read_csv(train_filename, header =None)
X_train = train_df.iloc[1:, 0:152]
Y_train = train_df.iloc[1:, 152:]

##Delete zero variance features 
X_train, X_test = delete_zero_var(X_train, X_test) 

##Delete any duplicate features by checking correlation against each other
X_train, X_test = correlation_remove(X_train, X_test)

##Normalise the data
normalise = MinMaxScaler()
X_train = normalise.fit_transform(X_train)
X_test = normalise.transform(X_test)

##Standarise the data 
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)


##PCA to reduce dimensions of data
pca_model = PCA(.95)
pca_train_features = pd.DataFrame(pca_model.fit_transform(X_train)).add_prefix('pca_')
pca_test_features = pd.DataFrame(pca_model.transform(X_test)).add_prefix('pca_')
X_train = pd.concat((pd.DataFrame(X_train), pca_train_features), axis = 1)
X_test = pd.concat((pd.DataFrame(X_test), pca_test_features), axis = 1)

##Autoencoder to generate features 
# input_size = X_train.shape[1] 
# hidden_1_size = input_size//2
# code_size = hidden_1_size//4

# input_layer = Input(shape = (input_size,))
# hidden_1 = Dense(hidden_1_size, activation = 'relu')(input_layer)
# code = Dense(code_size, activation = 'relu', activity_regularizer=l1(10e-6))(hidden_1)

# hidden_2 = Dense(hidden_1_size, activation = 'relu')(code)
# output_layer = Dense(input_size, activation = 'sigmoid')(hidden_2)

# autoencoder = Model(input_layer, output_layer)
# encoded = Model(input_layer, code)

# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# autoencoder.fit(X_train, X_train, epochs=3,
#                 validation_split = 0.1)

# X_train_auto = encoded.predict(X_train)
# X_test_auto = encoded.predict(X_test)

# X_train = np.concatenate((X_train, X_train_auto), axis = 1)
# X_test = np.concatenate((X_test, X_test_auto), axis = 1)


##Plot scatter plot of two variables and color code. 
# plt.scatter(x = X_train[:,0], y = X_train[:, 1], c = Y_train['155'])

##ExtraTreesClassifier to evaluate importance of features  
model = ExtraTreesClassifier()
extra_trees_class = model.fit(X_train, Y_train)
features_normalised = np.std([trees.feature_importances_ for trees
                              in extra_trees_class.estimators_], axis = 0)
features = [index for index, i in enumerate(features_normalised)
            if i >0.05]

top_features_train = pd.DataFrame(X_train.iloc[:, features])
top_features_test = pd.DataFrame(X_test.iloc[:, features])


##KMeans clustering on features and adding cluster number as feature
kmeans = KMeans(n_clusters = 2).fit(X_train.iloc[:, [4,13,48]])
labels = kmeans.labels_
X_train = np.concatenate((X_train, np.reshape(labels,(labels.shape[0],1))),
                          axis = 1)

x_test_predi = kmeans.predict(X_test.iloc[:, [4,13,48]])
X_test = np.concatenate((X_test, np.reshape(x_test_predi,
                                            (x_test_predi.shape[0], 1))),
                        axis = 1)
labels = pd.DataFrame(labels)
plt.scatter(x = labels.index, y = labels[0])
plt.show()


# ##Standarise the data 
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# ##SelectKBest on training data
# # X_train = pd.DataFrame(X_train)
# # X_test = pd.DataFrame(X_test)
# # select_best = SelectKBest(k=10)
# # best_model = select_best.fit(X_train, Y_train)
# # feature_index = best_model.get_support(indices=(True))
# # X_train = X_train.iloc[:, feature_index]
# # X_test = X_test.iloc[:, feature_index]


# ##Logistic reg model - make sure standardisation happens after norm.
# Y_train = np.array(np.ravel(Y_train))
# Y_test = np.array(np.ravel(Y_test))
# log_model = LogisticRegression()
# kfold = KFold(n_splits = 10)
# results = cross_val_score(log_model, X_train, Y_train, cv = kfold)
# print(results.mean())

# log_fit = log_model.fit(X_train, Y_train)
# log_res = log_fit.predict(X_test)
# print(confusion_matrix(Y_test, log_res))
# print(log_fit.score(X_test, Y_test))


##SVC Classifier - need to reduce number of features to work. 
# svm_model = SVC()
# kfold = KFold(n_splits = 5)
# results = cross_val_score(svm_model, X_train, Y_train, cv = kfold)
# print(results.mean())
# svm_model.fit(X_train, Y_train)
# print(svm_model.score(X_test, np.array(np.ravel(Y_test))))


##Random Forrest Clasifier 
#rf_model = RandomForestClassifier()

# features = [] 
# features.append(('standardize', StandardScaler()))
# features.append(('PCA', PCA(n_components=2)))
# features.append(('select_best', SelectKBest()))
# feature_union = FeatureUnion(features)

# estimators = []
# estimators.append(('features', feature_union))
# #estimators.append(('Logistic reg', LogisticRegression(C=10)))
# estimators.append(('svm', SVC()))
# model = Pipeline(estimators)


# model.fit(X_train, Y_train)
# predictions = model.predict(X_test)
#print(predictions)
# print(confusion_matrix(Y_test, predictions))
# print(model.score(X_test, Y_test))

# features_selected = model.named_steps['select_best'].support_
# print(features_selected)

# X_train, list_to_remove = delete_zero_cols(X_train)
# X_test = X_test.drop(X_test.columns[list_to_remove], axis = 1)




# features = [] 
# features.append(('standardize', StandardScaler()))
# features.append(('PCA', PCA(.95)))
# features.append(('select_best', SelectKBest()))
# feature_union = FeatureUnion(features)

# estimators = []
# estimators.append(('features', feature_union))
# estimators.append(('RF', RandomForestClassifier()))
# model = Pipeline(estimators)

# k_fold = KFold(n_splits = 5)
# results = cross_val_score(model, X_train, Y_train, cv = k_fold)
# print(results.mean())





##Train whole dataset on a RF to set baseline.Gives 53% accuracy


##Train on data with no correlated variables
# stan_X_train, stan_X_test = standardise(X_train, X_test) 
# rf_stan_model = RandomForestClassifier().fit(stan_X_train, np.ravel(Y_train))
# rf_stan_predictions = rf_stan_model.predict(stan_X_test)
# rf_stan_disp = RocCurveDisplay.from_estimator(rf_stan_model, stan_X_test, Y_test)
# plt.show()
# print(confusion_matrix(np.ravel(Y_test), rf_stan_predictions))

##Perform PCA to reduce number of features. Includes components that explain
# ~95% of the variance. Fit RF on PC's. Gives 53% accuracy 
# pca_train, pca_test = pca(X_train, X_test)
# rf_model = RandomForestClassifier().fit(pca_train, Y_train)
# rf_predictions = rf_model.predict(X_test)
# rf_disp = RocCurveDisplay.from_estimator(rf_model, X_test, Y_test)
# plt.show()
# print(confusion_matrix(Y_test, rf_predictions))

##Perform the same PCA above but standardize data before
# scaled_X_train, scaled_X_test = standardise(X_train, X_test)
# pca_train, pca_test = pca(scaled_X_train, scaled_X_test)
# print(randomforrest(pca_train, Y_train, pca_test, Y_test))
 




                                                        
    