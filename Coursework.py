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
import random
import time
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from tensorflow import keras
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
    
def plot_all_features(data):
    for i in range(0, data.shape[1]):
        series = data.iloc[:, i]
        plt.figure(i)
        series.plot()
        plt.title(data.columns[i])

def plot_hist(data):
    for i in range(0, data.shape[1]):
        plt.figure(i)
        plt.hist(data.iloc[:,i])

def plot_3d(x, y, z, color):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    ax.scatter(x, y, z, c = color)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.azim = 90
    
    plt.show()
  
# =============================================================================

#import test data 
test_filename = 'C:/Users/Max/Documents/Max/Birkbeck/2nd Year/Applied Machine Learning/Coursework/test_imperson_without4n7_balanced_data.csv'
test_df = read_csv(test_filename, header = None)
X_test = test_df.iloc[1:, 0:152]
Y_test = test_df.iloc[1:, 152:]


#import training data
train_filename = 'C:/Users/Max/Documents/Max/Birkbeck/2nd Year/Applied Machine Learning/Coursework/train_imperson_without4n7_balanced_data.csv'
train_df = read_csv(train_filename, header =None)
X_train = train_df.iloc[1:, 0:152]
Y_train = train_df.iloc[1:, 152:]

##Delete zero variance features 
X_train, X_test = delete_zero_var(X_train, X_test) 

##Standarise the data 
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train))
X_test = pd.DataFrame(scaler.fit_transform(X_test))

##Kmeans on whole dataset with decision tree to decide how many clusters
num_clusters = [2, 3, 5, 8, 12, 16, 20, 30]
scores = []
for i in range(0,8):
    kmeans = KMeans(n_clusters=num_clusters[i])
    kmeans_train = kmeans.fit(X_train)
    kmeans_test = kmeans.predict(X_test)
    dt = DecisionTreeClassifier().fit(kmeans_train.labels_.reshape(-1,1), Y_train)
    score = dt.score(kmeans_test.reshape(-1,1), Y_test)
    scores.append(score)
    print(score)
    
plt.scatter(num_clusters, scores)
plt.title('Decision tree classifier accuracy with increasing cluster number')
plt.xlabel('Cluster number') 
plt.ylabel('Decision tree accuracy')  

##Optimum cluster number found so append to main test and train sets
kmeans = KMeans(n_clusters = 8)
kmeans_train = kmeans.fit(X_train)
kmeans_test = kmeans.predict(X_test)

cluster_train = pd.DataFrame(kmeans_train.labels_)
cluster_test = pd.DataFrame(kmeans_test)


##PCA to reduce dimensions of data
pca_model = PCA(.97)
pca_train_features = pca_model.fit_transform(X_train)
pca_test_features = pca_model.transform(X_test)

#pca_train_features = normalise.fit_transform(pca_train_features)
#pca_test_features = normalise.fit_transform(pca_test_features)

pca_train_features = pd.DataFrame(pca_train_features).add_prefix('pca_')
pca_test_features = pd.DataFrame(pca_test_features).add_prefix('pca_')


##Autoencoder to generate features
input_size = X_train.shape[1] 
hidden_1_size = input_size//2
hidden_2_size = hidden_1_size//2
code_size = 2

input_layer = Input(shape = (input_size,))
hidden_1 = Dense(hidden_1_size, activation = 'relu')(input_layer)
hidden_2 = Dense(hidden_2_size, activation = 'relu')(hidden_1)
code = Dense(code_size, activation = 'relu', activity_regularizer=l1(10e-6))(hidden_2)

hidden_3 = Dense(hidden_2_size, activation = 'relu')(code)
hidden_4 = Dense(hidden_1_size, activation = 'relu')(hidden_3)
output_layer = Dense(input_size, activation = 'sigmoid')(hidden_4)

autoencoder = Model(input_layer, output_layer)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(X_train, X_train, epochs=5)
                
encoded = Model(input_layer, code)

X_train_auto = encoded.predict(X_train)
X_test_auto = encoded.predict(X_test)

X_train_auto = pd.DataFrame(X_train_auto).add_prefix('auto_')
X_test_auto = pd.DataFrame(X_test_auto).add_prefix('auto_')


## Concatanate all generated features
X_train = pd.concat((pd.DataFrame(X_train), pca_train_features), axis = 1)
X_test = pd.concat((pd.DataFrame(X_test), pca_test_features), axis = 1)

X_train = pd.concat((pd.DataFrame(X_train), X_train_auto), axis = 1)
X_test = pd.concat((pd.DataFrame(X_test), X_test_auto), axis = 1)
 
X_train['cluster'] = cluster_train
X_test['cluster'] = cluster_test


##KMeans clustering on features and adding cluster number as feature
#kmeans = KMeans(n_clusters = 3).fit(X_train.iloc[:, [4,13,48]])
n_clusters = 3
kmeans = KMeans(n_clusters = n_clusters).fit(X_train.loc[:,['pca_0', 'pca_1', 'pca_2']])
#labels = kmeans.labels_/n_clusters #Use if data has been normalised
labels = kmeans.labels_ #Use if data has been standardised
X_train['clusters'] = pd.DataFrame(labels)

x_test_pred = kmeans.predict(X_test.loc[:, ['pca_0', 'pca_1', 'pca_2']])
#x_test_pred = x_test_pred/n_clusters
X_test['clusters'] = pd.DataFrame(x_test_pred)

##ExtraTreesClassifier to evaluate importance of features  
model = ExtraTreesClassifier()
extra_trees_class = model.fit(X_train, Y_train)
features_normalised = np.std([trees.feature_importances_ for trees
                              in extra_trees_class.estimators_], axis = 0)
features = [index for index, i in enumerate(features_normalised)
            if i >0.05]

top_features_train = pd.DataFrame(X_train.iloc[:, features])
top_features_test = pd.DataFrame(X_test.iloc[:, features])


# ##SelectKBest on training data
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
select_best = SelectKBest(k=100)
best_model = select_best.fit(X_train, Y_train)
feature_index = best_model.get_support(indices=(True))
X_train = X_train.iloc[:, feature_index]
X_test = X_test.iloc[:, feature_index]


##Pipeline to evaluate different models
models = [] 
models.append(('LogReg', LogisticRegression(random_state=(7))))
#models.append(('SVC', SVC(random_state=(7))))
#models.append(('RandTClass', RandomForestClassifier(random_state=(7))))
#models.append(('LinDisAnal', LinearDiscriminantAnalysis(random_state=(7))))
models.append(('MLPClass', MLPClassifier(hidden_layer_sizes =(80,40,5),
                                          activation = 'tanh',
                                          solver='adam',
                                          random_state=(7))))

names = []
reports = []
scoring = 'accuracy'

for name, model in models:
    start = time.time()
    model.fit(X_train, Y_train)
    end = time.time()
    ttb = end-start
    train_score = model.score(X_train, Y_train)
    test_score = model.score(X_test, Y_test)
    names.append(name)
    reports.append(classification_report(Y_test, model.predict(X_test)))
    msg = "%s: %f %f %f" % (name, train_score, test_score, ttb)
    print(msg)










##Train whole dataset on a RF to set baseline.Gives 53% accuracy


##Train on data with no correlated variables
# stan_X_train, stan_X_test = standardise(X_train, X_test) 
# rf_stan_model = RandomForestClassifier().fit(stan_X_train, np.ravel(Y_train))
# rf_stan_predictions = rf_stan_model.predict(stan_X_test)
# rf_stan_disp = RocCurveDisplay.from_estimator(rf_stan_model, stan_X_test, Y_test)
# plt.show()
# print(confusion_matrix(np.ravel(Y_test), rf_stan_predictions))


 




                                                        
    