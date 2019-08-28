#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np  
import pandas as pd  
from sklearn import utils  

# import the CSV from http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
# https://thisdata.com/blog/unsupervised-machine-learning-with-one-class-support-vector-machines/
# this will return a pandas dataframe.

data_names = pd.read_csv('kddcup.names', skiprows = 1,header=None,sep=':')
data_names = dict(data_names[0])
data_names[41] = 'label'

dictlist = []

for key, value in data_names.items():
    temp = value
    dictlist.append(temp)

data = pd.read_csv('kddcup.data_10_percent', low_memory=False,
                   header=None,names = dictlist)

# extract just the logged-in HTTP accesses from the data
data = data[data['service'] == "http"]  
data = data[data["logged_in"] == 1]

# let's take a look at the types of attack labels are present in the data.
data.label.value_counts().plot(kind='bar')  


# here we'll grab just the relevant features for HTTP.
relevant_features = [  
    "duration",
    "src_bytes",
    "dst_bytes",
    "label"
]

# replace the data with a subset containing only the relevant features
data = data[relevant_features]

# normalise the data - this leads to better accuracy and reduces numerical instability in
# the SVM implementation
#data["duration"] = np.log((data["duration"] + 0.1).astype(float))  
#data["src_bytes"] = np.log((data["src_bytes"] + 0.1).astype(float))  
#data["dst_bytes"] = np.log((data["dst_bytes"] + 0.1).astype(float))  

# we're using a one-class SVM, so we need.. a single class. the dataset 'label'
# column contains multiple different categories of attacks, so to make use of 
# this data in a one-class system we need to convert the attacks into
# class 1 (normal) and class -1 (attack)
data.loc[data['label'] == "normal.", "attack"] = 1  
data.loc[data['label'] != "normal.", "attack"] = -1


# grab out the attack value as the target for training and testing. since we're
# only selecting a single column from the `data` dataframe, we'll just get a
# series, not a new dataframe
target = data['attack']



# find the proportion of outliers we expect (aka where `attack == -1`). because 
# target is a series, we just compare against itself rather than a column.
outliers = target[target == -1]  
print("outliers.shape", outliers.shape)  
print("outlier fraction", outliers.shape[0]/target.shape[0])


# drop label columns from the dataframe. we're doing this so we can do 
# unsupervised training with unlabelled data. we've already copied the label
# out into the target series so we can compare against it later.
data.drop(["label", "attack"], axis=1, inplace=True)


from sklearn.model_selection import train_test_split  
train_data, test_data, train_target, test_target = train_test_split(data, target, train_size = 0.8)  
train_data.shape 


from sklearn import svm


# set nu (which should be the proportion of outliers in our dataset)
nu = outliers.shape[0] / target.shape[0]  
print("nu", nu)


print("#######################################")
print("One Class SVM") #work for both Outlier detection & Novelty detection
print("#######################################")
      
model = svm.OneClassSVM(nu=nu, kernel='rbf', gamma=0.00005)  
model.fit(train_data) 


from sklearn import metrics  
preds = model.predict(train_data)  
targs = train_target


print("===================")
print("For Training Data: ")
print("===================")
print("accuracy: ", metrics.accuracy_score(targs, preds))  
print("precision: ", metrics.precision_score(targs, preds))  
print("recall: ", metrics.recall_score(targs, preds))  
print("f1: ", metrics.f1_score(targs, preds))  


preds = model.predict(test_data)  
targs = test_target


print("===============")
print("For Test Data: ")
print("===============")
print("accuracy: ", metrics.accuracy_score(targs, preds))  
print("precision: ", metrics.precision_score(targs, preds))  
print("recall: ", metrics.recall_score(targs, preds))  
print("f1: ", metrics.f1_score(targs, preds))  

'''
import matplotlib.pyplot as plt
plt.figure()
colors = np.array(['#377eb8', '#ff7f00'])
plt.scatter(data['dst_bytes'], data['src_bytes'], s=10, color=colors)
plt.show()        
'''

print("#######################################")
print("Isolation Forest")
print("#######################################")

from sklearn.ensemble import IsolationForest
model1 = IsolationForest(contamination=nu,random_state=42)
model1.fit(train_data)

preds1 = model1.predict(train_data) 
targs = train_target 


print("===================")
print("For Training Data: ")
print("===================")
print("accuracy: ", metrics.accuracy_score(targs, preds1))  
print("precision: ", metrics.precision_score(targs, preds1))  
print("recall: ", metrics.recall_score(targs, preds1))  
print("f1: ", metrics.f1_score(targs, preds1))  

preds1 = model1.predict(test_data) 
targs = test_target 

print("===============")
print("For Test Data: ")
print("===============")
print("accuracy: ", metrics.accuracy_score(targs, preds1))  
print("precision: ", metrics.precision_score(targs, preds1))  
print("recall: ", metrics.recall_score(targs, preds1))  
print("f1: ", metrics.f1_score(targs, preds1))  


print("#######################################")
print("Local Outlier Factor") #work for both Outlier detection & Novelty detection
print("#######################################")

from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=35, contamination=nu)

preds2 = lof.fit_predict(train_data) 
targs = train_target

print("===================")
print("For Training Data: ")
print("===================")
print("accuracy: ", metrics.accuracy_score(targs, preds2))  
print("precision: ", metrics.precision_score(targs, preds2))  
print("recall: ", metrics.recall_score(targs, preds2))  
print("f1: ", metrics.f1_score(targs, preds2))  

preds2 = lof.fit_predict(test_data) 
targs = test_target 

print("===============")
print("For Test Data: ")
print("===============")
print("accuracy: ", metrics.accuracy_score(targs, preds2))  
print("precision: ", metrics.precision_score(targs, preds2))  
print("recall: ", metrics.recall_score(targs, preds2))  
print("f1: ", metrics.f1_score(targs, preds2))  


'''
data_names = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment',
 'urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell',
'su_attempted', 'num_root','num_file_creations','num_shells','num_access_files',
'num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate',
'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
'srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
'dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
'dst_host_srv_rerror_rate','label']'''