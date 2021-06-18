# -*- coding: utf-8 -*-
"""
Created on Mon May 17 18:02:23 2021

@author: DELL
"""
#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#loading the dataset
crime = pd.read_csv("crime_data.csv")
crime.head()
crime.shape

#Normalization function
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)

#Normalized dataframe (considering the numerical part of the data)
df_norm = norm_func(crime.iloc[:,1:])
df_norm.describe()

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch #for creating the dendrogram

z = linkage(df_norm, method="complete", metric="euclidean")
plt.figure(figsize=(15, 5))
plt.title("Hierarchial Clustering Dendrogram")
plt.xlabel("Features")
plt.ylabel("crime")
sch.dendrogram(
    z,
    leaf_rotation=0.,
    leaf_font_size=8.,
)
plt.show()
crime.corr()


#Kmeans
#screw plot or elbow curve
k = list(range(2,15))
k
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from numpy import random, float, array
TWSS = [] #variable for storing total within sum of squares for each Kmeans
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
#Scree plot
plt.figure(figsize=(16,6))
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_clusters");plt.ylabel("total_within_SS");plt.xticks(k)
#Here the elbow appears to be smoothening out after four clusters indicating that the optimal number of cluster is 4
#selecting 4 clusters from the above scree plot which is the optimum number of clusters
model=KMeans(n_clusters=4)
model.fit(df_norm)
model.labels_ # getting the labels of clusters assigned to each row
model.cluster_centers_

import seaborn as sns
X = crime[["Murder","Assault","UrbanPop","Rape"]]
clusters = KMeans(4) # 4 clusters
clusters.fit(X)
clusters.cluster_centers_
clusters.labels_
crime["Crime_clusters"] = clusters.labels_
crime.head()
crime.sort_values(by=["Crime_clusters"],ascending = True)
X.head()

stats = crime.sort_values("Murder", ascending= True)
stats

#plot between pairs Murder~Assault
sns.lmplot("Murder","Assault", data=crime,
           hue = "Crime_clusters",
           fit_reg=False, size = 6 );
#plot between pairs Murder~Rape
sns.lmplot("Murder","Rape",data=crime,
           hue = "Crime_clusters",
           fit_reg=False, size = 6);
#plot between pairs Assault~Rape
sns.lmplot("Assault","Rape",data=crime,
           hue = "Crime_clusters",
           fit_reg=False, size = 6);
#All the dots are states of US and the different colours are one cluster showing clustering for the crime data.


#DBSCAN
from sklearn.cluster import DBSCAN
df= crime.iloc[:,1:5]
array = df.values  
array       
stscaler = StandardScaler().fit(array)
X = stscaler.transform(array)           
X
dbscan = DBSCAN(eps=0.8, min_samples=6)
dbscan.fit(X)
#Noisy samples are given the label -1.
dbscan.labels_
cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])
cl
pd.concat([df,cl],axis=1)
