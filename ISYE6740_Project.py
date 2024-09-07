#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 07:43:30 2022

@author: MKK
"""

import pandas as pd
import gower
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS


pd.set_option('display.float_format', lambda x: '%.9f' % x)
pd.set_option("display.precision", 20)

# Load data
rrfData = pd.read_csv('rrf_foia_210817.csv',on_bad_lines='skip').fillna('')
rrfData['zipcode'] = rrfData['BusinessZip']

# analyze outliers
extremes=rrfData[rrfData['GrantAmount']==10000000]
extremes['HH_Income'].mean()
extremes.groupby(['WomenOwnedIndicator'])['WomenOwnedIndicator'].count()
extremes.groupby(['SocioeconmicIndicator'])['SocioeconmicIndicator'].count()
extremes.groupby(['VeteranIndicator'])['VeteranIndicator'].count()

# analyze overall data
rrfData['GrantAmount'].mean()
rrfData['HH_Income'].mean()
rrfData.count()

# analyze subgroups
rrfData[rrfData['VeteranIndicator']=='N']['GrantAmount'].mean()
rrfData.groupby(['RuralUrbanIndicator'])['RuralUrbanIndicator'].count()
rrfData.groupby(['VeteranIndicator'])['VeteranIndicator'].count()
rrfData.groupby(['SocioeconmicIndicator'])['SocioeconmicIndicator'].count()
rrfData.groupby(['WomenOwnedIndicator'])['WomenOwnedIndicator'].count()

# collect zip code median income and clean up data
zipCodeData = pd.read_csv('acs_median_income_1.csv',on_bad_lines='skip')
zipCodeData = zipCodeData.iloc[1: , :] # delete first row
zipCodeData=zipCodeData[zipCodeData['S1901_C01_012E']!='-'] # weird problematic zip codes

# clean up text columns and file
zipCodeData['S1901_C01_012E']= np.where(zipCodeData['S1901_C01_012E']=='250,000+',250000,zipCodeData['S1901_C01_012E'])
zipCodeData['S1901_C01_012E']= np.where(zipCodeData['S1901_C01_012E']=='2,500-',np.nan,zipCodeData['S1901_C01_012E'])
zipCodeData=zipCodeData.dropna()
zipCodeData['zipcode'] = zipCodeData['NAME'].str[-5:].astype(int)
zipCodeData['HH_Income'] = zipCodeData['S1901_C01_012E'].astype(int)
zipCodeDataRelevant = zipCodeData[['zipcode','HH_Income']]

# merge restaurant relief data and zip code data. clean up
rrfData=pd.merge(rrfData,zipCodeDataRelevant, on='zipcode',how='left')
rrfData=rrfData.dropna() # lose 1000 data points with no income data

# analyze restaurant relief data
rrfData['RestaurantType'].value_counts()
rrfData.groupby('LegalOrganizationType').agg({'GrantAmount':['count', 'mean', max, min]})
rrfData.groupby('LMIIndicator').agg({'GrantAmount':['count', 'mean', max, min]})
rrfData.groupby('SocioeconmicIndicator').agg({'GrantAmount':['count', 'mean', max, min]})
rrfData.groupby('WomenOwnedIndicator').agg({'GrantAmount':['count', 'mean', max, min]})
rrfData.groupby('RestaurantType').agg({'GrantAmount':['count', 'mean', max, min]})
rrfData.groupby('BusinessState').agg({'GrantAmount':['count', 'mean', max, min]})

# filter data to relevant categories, sample
analyzableData = rrfData[['RuralUrbanIndicator','VeteranIndicator', 'SocioeconmicIndicator', 'WomenOwnedIndicator','GrantAmount', 'HH_Income']]
analyzableDataDownSampled = analyzableData.sample(10000)

# use a matrix to make a DBSCAN cluster of data. DBSCAN helps describe groups of restaurants in relief program.
analyzableData_matrix=np.asarray(analyzableDataDownSampled)
distance_matrix = gower.gower_matrix(analyzableData_matrix)

silhouette_score_calc = np.zeros([18,4]) # silhouette_score, eps, min_sample

for i in range(1,18):
    # Adding the results to a new column in the dataframe
    db = DBSCAN(eps=i, min_samples=60).fit(distance_matrix)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    analyzableDataDownSampled["cluster"] = labels
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    silhouette_score_calc[i][0] = metrics.silhouette_score(distance_matrix, labels)
    silhouette_score_calc[i][1] = db.eps
    silhouette_score_calc[i][2] = db.min_samples
    silhouette_score_calc[i][3] = n_clusters_


# Plot and organize the DB SCAN
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(silhouette_score_calc[1:17,1],label="Eps")
ax.plot(silhouette_score_calc[1:17,2],label="Min samples")
ax.plot(silhouette_score_calc[1:17,3],label="Number of Clusters")
ax.set_ylabel("values",fontsize=14)
plt.legend(loc='lower center')

ax2=ax.twinx()

# make a plot with different y-axis using second axis object
ax2.plot(silhouette_score_calc[1:17,0],label="Silhouette Score",color="blue",marker="o")
plt.title("Silhouette Score at various eps, min samples = 60", fontsize=14)
plt.legend()
ax2.set_ylabel("Silhouette Score",color="blue",fontsize=14)
# plt.show()
ax.xaxis.set_ticks([])
ax2.xaxis.set_ticks([])

plt.tick_params(bottom = False)
plt.rcParams['savefig.dpi'] = 300
plt.savefig('MinSamples is 60, silhouette scores.png')


# Adding the results to a new column in the dataframe
db = DBSCAN(eps=11, min_samples=12).fit(distance_matrix)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
analyzableDataDownSampled["cluster"] = labels

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(distance_matrix, labels))

fig, ax = plt.subplots(figsize=(8, 6))

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = analyzableData_matrix[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 4],
        xy[:, 5],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
        label=k
    )

    xy = analyzableData_matrix[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 4],
        xy[:, 5],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6
    )

ax.set_xlim([0, 2000000])
plt.title("Clusters on HH income and grant $ features")
plt.ticklabel_format(style='plain')
plt.rcParams['savefig.dpi'] = 300
plt.xlabel('Grant amount ($)', fontsize=14)
plt.ylabel('Median HH income by zip', fontsize=14)
plt.legend()
plt.savefig('Clustering income and grant amount zoomed in.png')
# plt.show()

# Create embeddings of 2 components to get 2-D representation of the groupings.
embedding = MDS(n_components=2,dissimilarity='precomputed', metric=True,n_init=2)
transformed = embedding.fit_transform(distance_matrix)

df = pd.DataFrame(transformed, columns = ['dim1','dim2'])
df['labels']=labels

transformed_downsampled=df
labels_downsampled = transformed_downsampled['labels']

cdict = {-1: 'black', 0: 'blue', 1: 'green', 2: 'yellow', 3:'orange',4:'purple',5:"brown", 6:'red',7:'pink',8:'violet',9:'gray'}

# MDS representation of downsampled dataset
fig, ax = plt.subplots()
for g in np.unique(labels_downsampled):
    ix = np.where(labels_downsampled == g)
    ax.scatter(transformed_downsampled['dim1'].to_numpy()[ix], transformed_downsampled['dim2'].to_numpy()[ix], c = cdict[g], label = g, s = 100)
ax.legend()
plt.title('MDS into two dimensions of downsampled RRF dataset')
plt.savefig('MDS represntation.png')
# plt.show()

# gather additional data points on the clusters
organizedCluster=analyzableDataDownSampled.groupby("cluster").agg({'GrantAmount':['count', 'mean', max, min], 'HH_Income':['mean']})
organizedCluster.to_csv('data_dissection.csv')

rural = analyzableDataDownSampled.groupby(['cluster', 'RuralUrbanIndicator'])['RuralUrbanIndicator'].count().rename('rural').reset_index()
rural=rural.pivot(index='cluster',columns='RuralUrbanIndicator')
rural_f = rural.iloc[:, 1]/(rural.iloc[:, 1]+rural.iloc[:, 0])

vet = analyzableDataDownSampled.groupby(['cluster', 'VeteranIndicator'])['VeteranIndicator'].count().rename('vet').reset_index()
vet=vet.pivot(index='cluster',columns='VeteranIndicator')
vet_f = vet.iloc[:, 1]/(vet.iloc[:, 1]+vet.iloc[:, 0])

socio = analyzableDataDownSampled.groupby(['cluster', 'SocioeconmicIndicator'])['SocioeconmicIndicator'].count().rename('socio').reset_index()
socio=socio.pivot(index='cluster',columns='SocioeconmicIndicator').fillna(0)
socio_f = socio.iloc[:, 1]/(socio.iloc[:, 1]+socio.iloc[:, 0])

women = analyzableDataDownSampled.groupby(['cluster', 'WomenOwnedIndicator'])['WomenOwnedIndicator'].count().rename('women').reset_index()
women=women.pivot(index='cluster',columns='WomenOwnedIndicator').fillna(0)
women_f = women.iloc[:, 1]/(women.iloc[:, 1]+women.iloc[:, 0])

analyzableDataDownSampled.groupby(['RuralUrbanIndicator'])['RuralUrbanIndicator'].count()
analyzableDataDownSampled.groupby(['VeteranIndicator'])['VeteranIndicator'].count()
analyzableDataDownSampled.groupby(['SocioeconmicIndicator'])['SocioeconmicIndicator'].count()
analyzableDataDownSampled.groupby(['WomenOwnedIndicator'])['WomenOwnedIndicator'].count()


data_frames = [rural_f, vet_f, socio_f, women_f]

pd.concat(data_frames,axis=1).to_csv('mergeddata.csv')

df_merged = pd.merge(lambda  left,right: pd.merge(left,right,on=['cluster'],
                                            how='inner'), data_frames)

rrfData['HH_Income'].corr(rrfData['GrantAmount'])


### PCA analysis
scaler = StandardScaler()
pca = PCA(n_components=2)
scaledData = scaler.fit(distance_matrix)
pca_data = pca.fit_transform(distance_matrix)
pca_data


fig, ax = plt.subplots()
for g in np.unique(labels):
    ix = np.where(labels == g)
    ax.scatter(pca_data[:, 0][ix], pca_data[:, 1][ix], c = cdict[g], label = g, s = 100)
ax.legend()
plt.show()


