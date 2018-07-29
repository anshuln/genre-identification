# -*- coding: utf-8 -*-
"""
Created on Tue May 29 22:48:18 2018

@author: Anshul
"""

import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
import sys
'''
def get_indices_of_features(features,data):
    indices=[]
    for feat in features:
        indices.append(data.columns.get_loc(feat))
    return indices
'''     
def random_color():
    r=random.random()
    b=random.random()
    g=random.random()
    return (r,g,b)

def get_graphing_space(x1,x2,number=300):      #returns a number element space to be graphed
    indices=set()
    X1=[]
    X2=[]
    while (len(indices)<min(number,len(x1))):
        n=random.randint(0,len(x1)-1)
        indices.add(n)
        X1.append(x1[n])
        X2.append(x2[n])
    return X1,X2

#performs PCA using the l best features
def pca(l):
    data=pd.read_csv('MSD_subset.csv')       #put the path to MSD.csv here 
    feat=open('Features/Best_features_1.csv')
    reader=csv.reader(feat)
    f=list(reader)[0:l]
    features=[]
    for fe in f:
        #print(fe[0])
        features.append(fe[0])
    X=data.loc[:,features].values
    y=data.iloc[:,0]

    print('Read data')
    from sklearn.preprocessing import StandardScaler
    sc_X=StandardScaler()
    X=sc_X.fit_transform(X)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)

    print('PCA done')
    y_dist=set()
    for data in y:
        y_dist.add(data)

    y_dist=list(sorted(y_dist))
    color_scheme={}
    for dat in y_dist:
        color_scheme[dat]=random_color()

    k=1
    ax,fig=plt.subplots(nrows=3,ncols=4)
    for dat in y_dist:
        
        plt.subplot(3,4,k)
        X1=[]
        X2=[]
        for i in range(0,len(y)):
            if(y[i]==dat):
                X1.append(principalComponents[i,0])
                X2.append(principalComponents[i,1])
        #X1,X2=get_graphing_space(X1,X2,500)
        #plt.scatter(X1,X2,c=color_scheme[dat],marker='.',alpha=0.3,label=dat)
        plt.scatter(X1,X2,c=color_scheme[dat],marker='.',alpha=0.5,label=dat)
        axes=plt.gca()
        axes.set_xlim([-10,20])
        axes.set_ylim([-10,10])
        plt.grid(True)
        plt.ylabel((dat[:6]+'('+str(len(X1))+')'))
        k+=1
    print(pca.explained_variance_ratio_.sum())
    #plt.legend(list(y_dist))
    plt.show()
    
if __name__=="__main__":
    l=int(sys.argv[1])
# for p in range(2,l):
    pca(l)

