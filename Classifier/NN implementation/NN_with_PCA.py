import sklearn
import pandas as pd
import numpy as np
import csv
import time
data=pd.read_csv('MSD_subset.csv')       #put the path to MSD.csv here 
feat=open('Best_features_1.csv')
reader=csv.reader(feat)
f=list(reader)[0:10]
features=[]
for fe in f:
    #print(fe[0])
    features.append(fe[0])
X=data.loc[:,features].values
y=data.iloc[:,0]
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y=y.reshape(-1,1)
y[:,0] = labelencoder_y.fit_transform(y[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
y = onehotencoder.fit_transform(y).toarray()

print('Read data')
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X=sc_X.fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(principalComponents, y, test_size = 0.2,random_state=int(time.time()))

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras import regularizers

model=Sequential()
model.add(Dense(1000,input_dim=2,activation='relu'))
model.add(Dropout(0.5))
# model.add(Dense(100,activation='relu'))
# model.add(Dense(100,activation='relu'))
model.add(Dense(1000,activation='relu'))    
model.add(Dropout(0.5))
model.add(Dense(1000,activation='linear'))
model.add(Dropout(0.5))
model.add(Dense(1000,activation='relu'))    
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=X_train,y=y_train, batch_size=64, epochs=200,verbose=1, validation_data=(X_test,y_test))
