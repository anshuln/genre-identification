import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
import sys
import keras
import time
def decode(y):
	ret=np.zeros((y.shape[0],1))
	for i in range(len(y)):
		ret[i]=int((y[i].argmax())) 
	return ret
def recall_and_prec(cm):
	fp=np.sum(cm,axis=0)
	fn=np.sum(cm,axis=1)
	prec=np.zeros((len(cm),1))
	rec=np.zeros((len(cm)+1,1))
	for i in range(len(cm[0])):
		prec[i]=cm[i][i]/fp[i]
		rec[i]=cm[i][i]/fn[i]
	rec[-1]=0
	print(rec.T)
	print(np.sum(rec)/len(rec))
	print(prec.T)
	print(np.sum(prec)/len(prec))

def select_genres(genres,y):
	indices=[]
	for i in range(len(y)):
		if(y[i] in genres):
			indices.append(i)
		# else:
		# 	print(y[i])
	return(indices)
data=pd.read_csv('MSD_subset.csv')       #put the path to MSD.csv here 
feat=open('Best_features_1.csv')
reader=csv.reader(feat)
f=list(reader)[0:]
features=[]
for fe in f:
    #print(fe[0])
    features.append(fe[0])
X_=data.loc[:,features].values
y_=data.iloc[:,0]
# y=np.append(y,data.iloc[654:,0])
# X = np.delete(X,np.s_[300:654],0)
def train_and_eval(genre,X_=X_,y_=y_):	
	# print(genre)
	# genre.append('soul and reggae')
	indices=select_genres(genre,y_)
	# print(len(indices))
	X=X_[indices,:]
	y=y_[indices]
	# print("len of y ",len(y))
	# print('Read data')
	from sklearn.preprocessing import StandardScaler
	sc_X=StandardScaler()
	X=sc_X.fit_transform(X)

	from sklearn.preprocessing import LabelEncoder, OneHotEncoder
	labelencoder_y = LabelEncoder()
	y=y.reshape(-1,1)
	y[:,0] = labelencoder_y.fit_transform(y[:,0])
	onehotencoder = OneHotEncoder(categorical_features = [0])
	y = onehotencoder.fit_transform(y).toarray()

	from sklearn.cross_validation import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=int(time.time()))

	from keras.layers import Dense,	Dropout
	from keras.models import Sequential
	from keras import regularizers

	model=Sequential()
	model.add(Dense(100,input_dim=len(f),activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(100,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(100,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1000,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1000,activation='relu'))	
	model.add(Dropout(0.5))
	model.add(Dense(len(labelencoder_y.classes_),activation='softmax'))
	model.compile(optimizer='rmsprop',
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])
	m=model.fit(x=X_train,y=y_train, batch_size=64, epochs=200,verbose=0, validation_data=(X_test,y_test))
	y_pred=decode(model.predict(X_test))
	y_t=decode(y_test)
	print(labelencoder_y.classes_)
	from sklearn.metrics import confusion_matrix as cma
	cm=(cma(y_t,y_pred))
	y_t=decode(y_train)
	y_pred=decode(model.predict(X_train))
	cm_=(cma(y_t,y_pred))
	print(cm)
	(recall_and_prec(cm))
	print(cm_)
	recall_and_prec(cm_)
	# model.fit(x=X_train,y=y_train, batch_size=64, epochs=1,verbose=1, validation_data=(X_test,y_test))
	x=model.evaluate(X_test,y_test,verbose=0)
	print(x)
	return([genre[0],genre[1],x[1]])
genres=['metal','classic pop and rock','classical','pop','punk','jazz and blues','folk','dance and electronica','hip-hop','soul and reggae']

import csv
file=open('Database.csv','a',newline='')
writer = csv.writer(file)
train_and_eval(genres)
# for g1 in range(len(genres)):
# 	for g2 in range(g1,len(genres)):
# 		if(g2!=g1):
# 			row=train_and_eval([genres[g1],genres[g2]])
# 			writer.writerow(row)
# 			print((g1+g2)/45)