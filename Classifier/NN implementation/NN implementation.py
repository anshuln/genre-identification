import tensorflow as tf 
import numpy as np 
import pandas as pd
from tf_utils import random_mini_batches
nx=30
ny=10
l1=255
l2=255
def get_cost(z3,y):
	logits=tf.transpose(z3)
	labels=tf.transpose(y)

	cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
	return cost

def create_placeholders(n_x,n_y):

	x=tf.placeholder(tf.float32,[n_x,None],name="x")
	y=tf.placeholder(tf.float32,[n_y,None],name="y")

	return x,y

def initialize_params(nx,ny,l1=l1,l2=l2):
	tf.set_random_seed(1)
	W1 = tf.get_variable("W1", [l1,nx], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
	b1 = tf.get_variable("b1", [l1,1], initializer = tf.zeros_initializer())
	W2 = tf.get_variable("W2", [l2,l1], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
	b2 = tf.get_variable("b2", [l2,1], initializer = tf.zeros_initializer())
	W3 = tf.get_variable("W3", [ny,l2], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
	b3 = tf.get_variable("b3", [ny,1], initializer = tf.zeros_initializer())

	return {"W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2,
            "W3": W3,
            "b3": b3}
'''
def forward_prop(X,parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
	
	z1=tf.add(tf.matmul(w1,X),b1)
	a1=tf.nn.relu(z1)

	z2=tf.add(tf.matmul(w2,a1),b2)
	a2=tf.nn.relu(z2)

	z3=tf.add(tf.matmul(w3,a2),b3)

	return z3
'''
def forward_prop(X,parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    z1=tf.add(tf.matmul(W1,X),b1)
    a1=tf.nn.relu(z1)
    z2=tf.add(tf.matmul(W2,a1),b2)
    a2=tf.nn.relu(z2)
    z3=tf.add(tf.matmul(W3,a2),b3)
    return z3

def model(X_train,y_train,X_test,y_test,num_epochs=1500,minibatch_size=64,learning_rate=0.0001,print_cost=True):
	tf.set_random_seed(1)
	seed=3
	(n_x,m)=X_train.shape
	n_y=y_train.shape[0]

	#costs=[]

	X,y=create_placeholders(n_x,n_y)

	parameters=initialize_params(n_x,n_y,l1=20,l2=15)

	z3=forward_prop(X,parameters)
	cost=get_cost(z3,y)

	optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)

		for epoch in range(0,num_epochs):
			epoch_cost=0.
			num_minibatches = int(m / minibatch_size)
			num_minibatches = int(m / minibatch_size)
			seed+=1
			minibatches = random_mini_batches(X_train, y_train, minibatch_size, seed)

			for minibatch in minibatches:
				(minibatch_X, minibatch_Y) = minibatch
				_ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, y: minibatch_Y})
				epoch_cost += minibatch_cost / num_minibatches

			if print_cost==True and epoch%100==0:
				print(str(epoch_cost)+' at epoch number '+str(epoch))

	# sess=tf.Session()
		parameters=sess.run(parameters)
		print('Training done')

		correct_prediction = tf.equal(tf.argmax(z3), tf.argmax(y))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		print("Train Accuracy:", str((accuracy.eval({X: X_train, y: y_train}))))
		print("Test Accuracy:", str((accuracy.eval({X: X_test, y: y_test}))))		
	# sess.close()
	return(parameters)

def get_data(path):
	data=pd.read_csv(path)
	X=data.iloc[:,4:].values
	y=data.iloc[:,0].values

	from sklearn.preprocessing import LabelEncoder, OneHotEncoder
	labelencoder_y = LabelEncoder()
	y=y.reshape(-1,1)
	y[:,0] = labelencoder_y.fit_transform(y[:,0])
	onehotencoder = OneHotEncoder(categorical_features = [0])
	y = onehotencoder.fit_transform(y).toarray()
	
    	
	from sklearn.preprocessing import StandardScaler
	sc_X=StandardScaler()
	X=sc_X.fit_transform(X)

	from sklearn.cross_validation import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

	return(X_train.T,y_train.T,X_test.T,y_test.T)

if __name__=="__main__":
	nx=30
	ny=10
	l1=255
	l2=255
	X_train,y_train,X_test,y_test=get_data('msd_genre_dataset.csv')

	parameters=model(X_train,y_train,X_test,y_test)








