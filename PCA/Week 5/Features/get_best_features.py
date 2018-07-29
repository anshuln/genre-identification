from collections import OrderedDict as OD
import csv
import numpy as np

file=open('G:\\SoC\\Week 5\\Features.txt')
contents=str(file.read())
file.close()
file=open('G:\\SoC\\Week 5\\features_with_values.txt')
values=str(file.read())
file.close()

list_of_feature_matrices=contents.split('\n')
list_of_feature_matrices=[word for word in list_of_feature_matrices if len(word)>1]
list_of_values=values.split('\n')
list_of_values=[word for word in list_of_values if len(word)>1]
for i in range(0,len(list_of_values)):
	list_of_values[i]=[float(word) for word in list_of_values[i].split('\t') if len(word)>2]

values=np.array(list_of_values)
for i in range(0,len(values)):
	values[i]/=values[i].sum()

ranks={}
flag=0

for i in range(0,len(list_of_feature_matrices)):
	features=list_of_feature_matrices[i].split('\t')
	for counter in range(0,len(values[i])):
		if(flag==0):
			ranks[features[counter]]=values[i][counter]
		else:
			ranks[features[counter]]+=values[i][counter]
	flag=1

od=OD(sorted(ranks.items(),key=lambda t: t[1],reverse=True))

features=list(od.items())

# for feature_matrix in list_of_feature_matrices:
# 	features=feature_matrix.split('\t')
	
# 	for counter in range(0,len(features)):
# 		if(flag==0):
# 			ranks[features[counter]]=counter
# 		else:
# 			ranks[features[counter]]+=counter
# 	flag=1


file=open('Best_features_1.csv','w+',newline='')
writer = csv.writer(file)
for feat in features:
	writer.writerow(list(feat))
