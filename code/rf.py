import pandas as pd
import numpy as np
from sklearn.utils import shuffle
data = pd.read_csv('drebinAndroidDataset.csv', sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
malware = {'S' : 0, 'B' : 1}
#df = pd.DataFrame(data)
#df['class'] = df['class'].replace( '[\S,)]','', regex=True )
data = data.replace(["S","B"],[1,0]) 
data = shuffle(data)
print(data)
#data.replace(to_replace ='B',value ='1')
#df['class'] = df['class'].astype(int)
X = data.iloc[:,0:215]  #independent columns
y = data.iloc[:,215]    #target column i.e price range
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.3,random_state=0)

print('X_train- ',X_train.shape)
print(X_train)
print('Y_train- ',Y_train.shape)
print(Y_train)
print('X_test- ',X_test.shape)
print(X_test)
print('Y_test- ',Y_test.shape)
print(Y_test)
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X_train,Y_train)
features = model.feature_importances_
feature_dict = {}
for i in range(features.size):
	feature_dict[i] = features[i]

#for i in sorted (feature_dict.keys()) :  
#     print(i, end = " ") 
#print(sorted(feature_dict.items(), key = lambda kv:(kv[1], kv[0])))  

final_keys = []
for keys in sorted(feature_dict.items(), key = lambda kv:(kv[1], kv[0])): 
	final_keys.append(keys[0])
	print(keys) 

print(final_keys)
x = 200
selected_keys = []
for i in range(15):
	selected_keys.append(final_keys[x])
	x = x+1

selected_keys.sort()
print(selected_keys)

f = open("rf.txt", "a")
for i in range(len(selected_keys)):
	f.write(str(selected_keys[i]))
	f.write("\n")
f.close()


#feature_dict.sort(key=sortSecond)
#features.sort(reverse=True)

#print(feature_dict) 
