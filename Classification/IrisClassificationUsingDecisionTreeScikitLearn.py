#Load dataset
from sklearn import datasets
iris = datasets.load_iris()

print(iris['feature_names'])
print(iris['target_names'])
print(iris['data'][0])
print(iris['target'][0])

#split data into train and test data
#currently taking only 3 records for testing one for each
# flower type located at 0, 50 and 100 line in dataset
test_idx=[0,50,100]

import numpy as np
#training data
train_target = np.delete(iris['target'], test_idx)
train_data = np.delete(iris['data'], test_idx, axis=0)

#testing data
test_target = iris['target'][test_idx]
test_data = iris['data'][test_idx]

#Training
from sklearn import tree
clfr = tree.DecisionTreeClassifier()
clfr.fit(train_data, train_target)

#print the expected result
print(test_target)

#print actual results
print(clfr.predict(test_data))