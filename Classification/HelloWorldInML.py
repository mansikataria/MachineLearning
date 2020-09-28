#Write a Classifier to identify whether the fruit is
# Apple or Orange based on two features weight and texture
from sklearn import tree

#training data "texture"=0 for "smooth" and "texture"=1 for "bumpy"
features = [[140, 0], [130, 0], [150, 1], [170, 1]]
# label=0 for Apple and 1 for Orange
labels = [0 , 0, 1,1]

#training
clfr = tree.DecisionTreeClassifier()
clfr.fit(features, labels)

#testing
print(clfr.predict([[150, 0]]))