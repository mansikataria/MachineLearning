#import dataset
from sklearn import datasets
iris = datasets.load_iris()

X = iris['data']
y = iris['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5)

#Decision tree classifier
from sklearn import tree
clfr = tree.DecisionTreeClassifier()

clfr.fit(X_train , y_train)

predictions = clfr.predict(X_test)

#calculate accuracy using built-in fuction
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))

#KNN classifier
from sklearn.neighbors import KNeighborsClassifier
clfr = KNeighborsClassifier()

clfr.fit(X_train , y_train)

predictions = clfr.predict(X_test)

#calculate accuracy using built-in fuction
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))