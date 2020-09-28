from sklearn import datasets
import numpy as np

#get iris dataset from scikit-learn
iris = datasets.load_iris()

#the 4 features are sepal length, sepal width, petal length, petal width
#fetching petal length and width
X = iris.data[:, [2,3]]
y = iris.target

#get unique target values corresponding to setoda, versicolor, virginica
print('Class labels:', np.unique(y))

#divide the data in train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

#check the stratification of class labels in train and test dataset using np.bincount
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train', np.bincount(y_train))
print('Labels counts in y_test', np.bincount(y_test))

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#Perceptron training
from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter_no_change=40, eta0=0.01, random_state=1)
ppn.fit(X_train_std, y_train)

#predict output for test data based on training
y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test!=y_pred).sum())

#Calculate accuracy
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

#Calculate accuracy using score method that combines "predict" with "accuracy_score"
print('Accuracy: %.2f' % ppn.score(X_test_std, y_test))




