import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn import datasets

#get iris dataset from scikit-learn
iris = datasets.load_iris()

#the 4 features are sepal length, sepal width, petal length, petal width
#fetching petal length and width
X = iris.data[:, [2,3]]
y = iris.target


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


#Plotting a sample sigmoid function
z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)

plt.plot(z, phi_z)
plt.xlabel('z')
plt.ylabel('$\phi(z)$')
plt.ylim(-0.1, 1.1)
plt.axvline(0.0, color='k')

#add y axis ticks and gridline
plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
ax.yaxis.grid(True)

plt.show()


#cost for y=1 and y=0
def cost_1(z):
    return -np.log(sigmoid(z))


def cost_0(z):
    return -np.log(1-sigmoid(z))


z = np.arange(-10, 10, 0.1)
phi_z = sigmoid(z)
c1 = [cost_1(x) for x in z]
plt.plot(phi_z, c1, label = 'J(w) if y=1')
c0 = [cost_0(x) for x in z]
plt.plot(phi_z, c0, linestyle='--', label = 'J(w) if y=0')

plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
plt.xlabel('$\phi$(z)')
plt.ylabel('J(w)')
plt.legend(loc='best')
plt.show()


class LogisticRegressionGD(object):

    """
    parameters:
    lr : float(between 0 and 1)
    learning rate

    n_iter : int
    Number of iterations over training data

    random_state : int
    Used to generate random weights


    attributes:
    w_ : 1-d array
         Weights at the end of training
    cost_ : list

    """

    def __init__(self, lr=0.01, n_iter=50, random_state=1):
        self.lr = lr
        self.n_iter = n_iter
        self.random_state = random_state

    def train(self, X, y):
        """
    Train model
    parameters:
    X: {array-like}, shape = [n_samples, m_features]
    Training data input

    y: array-like, shape = [n_samples_output]
    Training data output

    returns:
    self: object
    """

    # Random generation of weights
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size = 1 + X.shape[1])

        self.cost_ = []
        for n in range(self.n_iter) :
            output = self.activation(self.net_input(X))
            errors = (y - output)
            self.w_[1:] += self.lr * X.T.dot(errors)
            self.w_[0] += self.lr * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
            print(cost , " for lr: " , self.lr)
        return self

    def predict(self, X):
        """Return the predicted output at this point"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Logistic Activation function"""
        return sigmoid(np.clip(X, -250, 250))



def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')


#divide the data in train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)
X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
lrgd = LogisticRegressionGD(lr=0.01, n_iter=1000, random_state=1)
lrgd.train(X_train_01_subset, y_train_01_subset)
plot_decision_regions(X_train_01_subset, y_train_01_subset, classifier=lrgd)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()