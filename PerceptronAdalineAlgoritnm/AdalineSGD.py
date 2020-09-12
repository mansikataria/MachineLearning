import numpy as np


class AdalineSGD(object):

    """
    parameters:
    lr : float(between 0 and 1)
    learning rate

    n_iter : int
    Number of iterations over training data

    random_state : int
    Used to generate random weights

    shuffle: boolean
    Shuffle training samples before each iteration if set to true


    attributes:
    w_ : 1-d array
         Weights at the end of training
    cost_ : list
          Sum of squares cost function value in each epoch
    """

    def __init__(self, lr=0.01, n_iter=50, random_state=1, shuffle=True):
        self.lr = lr
        self.n_iter = n_iter
        self.random_state = random_state
        self.shuffle = shuffle
        self.w_initialized = False



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
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for n in range(self.n_iter) :
            if(self.shuffle) :
                X, y = self._shuffle(X, y)

            cost = []
            for xi, target in zip(X, y):
                new_w = self._update_weights(xi, target)
                cost.append(new_w)
            avg_cost = sum(cost) / len(cost)
            self.cost_.append(avg_cost)
        return self


    def partial_fit(self, X, y):
        """Fit training data without re-initializing weight"""
        if not self.w_initialized :
            self._initialize_weights(X)
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
             self._update_weights(X, y)
        return self


    def _update_weights(self, x, y):
        output = self.activation(self.net_input(x))
        error = (y - output)
        self.w_[1:] += self.lr * x.dot(error)
        self.w_[0] += self.lr * error
        cost = (error ** 2).sum() / 2.0
        return cost


    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]


    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size = 1 + m)
        self.w_initialized = True

    def predict(self, X):
        """Return the predicted output at this point"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Linear(Identity) Activation function"""
        return X