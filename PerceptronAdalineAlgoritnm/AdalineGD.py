import numpy as np

class AdalineGD(object):

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
          Sum of squares cost function value in each epoch
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
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Linear(Identity) Activation function"""
        return X