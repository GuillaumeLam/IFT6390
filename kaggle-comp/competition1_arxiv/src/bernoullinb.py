import numpy as np

class BernoulliNB:

    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        n_classes = self.n_classes

        self.counts = np.zeros(n_classes)
        for i in y:
            self.counts[int(i)] += 1
        self.counts /= len(y)

        self.params = np.zeros((n_classes, X.shape[1]))
        for idx in range(len(X)):
            self.params[int(y[idx])] += X[idx]
        self.params += self.alpha 

        self.class_sums = np.zeros(self.n_classes)
        for i in y:
            self.class_sums[int(i)] += 1
        self.class_sums += self.n_classes*self.alpha 

        self.params = self.params / self.class_sums[:, np.newaxis]

    def predict(self, X):
        neg_prob = np.log(1 - self.params)
        jll = np.dot(X, (np.log(self.params) - neg_prob).T)
        jll += np.log(self.counts) + neg_prob.sum(axis=1)
        return np.argmax(jll, axis=1)
