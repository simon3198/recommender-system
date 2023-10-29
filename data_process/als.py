import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm_notebook as tqdm


# Base code : https://github.com/mickeykedia/Matrix-Factorization-ALS/blob/master/ALS%20Python%20Implementation.py
class AlternatingLeastSquares:
    def __init__(self, R, k, reg_param, epochs, verbose=False):
        """
        :param R: rating matrix
        :param k: latent parameter
        :param learning_rate: alpha on weight update
        :param reg_param: beta on weight update
        :param epochs: training epochs
        :param verbose: print status
        """
        self._R = R
        self._num_users, self._num_items = R.shape
        self._k = k
        self._reg_param = reg_param
        self._epochs = epochs
        self._verbose = verbose

    def fit(self):
        # init latent features
        self._users = np.random.normal(size=(self._num_users, self._k))
        self._items = np.random.normal(size=(self._num_items, self._k))

        # train while epochs
        self._training_process = []
        self._user_error = 0
        self._item_error = 0
        for epoch in range(self._epochs):
            for i, Ri in enumerate(self._R):
                self._users[i] = self.user_latent(i, Ri)
                self._user_error = self.cost()

            for j, Rj in enumerate(self._R.T):
                self._items[j] = self.item_latent(j, Rj)
                self._item_error = self.cost()

            cost = self.cost()
            self._training_process.append((epoch, cost))

            # print status
            if self._verbose == True and ((epoch + 1) % 10 == 0):
                print("Iteration: %d ; cost = %.4f" % (epoch + 1, cost))

    def cost(self):
        """
        compute root mean square error
        :return: rmse cost
        """
        xi, yi = self._R.nonzero()
        cost = 0
        for x, y in zip(xi, yi):
            cost += pow(self._R[x, y] - self.get_prediction(x, y), 2)
        return np.sqrt(cost / len(xi))

    def user_latent(self, i, Ri):
        """
        :param error: rating - prediction error
        :param i: user index
        :param Ri: Rating of user index i
        :return: convergence value of user latent of i index
        """

        du = np.linalg.solve(
            np.dot(self._items.T, np.dot(np.diag(Ri), self._items)) + self._reg_param * np.eye(self._k),
            np.dot(self._items.T, np.dot(np.diag(Ri), self._R[i].T)),
        ).T
        return du

    def item_latent(self, j, Rj):
        """
        :param error: rating - prediction error
        :param j: item index
        :param Rj: Rating of item index j
        :return: convergence value of itemr latent of j index
        """

        di = np.linalg.solve(
            np.dot(self._users.T, np.dot(np.diag(Rj), self._users)) + self._reg_param * np.eye(self._k),
            np.dot(self._users.T, np.dot(np.diag(Rj), self._R[:, j])),
        )
        return di

    def get_prediction(self, i, j):
        """
        get predicted rating: user_i, item_j
        :return: prediction of r_ij
        """
        return self._users[i, :].dot(self._items[j, :].T)

    def get_complete_matrix(self):
        """
        :return: complete matrix R^
        """
        return self._users.dot(self._items.T)
