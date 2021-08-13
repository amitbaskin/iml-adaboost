"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Gad Zalcberg
Date: February, 2019

"""
from ex4_tools import *


TRAIN_SAMPLES_AMOUNT = 5000
TEST_SAMPLES_AMOUNT = 200
DEFAULT_NOISE = 0
Q14_NOISES = [0.01, 0.4]
DEFAULT_CLASSIFIERS_AMOUNT = 500
CLASSIFIERS_AMOUNTS_LST = [5, 10, 50, 100, 200, 500]


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None]*T     # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the
        last iteration.
        """
        m = len(X)
        D = np.ones(m) / m
        for i in range(self.T):
            h = self.WL(D, X, y)
            h.train(D, X, y)
            y_pred = h.predict(X)
            self.h[i] = h
            zero_indices = np.where(y_pred == 0)[0]
            y_pred[zero_indices] = -1
            compare = (y != y_pred).astype(int)
            epsilon = D @ compare
            epsilon_inverse = 1 / epsilon
            in_log = epsilon_inverse - 1
            curr_log = np.log(in_log)
            w = 0.5 * curr_log
            self.w[i] = w
            y_and_y_pred_product = np.multiply(y, y_pred)
            in_exp = -w * y_and_y_pred_product
            exp_arr = np.exp(in_exp)
            D = np.multiply(D, exp_arr)
            D /= np.sum(D)
        return D

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for
        the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        lst = [self.w[i] * self.h[i].predict(X) for i in range(max_t)]
        sign = np.sign(np.sum(lst, axis=0))
        return sign

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for
        the classification
        :return: error : the ratio of the wrong predictions when predict only
        with max_t weak learners (float)
        """
        pred = self.predict(X, max_t)
        return np.sum(pred != y) / len(y)


class ExeSolver:
    def __init__(self, noise):
        self.noise = noise
        self.train_samples = generate_data(TRAIN_SAMPLES_AMOUNT, noise)
        self.X_train, self.y_train = self.train_samples[0], \
                                     self.train_samples[1]
        self.adaboost = AdaBoost(DecisionStump, DEFAULT_CLASSIFIERS_AMOUNT)
        self.D = self.adaboost.train(self.X_train, self.y_train)
        self.test_samples = generate_data(TEST_SAMPLES_AMOUNT, noise)
        self.X_test = self.test_samples[0]
        self.y_test = self.test_samples[1]


def q10(exe_solver):
    T = range(1, DEFAULT_CLASSIFIERS_AMOUNT+1)
    train_errors = \
        [exe_solver.adaboost.error(
            exe_solver.X_train, exe_solver.y_train, i) for i in T]
    test_errors = \
        [exe_solver.adaboost.error(
            exe_solver.X_test, exe_solver.y_test, i) for i in T]
    plt.plot(T, train_errors, label='train set')
    plt.plot(T, test_errors, label='test set')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('error ratio of an Adaboost classifier\n'
              'based on a DecisionStump weak learner,\n'
              'as a function of the number of weak learners\n'
              'the samples are taken with {} noise'.format(exe_solver.noise))
    plt.xlabel('number of weak learners')
    plt.ylabel('error ratio')
    plt.show()


def q11(exe_solver):
    counter = 1
    errors_and_classifiers_amounts = []
    for t in CLASSIFIERS_AMOUNTS_LST:
        plt.subplot(2, 3, counter)
        error = decision_boundaries(exe_solver.adaboost, exe_solver.X_test,
                                    exe_solver.y_test, t)
        errors_and_classifiers_amounts.append((error, t))
        counter += 1
    plt.show()
    return min(errors_and_classifiers_amounts, key=lambda x: x[0])[1]


def q12(exe_solver):
    amount = q11(exe_solver)
    decision_boundaries(
        exe_solver.adaboost, exe_solver.X_train, exe_solver.y_train, amount)
    plt.show()


def q13(exe_solver):
    new_D = (exe_solver.D / np.max(exe_solver.D) * 10).T
    decision_boundaries(exe_solver.adaboost, exe_solver.X_train,
                        exe_solver.y_train, DEFAULT_CLASSIFIERS_AMOUNT,
                        weights=new_D)
    plt.show()


def run_questions(noise):
    exe_solver = ExeSolver(noise)
    q10(exe_solver)
    q12(exe_solver)
    q13(exe_solver)


def run_zero_noise():
    run_questions(0)


def q14():
    for noise in Q14_NOISES:
        run_questions(noise)


# run questions 10-13 with zero noise
# run_zero_noise()

# run question 14
# q14()
