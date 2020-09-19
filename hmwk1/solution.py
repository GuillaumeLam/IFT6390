import numpy as np

banknote = np.genfromtxt('data_banknote_authentication.txt', delimiter=',')

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
	seed = abs(np.sum(x))
	while seed < 1:
		seed = 10 * seed
	seed = int(1000000 * seed)
	np.random.seed(seed)
	return np.random.choice(label_list)
#############################################


class Q1:

	def feature_means(self, banknote):
		return np.mean(banknote, axis=0)[:-1]

	def covariance_matrix(self, banknote):
		banknote = np.transpose(banknote[:,:-1])
		return np.cov(banknote)

	def feature_means_class_1(self, banknote):
		banknote = np.array([np.array(line) for line in banknote if line[-1] ==1])
		return self.feature_means(banknote)

	def covariance_matrix_class_1(self, banknote):
		banknote = np.array([np.array(line) for line in banknote if line[-1] ==1])
		return self.covariance_matrix(banknote)


class HardParzen:
	def __init__(self, h):
		self.h = h

	def train(self, train_inputs, train_labels):
		# self.label_list = np.unique(train_labels)
		pass

	def compute_predictions(self, test_data):
		pass


class SoftRBFParzen:
	def __init__(self, sigma):
		self.sigma  = sigma

	def train(self, train_inputs, train_labels):
		# self.label_list = np.unique(train_labels)
		pass

	def compute_predictions(self, test_data):
		pass


def split_dataset(banknote):
	pass


class ErrorRate:
	def __init__(self, x_train, y_train, x_val, y_val):
		self.x_train = x_train
		self.y_train = y_train
		self.x_val = x_val
		self.y_val = y_val

	def hard_parzen(self, h):
		pass

	def soft_parzen(self, sigma):
		pass


def get_test_errors(banknote):
	pass


def random_projections(X, A):
	pass


print(Q1().covariance_matrix_class_1(banknote))