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
		banknote = np.array([np.array(line) for line in banknote if line[-1] == 1])
		return self.feature_means(banknote)

	def covariance_matrix_class_1(self, banknote):
		banknote = np.array([np.array(line) for line in banknote if line[-1] == 1])
		return self.covariance_matrix(banknote)

class HardParzen:
	def __init__(self, h):
		self.h = h

	def train(self, train_inputs, train_labels):
		self.label_list = np.unique(train_labels)
		self.train_data = np.append(np.array(train_inputs), np.transpose(np.array([train_labels])), axis=1)

	def compute_predictions(self, test_data):
		Y_test = []
		for X_test in test_data:
			dist_h_points = []
			for point in self.train_data:
				if np.linalg.norm(point[:-1]-X_test) <= self.h:
					dist_h_points.append(point[-1])
			if not len(dist_h_points) == 0:
				Y_test.append(np.argmax(np.bincount(dist_h_points)))
			else: 
				Y_test.append(draw_rand_label(X_test, self.label_list))

		return Y_test


class SoftRBFParzen:
	def __init__(self, sigma):
		self.sigma  = sigma

	def train(self, train_inputs, train_labels):
		self.label_list = np.unique(train_labels)
		self.train_data = np.append(np.array(train_inputs), np.transpose(np.array([train_labels])), axis=1)

	def compute_predictions(self, test_data):
		Y_test = []
		for X_test in test_data:
			kernel_dist = {}
			for point in self.train_data:
				dist = np.linalg.norm(point[:-1]-X_test)

				rbf_w = np.exp(-dist**2/(2*self.sigma**2))

				if point[-1] in kernel_dist:
					kernel_dist[point[-1]] += rbf_w
				else:
					kernel_dist[point[-1]] = rbf_w

			Y_test.append(max(kernel_dist, key=kernel_dist.get))
		return Y_test


def split_dataset(banknote):
	train = np.array([line for i, line in enumerate(banknote) if (i%5 == 0 or i%5 == 1 or i%5 == 2)])
	val = np.array([line for i, line in enumerate(banknote) if (i%5 == 3)])
	test = np.array([line for i, line in enumerate(banknote) if (i%5 == 4)])
	return train, val, test


class ErrorRate:
	def __init__(self, x_train, y_train, x_val, y_val):
		self.x_train = x_train
		self.y_train = y_train
		self.x_val = x_val
		self.y_val = y_val

	def hard_parzen(self, h):
		p = HardParzen(h)
		p.train(self.x_train, self.y_train)

		y_pred = p.compute_predictions(self.x_val)

		res = np.array([0 if pred[0] == pred[1] else 1 for pred in zip(y_pred, self.y_val)])

		return np.sum(res)/len(res)

	def soft_parzen(self, sigma):
		p = SoftRBFParzen(sigma)
		p.train(self.x_train, self.y_train)

		y_pred = p.compute_predictions(self.x_val)

		res = np.array([0 if pred[0] == pred[1] else 1 for pred in zip(y_pred, self.y_val)])

		return np.sum(res)/len(res)


def get_test_errors(banknote):
	train, val, test = split_dataset(banknote)

	er = ErrorRate(np.array(train)[:,:-1], np.array(train)[:,-1], np.array(val)[:,:-1], np.array(val)[:,-1])

	hyperparam = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 3, 10, 20]
	
	hard_p_err = []
	soft_p_err = []

	for val in hyperparam:
		hard_p_err.append(er.hard_parzen(val))
		soft_p_err.append(er.soft_parzen(val))

	h_star = hyperparam[np.argmin(hard_p_err)]
	s_star = hyperparam[np.argmin(soft_p_err)]

	er = ErrorRate(np.array(train)[:,:-1], np.array(train)[:,-1], np.array(test)[:,:-1], np.array(test)[:,-1])

	return [er.hard_parzen(h_star), er.soft_parzen(s_star)]

def random_projections(X, A):
	proj_X = np.empty((0,2), int)
	
	for x in X: 
		proj_X = np.append(proj_X, np.multiply(1/np.sqrt(2), np.array([np.dot(np.transpose(A),x)])), axis=0)
		
	return proj_X


def get_test_projection_error(banknote):
	A = np.random.normal(0, 1, size=(4, 2))

	hyperparam = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 3, 10, 20]

	hard_p_val = np.empty((0,10), float)
	soft_p_val = np.empty((0,10), float)

	for i in range(500):
		A = np.random.normal(0, 1, size=(4, 2))
		proj_banknote = np.array(random_projections(banknote[:,:-1], A))

		proj_banknote = np.append(proj_banknote, np.transpose([banknote[:,-1]]), axis=1)

		train, val, test = split_dataset(proj_banknote)

		er = ErrorRate(np.array(train)[:,:-1], np.array(train)[:,-1], np.array(val)[:,:-1], np.array(val)[:,-1])

		hard_p_err = []
		soft_p_err = []

		for val in hyperparam:
			hard_p_err.append(er.hard_parzen(val))
			soft_p_err.append(er.soft_parzen(val))

		hard_p_val = np.append(hard_p_val, [hard_p_err], axis=0)
		soft_p_val = np.append(soft_p_val, [soft_p_err], axis=0)

	return (hard_p_val, soft_p_val)

print(get_test_errors(banknote))


# TODO:
# Q1		DONE
# Q2		DONE
# Q3		DONE
# Q4		DONE
# Q5		DONE
# Q5 report
# Q6		DONE
# Q7 report		
# Q8		DONE
# Q9		DONE
# Q9 report		