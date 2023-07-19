from sklearn.datasets import make_blobs
from keras.utils import to_categorical

### Multi-class classification problem


def create_dataset(n_samples, centers, n_features, cluster_std, random_state):

	# generate 2d classification dataset
	X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features, cluster_std=cluster_std, 
		   random_state=random_state)
	# one hot encode output variable
	y = to_categorical(y)
	# split into train and test
	n_train = 500
	trainX, testX = X[:n_train, :], X[n_train:, :]
	trainy, testy = y[:n_train], y[n_train:]
	return trainX, trainy, testX, testy