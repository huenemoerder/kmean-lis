#from k_means_constrained import KMeansConstrained
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.neighbors import KDTree

# Just use the closest centroid to identify the partition
class BaselineClassifier:
    def __init__(self, centers):
        self.centers = centers
        
    def fit(self, X, y):
        pass
        
    def predict(self, X):
        y_pred = np.argmin(euclidean_distances(X, self.centers), axis=1)
        return y_pred

def run(n_samples, n_features, n_clusters, cluster_std, k, n_test, random_seed, kd_tree=False):

    np.random.seed(random_seed)

    X_org, _ = make_blobs(n_samples=n_samples + n_test, centers=n_clusters, n_features=n_features, cluster_std=cluster_std, random_state=random_seed)
    X_index = X_org[:n_samples]
    X_query = X_org[n_samples:]

    experiment = {}
    experiment['seed'] = random_seed
    experiment['n_samples'] = n_samples
    experiment['n_features'] = n_features
    experiment['n_clusters'] = n_clusters
    experiment['k'] = k
    experiment['std'] = cluster_std

    X_clust = None
    centers = None
    y = None

    if kd_tree:
        tree = KDTree(X_index, leaf_size=1000)   
        kd_clustering = pd.DataFrame(tree.get_arrays()[2])
        clustering = kd_clustering[kd_clustering['is_leaf'] == True][['idx_start', 'idx_end']]
        label_list = []
        for c in range(clustering.shape[0]):
            row = clustering.iloc[c]
            label_list.append(np.full(row[1] - row[0], c))
        y = np.concatenate(label_list, axis=0)

        X_index = X_index[tree.get_arrays()[1]]

        X_clust = np.zeros((n_samples, n_features + 1))
        X_clust[:, :-1] = X_index
        X_clust[:, -1] = y

        centers = pd.DataFrame(X_clust).groupby(n_features).mean().values

    else:
        clust = KMeans(n_clusters=k, random_state=0, n_jobs=-1)
        clust.fit_predict(X_index)
        y = clust.labels_
        centers = clust.cluster_centers_

        X_clust = np.zeros((n_samples, n_features + 1))
        X_clust[:, :-1] = X_index
        X_clust[:, -1] = y

    X_train, X_valid, y_train, y_valid = train_test_split(X_index, y, test_size=0.2)

    base = BaselineClassifier(centers)
    dct = DecisionTreeClassifier()
    nb = GaussianNB()
    linear_svm = svm.SVC(kernel='linear')
    radial_svm = svm.SVC(kernel='rbf')
    rf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    mlp = MLPClassifier(hidden_layer_sizes=(150,100,50), activation='relu', solver='adam', max_iter=500)
    classifiers = [base, nb, dct, rf, linear_svm, radial_svm, mlp]
    names = ['Base Model', 'Naive Bayes', 'Decision Tree', 'Random Forest','Linear SVM', 'RBF SVM',
            'MLP']

    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        y_pred_train = clf.predict(X_train)
        y_pred_valid = clf.predict(X_valid)
        errors = evaluate(X_index, X_clust, X_query, clf)
        train_acc = accuracy_score(y_train, y_pred_train)
        experiment[f'{name}_train'] = train_acc
        valid_acc = accuracy_score(y_valid, y_pred_valid)
        experiment[f'{name}_valid'] = valid_acc
        experiment[f'{name}_mean_error'] = np.mean(errors)
        #test_acc = nn_accuracy(errors)
        #experiment[f'{name}_test'] = test_acc
        #print(random_seed)
        #print(f'{name}: train: {train_acc}, valid: {valid_acc}, error: {np.mean(errors)}')
        #pd.DataFrame(errors).plot()
    
    return experiment, X_clust


def evaluate(X_index, X_clust, X_query, clf):
    n_features = X_index.shape[1]

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(X_index)
    n_1 = nbrs.kneighbors(X_query, return_distance=False)

    errors = []
    for i in range(0, X_query.shape[0]):
        X_found = X_clust[X_clust[:, n_features] == np.asarray(clf.predict(X_query[i].reshape(1, -1)), dtype=np.integer)][:, :n_features]
        nbrs_2 = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(X_found)
        n_2 = nbrs_2.kneighbors([X_query[i]])[1][0][0]
        real_dist = np.linalg.norm(X_index[n_1[i]] - X_query[i])
        #print(X_index[n_1[i]][0])
        approx_dist = np.linalg.norm(X_found[n_2] - X_query[i])
        #print(X_found[n_2])
        error = np.abs(1 - (approx_dist / real_dist))
        #print(error)
        #print(np.sum(np.abs(X_index[n_1[i]] - X_found[n_2])))
        errors.append(error)
    return errors

def nn_accuracy(errors):
    return (len(errors) - np.count_nonzero(errors)) / len(errors)