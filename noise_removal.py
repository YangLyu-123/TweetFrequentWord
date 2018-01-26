from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from process_tweets import *
import numpy as np


# for TF-IDF
vectorizer = TfidfVectorizer()

# vectorize text
# input : a nested word list
# output : a score matrix
def tf_idf(word_lists):
    return vectorizer.fit_transform(word_lists)

# kmeans
# input : number of clusters, feature matrix
# output : cluster labels
def kmeans(num_cluster, X):
    return KMeans(n_clusters=num_cluster).fit_predict(X)

# dbscan
# input : feature matrix, eps, min_port
# output : cluster labels
def my_dbscan(X, eps, min_p):
    return DBSCAN(eps=eps, min_samples=min_p).fit_predict(X)

# consensus matrix
# input : the result labels of several times clusters
# output : consensus matrix
def get_consensus_matrix(labels_lists):
    # n samples
    n = len(labels_lists[0])
    consensus_m = np.zeros((n, n))
    for label in labels_lists:
        for i in range(consensus_m.shape[0]):
            for j in range(i, consensus_m.shape[1]):
                if label[i] == label[j]:
                    consensus_m[i][j] += 1
                    if i != j:
                        consensus_m[j][i] += 1
    consensus_m = np.array(consensus_m)
    return consensus_m


# noise removal
# input : consensus matrix, the number of runs, threshold
# output : filtered consensus matrix, indices remained
def remove_noise(c_m, n_runs, threshold):
    # set c_m[i][j] = 0 when a pair (i, j) of tweets did not cluster together more than 10% of the total number of runs
    m = c_m.shape[0]
    index_f = []
    for i in range(m):
        c_m[i][(c_m[i] < 0.1 * n_runs)] = 0
        if sum(c_m[i]) >= threshold * 0.7:
            index_f.append(i)
    return [c_m[index_f][:], index_f]

# process tweets
tweets = get_tweets()
processed_tweets = tokenize_tweets(tweets)
sentence_tweets = []
for line in processed_tweets:
    sentence_tweets.append(words_to_sentence(line))
X_train = tf_idf(sentence_tweets)