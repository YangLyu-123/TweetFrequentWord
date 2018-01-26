from noise_removal import *
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

# noise removal
X = normalize(X_train, axis=0)

# dbscan for noise removal
dbscan_labels = []
n_runs = 17 - 6
for i in range(6, 17):
    dbscan_labels.append(my_dbscan(X, i * 0.2, 3))
dbscan_labels = np.array(dbscan_labels)
n = dbscan_labels.shape[0] # the number of runs
m = dbscan_labels.shape[1] # the number of data
count = [0 for i in range(m)]
for i in range(n):
    for j in range(m):
        if dbscan_labels[i][j] == -1:
            count[j] += 1
outlier = [True for i in range(m)]
index_f = []
num = 0
# remove outlier if the number of noise labels > 0.5 * runs
for i in range(m):
    if count[i] > 0.5 * n_runs:
        outlier[i] = False
        num += 1
    else:
        index_f.append(i)

x = np.array(X_train.todense())
dbscan_filtered = x[outlier][:]
print(num)
labels = []
# consensus matrix
for i in range(2, 13):
    labels.append(kmeans(i, dbscan_filtered))
consensus_m = get_consensus_matrix(labels)
# kmeans for clustering
nc = 9
labels_9 = kmeans(nc, consensus_m)

# histogram of clustering
labels = ["cluster_" + str(x) for x in range(nc)]
population = [np.sum(labels_9 == x) for x in range(nc)]
y_pos = np.arange(len(labels))
barlist = plt.bar(y_pos, population, align='center', width=0.3)
plt.xticks(y_pos, labels)
plt.ylabel('Number of examples')
plt.title('Clustering of Tweets by DBSCAN noise removal')
plt.show()

# get the index of remaining data
index_f = np.array(index_f)
clusters_index = []
ind = np.array([i for i in range(0, len(consensus_m))])
c_i = []
for i in range(nc):
    clusters_index.append(index_f[labels_9 == i])
    c_i.append(ind[labels_9 == i])
stemed_word_set = set()
for line in processed_tweets:
    for word in line:
        stemed_word_set.add(word.lower())
# for word counting and find the top 5 popular word
count_word = [dict() for i in range(nc)]
for item in stemed_word_set:
    for i in range(nc):
        count_word[i][item] = 0
for i in range(nc):
    for index in clusters_index[i]:
        for word in processed_tweets[index]:
            count_word[i][word] += 1
count_sorted = []
for i in range(nc):
    count_sorted.append(sorted(count_word[i].items(), key=lambda x: x[1]))
# display top 5 word
num = 5
for i in range(nc):
    print(count_sorted[i][-num:])
# for nodes and edges information in visualization
iiiii = np.random.choice(len(consensus_m), 1500, replace=False)
grephi = open('./data/cluster_dbscan.csv', 'w')
for i in range(9):
    for line in c_i[i]:
        if line in iiiii:
            grephi.write(str(line) + ',')
            grephi.write(str(count_sorted[i][-1][0]) + ',')
            grephi.write(str(i) + '\n')
grephi.close()

c_m = open('./data/consensus_matrix_dbscan.csv', 'w')
for i in range(len(consensus_m)):
    if i in iiiii:
        temp = consensus_m[i][:]
        for j in range(i, len(temp)):
            if j in iiiii:
                if labels_9[i] == labels_9[j]:
                    c_m.write(str(i) + ',' + str(j))
                    c_m.write('\n')
                elif temp[j] > 6:
                    c_m.write(str(i) + ',' + str(j))
                    c_m.write('\n')
c_m.close()