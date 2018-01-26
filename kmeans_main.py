from noise_removal import *
import matplotlib.pyplot as plt

# kmeans k from 2 to 12
kmeans_labels = []
for i in range(2, 10):
    kmeans_labels.append(kmeans(i, X_train))
# consensus matrix
n_runs = 13 - 2
kmeans_cm = get_consensus_matrix(kmeans_labels)
for i in range(len(kmeans_cm)):
    for j in range(len(kmeans_cm[i])):
        if kmeans_cm[i][j] < 0.1 * n_runs:
            kmeans_cm[i][j] = 0
kmeans_cm = np.array(kmeans_cm)
threshold = np.mean(np.sum(kmeans_cm, axis=0))
num = 0
outlier_f = [True for i in range(kmeans_cm.shape[0])]
index_f = []
for i in range(kmeans_cm.shape[0]):
    if sum(kmeans_cm[i]) < threshold * 0.6:
        num += 1
        outlier_f[i] = False
    else:
         index_f.append(i)
print(num)
x = np.array(X_train.todense())
kmeans_filtered = x[outlier_f][:]
labels = []
for i in range(2, 13):
    labels.append(kmeans(i, kmeans_filtered))
consensus_m = get_consensus_matrix(labels)

labels_9 = kmeans(9, consensus_m)

# histogram of clustering
labels = ["cluster_" + str(x) for x in range(9)]
population = [np.sum(labels_9 == x) for x in range(9)]
y_pos = np.arange(len(labels))
barlist = plt.bar(y_pos, population, align='center', width=0.3)
plt.xticks(y_pos, labels)
plt.ylabel('Number of examples')
plt.title('Clustering of Tweets by Kmeans noise removal')
plt.show()

index_f = np.array(index_f)
ind = [i for i in range(0,len(consensus_m))]
clusters_index = []
c_i = []
ind = np.array([i for i in range(len(consensus_m))])
for i in range(9):
    clusters_index.append(index_f[labels_9 == i])
    c_i.append(ind[labels_9 == i])
stemed_word_set = set()
for line in processed_tweets:
    for word in line:
        stemed_word_set.add(word.lower())
# for word counting and find the top 5 popular word
count_word = [dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict()]
for item in stemed_word_set:
    for i in range(9):
        count_word[i][item] = 0
for i in range(9):
    for index in clusters_index[i]:
        for word in processed_tweets[index]:
            count_word[i][word] += 1
count_sorted = []
for i in range(9):
    count_sorted.append(sorted(count_word[i].items(), key=lambda x: x[1]))


num = 5
for i in range(9):
    print(count_sorted[i][-num:])
iiiii = np.random.choice(len(consensus_m), 1500, replace=False)

grephi = open('./data/cluster_kmeans.csv', 'w')
for i in range(9):
    for line in c_i[i]:
        if line in iiiii:
            grephi.write(str(line) + ',')
            grephi.write(str(count_sorted[i][-1][0]) + ',')
            grephi.write(str(i) + '\n')
grephi.close()

c_m = open('./data/consensus_matrix_kmeans.csv', 'w')
for i in range(len(consensus_m)):
    if i in iiiii:
        temp = consensus_m[i][:]
        for j in range(i, len(temp)):
            if j in iiiii:
                if labels_9[i] == labels_9[j]:
                    c_m.write(str(i) + ',' + str(j))
                    c_m.write('\n')
                elif temp[j] > 7:
                    c_m.write(str(i) + ',' + str(j))
                    c_m.write('\n')
c_m.close()