from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn import preprocessing

# functions only used to compare data values to discuss socioeconomics patters
def plot_dataComparison(data_values):
    plt.plot()
    plt.scatter(data_values[:,0], data_values[:,1])
    plt.xlabel("Sexo")
    plt.ylabel("Coeficiente")
    plt.show()
    plt.scatter(data_values[:,2], data_values[:,3])
    plt.xlabel("Escola")
    plt.ylabel("Enem")
    plt.show()

    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(data_values[:,[2,3]])
    plt.scatter(data_values[:,2], data_values[:,3])
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='red')
    plt.xlabel("Escola")
    plt.ylabel("Enem")
    plt.show()
    exit()



def plot_WCSS(data, k_range):
    # calculate WCSS for each increasing number of clusters
    WCSS = []
    # we can use this loop to measure the score for each value in a file
    silhouette_score = []
    for i in range(1,k_range):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
        kmeans.fit(data)
        if i > 1: # so the number of kmeans.label_ is higher than 1
            silhouette_score.append("n_cluster: {} - score: {:.2f}".format(i,metrics.silhouette_score(data, kmeans.labels_)))
            sorted(silhouette_score, reverse=True)
        WCSS.append(kmeans.inertia_)

    # analyzing the top 5 best socres
    for ss in range(0,5): print(silhouette_score[ss])
    plt.plot(range(1,k_range), WCSS) # analysing the graph to get the best value for K
    plt.xlim([0,10])
    plt.show()



data_csv = pd.read_csv('alunos_engcomp-2023.csv')
# column 'Sexo': M=1 F=0
data_csv["Sexo"] = data_csv['Sexo'].apply(lambda x: 1 if x=='M' else 0)
# column 'Escola': Particular=1, Publica=0
data_csv["Escola"] = data_csv['Escola'].apply(lambda x: 1 if x=='Particular' else 0)


# rows count for testing values for K
K_range = len(data_csv.index)
# getting the values
data_values = data_csv.iloc[:,[0,1,2,3]].values
#plot_dataComparison(data_values)

# normalizing the data so all points has the same weight, no matter the distance
data_values = preprocessing.normalize(data_values)

# to minize the number of dimensions, PCA is used
pca = PCA(n_components=2)
pca_data = pca.fit_transform(data_values)


# The function below generates the image 'WCSS.png' and creates an array of best
# silhouette scores for each n_cluster
plot_WCSS(pca_data, K_range)

# So with the WCSS graph and the best silhouette score being 0.84, it is safe to conclude that
# K = 4 is the best value for K
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
Y = kmeans.fit_predict(pca_data)


plt.plot()
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=Y)
plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1], s=100, c='red')
plt.show()
