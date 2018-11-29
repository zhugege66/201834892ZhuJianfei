from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score
word_list = []
label_list = []


for line in open('Tweets.txt','r').readlines():
    dic = eval(line)
    word_list.append(dic["text"])
    label_list.append(dic['cluster'])

vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(word_list))
print(label_list)
print(word_list)
#K-means算法
km = KMeans(n_clusters=89,max_iter=300,n_init=40,init='k-means++',n_jobs=1)
result_kmeans = km.fit_predict(tfidf.toarray())
#AffinityPropagation算法
ap = AffinityPropagation(damping=0.55, max_iter=575, convergence_iter=575, copy=True, preference=None, affinity='euclidean', verbose=False)
result_ap = ap.fit_predict(tfidf.toarray())
#meanshift算法
ms = MeanShift(bandwidth = 0.65, bin_seeding = True)
result_ms = ms.fit_predict(tfidf.toarray())
#SpectralClustering算法
sc = SpectralClustering(n_clusters=89,affinity='nearest_neighbors',n_neighbors=4,eigen_solver='arpack',n_jobs=1)
result_sc = sc.fit_predict(tfidf.toarray())
#DBSCAN算法
db = DBSCAN(eps=0.7, min_samples=1)
result_db = db.fit_predict(tfidf.toarray())
#AgglomerativeClustering算法
ac = AgglomerativeClustering(n_clusters = 89, affinity = 'euclidean', linkage = 'ward')
result_ac = ac.fit_predict(tfidf.toarray())
#GaussianMixture算法
gm = GaussianMixture(n_components=89,covariance_type='diag', max_iter=20, random_state=0)
#for cov_type in ['spherical', 'diag', 'tied', 'full']
gm.fit(tfidf.toarray())
result_gm = gm.predict(tfidf.toarray())
print('K-means的准确率:',normalized_mutual_info_score(result_kmeans,label_list))
print('AffinityPropagation算法的准确率:',normalized_mutual_info_score(result_ap,label_list))
print('meanshift算法的准确率:',normalized_mutual_info_score(result_ms,label_list))
print('SpectralClustering算法的准确率:',normalized_mutual_info_score(result_sc,label_list))
print('DBSCAN算法的准确率:',normalized_mutual_info_score(result_db,label_list))
print('AgglomerativeClustering算法的准确率:',normalized_mutual_info_score(result_ac,label_list))
print('GaussianMixture算法的准确率:',normalized_mutual_info_score(result_gm,label_list))