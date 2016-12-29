from __future__ import division
from gensim import corpora, models, similarities
import logging
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import numpy
from scipy import sparse
from scipy.sparse.linalg.eigen import arpack
import time
from sklearn import cluster

def calculate_similarities():
    items = [line.strip() for line in open('new2000')]

    print("begin to calculate similarity")

    '''
    texts_tokenized = [[word.lower() for word in word_tokenize(document)] for document in items]

    english_stopwords = stopwords.words('english')
    print("filter stopwords")
    texts_filtered_stopwords = [[word for word in document if not word in english_stopwords] for document in texts_tokenized]

    print("remove punctuations")
    english_punctuations = [',','.',':',';','(',')','[',']','{','}','&','!','@','#','%','$','*']
    texts_filtered = [[word for word in document if not word in english_punctuations]for document in texts_filtered_stopwords]


    #stemmed words
    print("stemmed")
    st = LancasterStemmer()
    texts_stemmed = [[st.stem(word) for word in document]for document in texts_filtered]
    '''

    #new2000 has been processed
    texts = [[word for word in document] for document in items]
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    dictionary = corpora.Dictionary(texts)

    corpus = [dictionary.doc2bow(text) for text in texts]

    print("perform tfidf")
    tfidf = models.TfidfModel(corpus)

    corpus_tfidf = tfidf[corpus]

    print("perform lsi")
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10)

    index = similarities.MatrixSimilarity(lsi[corpus])

    return index

'''
def spectral_clustering(similarity, sigma, clusters):
    print("get matrix S")
    sparse_similarity = sparse.csr_matrix(similarity)
    data = sparse.csc_matrix.multiply(sparse_similarity, sparse_similarity)
    data = -data / (2 * sigma * sigma)
    S = sparse.csc_matrix.expm1(data) + sparse.csc_matrix.multiply(sparse.csc_matrix.sign(data),sparse.csc_matrix.sign(data))

    print("get matrix L")
    D = S.sum(1)
    D = sqrt(1 / D)
    n = len(D)
    D = D.T
    D = sparse.spdiags(D, 0, n, n)
    L = D * S * D

    print("get the eigenvalue and eigenvector")
    values, vectors = arpack.eigs(L, k=clusters,tol=0,which="LM")

    print(" vectors normalization")

    sqrt_sum = sqrt(multiply(vectors,vectors).sum(1))
    row, column = shape(vectors)

    for i in range(row):
        for j in range(column):
            vectors[i][j] = vectors[i][j] / sqrt_sum[i]
    print("perform k-means")

    clusters_category = sklearn_kmeans(vectors, clusters)

    return clusters_category

def sklearn_kmeans(data, clusters):
    k_means = cluster.KMeans(n_clusters=clusters)
    #data_int =data.astype(int)
    k_means.fit(data)

    return k_means.labels_

'''

def get_presort_cluster():
    f = open('cluster')
    result = []
    for line in f.readlines():
        result.append(int(line.strip()))
    return result



if __name__ == '__main__':
    presort_cluster = get_presort_cluster()

    '''
    a = numpy.array([[1,2,3],
            [4,5,6]])
    print(type(a))
    '''

    clusters = 10
    sigma = 20
    similarity = calculate_similarities()
    #print(similarity)
    ndarray_similarity = numpy.array(similarity)
    float_ndarray_similarity= ndarray_similarity.astype(numpy.float)
    labels = cluster.spectral_clustering(float_ndarray_similarity, n_clusters=10, eigen_solver='arpack')
    #print(type(labels[0]))
    n = numpy.shape(presort_cluster)[0]
    check_result = [[0 for i in range(clusters)] for j in range(clusters)]
    for i in range(n):
        check_result[presort_cluster[i]][labels[i]] += 1

    sum = 0
    for i in range(clusters):
        for j in range(clusters):
            print(check_result[i][j], end="")
            print(",", end="")
            sum += check_result[i][j]
        print()

    print(sum)

    print("for echars")
    for i in range(clusters):
        for j in range(clusters):
            if (i < j):
                check_result[i][j], check_result[j][i] = check_result[j][i], check_result[i][j]
            print(check_result[i][j], end="")
            print(",", end="")
        print()
    '''
    clusters_category = spectral_clustering(similarity,sigma,clusters)

    print(clusters_category)

    n = shape(clusters_category)[0]
    check_result = [[0 for i in range(clusters)] for j in range(clusters)]
    for i in range(n):
        check_result[presort_cluster[i]][clusters_category[i]] += 1

    sum = 0
    for i in range(clusters):
        for j in range(clusters):
            print(check_result[i][j],end="")
            print(",",end="")
            sum+=check_result[i][j]
        print()

    print(sum)

    print("for echars")
    for i in range(clusters):
        for j in range(clusters):
            if(i<j):
                check_result[i][j],check_result[j][i] = check_result[j][i],check_result[i][j]
            print(check_result[i][j], end="")
            print(",", end="")
        print()
    '''




