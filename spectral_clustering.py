from __future__ import division
from gensim import corpora, models, similarities
import logging
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from numpy import *
from scipy import sparse
from scipy.sparse.linalg.eigen import arpack
import time

def calculate_similarities():
    courses = [line.strip() for line in open('new2000')]
    #courses_name = [course.split('\t')[0] for course in courses]
    #print(courses_name[0:10])

    #texts_lower = [[word for word in document.lower().split()]for document in courses]
    #print(texts_lower[0])
    print("begin to calculate similarity")

    '''
    texts_tokenized = [[word.lower() for word in word_tokenize(document)] for document in courses]
    print(texts_tokenized[0])

    english_stopwords = stopwords.words('english')
    #print(english_stopwords)
    print("filter stopwords")
    texts_filtered_stopwords = [[word for word in document if not word in english_stopwords] for document in texts_tokenized]
    print(texts_filtered_stopwords[0])

    print("remove punctuations")
    english_punctuations = [',','.',':',';','(',')','[',']','{','}','&','!','@','#','%','$','*']
    texts_filtered = [[word for word in document if not word in english_punctuations]for document in texts_filtered_stopwords]
    print(texts_filtered[0])

    print("stemmed")
    st = LancasterStemmer()
    texts_stemmed = [[st.stem(word) for word in document]for document in texts_filtered]
    print(texts_stemmed[0])

    all_stems = sum(texts_stemmed, [])
    stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
    texts = [[stem for stem in text if stem not in stems_once]for text in texts_stemmed]

    texts = texts_stemmed

    '''

    texts = [[word for word in document] for document in courses]

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
    #print(row, column)
    for i in range(row):
        for j in range(column):
            vectors[i][j] = vectors[i][j] / sqrt_sum[i]
            #print(vectors[i][j])
        #print(" ")
    print("perform k-means")
    centers, clusters_category = kMeans(vectors, clusters)

    '''
    for i in range(shape(clusters_category)[0]):
        print(clusters_category[i,0])
    '''
    return clusters_category

def random_centers(data, clusters):
    k = shape(data)[1]
    centers = mat(zeros((clusters,k)))
    for j in range(k):
        minum = min(data[:,j])
        #print(minum)
        distance = max(data[:,j]) - minum
        #print(distance)
        centers[:,j] = mat(minum.real + distance.real * random.rand(k,1))
    return centers

def euclidean_metric(vec1, vec2):
    return sqrt(sum(power(vec1 - vec2, 2)))

def kMeans(data, clusters):
    n = shape(data)[0]
    clusters_category = mat(zeros((n,2)))
    centers = random_centers(data, clusters)
    changed = True
    while changed:
        print("perform k-means", time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))
        changed = False
        for i in range(n):
            min_distance = inf; cluster_index = -1
            for j in range(clusters):
                distance = euclidean_metric(centers[j,:],data[i,:])
                if distance < min_distance:
                    min_distance = distance; cluster_index = j

            if clusters_category[i,0] != cluster_index:
                changed = True
                clusters_category[i,:] = cluster_index, min_distance.real**2

        for center in range(clusters):
            points_of_cluster = data[nonzero(clusters_category[:,0].A==center)[0]]
            centers[center,:] = mean(points_of_cluster.real,axis=0)

    return centers, clusters_category

if __name__ == '__main__':
    '''
    similarity = calculate_similarities()
    clusters_category = spectral_clustering(similarity,20,10)
    #clusters_category.dtype = 'int32'
    n = shape(clusters_category)[0]
    result = [0]*10
    clusters_category_int = clusters_category.astype(int)
    for i in range(n):
        print(clusters_category_int[i,0])
        result[clusters_category_int[i, 0]]+=1
    '''
    result = [172, 292, 269, 265, 87, 113, 257, 144, 201, 81]
    result.sort()
    print(result)

    '''
    n = shape(clusters_category)[0]
    name_cluster = [("none", 0)] * n
    for i in range(n):
        tup = (clusters_category[i,0], courses_name[i])
        name_cluster[i] = tup
        #print(tup)
    name_cluster.sort(key=lambda x:x[0])
    #print(name_cluster)
    print("write result to file")
    f = open("result", "w")
    for i in range(n):
        tup = name_cluster[i]
        cluster = tup[0]; name = tup[1]
        item = str(cluster) + " " + str(name) + "\n"
        f.write(item)
    f.close()
    '''
    print("complete")



