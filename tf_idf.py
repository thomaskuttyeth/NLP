
'''
TF-IDF method
========================================
it shows how relevant a word in a series or corpus is to a text. 
TF-IDF is a frequency-based method that takes into account the frequency with
which a word appears in a corpus. This is a word representation in the sense that it
represents the importance of a specific word in a given document. Intuitively, the
higher the frequency of the word, the more important that word is in the document.

TF stands for term frequency and IDF stands for inverse document frequency:

TF(wi) = number of times wi appear / total number of words
IDF(wi) = log(total number of documents / number of documents with wi in it)

TF-IDF(wi) = TF(wi) x IDF(wi)

for example :
 sentence 1 = 'This is about cats. Cats are great companions'
 sentence 2 = 'This is about dogs. Dogs are very loyal'

tf-idf(cats,sentenc1) = (2/8) * log(2/1) = 0.075
tf-idf(this, senetence2) = (1/8) * log(2/2) = 0.0
'''

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
corpus = ['This is about cats. cats are great companions',
          'This is about dogs. dogs are very loyal']

tfidf = TfidfVectorizer()
tfidf_vector = tfidf.fit_transform(corpus)
result = tfidf.fit_transform(corpus).toarray()

result_arr = np.round(result, decimals=2).T

feature_names = tfidf.get_feature_names()

for name, value in zip(feature_names, result_arr):
    print(name, value)
