
'''
One hot encoded representation
=============================================
This means that if we have a vocabulary of V size, for each i
th word wi, we will represent the word wi with a V-long vector [0, 0, 0, …, 0, 1, 0, …, 0, 0,0] where the ith element is 1 and other elements are zero
'''
from sklearn.feature_extraction.text import CountVectorizer

corpus = ['Time flies flies like an arrow', 'Fruit flies like banana']

# one hot vectorizer
one_hot_vectorizer = CountVectorizer(binary=True)
one_hot = one_hot_vectorizer.fit_transform(corpus).toarray()


'''
Problems with the one hot representation
=============================================
1. This representation does not include the similarity between words in any way and
   completely ignores the context in which the words are used. Let's consider the dot
product between the word vectors as the similarity measure. The more similar two
vectors are, the higher the dot product is for those two vectors.

2. extremly ineffective for large vocabularies. if there are 50,000 words in the corpus, then this representation generates 50,000* 50,000 sparse matrix.
'''
