
import nltk 
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns

corpus = ['Time flies flies like an arrow.','Fruit flies like  banana.']


# one hot vectorizer   
one_hot_vectorizer = CountVectorizer(binary = True) 
one_hot = one_hot_vectorizer.fit_transform(corpus).toarray() 

sns.heatmap(one_hot)


# tfidf 
from sklearn.feature_extraction.text import TfidfVectorizer 
import seaborn as sns 
tfidf_vectorizer = TfidfVectorizer()
ttfidf = tfidf_vectorizer.fit_transform(corpus).toarray()


