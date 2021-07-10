

import re
import nltk  
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


paragraph = '''
I have three visions for India. In 3000 years of our history, people from all over 
the world have come and invaded us, captuour lands, conquered our minds. 
From Alexander onwards, the Greeks, the Turthe Moguls, the Portuguese, the British,
the French, the Dutch, all of them came looted us, took over what was ours. 
Yet we have not done this to any other natiWe have not conquered anyone. 
'''

ps = PorterStemmer()
wordnet = WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []

for i in range(len(sentences)):
    english_stopwords = stopwords.words('english')
    review = re.sub('[^a-zA-Z]', ' ',sentences[i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(english_stopwords)]
    review = ' '.join(review)
    corpus.append(review) 

# print(corpus[0:5]) 
print()
# print(sentences[0:5]) 


# creating the bag_of_words 
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer() 
X = cv.fit_transform(corpus).toarray() 

print(X) 
