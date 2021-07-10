# importing libraries 
import nltk 
from gensim.models import Word2Vec
from nltk.corpus import stopwords 
import re 

paragraph = '''
I have three visions for India. In 3000 years of our history, people from all over 
the world have come and invaded us, captuour lands, conquered our minds. 
From Alexander onwards, the Greeks, the Turthe Moguls, the Portuguese, the British,
the French, the Dutch, all of them came looted us, took over what was ours. 
Yet we have not done this to any other natiWe have not conquered anyone. 
We have not grabbed their land, their culture, 
their history and tried to enforce our waylife on them. 
Why? Because we respect the freedom of othThat is why my 
first vision is that of freedom. I belithat India got its first vision of 
this in 1857, when we started the WarIndependence. It is this freedom that
we must protect and nurture and build on.we are not free, no one will respect us.
'''

# preprocessing 
text = re.sub(r'\[[0-9]*\]',' ', paragraph) 
text = re.sub(r'\s+', ' ', text) 
text = text.lower() 
text = re.sub(r'\d', ' ', text) 
text = re.sub(r'\s+', ' ', text ) 

# sentence tokenization 
sentences = nltk.sent_tokenize(text) 

sentences = [nltk.word_tokenize(sentence) for sentence in sentences] 

for i in range(len(sentences)): 
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')] 
    
# training the word2vec model 
model = Word2Vec(sentences, min_count = 1) 

words = list(model.wv.index_to_key) 

# finishing words vectors 
vector = model.wv['culture'] 

# most similar words
similar = model.wv.most_similar('culture') 
