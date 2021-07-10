
 

import nltk 
# nlkt.download() 
from nltk.stem import PorterStemmer  # stemming 
from nltk.corpus import stopwords    # removing stopwords



# reading the file 
text_file='''
I have three visions for India.
In 3000 years of our history, people from allover the world have come and invaded us, captured our lands, conquered our minds. From Alexander onwards. The Greeks, the Portuguese, the British, the French, the Dutch, all of them came and looted us, took over what was ours. Yet we have not done this to any other nation. We have not conquered anyone. We have not grabbed their land, their culture, their history tried to enforce our way of life on them. Why? Because we respect the freedom of others.
'''


# tokenizing words 
words = nltk.word_tokenize(text_file) 

# getting the stop  words of english language 
stopwords.words('german') 


# stemming  

# tokenizing sentences               
sentences = nltk.sent_tokenize(text_file) 

stemmer = PorterStemmer() 
english_stopwords = stopwords.words('english') 
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word)   for word in words if word not in set(english_stopwords)] 
    sentences[i] = ' '.join(words) 

print(sentences[0:2]) 


    