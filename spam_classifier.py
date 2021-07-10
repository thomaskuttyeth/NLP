import os 
os.listdir()
import pandas as pd 

# loading the data 
messages = pd.read_csv('data/SMSSpamCollection', 
                       sep = '\t', 
                       names = ['label', 'message']) 
# data cleaning and processing 
import re 
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords 


# stemmig part 
ps = PorterStemmer() 
corpus = []
for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]',' ', messages['message'][i])
    review = review.lower() 
    review = review.split() 
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review) 
    
# creating bag of words model 
from sklearn.feature_extraction.text import CountVectorizer 
# taking top 5000 occurings 
cv = CountVectorizer(max_features=5000) 
 
#independent feature 
X = cv.fit_transform(corpus).toarray()    


# get dummies for lables (dependent featue) 
y = pd.get_dummies(messages['label']) 
y = y['spam'] 

# train test split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state=0) 

# training model using naive bayes classifier 
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train,y_train)
 
y_pred = spam_detect_model.predict(X_test)

# evaluation of model 
from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test, y_pred)

# checking accuracy scores 
from sklearn.metrics import accuracy_score 
accuracy = accuracy_score(y_test, y_pred)


from sklearn.metrics import classification_report
model_report = classification_report(y_test, y_pred)


