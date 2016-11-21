import re, nltk, sklearn, matplotlib
import numpy 					as np 
import pandas 					as pd 
import matplotlib.pyplot 		as plt
from sklearn.cross_validation 	import train_test_split
from nltk.corpus 				import stopwords
from sklearn.linear_model 	import LogisticRegression
from sklearn.neighbors 		import KNeighborsClassifier
from sklearn.svm 			import SVC
from sklearn.tree 			import DecisionTreeClassifier

from sklearn.naive_bayes 	import GaussianNB
from sklearn.metrics 		import accuracy_score
from sklearn.metrics 		import roc_curve, auc
from ggplot 				import *
from nltk.stem.porter 		import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
stemmer = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


df = pd.read_csv('stocknews/Combined_News_DJIA.csv')
print(df.shape)

matplotlib.rcParams["figure.figsize"] = "8, 8"
df['Combined']=df.iloc[:,2:27].apply(lambda row: ''.join(str(row.values)), axis=1)


train,test = train_test_split(df,test_size=0.2,random_state=42)

non_decrease = train[train['Label']==1]
decrease = train[train['Label']==0]
print(len(non_decrease)/float(len(df)))


def to_words(content):
    letters_only = re.sub("[^a-zA-Z]", " ", content) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return( " ".join( meaningful_words ))

non_decrease_word=[]
decrease_word=[]
for each in non_decrease['Combined']:
    non_decrease_word.append(to_words(each))

for each in decrease['Combined']:
    decrease_word.append(to_words(each))


tfidf = CountVectorizer()
train_text = []
test_text = []
for each in train['Combined']:
    train_text.append(to_words(each))

for each in test['Combined']:
    test_text.append(to_words(each))
train_features = tfidf.fit_transform(train_text)
test_features = tfidf.transform(test_text)

non_decrease_list = []
decrease_list = []

for ele in non_decrease_word:
	newList = ele.split(" ")
	for newEle in newList:
		decrease_list.append(newEle)

for ele in decrease_word:
	newList = ele.split(" ")
	for newEle in newList:
		non_decrease_list.append(newEle)

'''
count = 0
for ele in decrease_list:
	if ele in non_decrease_list:
		count += 1
print count

print count/float(len(decrease_list))
'''

Classifiers = [
    KNeighborsClassifier(n_neighbors=150),
    SVC(kernel="rbf", C=0.025, probability=True),
    ]


dense_features=train_features.toarray()
dense_test= test_features.toarray()
Accuracy=[]
Model=[]
for classifier in Classifiers:
    try:
        fit = classifier.fit(train_features,train['Label'])
        pred = fit.predict(test_features)
        prob = fit.predict_proba(test_features)[:,1]
    except Exception:
        fit = classifier.fit(dense_features,train['Label'])
        pred = fit.predict(dense_test)
        prob = fit.predict_proba(dense_test)[:,1]
    accuracy = accuracy_score(pred,test['Label'])
    Accuracy.append(accuracy)
    Model.append(classifier.__class__.__name__)
    print('Accuracy of '+classifier.__class__.__name__+' is '+str(accuracy))
    fpr, tpr, _ = roc_curve(test['Label'],prob)
    print('AUC of '+classifier.__class__.__name__+' is '+str(auc(tpr,fpr)))
    print("\n")
    tmp = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
    g = ggplot(tmp, aes(x='fpr', y='tpr')) +geom_line() +geom_abline(linetype='dashed')+ ggtitle('Roc Curve of '+classifier.__class__.__name__)
    filename = str("AUC/")+str(classifier.__class__.__name__)+".png"
    g.save(filename)

    #print(g)

