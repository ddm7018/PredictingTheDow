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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem import PorterStemmer
from sklearn.ensemble 					import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
import pandas
from sklearn.naive_bayes import BernoulliNB
from sklearn.decomposition import TruncatedSVD
from sklearn.svm 						import SVC
import pickle
df = pd.read_csv('stocknews/Combined_News_DJIA.csv')
print(df.shape)

matplotlib.rcParams["figure.figsize"] = "8, 8"
df['Combined']=df.iloc[:,2:28].apply(lambda row: ''.join(str(row.values)), axis=1)


#train,test = train_test_split(df,test_size=0.2,random_state=42)
train = df[df['Date'] < '2015-01-01']
test = df[df['Date'] > '2014-12-31']




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



testheadlines 	= []

for each in test['Combined']:
    #testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
    testheadlines.append(to_words(each))

trainheadlines = []
for each in train['Combined']:
	#trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))
	trainheadlines.append(to_words(each))


vector = CountVectorizer(ngram_range = (1,2), min_df = 2)
trainvector 		= vector.fit_transform(trainheadlines)
testvector 			= vector.transform(testheadlines)
tsvd 				= TruncatedSVD(n_components=2)
t 					= tsvd.fit_transform(trainvector)
t1 					= tsvd.transform(testvector)


testLines = t1
trainLines = t


'''
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
'''
count = 0
for ele in decrease_list:
	if ele in non_decrease_list:
		count += 1
print count

print count/float(len(decrease_list))
'''

Classifiers = [
    KNeighborsClassifier(n_neighbors=50),
    #SVC(kernel="rbf", C=0.025,probability=True),
    AdaBoostClassifier(ExtraTreesClassifier()),
    DecisionTreeClassifier(),
    ExtraTreesClassifier(),
   	]

predDict = {}
Accuracy=[]
Model=[]
label = 'Label'
for clf in Classifiers:
	
	clf.fit(trainLines, train[label])
	predictions = clf.predict(testLines)
	matrix 			= pandas.crosstab(test[label], predictions, rownames=["Actual"], colnames=["Predicted"])
	#print "Running Ada Boost gives accuracy of \t\t ",
	accuracy = accuracy_score(test[label], predictions),
	print " --- ",
	prob = clf.predict_proba(testLines)[:,1]
	fpr, tpr, _ = roc_curve(test[label],prob)
	print "\t",
	print auc(fpr,tpr),
	#print "Running Ada Boost gives accuracy of \t\t ",
	
	Accuracy.append(accuracy)
  	Model.append(clf.__class__.__name__)
  	print('Accuracy of '+clf.__class__.__name__+' is '+str(accuracy))
  	fpr, tpr, _ = roc_curve(test['Label'],prob)
  	print('AUC of '+clf.__class__.__name__+' is '+str(auc(fpr,tpr)))
 	print("\n")
	tmp = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
 	g = ggplot(tmp, aes(x='fpr', y='tpr')) +geom_line() +geom_abline(linetype='dashed')+ ggtitle('Roc Curve of '+clf.__class__.__name__ + " Accuracy("+str(round(accuracy[0],4))+") with AUC of "+ str(round(auc(fpr,tpr),4)))
 	filename = str("AUC/")+str(clf.__class__.__name__)+".png"
 	g.save(filename)
 	predDict[str(clf.__class__.__name__)] = predictions
pickle.dump(predDict, open("pickle/prediction.p","wb"))

    #print(g)

