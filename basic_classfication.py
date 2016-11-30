import pandas
from sklearn.feature_extraction.text 	import CountVectorizer, TfidfVectorizer
from sklearn.linear_model 				import LogisticRegression
from sklearn.neighbors 					import KNeighborsClassifier
from sklearn.svm 						import SVC
from sklearn.metrics					import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree 						import DecisionTreeClassifier
from sklearn.ensemble 					import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes 				import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics 		import roc_curve, auc
from nltk.corpus 				import stopwords
import pickle, os, re, multiprocessing
from multiprocessing import Manager
from sklearn.decomposition import TruncatedSVD




import warnings
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	from sklearn.lda 						import LDA
	from sklearn.qda  						import QDA
	from sklearn.cross_validation 			import train_test_split


def to_words(content):
    letters_only = re.sub("[^a-zA-Z]", " ", content) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return( " ".join( meaningful_words ))


Vectorizers = [
CountVectorizer(min_df = 5),
CountVectorizer(ngram_range = (2,2), min_df = 5),
TfidfVectorizer(min_df = 5),
TfidfVectorizer(ngram_range=(2,2), min_df =5)
]



Classifiers = [
KNeighborsClassifier(n_neighbors=5),
BernoulliNB(),

AdaBoostClassifier(LogisticRegression()),
AdaBoostClassifier(RandomForestClassifier()),
AdaBoostClassifier(ExtraTreesClassifier()),
AdaBoostClassifier(BernoulliNB()),
AdaBoostClassifier(),
#AdaBoostClassifier(SVC(probability=True)),

DecisionTreeClassifier(),
RandomForestClassifier(),
LogisticRegression(),
SVC(kernel="rbf", C=0.025,probability=True),
ExtraTreesClassifier(),
]


#def main():
	# the labels in this file are either 0,1 (will modify accordinly since this not binary classficattion)
data 	= 	pandas.read_csv("stocknews/full-table.csv")

#dividing up the training data per Kaggle instructions, will modify later
#train = data[data['Date'] < '2015-01-01']
#test = data[data['Date'] > '2014-12-31']

data['Combined']=data.iloc[:,9:33].apply(lambda row: ''.join(str(row.values)), axis=1)
data['Tomm_Label'] = data.Label.shift(-1)
data = data[0:len(data)-1]

train,test 		= train_test_split(data,test_size=0.2,random_state=42)

### Dividing the data into test and train by dates (as specified)
#train = data[data['Date'] < '2015-01-01']
#test = data[data['Date'] > '2014-12-31']

testheadlines 	= []

for each in test['Combined']:
    #testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
    testheadlines.append(to_words(each))

trainheadlines = []
for each in train['Combined']:
	#trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))
	trainheadlines.append(to_words(each))

vectorDict = {}
if not os.path.isfile("pickle/vectors.p"):
	print "Starting Vectorizing"

	for vector in Vectorizers:
		trainvector 		= vector.fit_transform(trainheadlines)
		testvector 			= vector.transform(testheadlines)
		name				= vector.__class__.__name__ +" "+ str(vector.ngram_range)
		tsvd 				= TruncatedSVD(n_components=2)
		t 					= tsvd.fit_transform(trainvector)
		t1 					= tsvd.transform(testvector)

		vectorDict[name]    = [t,t1]


	print "Finished Vectorizing"

	pickle.dump(vectorDict, open("pickle/vectors.p","wb"))

	data = {
				  "Test":test,
				  "Train":train,
				  
				  }
	pickle.dump(data, open("pickle/data.p","wb"))

	print 'Saving to pickle'
else:
	vectorDict = pickle.load( open( "pickle/vectors.p", "rb" ))

label = 'Label'
jobs = []
manager = Manager()
return_dict = manager.dict()



def runModel(trainLines,testLines,train,test, label,key):
		clf.fit(trainLines, train[label])
		predictions = clf.predict(testLines)
		matrix 			= pandas.crosstab(test[label], predictions, rownames=["Actual"], colnames=["Predicted"])
		#print "Running Ada Boost gives accuracy of \t\t ",
		print accuracy_score(test[label], predictions),
		print " --- ",
		prob = clf.predict_proba(testLines)[:,1]
		fpr, tpr, _ = roc_curve(test[label],prob)
		print "\t",
		print auc(fpr,tpr),
		print " --- ",
		print "\t" + str(key) + " --- ",
		print "\t" + str(clf.__class__.__name__)
		return predictions


predDict = {}
for key,value in vectorDict.iteritems():
	for clf in Classifiers:
		p = multiprocessing.Process(target=runModel, args=(value[0],value[1],train,test, label,key,))
		jobs.append(p)
		p.start()
		p.join()
		#pred = runModel(value[0],value[1],train,test, label,key)
		#names = str(clf.__class__.__name__) + " " + str(clf.n_neighbors)
		#predDict[names] = pred
#pickle.dump(predDict, open("pickle/prediction.p","wb"))
