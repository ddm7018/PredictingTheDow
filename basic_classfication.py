import pandas
from sklearn.feature_extraction.text 	import CountVectorizer, TfidfVectorizer
from sklearn.linear_model 				import LogisticRegression
from sklearn.neighbors 					import KNeighborsClassifier
from sklearn.svm 						import LinearSVC
from sklearn.metrics					import accuracy_score
from sklearn.cross_validation 			import train_test_split
from sklearn.tree 						import DecisionTreeClassifier
from sklearn.ensemble 					import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes 				import GaussianNB
import multiprocessing
import pickle, os
from multiprocessing import Manager
import warnings
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	from sklearn.lda 						import LDA
	from sklearn.qda  						import QDA

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

def countVectorize(trainheadlines, testheadlines):
	
	basicvectorizer = CountVectorizer(stop_words='english',  min_df = 5 ,ngram_range=(1,1),analyzer=stemmed_words)
	basictrain 		= basicvectorizer.fit_transform(trainheadlines)
	basictest 		= basicvectorizer.transform(testheadlines)
	return basictrain, basictest, basicvectorizer

def tdIdfVectorize(trainheadlines, testheadlines):
	#td 				= TfidfVectorizer()
	td 				= TfidfVectorizer(stop_words='english',max_df=0.95, min_df=1, ngram_range=(1,1),analyzer=stemmed_words)
	tdTrain 		= td.fit_transform(trainheadlines)
	tdTest 			= td.transform(testheadlines)
	return tdTrain, tdTest, td


def countVectorize1(trainheadlines, testheadlines):
	basicvectorizer = CountVectorizer(stop_words='english', ngram_range=(2,2), analyzer =stemmed_words)
	#basicvectorizer = CountVectorizer()
	basictrain 		= basicvectorizer.fit_transform(trainheadlines)
	basictest 		= basicvectorizer.transform(testheadlines)
	return basictrain, basictest, basicvectorizer

def tdIdfVectorize1(trainheadlines, testheadlines):
	#td 				= TfidfVectorizer()
	td 				= TfidfVectorizer(stop_words='english',max_df=0.95, min_df=2,
                                max_features=2000, ngram_range=(2,2), analyzer = stemmed_words)
	tdTrain 		= td.fit_transform(trainheadlines)
	tdTest 			= td.transform(testheadlines)
	return tdTrain, tdTest, td

def runKNN(basictrain,basictest, train,test, label, key):
	maxAccuracy = 0
	val = 0
	#print "Beginning KNN runs"
	for x in range(1,250):
		neigh = KNeighborsClassifier(n_neighbors=x)
		neigh.fit(basictrain, train[label])
		predictions 	= neigh.predict(basictest)
		matrix 			= pandas.crosstab(test[label], predictions, rownames=["Actual"], colnames=["Predicted"])

		if accuracy_score(test[label], predictions) > maxAccuracy:
			maxAccuracy = accuracy_score(test[label], predictions)
			val = x
	print "KNN of n = "+str(val),
	print "gives an accuracy of \t\t  " + str(maxAccuracy) ,
	print "\t" + str(key)
	return neigh

def runLogisticReegresion(basictrain,basictest, train,test,label,key):
	logModel 		= LogisticRegression()
	logModel 		= logModel.fit(basictrain, train[label])
	predictions 	= logModel.predict(basictest)
	matrix 			= pandas.crosstab(test[label], predictions, rownames=["Actual"], colnames=["Predicted"])
	print "Running Logistical Regression gives accuracy of  ",
	print accuracy_score(test[label], predictions),
	print "\t" + str(key)
	return logModel

def runLinearSVC(basictrain,basictest, train,test,label,key):
	clf = LinearSVC()
	clf.fit(basictrain, train[label])
	predictions = clf.predict(basictest)
	matrix 			= pandas.crosstab(test[label], predictions, rownames=["Actual"], colnames=["Predicted"])
	print "Running LinearSVC gives accuracy of\t\t ",
	print accuracy_score(test[label], predictions) ,
	print "\t" + str(key)
	return clf

def runRandomForestClassifier(basictrain,basictest, train,test,label,key):
	rfc = RandomForestClassifier()
	rfc.fit(basictrain, train[label])
	predictions 	= rfc.predict(basictest)
	print "Running RandomForestClassifier gives accuracy of ",
	print accuracy_score(test[label], predictions),
	print "\t" + str(key)
	return rfc

def decisionTree(basictrain,basictest, train,test,label,key):
	clf = DecisionTreeClassifier()
	clf.fit(basictrain, train[label])
	predictions = clf.predict(basictest)
	matrix 			= pandas.crosstab(test[label], predictions, rownames=["Actual"], colnames=["Predicted"])
	print "Running Decision Tree gives accuracy of    \t ",
	print accuracy_score(test[label], predictions) ,
	print "\t" + str(key)
	return clf

def runLDA(basictrain,basictest, train,test,label, key):
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		clf = LDA()
		clf.fit(basictrain, train[label])
		predictions = clf.predict(basictest)
		matrix 			= pandas.crosstab(test[label], predictions, rownames=["Actual"], colnames=["Predicted"])
		print "Running LDA gives accuracy of ",
		print accuracy_score(test[label], predictions) ,
		print "\t" + str(key)
		return clf


def runQDA(basictrain,basictest, train,test,label,key):
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		clf = QDA()
		clf.fit(basictrain, train[label])
		predictions = clf.predict(basictest)
		matrix 			= pandas.crosstab(test[label], predictions, rownames=["Actual"], colnames=["Predicted"])
		print "Running QDA gives accuracy of ",
		print accuracy_score(test[label], predictions),
		print "\t" + str(key)
		return clf


def addBoost(basictrain,basictest, train,test,label,key):
	clf = AdaBoostClassifier()
	clf.fit(basictrain, train[label])
	predictions = clf.predict(basictest)
	matrix 			= pandas.crosstab(test[label], predictions, rownames=["Actual"], colnames=["Predicted"])
	print "Running Ada Boost gives accuracy of \t\t ",
	print accuracy_score(test[label], predictions),
	print "\t" + str(key)
	return clf

def CoefToHTML(basicvectorizer,basicmodel,filename):
	basicwords = basicvectorizer.get_feature_names()
	basiccoeffs = basicmodel.coef_.tolist()[0]
	coeffdf = pandas.DataFrame({'Word' : basicwords, 
	                        'Coefficient' : basiccoeffs})
	coeffdf = coeffdf.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
	coeffdf.head(10).to_html(filename+'_head.html')
	coeffdf.tail(10).to_html(filename+'_tail.html')

#def main():
	# the labels in this file are either 0,1 (will modify accordinly since this not binary classficattion)
data 	= 	pandas.read_csv("stocknews/Combined_News_DJIA.csv")

#dividing up the training data per Kaggle instructions, will modify later
#train = data[data['Date'] < '2015-01-01']
#test = data[data['Date'] > '2014-12-31']

data['Combined']=data.iloc[:,2:27].apply(lambda row: ''.join(str(row.values)), axis=1)
data['Tomm_Label'] = data.Label.shift(-1)
data = data[0:len(data)-1]

train,test 		= train_test_split(data,test_size=0.2,random_state=42)
testheadlines 	= []

for row in range(0,len(test.index)):
    #testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))

trainheadlines = []
for row in range(0,len(train.index)):
	#trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))
	trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))

vectorDict = {}
if not os.path.isfile("pickle/vectors.p"):
	print "Starting Vectorizing"
	cvtrain, cvtest, cvVector = countVectorize(trainheadlines,testheadlines)
	tdTrain, tdTest, tdVector = tdIdfVectorize(trainheadlines,testheadlines)
	cvtrain1, cvtest1, cvVector1 = countVectorize1(trainheadlines,testheadlines)
	tdTrain1, tdTest1, tdVector1 = tdIdfVectorize1(trainheadlines,testheadlines)

	print "Finished Vectorizing"

	vectorDict = {
				  "Count Vector":[cvtrain, cvtest],
				  "TD-IDF":[tdTrain,tdTest],
				  "Count Vector with ngram of 2,2":[cvtrain1, cvtest1],
				  "TD-IDF with ngram of 2,2":[tdTrain1,tdTest1]	
				  }
	pickle.dump(vectorDict, open("pickle/vectors.p","wb"))

	data = {
				  "Test":test,
				  "Train":train,
				  
				  }
	pickle.dump(data, open("pickle/data.p","wb"))

	print 'Saving to pickle'
else:
	vectorDict = pickle.load( open( "pickle/vectors.p", "rb" ))

label = 'Tomm_Label'
jobs = []
manager = Manager()
return_dict = manager.dict()

for key,value in vectorDict.iteritems():
	#print 'Running with ' + str(key)
	p = multiprocessing.Process(target=runKNN, args=(value[0],value[1],train,test, label,key,))
	jobs.append(p)
	p.start()


	p = multiprocessing.Process(target=runLogisticReegresion, args=(value[0],value[1],train,test, label,key,))
	jobs.append(p)
	p.start()

	p = multiprocessing.Process(target=runLinearSVC, args=(value[0],value[1],train,test, label,key,))
	jobs.append(p)
	p.start()

	p = multiprocessing.Process(target=runRandomForestClassifier, args=(value[0],value[1],train,test, label,key,))
	jobs.append(p)
	p.start()

	p = multiprocessing.Process(target=decisionTree, args=(value[0],value[1],train,test, label,key,))
	jobs.append(p)
	p.start()


	if key != 'Count Vector with ngram of 2,2':
	
		p = multiprocessing.Process(target=runLDA, args=(value[0].toarray(),value[1].toarray(),train,test, label,key,))
		jobs.append(p)
		p.start()

		p = multiprocessing.Process(target=runQDA, args=(value[0].toarray(),value[1].toarray(),train,test, label,key,))
		jobs.append(p)
		p.start()

	p = multiprocessing.Process(target=addBoost, args=(value[0],value[1],train,test, label,key,))
	jobs.append(p)
	p.start()

	for proc in jobs: proc.join()

        '''
		runKNN(value[0],value[1],train,test, label)
		runLogisticReegresion(value[0],value[1],train,test,label)
		runLinearSVC(value[0],value[1],train,test,label)
		runRandomForestClassifier(value[0],value[1],train,test,label)
		decisionTree(value[0],value[1],train,test,label)
		if key != 'Count Vector with ngram of 2,2':
			runLDA(value[0].toarray(),value[1].toarray(),train,test,label)
			runQDA(value[0].toarray(),value[1].toarray(),train,test,label)
		#gaussianNB(value[0],value[1],train,test)
		addBoost(value[0],value[1],train,test,label)
		print "\n"
		'''

#run("Label")	
'''

#CoefToHTML(cvVector,logCV,"log-countVector")
#CoefToHTML(tdVector,svmidf,"SVM-IDF")
#matrix 			= pandas.crosstab(test[label], predictions, rownames=["Actual"], colnames=["Predicted"])



#if __name__ == "__main__":
#	main()
'''

