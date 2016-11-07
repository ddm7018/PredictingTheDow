import pandas
from sklearn.neighbors 					import KNeighborsClassifier
from sklearn.metrics					import accuracy_score
import pickle, os
import numpy as np
from sklearn.feature_extraction.text 	import CountVectorizer

from nltk.stem.snowball import FrenchStemmer
stemmer = FrenchStemmer()
analyzer = CountVectorizer().build_analyzer()

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

def countVectorize(trainheadlines,testheadlines):
	
	basicvectorizer = CountVectorizer(stop_words='english', ngram_range=(1,1),analyzer=stemmed_words)
	basictrain 		= basicvectorizer.fit_transform(trainheadlines)
	basictest 		= basicvectorizer.transform(testheadlines)
	return basictrain, basictest, basicvectorizer


def runKNN(basictrain,basictest, train,test):
	knnDict = {}
	maxAccuracy = 0
	val = 0
	print "Beginning KNN runs"
	for x in range(1,200):
		neigh = KNeighborsClassifier(n_neighbors=x, algorithm = 'auto')
		neigh.fit(basictrain, train["Label"])
		predictions 	= neigh.predict(basictest)
		matrix 			= pandas.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"])
		knnDict[x] 		= accuracy_score(test["Label"], predictions) 

	return knnDict


def vectorize(train, test):

	testheadlines = []
	trainheadlines = []
	for row in range(0,len(test.index)):
	    #testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
	    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))

	trainheadlines = []
	for row in range(0,len(train.index)):
		#trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))
		trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))

	cvtrain, cvtest, cvVector = countVectorize(trainheadlines,testheadlines)
	return runKNN(cvtrain,cvtest, train,test)


data 	= 	pandas.read_csv("stocknews/Combined_News_DJIA.csv")
data['Combined']=data.iloc[:,2:27].apply(lambda row: ''.join(str(row.values)), axis=1)
	
shuffledata = data.reindex(np.random.permutation(data.index))


n1 = shuffledata[0:400]
n2 = shuffledata[400:800]
n3 = shuffledata[800:1200]
n4 = shuffledata[1200:1600]
n5 = shuffledata[1600:]


train1 = n1.append(n2).append(n3).append(n4)
test1 = n5

train2 = n1.append(n2).append(n3).append(n5)
test2 = n4

train3 = n1.append(n2).append(n5).append(n4)
test3= n3

train4 = n1.append(n5).append(n3).append(n4)
test4 = n2

train5 = n5.append(n2).append(n3).append(n4)
test5 = n1


print '1' 
r1 = vectorize(train1, test1) 
print '2'
r2 = vectorize(train2, test2)
print '3'
r3 = vectorize(train3, test3) 
print '4'
r4 = vectorize(train4, test4) 
print '5'
r5 = vectorize(train5, test5) 

best = 0
bestI = 0

for x in range(1,200):
	sumR = r1[x] + r2[x] + r3[x] + r4[x] + r5[x]
	if sumR > best:
            best = sumR
            bestI = x
 
print "KNN with neigh " + str(bestI) +  " gave the best accuracies of " + str(best/5)
