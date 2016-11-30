import pandas
from sklearn.neighbors 					import KNeighborsClassifier
from sklearn.metrics					import accuracy_score
import pickle, os, re
import numpy as np
from sklearn.feature_extraction.text 	import CountVectorizer, TfidfVectorizer
import multiprocessing
from nltk.stem.snowball import EnglishStemmer
from multiprocessing import Manager
from nltk.corpus 				import stopwords
from sklearn.metrics 		import roc_curve, auc
from sklearn.decomposition import TruncatedSVD



stemmer = EnglishStemmer()
analyzer = CountVectorizer().build_analyzer()

def to_words(content):
    letters_only = re.sub("[^a-zA-Z]", " ", content) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return( " ".join( meaningful_words ))



def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

def countVectorize(trainheadlines,testheadlines):
	
	basicvectorizer = CountVectorizer(min_df = 5)
	tsvd 				= TruncatedSVD(n_components=2)
	basictrain 		= tsvd.fit_transform(basicvectorizer.fit_transform(trainheadlines))
	basictest 		=  tsvd.transform(basicvectorizer.transform(testheadlines))
	
	#tsvd 				= TruncatedSVD(n_components=2)
	#t 					= tsvd.fit_transform(trainvector)
	#t1 					= tsvd.transform(testvector)

	return basictrain, basictest, basicvectorizer


def runKNN(basictrain,basictest, train,test, ):



	label = 'Label'
	knnDict = {}
	maxAccuracy = 0
	val = 0
	#print "Beginning KNN runs"
	for x in range(1,300):
		neigh = KNeighborsClassifier(n_neighbors=x)
		neigh.fit(basictrain, train[label])
		predictions 	= neigh.predict(basictest)
		matrix 			= pandas.crosstab(test[label], predictions, rownames=["Actual"], colnames=["Predicted"])
		
		prob = neigh.predict_proba(basictest)[:,1]
		fpr, tpr, _ = roc_curve(test[label],prob)
		knnDict[x] 		= [auc(fpr,tpr), accuracy_score(test["Label"], predictions) ]

	return knnDict
	

def vectorize(train, test,num):
	testheadlines = []
	trainheadlines = []
	for each in test['Combined']: testheadlines.append(to_words(each))
	for each in train['Combined']: trainheadlines.append(to_words(each))
	cvtrain, cvtest, cvVector = countVectorize(trainheadlines,testheadlines)
	return runKNN(cvtrain,cvtest, train,test)


data 	= 	pandas.read_csv("stocknews/Combined_News_DJIA.csv")
data['Combined']=data.iloc[:,2:27].apply(lambda row: ''.join(str(row.values)), axis=1)
data['Tomm_Label'] = data.Label.shift(-1)
data = data[0:len(data)-1]

	
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


k1 = vectorize(train1, test1,1)
k2 = vectorize(train2, test2,2)
k3 = vectorize(train3, test3,3)
k4 = vectorize(train4, test4,4)
k5 = vectorize(train5, test5,5)


'''
jobs = []
manager = Manager()
return_dict = manager.dict()



p = multiprocessing.Process(target=vectorize, args=(train1, test1,1,return_dict,))
jobs.append(p)
p.start()

p = multiprocessing.Process(target=vectorize, args=(train2, test2,2,return_dict,))
jobs.append(p)
p.start()

p = multiprocessing.Process(target=vectorize, args=(train3, test3,3,return_dict,))
jobs.append(p)
p.start()

p = multiprocessing.Process(target=vectorize, args=(train4, test4,4,return_dict,))
jobs.append(p)
p.start()

p = multiprocessing.Process(target=vectorize, args=(train5, test5,5,return_dict,))
jobs.append(p)
p.start()

'''

best = 0
bestI = 0
bestA = 0


for x in range(1,300):
	#sumR = r1[x] + r2[x] + r3[x] + r4[x] + r5[x]
	sumAUC = k1[x][0] + k2[x][0] + k3[x][0] + k4[x][0] + k5[x][0]
	sumAccuracy = k1[x][1] + k2[x][1] + k3[x][1] + k4[x][1] + k5[x][1]


	#print (sumAUC/5)
	print sumAUC/5
	if sumAUC > best:
            best = sumAUC
            bestI = x
            bestAUC = sumAUC/float(5)
            bestAccuracy = sumAccuracy/float(5) 
 
print "KNN using CountVector with neigh " + str(bestI) +  " had the best AUC  of " + str(bestAUC) + "an accuracy of " + str(bestAccuracy)

