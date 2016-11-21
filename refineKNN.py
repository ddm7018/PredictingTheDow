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
	
	basicvectorizer = TfidfVectorizer()
	basictrain 		= basicvectorizer.fit_transform(trainheadlines)
	basictest 		= basicvectorizer.transform(testheadlines)
	return basictrain, basictest, basicvectorizer


def runKNN(basictrain,basictest, train,test, ):
	label = 'Label'
	knnDict = {}
	maxAccuracy = 0
	val = 0
	print "Beginning KNN runs"
	for x in range(1,300):
		neigh = KNeighborsClassifier(n_neighbors=x)
		neigh.fit(basictrain, train[label])
		predictions 	= neigh.predict(basictest)
		matrix 			= pandas.crosstab(test[label], predictions, rownames=["Actual"], colnames=["Predicted"])
		
		prob = neigh.predict_proba(basictest)[:,1]
		fpr, tpr, _ = roc_curve(test[label],prob)
		knnDict[x] 		= [auc(tpr,fpr), accuracy_score(test["Label"], predictions) ]

	return knnDict
	

def vectorize(train, test,num, return_dict):
	testheadlines = []
	trainheadlines = []
	for each in test['Combined']: testheadlines.append(to_words(each))
	for each in train['Combined']: trainheadlines.append(to_words(each))
	cvtrain, cvtest, cvVector = countVectorize(trainheadlines,testheadlines)
	return_dict[num] = runKNN(cvtrain,cvtest, train,test)


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


print 'here'

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

for proc in jobs:
	proc.join()


best = 0
bestI = 0
bestA = 0


for x in range(1,500):
	#sumR = r1[x] + r2[x] + r3[x] + r4[x] + r5[x]
	sumR = return_dict.values()[0][x][0] + return_dict.values()[1][x][0] + return_dict.values()[2][x][0] + return_dict.values()[3][x][0] +return_dict.values()[4][x][0]
	sumA = return_dict.values()[0][x][1] + return_dict.values()[1][x][1] + return_dict.values()[2][x][1] + return_dict.values()[3][x][1] +return_dict.values()[4][x][1]
	if sumR > best:
            best = sumR
            bestI = x
            bestA = sumA/float(5)
 
print "KNN with neigh " + str(bestI) +  "had the best AUC  of " + str(best/5) + "an accuracy of " + str(bestA)
