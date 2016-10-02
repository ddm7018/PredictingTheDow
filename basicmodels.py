import pandas
from sklearn.feature_extraction.text 	import CountVectorizer, TfidfVectorizer
from sklearn.linear_model 				import LogisticRegression
from sklearn.neighbors 					import KNeighborsClassifier
from sklearn.svm 						import LinearSVC
from sklearn.ensemble 					import RandomForestClassifier


def countVectorize(trainheadlines, testheadlines):
	basicvectorizer = CountVectorizer()
	basictrain 		= basicvectorizer.fit_transform(trainheadlines)
	basictest 		= basicvectorizer.transform(testheadlines)
	return basictrain, basictest, basicvectorizer

def tdIdfVectorize(trainheadlines, testheadlines):
	td 				= TfidfVectorizer()
	tdTrain 		= td.fit_transform(trainheadlines)
	tdTest 			= td.transform(testheadlines)
	return tdTrain, tdTest, td


def runKNN(basictrain,basictest, train,test):
	neigh = KNeighborsClassifier(n_neighbors=1)
	neigh.fit(basictrain, train["Label"])
	predictions 	= neigh.predict(basictest)
	matrix 			= pandas.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"])
	print "Running KNN gives accuracy of ",
	accuracy(matrix)
	return neigh

def runLogisticReegresion(basictrain,basictest, train,test):
	logModel 		= LogisticRegression()
	logModel 		= logModel.fit(basictrain, train["Label"])
	predictions 	= logModel.predict(basictest)
	matrix 			= pandas.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"])
	print "Running Logistical Regression gives accuracy of ",
	accuracy(matrix)
	return logModel

def runLinearSVC(basictrain,basictest, train,test):
	clf = LinearSVC()
	clf.fit(basictrain, train["Label"])
	predictions = clf.predict(basictest)
	matrix 			= pandas.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"])
	print "Running LinearSVC gives accuracy of ",
	accuracy(matrix)
	return clf

def runRandomForestClassifier(basictrain,basictest, train,test):
	rfc = RandomForestClassifier()
	rfc.fit(basictrain, train["Label"])
	predictions 	= rfc.predict(basictest)
	matrix 			= pandas.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"])
	print "Running RandomForestClassifier gives accuracy of ",
	accuracy(matrix)
	return rfc


def accuracy(matrix):
	correct = matrix[0][0] + matrix[1][1]
	total 	= correct + matrix[1][0] + matrix[0][1]
	print float(correct)*100/total 


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
train	= data[data['Date'] < '20150101']
test 	= data[data['Date'] > '20141231']

testheadlines 	= []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))

trainheadlines = []
for row in range(0,len(train.index)):
	trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))

cvtrain, cvtest, cvVector = countVectorize(trainheadlines,testheadlines)
tdTrain, tdTest, tdVector = tdIdfVectorize(trainheadlines,testheadlines)

print 'Running with Count Vectorizer'
logCV = runLogisticReegresion(cvtrain,cvtest,train,test)
runKNN(cvtrain,cvtest,train,test)
runLinearSVC(cvtrain,cvtest,train,test)
runRandomForestClassifier(cvtrain,cvtest,train,test)

print "\nRunning with TD-IDF"
runLogisticReegresion(tdTrain,tdTest,train,test)
runKNN(tdTrain,tdTest,train,test)
svmidf = runLinearSVC(tdTrain,tdTest,train,test)
runRandomForestClassifier(tdTrain,tdTest,train,test)

CoefToHTML(cvVector,logCV,"log-countVector")
CoefToHTML(tdVector,svmidf,"SVM-IDF")




#if __name__ == "__main__":
#	main()


	