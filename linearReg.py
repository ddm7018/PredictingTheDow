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
import pickle, os


if os.path.isfile("vectors.p"):
	data 	= 	pandas.read_csv("stocknews/full-table.csv")
	data['Combined']=data.iloc[:,2:27].apply(lambda row: ''.join(str(row.values)), axis=1)
	train,test = train_test_split(data,test_size=0.2,random_state=42)

	vectorDict = pickle.load( open( "vectors.p", "rb" ))

	
	for key,value in vectorDict.iteritems():
		print 'Running with ' + str(key)
		from sklearn import linear_model as LR
		linearModelObj= LR.LinearRegression()
		linearModelObj= linearModelObj.fit(value[0], train["Open"])
		predictions= linearModelObj.predict(value[1])
		from sklearn.metrics import mean_squared_error
		print mean_squared_error(test['Open'], predictions)

			
else:
	print "No Vectors Pickle File- Can not run"



