
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation 			import train_test_split
from sklearn.neighbors 					import KNeighborsClassifier
from sklearn.svm 						import LinearSVC
from sklearn.ensemble 					import RandomForestClassifier
import pandas
from sklearn.metrics                                        import accuracy_score
from sklearn.cross_validation import cross_val_score

data 	= 	pandas.read_csv("stocknews/full-table.csv")
data 	= data[0:len(data)-1]
train,test = train_test_split(data,test_size=0.2,random_state=42)

#train_x = train[['Sentiment1', "SentimentAll"]]
'''
train_x = train[[ "Sentiment1"]]
train_y = train['Label']
 
#test_x = test[['Sentiment1',"SentimentAll"]]
test_x = test[[ "Sentiment1"]]
test_y = test['Label']
'''


clf = DecisionTreeClassifier()
scores = cross_val_score(clf, data[[ "Label","Sentiment1"]], data['Tomm_Label'], cv=5)
print scores.mean()

bestN = 0
bestVal = 0
'''
for i in range(1,200):
	neigh = KNeighborsClassifier(n_neighbors=i)
	scores = cross_val_score(neigh, data[["Sentiment1"]], data['Label'], cv=5)
	scores.mean()
	if scores.mean() > bestVal:
		bestN = i
		bestVal = scores.mean()

print bestN, bestVal
'''




clf1 = LinearSVC()
scores = cross_val_score(clf1, data[[ "Label","Sentiment1"]], data['Tomm_Label'], cv=5)
print scores.mean()

bestN= 0
bestVal = 0

'''
for i in range(100,130):
	clf = RandomForestClassifier(n_estimators = i)
	scores = cross_val_score(clf, data[[ "Sentiment1"]], data['Label'], cv=5)
	if scores.mean() > bestVal:
		bestN = i
		bestVal = scores.mean()

print bestN, bestVal
'''




