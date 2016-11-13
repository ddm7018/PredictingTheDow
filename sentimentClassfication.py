
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
scores = cross_val_score(clf, data[[ "Sentiment1"]], data['Label'], cv=5)
print scores.mean()

neigh = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(neigh, data[["Sentiment1"]], data['Label'], cv=5)
print scores.mean()

clf1 = LinearSVC()

scores = cross_val_score(clf1, data[[ "Sentiment1"]], data['Label'], cv=5)
print scores.mean()

clf = RandomForestClassifier()
scores = cross_val_score(clf, data[[ "Sentiment1"]], data['Label'], cv=5)
print scores.mean()

'''


clf = DecisionTreeClassifier()
clf = clf.fit(train_x, train_y)
predictions =  clf.predict(test_x)
print accuracy_score(test_y, predictions)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_x,train_y)
predictions1 	= neigh.predict(test_x)
print accuracy_score(test_y, predictions1)
#features = list(df2.columns[])
'''