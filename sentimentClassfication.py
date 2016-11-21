
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation 			import train_test_split
from sklearn.neighbors 					import KNeighborsClassifier
from sklearn.svm 						import SVC
from sklearn.ensemble 					import RandomForestClassifier
import matplotlib.pyplot 				as plt
import pandas
from sklearn.metrics                                        import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np

data 	= 	pandas.read_csv("stocknews/full-table.csv")
data 	= data[0:len(data)-1]
train,test = train_test_split(data,test_size=0.2,random_state=42)


indices1 = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Sentiment1', 'SentimentAll']
x = data[indices1]
y = data['Label']

clf = DecisionTreeClassifier()
scores = cross_val_score(clf, data[[ "Sentiment1"]], data['Label'], cv=5)
print scores.mean()

clf = KNeighborsClassifier(500)
scores = cross_val_score(clf, x, data['Label'], cv=5)
print scores.mean()

clf1 = SVC(kernel="rbf", C=0.025, probability=True)
scores = cross_val_score(clf, x, data['Label'], cv=5)
print scores.mean()

model = ExtraTreesClassifier()
scores = cross_val_score(model, x, data['Label'], cv=5)
print scores.mean()

model = ExtraTreesClassifier()
scores = cross_val_score(model, x, data['Tomm_Label'], cv=5)
print scores.mean()

def featureImportance(label):
	model = ExtraTreesClassifier()
	model.fit(x,data['Tomm_Label'])
	importances = model.feature_importances_
	std = np.std([tree.feature_importances_ for tree in model.estimators_],
	             axis=0)
	indices = np.argsort(importances)[::-1]


	# Print the feature ranking
	print("Feature ranking:")

	for f in range(x.shape[1]):
	    print indices1[f], importances[indices[f]]

	# Plot the feature importances of the forest
	plt.figure()
	plt.title("Feature importances")
	plt.bar(range(x.shape[1]), importances[indices],
	       color="r", yerr=std[indices], align="center")
	plt.xticks(range(x.shape[1]), indices1)
	plt.xlim([-1, x.shape[1]])
	plt.savefig("features/"+label+"-feature-selection.png")

featureImportance('Label')
featureImportance('TommLabel')
featureImportance('100Label')






