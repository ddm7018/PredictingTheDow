import pandas
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
import re
from nltk.corpus 				import stopwords


#adds in distribution of label, add more columns to csv
#generates a scatter plot of sentiment1 vs sentimentAll
#generated two histograms, one broken down by Label and the other broken down by Diff bins of 100



def to_words(content):
    letters_only = re.sub("[^a-zA-Z]", " ", content) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return( " ".join( meaningful_words ))


def classDistrubtion(combined):
	menMeans = (combined[combined.Label == 1].shape[0], combined[combined.Label == 0].shape[0])
	ind = np.arange(2)
	fig, ax = plt.subplots()
	rects1 = ax.bar(ind, menMeans,align='center', alpha=0.5, color='gr')

	ax.set_ylabel('# of days')
	ax.set_title('Class Distribution')
	ax.set_xticks(ind)
	ax.set_xticklabels(('Label = 1', 'Label = 0'))

	def autolabel(rects):
	    # attach some text labels
	    for rect in rects:
	        height = rect.get_height()
	        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
	                '%d' % int(height),
	                ha='center', va='bottom')


	plt.ylim(0,combined.shape[0])
	autolabel(rects1)
	plt.savefig("distribution/class-distrubtuion.png")



def greaterDistrubtion(full):
	listBin = []
	for ele in range(-8,8):
		listBin.append(ele*100)
	hist = full.Diff.hist(bins = listBin)
	plt.savefig("distribution/class-distribtion-greater.png")



combined 	= pandas.read_csv("stocknews/Combined_News_DJIA.csv")
stock  		= pandas.read_csv("stocknews/DJIA_table.csv")
full 		= stock.merge(combined, left_on = "Date", right_on = "Date", how = "outer")

s =[]
for row in full['Top1']: 
	s.append(TextBlob(to_words(row)).sentiment.polarity)
full['Sentiment1'] = s

s1 = []
for row in full.iloc[:,8:30].iterrows():
	rowText = ''
	for ele in row[1]:
		rowText += str(to_words(ele)) + " "
	s1.append(TextBlob(rowText).sentiment.polarity)
full['SentimentAll'] = s1
full['Tomm_Label'] = full.Label.shift(-1)


full[ "Sentiment1times"] = full[ "Sentiment1"] 
full[ "SentimentAll"] = full[ "SentimentAll"]  
full["Diff"] = full["Open"] - full["Close"]
def  diffLabel(x):
		if x > 100: return 2
		elif x >0 and x < 100: return 1
		elif x > -100 and x < 0: return -1 
		else: return -2
full['NewLabel'] = full['Diff'].apply(diffLabel)



full['100Label'] = full.Diff > 100 
full['100Label'] = full['100Label'].astype(int)
full.to_csv("stocknews/full-table.csv")
plt.scatter(full[[ "Sentiment1"]], full['SentimentAll'], c = full['Label'])
plt.savefig("distribution/scatter.png")
greaterDistrubtion(full)
classDistrubtion(combined)





