import pandas
from textblob import TextBlob

#adding in stemming and stop word
#polarity and other features

combined 	= pandas.read_csv("stocknews/Combined_News_DJIA.csv")
stock  		= pandas.read_csv("stocknews/DJIA_table.csv")
full 		= stock.merge(combined, left_on = "Date", right_on = "Date", how = "outer")



stop = set(stopwords.words('english'))
sentence = "Whathever worska as23"
" ".join([i for i in sentence.lower().split() if i not in stop])


s =[]
for row in full['Top1']: s.append(TextBlob(row).sentiment.polarity)
full['Sentiment1'] = s

s1 = []
for row in full.iloc[:,7:33].iterrows():
	rowText = ''
	for ele in row[1]:
		rowText += str(ele)
	s1.append(TextBlob(rowText).sentiment.polarity)
full['SentimentAll'] = s1
full['Tomm_Label'] = full.Label.shift(-1)
full.to_csv("stocknews/full-table.csv")


full[ "Sentiment1times"] = full[ "Sentiment1"]  * 4 
full[ "SentimentAll"] = full[ "SentimentAll"]  * 10 
full["Diff"] = full["Open"] - full["Close"]


import matplotlib.pyplot as plt
plt.scatter(full[[ "Sentiment1"]], full['SentimentAll'], c = full['Label'])
#plt.show()


down = full[full.Label == 0].shape[0]

up = full[full.Label == 1].shape[0]

print up/float((up+down))