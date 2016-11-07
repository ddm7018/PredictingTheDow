import numpy as np 
import pandas as pd 
import sklearn
import matplotlib.pyplot as pl
import re
import nltk
import matplotlib as mpl
from sklearn.cross_validation import train_test_split
from wordcloud import WordCloud,STOPWORDS
#nltk.download()
from nltk.corpus import stopwords
from os import path
from PIL import Image
	



dataframe = pd.read_csv('E:/RIT/S3Fall2016/KDD/Project/-PredictingTheDow-master/-PredictingTheDow-master/stocknews/Combined_News_DJIA.csv')
print(dataframe.shape)


mpl.rcParams["figure.figsize"] = "8, 8"

dataframe['Combined']=dataframe.iloc[:,2:27].apply(lambda row: ''.join(str(row.values)), axis=1)

training,testing = train_test_split(dataframe,test_size=0.2,random_state=42)

upOrSame = training[training['Label']==1]
down = training[training['Label']==0]
print(len(upOrSame)/len(dataframe))

def to_words(content):
    letters_only = re.sub("[^a-zA-Z]", " ", content) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return( " ".join( meaningful_words )) 
	
upOrSame_word=[]
down_word=[]
for each in upOrSame['Combined']:
    upOrSame_word.append(to_words(each))

for each in down['Combined']:
    down_word.append(to_words(each))


d = path.dirname(__file__)	
	
bear_mask = np.array(Image.open(path.join(d,  "E:/RIT/S3Fall2016/KDD/Project/-PredictingTheDow-master/-PredictingTheDow-master/bear.jpg")))

down_wordCloud = WordCloud(background_color='black',
                      mask = bear_mask
                     )
down_wordCloud.generate(down_word[0])
pl.imshow(down_wordCloud)
pl.axis('off')

pl.show()


bull_mask = np.array(Image.open(path.join(d,  "E:/RIT/S3Fall2016/KDD/Project/-PredictingTheDow-master/-PredictingTheDow-master/bull.jpg")))


upOrSame_wordCloud = WordCloud(background_color='black',
                      mask = bull_mask
                     )
upOrSame_wordCloud.generate(upOrSame_word[0])					 

				 
					 
pl.imshow(upOrSame_wordCloud)
pl.axis('off')

pl.show()
print()
print("Words that led to increase or remain same: ")
print(len(upOrSame_word))
print()
print("Words that led to decrease: ")
print(len(down_word))
