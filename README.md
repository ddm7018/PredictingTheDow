# \#PredictingTheDow

## Description
Group project using a kaggle dataset https://www.kaggle.com/aaron7sun/stocknews
  wqasdadThe goal of this  analysis is to predict the relationship along with the strenght of relationship between the popular
news articles of the day, the direction of the stock market.



## Exploring the DJIA
```cmd
python explore.py
```
Running the explore python file will produce the Japanese candlestick chart. Further information about candlestick graphs can be found here: http://www.investopedia.com/ask/answers/07/candlestickcolor.asp
You can also access the chart using this link: https://people.rit.edu/ddm7018/KDDD/djia_candlestick.png 

## Running the Basic Classfication Models
At this stage on our group project, we are treating this as binary classfication, the training and test Y label is 0 or 1 to indicate if the stock market went up or down. Later on we try regression and more specific classfication approach. We divide up the data randomly rather than following Kaggle's instructions.

```cmd
python basic_classifcation.py
```

This runs 
- KNN finding the optimal N
- Logistic Regression
- LinearSVC
- RandomForest
- Decision Tree
- QDA
- LDA
- Ada Boost

against four different kinds of vectors
- Count Vector
- Count Vector Ngram of 2
- TD-IDF
- TI-IDF Ngram of 2

and prints the resulting accaurcies. 

```cmd
python wordcloud.py
```
this file generated two word clouds. One in the shape of bear for when most frequent words for when the market goes down, and a bull for when the market stay the same or goes up. Two generated images were place in the wordcl

## Refining Core Algorithm (KNN)

```cmd
python refineKNN.py
```
This file runs k-fold validation and reports the optimal number of neighbors.

## Accuarcies
The screenshots of refineKNN.py and basic_classfication.py are include the screenshot folder
