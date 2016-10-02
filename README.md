# \#PredictingTheDow

## Description
Group project using a kaggle dataset https://www.kaggle.com/aaron7sun/stocknews
The goal of this  analysis is to predict the relationship along with the strenght of relationship between the popular
news articles of the day, the direction of the stock market.



## Exploring the DJIA
```cmd
python explore.py
```
Running the explore python file will produce the Japanese candlestick chart. Further information about candlestick graphs can be found here: http://www.investopedia.com/ask/answers/07/candlestickcolor.asp
You can also access the chart using this link: https://people.rit.edu/ddm7018/KDDD/djia_candlestick.png 

## Running the Basic Models
At this stage on our group project, we are treating this as binary classfication, the training and test Y label is 0 or 1 to indicate if the stock market went up or down. Later on we try regression and more specific classfication approach.

We divided up the training and testing data per Kaggle's instructions. We then transformed the test and training data using both CountVectors and TD-IDF. We then ran Logistical Regression, KNN, LinearSVC, and RandomForrest for basic binary classfication twice(once for countvectors and once for TD-IDF). The results for the 8 models are printed to console. We also save the coefficents for the Logistical Regression using CountVectors along with Linear SVC using TD-IDF. The results are save as html files in the directory. You can also examine them with the link below

```cmd
python basicmodel.py
```


https://people.rit.edu/ddm7018/KDDD/log-countVector_head.html  
https://people.rit.edu/ddm7018/KDDD/log-countVector_tail.html  
https://people.rit.edu/ddm7018/KDDD/SVM-IDF_head.html  
https://people.rit.edu/ddm7018/KDDD/SVM-IDF_tail.html  
