import pandas
combined 	= pandas.read_csv("stocknews/Combined_News_DJIA.csv")
stock  		= pandas.read_csv("stocknews/DJIA_table.csv")
full 		= stock.merge(combined, left_on = "Date", right_on = "Date", how = "outer")
full.to_csv("stocknews/full-table.csv")