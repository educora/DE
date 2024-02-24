# read data
import pandas as pd

try:
    customers = pd.read_csv("C:\\temp\\data\\Sales\\customer_data.csv",low_memory=False)
    reviews = pd.read_csv("C:\\temp\\data\\Sales\\customer_reviews.csv",low_memory=False)
    sales = pd.read_csv("C:\\temp\\data\\Sales\\sales_data.csv",low_memory=False)    
except Error as e:
    print(e)
    
	
# get information about data
customers.head()
reviews.head()
sales.head()

# dataframe information
customers.info()
reviews.info()
sales.info()

# describe dataframes
customers.describe()
reviews.describe()
sales.describe()

# data cleaning
from sklearn.feature_extraction.text import re

def clean_data(review):    
    # 1st pass - remove any reviews with errors
    if "err" in str(review).lower():
        review = ""        
    # 2nd pass - remove any numeric data
    no_punc = re.sub(r'[^\w\s]', '', str(review))
    no_digits = ''.join([i for i in no_punc if not i.isdigit()])    
    return(no_digits)
	
reviews["Review"] = reviews["Review"].apply(clean_data)
reviews.head()

from nltk.sentiment import SentimentIntensityAnalyzer

def sentiment(review):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(str(review))
    #return score
    return str(score["compound"])

#print(sentiment("this is an excellent product, I loved it"))
#print(sentiment("the product was not too good, I am sorry"))

reviews["Sentiment"] = reviews["Review"].apply(sentiment)
reviews.head()

import sqlite3
from sqlite3 import Error

conn = None;
try:
    conn = sqlite3.connect(':memory:')
    print(sqlite3.version)
except Error as e:
    print(e)
finally:
    if conn:
        conn.close()

customers.to_sql("customers",conn, if_exists="append", index = False)
customer_rows = cur.execute("select * from customers limit 10;")    

reviews_data.to_sql("reviews",conn, if_exists="append", index = False)    
#review_rows = cur.execute("select * from customer_reviews limit 10;")
sales_data.to_sql("sales",conn, if_exists="append", index = False)
#sales = cur.execute("select * from sales limit 10;")


# Data integration, joining 3 tables and looking at the item reviews
rows = conn.execute("select c.*,s.*,r.* from customers c join sales s on s.CustomerID = c.customer_id left join reviews r on r.customerid = c.customer_id and r.StockCode = s.StockCode limit 10")
for r in rows:
    print(r)	


	