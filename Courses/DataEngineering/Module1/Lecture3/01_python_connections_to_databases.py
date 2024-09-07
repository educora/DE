# Database	Python Code Snippet
# MySQL	
import mysql.connector
conn = mysql.connector.connect(host='localhost', user='your_username', password='your_password', database='your_database')
cursor = conn.cursor()

# PostgreSQL	
import psycopg2
conn = psycopg2.connect(host='localhost', dbname='your_database', user='your_username', password='your_password')
cursor = conn.cursor()

# SQLite	
import sqlite3
conn = sqlite3.connect('your_database.db')
cursor = conn.cursor()

# Oracle	
import cx_Oracle
conn = cx_Oracle.connect('your_username/your_password@localhost')
cursor = conn.cursor()

# SQL Server	
import pyodbc 
conn = pyodbc.connect('DRIVER={SQL Server};SERVER=localhost;DATABASE=your_database;UID=your_username;PWD=your_password')
cursor = conn.cursor()

# MariaDB	
import mariadb
conn = mariadb.connect(host='localhost', user='your_username', password='your_password', database='your_database')
cursor = conn.cursor()

### Semi Structured Sources

# JSON Files	
import json
with open('data.json', 'r') as file:
    data = json.load(file)
    print(data)
    
# XML Files	
import xml.etree.ElementTree as ET
tree = ET.parse('data.xml')
root = tree.getroot()
print(root.tag)

# YAML Files	
import yaml
with open('data.yaml', 'r') as file:
    data = yaml.safe_load(file)
    print(data)

# CSV Files	
import csv
with open('data.csv', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)

# MongoDB	
from pymongo import MongoClient
client = MongoClient('localhost', 27017)
db = client['your_database']
collection = db['your_collection']
doc = collection.find_one()
print(doc)

# Apache Cassandra	
from cassandra.cluster import Cluster
cluster = Cluster(['localhost'])
session = cluster.connect('your_keyspace')
rows = session.execute('SELECT * FROM your_table')
for row in rows:
    print(row)

# Elasticsearch	
from elasticsearch import Elasticsearch
es = Elasticsearch(['http://localhost:9200'])
response = es.get(index='your_index', doc_type='your_doc_type', id=1)
print(response['_source'])

# Redis	
import redis
r = redis.Redis(host='localhost', port=6379, db=0)
value = r.get('your_key')
print(value)


# Unstructured Sources
# Text Files	
with open('data.txt', 'r') as file:
    content = file.read()
    print(content)


# PDF Files	
import PyPDF2
with open('data.pdf', 'rb') as file:
    reader = PyPDF2.PdfFileReader(file)
    page = reader.getPage(0)
    text = page.extractText()
    print(text)

# Image Files	
from PIL import Image
img = Image.open('photo.jpg')
img.show()

# Audio Files	
import wave
with wave.open('sound.wav', 'r') as wav_file:
    frames = wav_file.readframes(-1)
    print(frames)

# Video Files	
import cv2
cap = cv2.VideoCapture('video.mp4')
ret, frame = cap.read()
cv2.imshow('Frame', frame)
cv2.waitKey(0)
cap.release()

# Emails	
import imaplib
mail = imaplib.IMAP4_SSL('imap.example.com')
mail.login('your_email@example.com', 'your_password')
mail.select('inbox')

# Social Media API	
import tweepy
auth = tweepy.OAuthHandler('consumer_key', 
                           'consumer_secret')
auth.set_access_token('access_token', 
                      'access_token_secret')
api = tweepy.API(auth)
public_tweets = api.home_timeline()
for tweet in public_tweets:
    print(tweet.text)

# Web Pages (HTML)	
import requests
from bs4 import BeautifulSoup
response = requests.get('http://example.com')
soup = BeautifulSoup(response.content, 'html.parser')
print(soup.prettify())


# APIs
# REST
import requests

# URL of the REST API endpoint
api_url = "https://api.example.com/data"

# Headers and parameters can be customized as needed
headers = {
    'Authorization': 'Bearer YOUR_ACCESS_TOKEN',
    'Accept': 'application/json'
}

# Making a GET request to the API
response = requests.get(api_url, headers=headers)

# Checking if the request was successful
if response.status_code == 200:
    # Parsing the JSON response
    data = response.json()
    print("Data retrieved successfully:")
    print(data)
else:
    print("Failed to retrieve data")
    print("Status Code:", response.status_code)
    print("Response:", response.text)

# GraphQL
import requests
import json

# URL of the GraphQL API endpoint
graphql_url = 'https://api.example.com/graphql'
# GraphQL query. For example, fetching user data by ID.
query = """
{
  user(id: "1") {
    name
    email
    posts {
      title
      content
    }
  }
}
"""
# Headers may need to include tokens or API keys for authentication
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer YOUR_ACCESS_TOKEN'  # Adjust as needed
}

# Making the POST request to the GraphQL endpoint
response = requests.post(graphql_url, json={'query': query}, headers=headers)

# Checking if the request was successful
if response.status_code == 200:
    # The response.data contains the results
    data = response.json()
    print("Data retrieved successfully:")
    print(json.dumps(data, indent=2))  # Pretty print the data
else:
    print("Failed to retrieve data")
    print("Status Code:", response.status_code)
    print("Response:", response.text)


3rd Floor, 86-90 Paul Street, London, England, United Kingdom, EC2A 4NE