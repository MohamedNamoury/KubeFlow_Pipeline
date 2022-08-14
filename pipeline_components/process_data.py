import pandas as pd
import pipeline_components.config as config
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk

df = pd.read_csv(config.data_path)

all_features = df.columns

# Let's drop some features
useless = ["selected_text",'Time of Tweet','Age of User', 'Country',
    'Population -2020', 'Land Area (Km²)', 'Density (P/Km²)',"sentiment"] # features that we expect are uninformative
drop_list = useless 

# Remove unused coloumns
target = df['sentiment']

le = LabelEncoder() 
target = le.fit_traansorm(target)

features = df.drop(columns=drop_list)
features = nltk.word_tokenize(df)
vectorizer = CountVectorizer(lowercase = True, stop_words='english', max_features= 2000)
features = vectorizer.fit_transform(features)
# concating target and features together
data_processed = pd.concat([features, target])

df.to_csv("data_processed.csv")
from google.cloud import storage
storage_client = storage.Client()
bucket = storage_client.bucket(config.gs_bucket_name)
bucket.blob('processed/data_processed.csv').upload_from_filename('data_processed.csv', content_type='text/csv')
print("Raw Data Processed Sucessfully")

