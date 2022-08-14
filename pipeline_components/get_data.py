import os
import pipeline_components.config as config
    
from google.cloud import storage
storage_client = storage.Client()
bucket = storage_client.bucket(config.gs_bucket_name)
bucket.blob('data/train.csv').upload_from_filename('train.csv', content_type='text/csv')
print("Data uploaded Sucessfully")