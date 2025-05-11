from pymongo import MongoClient
import ssl
import os
uri = os.environ.get("MONGO_URI")
client = MongoClient(uri, ssl=True, ssl_cert_reqs=ssl.CERT_REQUIRED)
db = client['fertilizer']
collection = db['predictions']

def store_prediction(data):
    collection.insert_one({"prediction": data})
