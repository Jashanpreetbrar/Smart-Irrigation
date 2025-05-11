from pymongo import MongoClient
import os

MONGO_URI = os.environ.get("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client['fertilizer']
collection = db['predictions']

def store_prediction(data):
    collection.insert_one({"prediction": data})
