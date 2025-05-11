import os
from pymongo import MongoClient
from datetime import datetime

client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
db = client["npk_predictions"]
collection = db["predictions"]

def store_prediction(prediction):
    collection.insert_one({
        "timestamp": datetime.utcnow(),
        "prediction": prediction
    })