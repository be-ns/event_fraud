from pymongo import MongoClient
import requests
import json


def post_to_mongo(loaded_json, db):
    '''
    INPUT: JSON formatted post, name of client MongoDB
    OUTPUT: NONE, sim
    '''
    posts = db.posts
    result = posts.insert_one(x_json)
    return None

if __name__ == '__main__':

    x = requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point').content
    x_json = json.loads(x)
    client = MongoClient()
    db = client.pymongo_test
    post_to_mongo(x_json, db)
