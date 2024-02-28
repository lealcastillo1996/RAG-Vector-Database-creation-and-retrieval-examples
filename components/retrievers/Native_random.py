"""Code for random retrieval of documents (reference for evaluation)"""
import pickle
import numpy as np

#Function to retrieve random documents from the local db
def Random_retrieve( k, query):
    query = query.lower()
    # Import csv from pickle file
    with open('components/db_builds/Native_db_KFC.pickle', 'rb') as f:
        db = pickle.load(f)
    #Generate a list of random indices between 0 and the lenght of db
    random_indices = np.random.choice(len(db), k, replace=False)
    text_joined = ""
    for index, row in db.iloc[random_indices].iterrows():
        text_joined += f"{{Match {index}: {row['combined']}, Price: {row['Price']} , Available: {row['Available']}, Keywords: {row['Keywords']}  }} "
    return text_joined



    
