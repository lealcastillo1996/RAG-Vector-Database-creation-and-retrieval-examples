"""Code for native retrieval of documents using Tfidf"""
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import scipy
#Function to retrieve the relevant documents from Tfidf
def Tfidf_retrieve( k, query):
    query = query.lower()
    # Import csv from pickle file
    tfidf_matrix = scipy.sparse.load_npz('components/db_builds/tf_idf_KFC.npz')
    with open('components/db_builds/tf_idf_vectorizer_KFC.pickle', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open('components/db_builds/Native_db_KFC.pickle', 'rb') as f:
        db = pickle.load(f)
    query = query.lower()
    query_vector = tfidf_vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix)
    top_k_indices = np.argpartition(similarities[0], -k)[-k:][::-1]
    #Build the string to be returned
    text_joined = ""
    for index, row in db.iloc[top_k_indices].iterrows():
        text_joined += f"{{Match {index}: {row['combined']}, Price: {row['Price']} , Available: {row['Available']}, Keywords: {row['Keywords']}  }} "
    return text_joined



    
