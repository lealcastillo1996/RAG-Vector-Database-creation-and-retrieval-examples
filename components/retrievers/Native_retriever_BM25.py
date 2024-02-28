"""Code for native retrieval of documents using BM25"""
import pickle
import numpy as np
import heapq
from nltk.tokenize import word_tokenize

#Function to retrieve the relevant documents from BM25
def bm25_retrieve( k, query, ):
    query = query.lower()
    # Import csv from pickle file
    with open('components/db_builds/bm25_KFC.pickle', 'rb') as f:
        bm25 = pickle.load(f)
    with open('components/db_builds/documents_all_KFC.pickle', 'rb') as f:
        documents = pickle.load(f)
    query = query.lower()
    tokenized_query = word_tokenize(query)
    # Calculate BM25 scores
    scores = bm25.get_scores(tokenized_query)
    # Use a heap to efficiently find top k matches
    heap = []
    for i, score in enumerate(scores):
        if len(heap) < k:
            heapq.heappush(heap, (score, documents[i]))
        else:
            heapq.heappushpop(heap, (score, documents[i]))
    # Extract top k matches from the heap
    top_k_matches = [item[1] for item in sorted(heap, reverse=True)]
    # Build the text representation of the top k matches
    text_joined = ""
    for index, document in enumerate(top_k_matches):
        text_joined += f"{{Match {index + 1}: {document} }}, "
    return text_joined


    
