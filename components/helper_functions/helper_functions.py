""" Helper functions for the app """
# Imports
import json
import sys
import os
two_levels_up = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../"))
sys.path.append(two_levels_up)
from components.retrievers.Faiss_retriever import Faiss_retrieve
from components.retrievers.qdrant_retriever import Qdrant_retrieve
from components.retrievers.Weviate_retriever import Weviate_retrieve_keyword, Weviate_retrieve_vector, Weviate_retrieve_hybrid
from components.retrievers.Native_retriever import Native_retrieve
from components.retrievers.Native_retriever_Tfidf import Tfidf_retrieve
from components.retrievers.Native_retriever_BM25 import bm25_retrieve
from components.retrievers.Azure_retriever import Azure_retrieve_vector, Azure_retrieve_keyword, Azure_retrieve_hybrid

# Function to get the result from the selected vector DB
def get_result(path_vs, embeddings,embeddings_Faiss, k, query_input, selected_sources, choice):
    results = ""
    if choice == "Faiss (local)":
        results = Faiss_retrieve(path_vs, embeddings_Faiss, k, query_input, selected_sources)
    elif choice == "Qdrant (cloud)":
        results = Qdrant_retrieve( k, query_input)
    elif choice == "Weviate_keyword (cloud)":
        results = Weviate_retrieve_keyword( k, query_input)
    elif choice == "Weviate_vector (cloud)":
        results = Weviate_retrieve_vector( k, query_input)
    elif choice == "Weviate_hybrid (cloud)":
        results = Weviate_retrieve_hybrid( k, query_input)
    elif choice == "Native_vector (local)":
        results = Native_retrieve( embeddings, k, query_input)
    elif choice == "Native_tfidf (local)":
        results = Tfidf_retrieve( k, query_input)
    elif choice == "Native_bm25 (local)":
        results = bm25_retrieve( k, query_input)
    elif choice == "Azure_keyword (cloud)":
        results = Azure_retrieve_keyword( k, query_input)
    elif choice == "Azure_vector (cloud)":
        results = Azure_retrieve_vector( embeddings, k, query_input)
    elif choice == "Azure_hybrid (cloud)":
        results = Azure_retrieve_hybrid( embeddings, k, query_input)
    return results

# Function to create the prompt for the LLM
def create_prompt(query, context, restaurant_name):
    return f"""
    You are a helpful assistant that give information about asked questions of products of this restaurant: {str(restaurant_name)}. 

    
    Answer the given query with the following context:

    
    Query: {str(query)}


    Context: {str(context)}
    """



