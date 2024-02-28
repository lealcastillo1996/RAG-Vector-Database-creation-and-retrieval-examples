"""code to build the vector stores and dbs for my native local approach"""
from dotenv import load_dotenv
import os
import tiktoken
tokenizer = tiktoken.get_encoding('p50k_base')
import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util
embedder = SentenceTransformer('all-MiniLM-L6-v2')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy
import pickle
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt') 

#Function to get the embeddings
def get_embedding(text):
   return embedder.encode(text, convert_to_tensor=True)

# Load the environment variables from .env file
load_dotenv()

#Function to extract all the sources of pdfs in a folder
def list_files(directory):
    paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                paths.append(os.path.join(root, file))
    return paths

class Native_json_loader:
    def __init__(self, file):
        self.file = file
        with open(self.file, 'r') as file:
            self.json_data = json.load(file)
    def create_document(self, item_id, name, price, nutritional_info, available, category, keywords):
        document = {
                "Item_id": str(item_id).lower(),
                "Product": str(name).lower(),   
                "Price": str(price).lower(),
                "Content": str(nutritional_info).lower(),
                "Available": str(available).lower(),
                "Category": str(category).lower(),
                "Keywords": str(keywords).lower()
            }
        return document

    def json_to_document_list(self):
        document_list = []
        for category, items in self.json_data.items():
            for item_id, item_info in items.items():
                try:
                    #Packs
                    if isinstance(item_info, dict):
                        name = item_info.get("name", "")
                        price = item_info.get("price", "")
                        nutritional_info = item_info.get("contents", {})
                        available = item_info.get("available", True)
                        keywords = ""
                    #Individual items
                    else:
                        name, price, details, keywords = item_info
                        nutritional_info = details.get("nutritionalInfo", {})
                        available = details.get("available", True)
                except:
                    name = ""
                    price = ""
                    nutritional_info = ""
                    available = False    
                    keywords = ""    
                document = self.create_document(item_id, name, price, nutritional_info, available, category, keywords)
                if price != "":
                    document_list.append(document)
        return document_list

#Function to build the nativa local vector store
def Native_build(directory, name_store):
    paths = list_files(directory)
    documents= []
    for path in paths:
            if path.endswith(".json"):
                loader = Native_json_loader(path)
                documents.extend(loader.json_to_document_list())
    db = pd.DataFrame(documents)
    db['combined'] = "Product: " + db['Product'] + "; " + "Content: " + db['Content'] + "; " + "Category: " + db['Category'] + ";" 
    #get embeddings columns
    db['embeddings'] = db.combined.apply(lambda x: get_embedding(x))
    #save db to a piclke file (pickle load faster than csv)
    db.to_pickle('components/db_builds/Native_db_'+ name_store + '.pickle')
    print(f'DB built successfully! , \n  lenght of the vector store: {len(documents)} chunks  \n  documents indexed: {paths}')
    return len(documents)

#Function to build the nativa local tfidf approach
def Native_build_tfidf(directory, name_store):
    paths = list_files(directory)
    documents= []
    for path in paths:
            if path.endswith(".json"):
                loader = Native_json_loader(path)
                documents.extend(loader.json_to_document_list())
    db = pd.DataFrame(documents)
    db['combined'] = "Product: " + db['Product'] + "; " + "Content: " + db['Content'] + "; " + "Category: " + db['Category'] + "; " + "Keywords: " + db['Keywords'] 
    #get embeddings columns
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    documents = db['combined'].tolist()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    #save db to a piclke file (pickle load faster than csv)
    scipy.sparse.save_npz('components/db_builds/tf_idf_'+ name_store + '.npz', tfidf_matrix)
    with open('components/db_builds/tf_idf_vectorizer_'+ name_store + '.pickle', 'wb') as fin:
        pickle.dump(tfidf_vectorizer, fin)
    with open('components/db_builds/documents_'+ name_store + '.pickle', 'wb') as fin:
        pickle.dump(documents, fin)
    print(f'DB built successfully! , \n  lenght of Tfidf: {len(documents)} chunks  \n  documents indexed: {paths}')
    return len(documents)

#Function to build the nativa local bm25 approach
def Native_build_bm25(directory, name_store):
    paths = list_files(directory)
    documents= []
    for path in paths:
            if path.endswith(".json"):
                loader = Native_json_loader(path)
                documents.extend(loader.json_to_document_list())
    db = pd.DataFrame(documents)
    db['combined'] = "Product: " + db['Product'] + "; " + " Content: " + db['Content'] + "; " + " Category: " + db['Category'] + "; " + " Keywords: " + db['Keywords'] 
    db['all'] = "Product: " + db['Product'] + "; " + " Content: " + db['Content'] + "; " + " Category: " + db['Category'] + "; " + " Keywords: " + db['Keywords'] + "; " + " Price: " + db['Price'] + "; " + " Available: " + db['Available'] + ";"
    documents_all = db['all'].tolist()
    with open('components/db_builds/documents_all_'+ name_store + '.pickle', 'wb') as fin:
        pickle.dump(documents_all, fin)
    tokenized_documents = [word_tokenize(doc) for doc in db['combined'].tolist()]
    bm25 = BM25Okapi(tokenized_documents,  k1=1.5, b=0.75)
    #get embeddings columns
    with open('components/db_builds/bm25_'+ name_store + '.pickle', 'wb') as fin:
        pickle.dump(bm25, fin)
    print(f'DB built successfully! , \n  lenght of bm25: {len(documents)} chunks  \n  documents indexed: {paths}')
    return len(documents)

