"""code to build the vector store with langchain-Faiss"""
#Imports
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import tiktoken
tokenizer = tiktoken.get_encoding('p50k_base')
import json
from langchain_core.documents.base import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
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

#Class to load the json files and create the documents for indexing in Faiss
class faiss_json_loader:
    def __init__(self, file):
        self.file = file
        with open(self.file, 'r') as file:
            self.json_data = json.load(file)

    def create_document(self, item_id, name, price, nutritional_info, available, category, keywords):
        text = f" Product: {name}, Content: {nutritional_info}, Category: {category}"
        text = text.lower()
        metadata = {
            "available": available,
            "source": self.file,
            "keywords": keywords,
            "price": price,
        }
        return Document(metadata=metadata, page_content=text)

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

#Function to build the vector store
def Faiss_build_vs(directory, name_store):
    paths = list_files(directory)
    documents= []
    for path in paths:
            if path.endswith(".json"):
                loader = faiss_json_loader(path)
                documents.extend(loader.json_to_document_list())
    
    #Defining the embeddings for Faiss with SentenceTransformer
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embeddings)
    #Save the vectorstore
    db.save_local(name_store)
    print(f'DB built successfully! , \n  lenght of the vector store: {len(documents)} chunks  \n  documents indexed: {paths}')
    return len(documents)
