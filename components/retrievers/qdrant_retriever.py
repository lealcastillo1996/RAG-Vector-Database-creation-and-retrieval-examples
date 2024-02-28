"""Code for retrieval of documents using Qdrant"""
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
load_dotenv()
#Set up the Qdrant client
client = QdrantClient(
        url= os.getenv("QDRANT_URL"),
        api_key= os.getenv("QDRANT_API_KEY")
    )
client.set_model("sentence-transformers/all-MiniLM-L6-v2")

#Function to retrieve documents from Qdrant
def Qdrant_retrieve( k , query):
    query = query.lower()
    results = client.query(
    collection_name= "KFC",
    query_filter=None,
    query_text=query,
    limit=k,
)
    docs = results
    #Build the string to be returned
    text_joined = """ """
    for index, doc in enumerate (docs):
        text_joined += f"{{Match {index}: {doc.metadata['document']},  Price: {doc.metadata['price']}, Available: {doc.metadata['available']}, Keywords: {doc.metadata['keywords']}  }}, "
    return text_joined



