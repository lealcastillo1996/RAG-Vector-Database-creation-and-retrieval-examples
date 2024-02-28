"""Code for building the Qdrant vector store in cloud"""
#imports
from qdrant_client import QdrantClient
import json
from .Faiss_builder import list_files
import time
from qdrant_client.models import VectorParams, Distance
from tqdm import tqdm
from dotenv import load_dotenv
import os

# Load the environment variables from .env file
load_dotenv()

#Class to load the json files and create the documents for indexing in Qdrant
class qdrant_json_loader:
    def __init__(self, file):
        self.file = file
        with open(self.file, 'r') as file:
            self.json_data = json.load(file)

    def create_document(self, item_id, name, price, nutritional_info, available, category, keywords):
        text = f"Product: {name}, Content: {nutritional_info}, Category: {category}"
        text = text.lower()
        metadata = {
            "available": available,
            "source": self.file,
            "keywords": keywords,
            "price": price,
            "item_id": item_id,
        }
        return text, metadata

    def json_to_document_list(self):
        document_list = []
        metadata_list = []
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
                document,meta = self.create_document(item_id, name, price, nutritional_info, available, category, keywords)
                if price != "":
                    document_list.append(document)
                    metadata_list.append(meta)
        return document_list, metadata_list

#Function to build the vector store
def Qdrant_build(directory, name_store):
    paths = list_files(directory)
    documents= []
    metadata = []
    for path in paths:
            if path.endswith(".json"):
                loader = qdrant_json_loader(path)
                document_list, metadata_list = loader.json_to_document_list()
                documents.extend(document_list)
                metadata.extend(metadata_list)

    
    client = QdrantClient(
        url=  os.getenv("QDRANT_URL"),
        api_key= os.getenv("QDRANT_API_KEY")
    )
    client.set_model("sentence-transformers/all-MiniLM-L6-v2")

    try:
        client.delete_collection(collection_name = name_store)
        time.sleep(5)
    except:
        pass
    
    try:
        client.add(
        collection_name=name_store,
        documents=documents,
        metadata=metadata,
        ids=tqdm(range(len(documents))))
    

    except:
        pass
    return len(documents)