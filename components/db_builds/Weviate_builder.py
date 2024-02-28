"""Code to create the Weviate vector store in cloud"""
import weaviate
import json
import time
from .Faiss_builder import list_files
import sys
import os
two_levels_up = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../"))
sys.path.append(two_levels_up)
from dotenv import load_dotenv
load_dotenv()

class weviate_json_loader:
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
                "Available": available,
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
                        keywords = item_info.get("keywords", True)
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
                document = self.create_document(item_id, name, price, nutritional_info, available, category,keywords)
                if price != "":
                    document_list.append(document)
        return document_list

   
#Function to build the vector store
def Weviate_build(directory, name_store):
    paths = list_files(directory)
    documents= []
    for path in paths:
            if path.endswith(".json"):
                loader = weviate_json_loader(path)
                documents.extend(loader.json_to_document_list())

    auth_config = weaviate.AuthApiKey(os.getenv("WEAVIATE_API_KEY"))
    client = weaviate.Client(
        
    url= os.getenv("WEAVIATE_URL"),
    additional_headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
    },
    auth_client_secret=auth_config
    )
    #Delete the class if it exists
    try:
        client.schema.delete_class(name_store) 
        time.sleep(1)
    except Exception as e:
        print(e)
        pass
    #Creating an empty class wih schema
    class_obj = {
    "class": name_store,
    "properties": [
        {
            "name": "item_id",
            "dataType": ["text"],
        },
        {
            "name": "product",
            "dataType": ["text"],
        },
        {
            "name": "price",
            "dataType": ["text"],
        },
        {
            "name": "content",
            "dataType": ["text"],
        },
        {
            "name": "available",
            "dataType": ["boolean"],
        },
        {
            "name": "category",
            "dataType": ["text"],
        },
         {
            "name": "keywords",
            "dataType": ["text"],
        }
    ],
  "vectorizer": "text2vec-openai",
      "moduleConfig": {
        "text2vec-openai": {
          "model": "text-embedding-3-small",}}
}
    try:
        client.schema.create_class(class_obj)
        #Adding the documents to the class from the list of objects (batch import)
        class_name = name_store
        # Replace with your class name
        client.batch.configure(batch_size=100)  # Configure batch
        with client.batch as batch:
            for data_obj in documents:
                batch.add_data_object(
                    data_obj,
                    class_name,
                    # tenant="tenantA"  # If multi-tenancy is enabled, specify the tenant to which the object will be added.
                )
    except Exception as e:
        print(e)
        pass
    return len(documents)