"""code to build the index in Azure"""
from dotenv import load_dotenv
import os
import tiktoken
tokenizer = tiktoken.get_encoding('p50k_base')
import json
import pandas as pd
import numpy as np
import math
import ast
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')
import pandas as pd
from azure.core.credentials import AzureKeyCredential
from tenacity import retry, wait_random_exponential, stop_after_attempt,wait_fixed
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswParameters,
    PrioritizedFields,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticSettings,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
    CorsOptions,
)
load_dotenv()

#ACS Credentials
space = os.getenv("AZURE_SEARCH_SPACE")
search_service_endpoint: str = f"https://{space}.search.windows.net"
search_service_api_key: str =  os.getenv("AZURE_SEARCH_API_KEY")
credential = AzureKeyCredential(search_service_api_key)

"""
"Item_id": str(item_id).lower(),
                "Product": str(name).lower(),   
                "Price": str(price).lower(),
                "Content": str(nutritional_info).lower(),
                "Available": str(available).lower(),
                "Category": str(category).lower(),
                "Keywords": str(keywords).lower(),
                "Restaurant": "KFC"


"""

# Decorate the feed function with a retry policy
@retry(
    stop=stop_after_attempt(10),  # Number of retry attempts
    wait=wait_fixed(1),  # Wait 2 seconds between retries
    reraise=True  # Reraise exceptions after retries are exhausted
)
def create_search_index_text(index_name):
    cors_options = CorsOptions(allowed_origins=["*"], max_age_in_seconds=60)
    index_client = SearchIndexClient(
        endpoint= search_service_endpoint,
        credential= credential,
    )
    if isinstance(index_name, str):
        index = SearchIndex(
            cors_options = cors_options,
            name= index_name,
            fields=[
                SimpleField(
                    name="Item_id",
                    type=SearchFieldDataType.String,
                    key=True,
                    filterable=True,
                    sortable=True,
                    facetable=True,
                ),
                SearchableField(name="Product", type=SearchFieldDataType.String),
                SearchableField(name="combined", type=SearchFieldDataType.String),
                SearchableField(name="Keywords", type=SearchFieldDataType.String),
                #Filter restaurant field
                SearchableField(name="Restaurant", type=SearchFieldDataType.String, filterable=True, retrievable = False, key = False),
                #Other info fields
                SearchableField(name="Price", type=SearchFieldDataType.String, filterable=True, ),
                SearchableField(name="Content", type=SearchFieldDataType.String, filterable=True, ),
                SearchableField(name="Available", type=SearchFieldDataType.String, filterable=True,),
                SearchableField(name="Category", type=SearchFieldDataType.String, filterable=True,),
                SearchField(
                    name="embeddings",
                    type= SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    vector_search_dimensions=384, 
                    vector_search_configuration="my-vector-config"  
                ),
              
            ],
            vector_search=VectorSearch(
                algorithm_configurations=[
                    VectorSearchAlgorithmConfiguration(
                        name="my-vector-config",
                        kind="hnsw",
                        hnsw_parameters=HnswParameters(
                            m=4, ef_construction=400, ef_search=500, metric="cosine"
                        ),
                    )
                ]
            ),
            semantic_settings=SemanticSettings(
                configurations=[
                    SemanticConfiguration(
                        name="my-semantic-config",
                        prioritized_fields=PrioritizedFields(
                            title_field=SemanticField(field_name="Product"),
                            prioritized_content_fields=[
                                SemanticField(field_name="combined")
                            ],
                            prioritized_keywords_fields=[
                                SemanticField(field_name="Keywords")
                            ],
                        ),
                    )
                ]
            ),
        )
        print(f"Creating {index_name} search index")
        index_client.create_index(index)
            
        print("created")
        return True
    else:
        print(f"Search index {index_name} already exists")
        return False

# Decorate the feed function with a retry policy
@retry(
    stop=stop_after_attempt(10),  # Number of retry attempts
    wait=wait_fixed(1),  # Wait 2 seconds between retries
    reraise=True  # Reraise exceptions after retries are exhausted
)
def delete_search_index(index_name):
    SearchIndexClient(
            endpoint= search_service_endpoint,
            credential= credential,
        ).delete_index(index_name)
    print(index_name, "deleted success")

# Decorate the feed function with a retry policy
@retry(
    stop=stop_after_attempt(10),  # Number of retry attempts
    wait=wait_fixed(1),  # Wait 2 seconds between retries
    reraise=True  # Reraise exceptions after retries are exhausted
)
def feed(index_name,df):
    # Create columns for cognitive search
    df
    # Define batch size
    batch_size = 1000
    # Calculate the number of batches
    num_batches = math.ceil(len(df) / batch_size)
    # Initialize the SearchClient/
    search_client = SearchClient(endpoint=search_service_endpoint, index_name=index_name, credential=credential)
    # Loop through batches
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = (batch_num + 1) * batch_size
        batch_df = df[start_idx:end_idx]
        documents = []

        embeddings = batch_df['embeddings'].tolist()
        for index, row in batch_df.iterrows():
            document = {
                #"@search.action": "upload",
                "Item_id": str(row['Item_id'].lower()),
                "Product": str(row['Product'].lower()),
                "combined": str(row['combined'].lower()),
                "Restaurant": str(row['Restaurant'].lower()),
                "embeddings": embeddings[index].tolist(),
                "Keywords": str(row['Keywords'].lower()),
                "Price": str(row['Price'].lower()),
                "Content": str(row['Content'].lower()),
                "Available": str(row['Available'].lower()),
                "Category": str(row['Category'].lower()),
            }
            documents.append(document)
        # Upload the batch of documents
        result = search_client.upload_documents(documents)
        print('Batch', batch_num + 1, 'uploaded (', len(documents), ' documents )')
    print('All documents uploaded to:', index_name)

def get_embedding(text):
   return embedder.encode(text, convert_to_tensor=False)

#Function to extract all the sources of pdfs in a folder
def list_files(directory):
    paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                paths.append(os.path.join(root, file))
    return paths

#Jspn loader class
class Azure_json_loader:
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
                "Keywords": str(keywords).lower(),
                "Restaurant": "KFC"

            }
        return document

    def json_to_document_list(self):
        document_list = []
        for category, items in self.json_data.items():
            for item_id, item_info in items.items():
                try:
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
def Azure_build(directory, name_store):
    paths = list_files(directory)
    documents= []

    for path in paths:
            if path.endswith(".json"):
                loader = Azure_json_loader(path)
                documents.extend(loader.json_to_document_list())
    db = pd.DataFrame(documents)
    db['combined'] = "Product: " + db['Product'] + "; " + "Content: " + db['Content'] + "; " + "Category: " + db['Category']
    #get embeddings columns
    db['embeddings'] = db.combined.apply(lambda x: get_embedding(x))
    try:
        db['embeddings'] =db['embeddings'].apply(ast.literal_eval)
    except: 
        pass    
    #Create a new azure search index
    #operate the fjunctions to create search index and its backup
    #1.- Try delete new
    try:
        delete_search_index(name_store)
    except Exception as e:
            print(e)
    #2.- Create new 
    try:
        create_search_index_text(name_store)
    except Exception as e:
        print(e)
    #3.- feed 
    try:
        feed(name_store, db)
    except Exception as e:
        print('error feeding')
        print(e)
    return len(db)



