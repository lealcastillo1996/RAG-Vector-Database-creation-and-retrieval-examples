
"""CODE TO RETRIEVE DATA FROM AZURE SEARCH SERVICE"""
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import Vector 
from dotenv import load_dotenv
import os
load_dotenv()


#ACS Credentials
space = os.getenv("AZURE_SEARCH_SPACE")
search_service_endpoint: str = f"https://{space}.search.windows.net"
search_service_api_key: str =  os.getenv("AZURE_SEARCH_API_KEY")
index_name: str = "kfc"
credential = AzureKeyCredential(search_service_api_key)

#Azure Search Service vector retrieval
def Azure_retrieve_vector(embeddings, k , query, ):
    query = query.lower()  
    query_embeddings = embeddings.encode(query, convert_to_tensor=False)
    search_client = SearchClient(search_service_endpoint, index_name, AzureKeyCredential(search_service_api_key))  
    vector = Vector(value= query_embeddings.tolist(), k=k)   
    results = search_client.search(  
    search_text = None,
    vector = vector.value,
    select=["Item_id", "Product", "Price", "combined", "Available", "Category", "Keywords"] ,
    vector_fields= "embeddings",
    top_k = k,
    filter= "Restaurant eq 'kfc'"
)
    #Build the response
    text_joined = """ """
    for index, result in enumerate(results):
        text_joined += f"{{Match {index}: {result['combined']}, Price: {result['Price']}, Available: {result['Available']}, Keywords: {result['Keywords']}  }}, "

    return text_joined

#Azure Search Service keyword retrieval
def Azure_retrieve_keyword( k , query):
    query = query.lower()
    search_client = SearchClient(search_service_endpoint, index_name, AzureKeyCredential(search_service_api_key))  
    results = search_client.search(  
    search_text= query,   
    search_fields= ["combined", "Keywords"],
    select=["Item_id", "Product", "Price", "combined", "Available", "Category", "Keywords"],
    top = k,
    filter= "Restaurant eq 'kfc'"

)
    #Build the response
    text_joined = """ """
    for index, result in enumerate(results):
        text_joined += f"{{Match {index}: {result['combined']}, Price: {result['Price']}, Available: {result['Available']}, Keywords: {result['Keywords']}  }}, "

    return text_joined

# Azure Search Service hybrid retrieval
def Azure_retrieve_hybrid(embeddings, k , query):
    query = query.lower()  
    query_embeddings = embeddings.encode(query, convert_to_tensor=False)
    search_client = SearchClient(search_service_endpoint, index_name, AzureKeyCredential(search_service_api_key))  
    vector = Vector(value= query_embeddings.tolist(), k=k) 
    results = search_client.search(  
    vector = vector.value,
    search_text= query,   
    search_fields= ["combined", "Keywords"],
    select=["Item_id", "Product", "Price", "combined", "Available", "Category", "Keywords"] ,
    vector_fields= "embeddings",
    top_k = k,
    top = k,
    filter= "Restaurant eq 'kfc'"

)
    #Build the response
    text_joined = """ """
    for index, result in enumerate(results):
        text_joined += f"{{Match {index}: {result['combined']}, Price: {result['Price']}, Available: {result['Available']}, Keywords: {result['Keywords']}  }}, "
    return text_joined
