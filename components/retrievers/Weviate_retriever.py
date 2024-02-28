"""Code for retrieval of documents using Weaviate Cloud"""
import weaviate
import os
from dotenv import load_dotenv
load_dotenv()
#Set up the Weaviate client
auth_config = weaviate.AuthApiKey(api_key= os.getenv("WEAVIATE_API_KEY"))
client = weaviate.Client(
url= os.getenv("WEAVIATE_URL"),
additional_headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
    },
auth_client_secret=auth_config
)

#Function to retrieve documents from Weaviate using keyword retrieval (BM25)
def Weviate_retrieve_keyword( k , query):
    query = query.lower()
    #Hardcode to extact source name
    response = (
    client.query
    .get("KFC", ["item_id", "product", "price", "content", "available", "category", "keywords"])
    .with_bm25(
      query= query
    )
    .with_limit(k)
    .do()
)
    #Build the response
    text_joined = """ """
    for index, doc in enumerate(response["data"]["Get"]["KFC"]):
        text_joined += f"{{Match {index}: {doc} }}, "
    return text_joined

#Function to retrieve documents from Weaviate using vector retrieval
def Weviate_retrieve_vector(k , query):
    query = query.lower()
    #Hardcode to extact source name
    response = (
    client.query
    .get("KFC", ["item_id", "product", "price", "content", "available", "category", "keywords"])
    .with_near_text({
        "concepts": [query]
    })
    .with_limit(k)
    .do()
)
    #Build the response
    text_joined = """ """
    for index, doc in enumerate(response["data"]["Get"]["KFC"]):
        text_joined += f"{{Match {index}: {doc} }}, "
    return text_joined

# Function to retrieve documents from Weaviate using hybrid retrieval
def Weviate_retrieve_hybrid( k , query):
    query = query.lower()
    #Hardcode to extact source name
    response = (
    client.query
    .get("KFC", ["item_id", "product", "price", "content", "available", "category", "keywords"])
    .with_hybrid(
        query= query
    )
    .with_limit(k)
    .do()
)
    text_joined = """ """
    for index, doc in enumerate(response["data"]["Get"]["KFC"]):
        text_joined += f"{{Match {index}: {doc} }}, "
    return text_joined



