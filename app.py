""" Code for running main the app """
# Imports
from langchain_openai import OpenAIEmbeddings
from components.main.main import main
from components.db_builds.Faiss_builder import Faiss_build_vs
from components.db_builds.qdrant_builder import Qdrant_build
from components.db_builds.Weviate_builder import Weviate_build
from components.db_builds.Native_builder import Native_build, Native_build_tfidf, Native_build_bm25
from components.db_builds.Azure_builder import Azure_build
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('punkt')  
#Activate GPU if available
import torch
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embeddings = SentenceTransformer('all-MiniLM-L6-v2', device=device)
directory_data = 'data/'   #Path to the pdf source
embeddings_Faiss = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
path_vs_Faiss = "components/db_builds/db_Faiss" #Path to save the vector store
path_vs_Native = "components/db_builds/Native_db_KFC.pickle" #Path to save the vector store

#Build the vector store if it does not exist
Faiss_success = None
Qdrant_success = None
Weviate_success = None
Native_success = None
Azure_success = None

#CREATE THE VECTOR STORES IF THEY DON'T EXIST   Uncomment the following lines to build the vector stores if required

#BUILD FAISS
#Faiss_success = Faiss_build_vs(directory_data, path_vs_Faiss)

#Build the vector store for Qdrant
#Qdrant_success = Qdrant_build(directory_data, "KFC")

#Build the vector store for Weviate
#Weviate_success = Weviate_build(directory_data, "KFC")

#Build the native approaches dbs
#Native_success = Native_build(directory_data, "KFC")
#Native_success = Native_build_tfidf(directory_data, "KFC")
#Native_success =Native_build_bm25(directory_data, "KFC")

#Build Azure Search Index
#Azure_success = Azure_build(directory_data, "kfc")

#Run the app
if __name__ == "__main__":
    main(path_vs_Faiss, embeddings,embeddings_Faiss,directory_data,Faiss_success, Qdrant_success, Weviate_success, Native_success, Azure_success)