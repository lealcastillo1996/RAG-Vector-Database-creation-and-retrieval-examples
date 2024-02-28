""" Code to retrieve the relevant documents from FAISS"""
from langchain_community.vectorstores import FAISS

#Function to retrieve the relevant documents from FAISS
def Faiss_retrieve(path_vs, embeddings, k , query, list_sources):
    query = query.lower()
    db = FAISS.load_local(path_vs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": k, 'filter': {"source": list_sources}})
    docs = retriever.get_relevant_documents(query)
    #Build the string to be returned
    text_joined = """ """
    for index, doc in enumerate (docs):
        text_joined += f"{{Match {index}: {doc.page_content} , Price: {doc.metadata['price']}, Available: {doc.metadata['available']}, Keywords: {doc.metadata['keywords']}  }}, "
    return text_joined



