"""Code for native retrieval of documents using Sentence Transformers and Cosine Similarity"""
import pickle
import torch
from sentence_transformers import util
def Native_retrieve(embeddings, k, query):
    query = query.lower()
    # Import csv from pickle file
    with open('components/db_builds/Native_db_KFC.pickle', 'rb') as f:
        db = pickle.load(f)
    #Activating GPU if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_embedding = embeddings.encode(query, convert_to_tensor=True).to(device)
    stacked_chunks = torch.stack(db['embeddings'].tolist()).to(device)
    cos_scores = util.cos_sim(query_embedding, stacked_chunks)[0]
    top_results = torch.topk(cos_scores, k=k)
    indices = top_results.indices.tolist()
    #Build the string to be returned
    text_joined = ""
    for index, row in db.iloc[indices].iterrows():
        text_joined += f"{{Match {index}: {row['combined']}, Price: {row['Price']} , Available: {row['Available']}, Keywords: {row['Keywords']}  }} "
    return text_joined


