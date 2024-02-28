"""Main app streamlit code"""
#imports
import streamlit as st
import json
from ..helper_functions.helper_functions import  get_result, create_prompt
from ..db_builds.Faiss_builder import list_files
from .config_css import setup_page_config
from ..llm_call.mistral_llm import call_llm
from ..llm_call.gpt4_llm import call_llm_gpt4
import time

#Function to run the main app
def main(path_vs, embeddings,embeddings_Faiss, directory_menus,Faiss_success  = None, Qdrant_success = None, Weviate_success = None, Native_success = None, Azure_success = None):
    #Set up the page configuration
    setup_page_config()
    #Initialize the session state vars
    if 'context_input' not in st.session_state:
        st.session_state.context_input = ""
    if 'time_retrieval' not in st.session_state:
        st.session_state.time_retrieval = ""
    if 'time_llm' not in st.session_state:
        st.session_state.time_llm = ""
    if 'response_llm' not in st.session_state:
        st.session_state.response_llm = ""
    if 'response_llm_gpt4' not in st.session_state:
        st.session_state.response_llm_gpt4 = ""
    if 'time_llm_gpt4' not in st.session_state:
        st.session_state.time_llm_gpt4 = ""
    #Display db buildup message
    if Faiss_success:
        st.success(f'Vector store built successfully for Faiss! , \n  lenght of the vector store: {Faiss_success} pages')
    if Qdrant_success:
        st.success(f'Vector store built successfully for Qdrant! , \n  lenght of the vector store: {Qdrant_success} pages')
    if Weviate_success:
        st.success(f'Vector store built successfully for Weviate! , \n  lenght of the vector store: {Weviate_success} pages')
    if Native_success:
        st.success(f'Vector store built successfully for Native! , \n  lenght of the vector store: {Native_success} pages')
    if Azure_success:
        st.success(f'Vector store built successfully for Azure! , \n  lenght of the vector store: {Azure_success} pages')

    #Page title
    st.title("Task VOX")
    #Sidebar
    with st.sidebar.form(key ='side1'):
        menu = ["Faiss (local)", "Qdrant (cloud)",  "Weviate_keyword (cloud)", "Weviate_vector (cloud)", "Weviate_hybrid (cloud)", "Native_vector (local)", "Native_tfidf (local)", "Native_bm25 (local)", "Azure_keyword (cloud)", "Azure_vector (cloud)", "Azure_hybrid (cloud)"]
        choice = st.selectbox("Choose a vector DB", menu)
        restaurant_menus = list_files(directory_menus)
        selected_sources = st.multiselect(
        "Select a restaurant :blue[Sources]",
        restaurant_menus,
        help="Select the menu JSON files you want to include as sources.")
        k = st.slider("Select the value of k for retrieval", min_value=1, max_value=20, value=10)
        query_input = st.text_area("Query Input")
        ask_button = st.form_submit_button(label='RAG retrieve')
        # Initialize submit_button
        submit_button = False
        if st.session_state.context_input != "":
            submit_button = st.form_submit_button(label='LLM prompt results')

    # Page containers with sections
    con1= st.container()
    con2= st.container()
    con3= st.container()
    
    # Container 1 Section: RAG retrieve generated prompt
    with con1:
        st.write("RAG latency [ms] ", st.session_state.time_retrieval)
        st.subheader("RAG Generated Prompt:")
        if st.session_state.context_input != "":
            st.markdown(st.session_state.context_input)
    # Handle Ask Button Press
    if ask_button:
        with st.spinner('Retrieving...'):
            #Retrieve the top k matching documents 
            #Cout time
            start = time.time()
            retrieved_docs = get_result(path_vs, embeddings,embeddings_Faiss, k, query_input, selected_sources, choice)
            end = time.time()
            retrieved_string = json.dumps(retrieved_docs)
            generated_prompt = create_prompt(query_input, retrieved_string, selected_sources)
            end = time.time()
            st.session_state.context_input = generated_prompt
            # Save the order of the retrieved docs in a list
            st.session_state.time_retrieval = f"{(end - start) * 1000} ms"
            # Update context input box
            st.rerun()
    
    # Container 2 Section: Mistral prompt results
    if st.session_state.context_input != "":  # Checks if context_input is not just whitespace
        with con2:
            st.write("Mistral latency [ms]: ", st.session_state.time_llm)
            st.subheader(f"Mistral Answer {choice}: ")
            if st.session_state.response_llm != "":
                st.markdown(st.session_state.response_llm)

    # Container 3 Section: GPT4 prompt results
        with con3:
            st.write("GPT4 latency [ms]: ", st.session_state.time_llm_gpt4)
            st.subheader(f"GPT4 Answer {choice}: ")
            if st.session_state.response_llm_gpt4 != "":
                st.markdown(st.session_state.response_llm_gpt4)
        # Handle Rerank Button Press
    if submit_button:
        with st.spinner('Processing...'):
            # Mistral
            start = time.time()
            result = call_llm(st.session_state.context_input)
            st.session_state.response_llm = result
            end = time.time()
            st.session_state.time_llm = f"{(end - start) * 1000} ms"
            # GPT4
            start = time.time()
            result = call_llm_gpt4(st.session_state.context_input)
            st.session_state.response_llm_gpt4 = result
            end = time.time()
            st.session_state.time_llm_gpt4 = f"{(end - start) * 1000} ms"
            # Update context input box
            st.rerun()

           
