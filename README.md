# VOX TASK 1 Project Structure and content:

## _RAG Retrievers UI APP
The main objective of this app is to show case the operation of different RAG retrievers used to build a prompt to an LLM for taking orders, with the one with the best tradeoff between speed, performance and scalability.

This project consists of a main application run by Streamlit, serving as a User Interface (UI). It enables visual inspection of the built RAG retrievers. The RAG retrieval options include Cloud, Local and Self-built solutions for this RAG task

## _DBs and files for retrieval
The functions for creating the different vector stores and files used for local retrieval are commented in the start of the app

## _Dataset enrichment
An important feature for this approach is the dataset enrichment, it was done by running menu_enriching_code.py in the data folder, it generates a list of keywords for improving keyword search. It also replaces the product ids with the names of the products inside the menus to have a better response from the llm responses.

## _LLM responses
An implementation of both mistral and gpt4 was done to get a sample of a response using the built prompt with the retriever

## _Evaluation
The notebook called evaluation displays the results of the evaluation for the different built retrievers for Task1

Metrics to evaluate:
- Performance: (Score with GPT4)
- Time of computation : (Miliseconds [ms])


# Operation

## _Instructions to run APP
1.- Pull the repository from Github

2.- create python enviroment in the main folder of the project
$command: python -m venv venv    

3.- activate python env
$command: source venv/bin/activate

4.- install requirements
$command: pip install -r requirements.txt

5.- Rename .env_sample to .env and fill the Keys of the retrievers you want to use.

6.- for running the app go to the main source folder and do in terminal:
$command: streamlit run app.py


## _Source of knowledge
The documents used to build the knowledge of the explored retrievers are stored in data/

## Adding or modifying sources:

1.- Add or remove files from data/

2.- Uncomment the lines of code for building the different DBs in App.py

3.- run the app

command: streamlit run app.py

Tip: After running the app for first time and successfully creating the DBs, I suggest to comment again the lines of code, otherwise streamlit will take to much time to load in every interaction within the app (Dbs, will be creating every time you run it)


## FAQ

Q: Why I used sentence transformers emebeddings rather then OpenAI embeddings?

A: As you can see in embeddings_speed_test.ipynb, Although the performance of OpenAI embeddings is well known to be high with a dimensionality of 1536, the latency time for generating them is a concern, as you can see in embeddings_speed_test.ipynb notebook, sentence transformers embeddings are computed faster. The performance of sentence transformers is lower (dimensionality of 384), but for this food menu simple task, is enough. 

# VOX_task1 conclusions:

# Conclusions


- Local implementation of BM25 (Keyword search) is the fastest retrieval method with an average proccessing time of 0.79 ms, while performance is acceptable compared to other retrieval methods. There are approaches to host .pkl files to scale this approach if required.

- Weaviate Keyword search is the cloud based fastest retriever with an average of 27.14 ms for retrieval, this approach can be scaled up easier and have an acceptable performance

- Faiss and my native implementation of vector search using Sentence Transformers are the fastest vector based retrievals, however performance for this use case is the same for keyword and vector based approach

- The preliminary dataset enrichment with keywords and product dictionaries was a key factor to equal the performance between keyword search and vector search

# Next steps:

- Validate the performance of the selected tool with real human labeling and stake-holders




