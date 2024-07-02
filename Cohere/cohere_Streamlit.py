"""
cohere_Streamlit.py is very similar to the cohere_rag.py except that this one was created to have a single LLM model answer the questions, whereas 
the bigger streamlit app has a toggle button that switches the LLM that is being used.
"""

import streamlit as st
from langchain.embeddings import CohereEmbeddings
from langchain_cohere import ChatCohere
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Set the Cohere API key
cohere_key = "8acOGXby2VEQ70UErIkns9I0qGLUu9QT6KIYfzpA"

# Load data and prepare embeddings
def load_data_and_prepare_embeddings():
    embeddings_model = CohereEmbeddings(cohere_api_key=cohere_key)
    #loader = CSVLoader(file_path="mtsamples_with_rand_names.csv")
    loader = CSVLoader(file_path="/Users/jean-sebastiengaultier/Desktop/UChicago/Academic/Hackathon/test_data.csv")
    #loader = CSVLoader(file_path="/Users/jean-sebastiengaultier/Desktop/UChicago/Academic/Hackathon/test_data_tran.csv")
    data = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    split_csv = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(documents=split_csv, embedding=embeddings_model)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': len(data)})
    return retriever

retriever = load_data_and_prepare_embeddings()
print("Done retrieving")
prompt = hub.pull("rlm/rag-prompt")
llm = ChatCohere(model="command-r", cohere_api_key=cohere_key)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_answer(query):
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = ""
    for chunk in rag_chain.stream(query):
        response += chunk
    return response

# Streamlit app interface
st.title("RAG Model with Cohere Embeddings")
query = st.text_input("Enter your query:")
if st.button("Get Answer"):
    with st.spinner("Retrieving documents and generating answer..."):
        answer = get_answer(query)
        st.write(answer)

#%%
