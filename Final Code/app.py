import os
#from langchain_cohere import CohereEmbeddings
from langchain_cohere import ChatCohere
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings



# Set the Cohere API key
cohere_key = "8acOGXby2VEQ70UErIkns9I0qGLUu9QT6KIYfzpA"

def custom_medical_text_splitter(documents, chunk_size=3000, chunk_overlap=500):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],  # Split by paragraphs, then lines, then sentences, then words
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs

# Load data and prepare embeddings
def load_data_and_prepare_embeddings():
    print("Starting to load")
    #embeddings_model = CohereEmbeddings(cohere_api_key=cohere_key)
    embeddings_model = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en", model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True}
    )
    loader = CSVLoader(file_path="/Users/jean-sebastiengaultier/Desktop/UChicago/Academic/Hackathon/test_data.csv")
    #loader = CSVLoader(file_path="/Users/jean-sebastiengaultier/Desktop/UChicago/Academic/Hackathon/mtsamples_with_rand_names.csv")
    data = loader.load()
    print("Done Loading!")
    # text_splitter = CharacterTextSplitter(
    #     separator="\n",
    #     chunk_size=2000,
    #     chunk_overlap=200,
    #     length_function=len,
    #     is_separator_regex=False,
    # )
    #split_csv = text_splitter.split_documents(data)
    #split_csv = custom_medical_text_splitter(data)
    print("Finished Splitting!")
    vectorstore = Chroma.from_documents(documents=data, embedding=embeddings_model)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': len(data)})
    print("Ready to ask questions!")
    return retriever

retriever = load_data_and_prepare_embeddings()
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

import streamlit as st
import sys
import os

# Streamlit app interface
st.set_page_config(
    page_title="RAG Model Chatbot for Medical Field",
    layout="wide",
)

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #000000;
        color: #ffffff;
    }
    .main {
        background-color: #000000;
        padding: 2rem;
        color: #ffffff;
    }
    .stTextInput label {
        font-size: 45px;
        display: block;
        text-align: center;
    }
    .stTextInput input {
        font-size: 24px;
        padding: 15px;
        width: 80%;
        margin: 0 auto;
        display: block;
        background-color: #333333;
        color: #ffffff;
    }
    .stButton button {
        font-size: 24px;
        padding: 15px;
        width: 40%;
        margin: 20px auto;
        display: block;
        background-color: #333333;
        color: #ffffff;
    }
    .chat-history {
        font-size: 26px;
        margin-top: 2rem;
        padding: 15px;
        background-color: #1e1e1e;
        border-radius: 5px;
        max-width: 80%;
        margin-left: auto;
        margin-right: auto;
        color: #ffffff;
    }
    .chat-history div {
        padding: 10px;
        margin: 5px 0;
        border: 1px solid #333333;
        border-radius: 5px;
        background-color: #333333;
        color: #ffffff;
    }
    .centered {
        display: flex;
        justify-content: center;
        align-items: flex-start;
        flex-direction: column;
        min-height: 100vh;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("RAG Model Chatbot for Medical Field")

# Model selection
#model_option = st.selectbox("Select a model", ("Cohere", "Mistral"))

# Chat history
if 'history' not in st.session_state:
    st.session_state.history = []

# Query input
query = st.text_input("Enter your query:", key="query_input")

if st.button("Get Answer"):
    with st.spinner("Retrieving documents and generating answer..."):
        answer = get_answer(query)
        st.session_state.history.append((query, answer))
        #st.session_state.query_input = ''  # Set the query input state
        #st.experimental_rerun()  # Rerun the script to clear the input box

# Display chat history
st.markdown('<div class="chat-history">', unsafe_allow_html=True)
st.subheader("Chat History")
for q, a in st.session_state.history:
    st.write(f"**Question:** {q}")
    st.write(f"**Answer:** {a}")
    st.write("---")
st.markdown('</div>', unsafe_allow_html=True)
