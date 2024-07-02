import os
from langchain import hub
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI

# Assuming you have an API key for MistralAI
mistral_api_key = "Uaj41DARGjquAh7l2HGuwVef2LAsVfeb"

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
    embeddings_model = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en", model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True}
    )
    loader = CSVLoader(file_path="/Users/jean-sebastiengaultier/Desktop/UChicago/Academic/Hackathon/test_data.csv")
    #loader = CSVLoader(file_path="/Users/jean-sebastiengaultier/Desktop/UChicago/Academic/Hackathon/mtsamples_with_rand_names.csv")
    data = loader.load()
    print("Done Loading!")
    split_csv = custom_medical_text_splitter(data)
    print("Finished Splitting!")
    vectorstore = Chroma.from_documents(documents=split_csv, embedding=embeddings_model)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 100})
    print("Ready to ask questions!")
    return retriever

retriever = load_data_and_prepare_embeddings()
prompt = hub.pull("rlm/rag-prompt")
llm = ChatMistralAI(api_key=mistral_api_key)

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
