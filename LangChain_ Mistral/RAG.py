from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.vectorstores import FAISS
#from langchain_openai import OpenAIEmbeddings
from langchain_mistralai import MistralAIEmbeddings
import os
from langchain_community.vectorstores import Chroma


DATA_PATH = "data_test"
os.environ['HF_TOKEN'] = "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"

# Load data using Markdown Document Loader
def load_markdown_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.md'):
            file_path = os.path.join(directory, filename)
            loader = UnstructuredMarkdownLoader(file_path)
            documents.append(loader.load())
    return documents

# Split them into chunks
def split_documents_by_paragraph(documents):
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    split_docs = []
    for doc in documents:
        chunks = text_splitter.split_text(doc[0].page_content)
        chunk_number = 1
        for chunk in chunks:
            chunk_dict = {
                'text': chunk,
                'metadata': {
                    'source': doc[0].metadata['source'],
                    'chunk_number': chunk_number
                }
            }
            split_docs.append(chunk_dict)
            chunk_number += 1
    #return split_docs
    return text_splitter.split_text(documents)


# Create embeddings for each document chunk
# def create_embeddings(documents, mistral_api_key):
#     embeddings = MistralAIEmbeddings(mistral_api_key=mistral_api_key)
#     texts = [doc['text'] for doc in documents]
#     metadatas = [doc['metadata'] for doc in documents]
#     embedded_documents = embeddings.embed_documents(texts)
#     print(embedded_documents)
#     # Create a VectorStore to store embeddings and metadata
#     #vector_store = FAISS()
#     #vector_store.add_vectors(embedded_documents, metadatas)
    
#     #return vector_store


def main():
    # openai_api_key = "sk-proj-1DPrTuEGiplWlDet8HF6T3BlbkFJwq96CTWeIoVTPegi2yao"
    mistral_api_key = "Uaj41DARGjquAh7l2HGuwVef2LAsVfeb"
    documents = load_markdown_documents(DATA_PATH)
    paragraph_split_docs = split_documents_by_paragraph(documents)
    # vector_store = create_embeddings(paragraph_split_docs, mistral_api_key)
    
    # Optionally save vector store to disk
    #vector_store.save("Vector Store")

    embeddings = MistralAIEmbeddings(api_key=mistral_api_key)

    # Create Chroma vector store from documents and embeddings
    vectorstore = Chroma.from_documents(documents=paragraph_split_docs, embedding=embeddings)

    # Create retriever
    retriever = vectorstore.as_retriever()

    # # Now you can use the retriever to retrieve documents based on queries
    # query = "Your query here"
    # results = retriever.retrieve(query)
    # for result in results:
    #     print(result)


if __name__ == '__main__':
    main()