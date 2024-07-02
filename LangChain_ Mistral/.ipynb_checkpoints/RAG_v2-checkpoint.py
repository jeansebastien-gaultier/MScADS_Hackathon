from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
import os

DATA_PATH = "data_test"

# Load Documents
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )
# docs = loader.load()

# documents = []
# for filename in os.listdir(DATA_PATH):
#     if filename.endswith('.md'):
#         file_path = os.path.join(DATA_PATH, filename)
#         loader = UnstructuredMarkdownLoader(file_path)
#         documents.append(loader.load())


loader = UnstructuredMarkdownLoader("data_test/0.md")
docs = loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
splits = text_splitter.split_documents(docs)


vectorstore = FAISS.from_documents(documents=splits, 
                                    embedding=MistralAIEmbeddings(api_key="Uaj41DARGjquAh7l2HGuwVef2LAsVfeb"))

retriever = vectorstore.as_retriever()

def main():
    print("Success")

if __name__ == '__main__':
    main()
