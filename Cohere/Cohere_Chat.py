"""
Cohere_Chat.py was used as a placeholder to test various functions, it is not used in production.
"""


from langchain.embeddings import CohereEmbeddings
from langchain_cohere import ChatCohere
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

cohere_key = "8acOGXby2VEQ70UErIkns9I0qGLUu9QT6KIYfzpA"

embeddings_model = CohereEmbeddings(cohere_api_key = cohere_key)
loader = CSVLoader(file_path="/Users/jean-sebastiengaultier/Desktop/UChicago/Academic/Hackathon/mtsamples_with_rand_names.csv")
data = loader.load()
data_test = data[:10]
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
split_csv = text_splitter.split_documents(data_test)
vectorstore = Chroma.from_documents(documents=split_csv, 
                                    embedding=embeddings_model)

#retriever = vectorstore.as_retriever()

retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.1})

retrieved_docs = retriever.invoke("How many people have allergies?")

prompt = hub.pull("rlm/rag-prompt")

llm = ChatCohere(model="command-r", cohere_api_key = cohere_key)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

for chunk in rag_chain.stream("How many people have allergies?"):
    print(chunk, end="", flush=True)