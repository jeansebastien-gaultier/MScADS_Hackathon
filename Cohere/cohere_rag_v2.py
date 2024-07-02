import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_cohere import ChatCohere
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer


# Set the Cohere API key
cohere_key = "8acOGXby2VEQ70UErIkns9I0qGLUu9QT6KIYfzpA"

# Initialize the embeddings model
embeddings_model = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en", model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True}
)

# Custom text splitter
def custom_medical_text_splitter(documents, chunk_size=5000, chunk_overlap=500):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs

# Load data and prepare embeddings
def load_data_and_prepare_embeddings(file_path):
    loader = CSVLoader(file_path=file_path)
    data = loader.load()
    split_docs = custom_medical_text_splitter(data)
    vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings_model)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': len(data)})
    return retriever

retriever = None
prompt = hub.pull("rlm/rag-prompt")
llm = ChatCohere(model="command-r", cohere_api_key=cohere_key)

def update_data(file_path):
    global retriever
    retriever = load_data_and_prepare_embeddings(file_path)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_answer(query):
    if not retriever:
        return "No data available. Please upload a CSV file."
    
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

def fine_tune_model(dataset, output_dir="fine_tuned_model"):
    tokenizer = AutoTokenizer.from_pretrained("gpt-2")
    model = AutoModelForCausalLM.from_pretrained("gpt-2")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )

    trainer.train()
    trainer.save_model(output_dir)