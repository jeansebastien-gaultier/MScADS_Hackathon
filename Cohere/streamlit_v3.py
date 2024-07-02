# app.py
import streamlit as st
from cohere_rag_v3 import update_data, get_answer
from tempfile import NamedTemporaryFile

st.title("Medical Data RAG Model")

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Save uploaded file to a temporary file
    with NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    
    # Update data with the uploaded CSV file
    update_data(tmp_file_path)
    st.success("CSV file has been uploaded and processed.")

# Input query
query = st.text_input("Enter your query:")

if st.button("Get Answer") and query:
    answer = get_answer(query)
    st.write("Answer:", answer)
