import streamlit as st
import sys
import os

# Add paths to sys.path to import cohere_rag and mistral_rag modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Cohere')))
import cohere_rag_v2

# Streamlit app interface
st.set_page_config(
    page_title="RAG Model Chatbot",
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

st.title("RAG Model Chatbot")

# Model selection
#model_option = st.selectbox("Select a model", ("Cohere", "Mistral"))

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    file_path = f"/tmp/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File {uploaded_file.name} uploaded successfully.")
    cohere_rag_v2.update_data(file_path)

# Chat history
if 'history' not in st.session_state:
    st.session_state.history = []

# Query input
query = st.text_input("Enter your query:", key="query_input")

if st.button("Get Answer"):
    with st.spinner("Retrieving documents and generating answer..."):
        answer = cohere_rag_v2.get_answer(query)
        st.session_state.history.append((query, answer))
        #st.experimental_rerun()  # Rerun the script to update the history

# Display chat history
st.markdown('<div class="chat-history">', unsafe_allow_html=True)
st.subheader("Chat History")
for q, a in st.session_state.history:
    st.write(f"**Question:** {q}")
    st.write(f"**Answer:** {a}")
    st.write("---")
st.markdown('</div>', unsafe_allow_html=True)
