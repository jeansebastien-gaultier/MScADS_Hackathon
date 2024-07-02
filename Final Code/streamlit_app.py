import streamlit as st
import sys
import os
import rag_model
#import mistral_rag

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
        answer = rag_model.get_answer(query)
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
