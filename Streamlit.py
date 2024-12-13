import os
import streamlit as st
from streamlit_chat import message
import pinecone

from pinecone import Pinecone

# Initialize the Pinecone client
pc = Pinecone(api_key="d8f32a14-b0b1-40bf-bbc1-b93f9f8b6c8d")

# Connect to the "taxease" index
index = pc.Index("taxease")

# Ensure conversation history is persistent
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "Welcome! I am TaxEase AI, your tax assistant. How can I help you today?"}
    ]

# Function to query Pinecone and generate response
def get_response(query):
    try:
        # Retrieve relevant documents from Pinecone
        retrieved_results = query_pinecone(query)

        # Generate a response using your RAG pipeline
        response = generate_response_with_rag(query, retrieved_results)

        return response
    except Exception as e:
        return f"Sorry, an error occurred: {str(e)}"

# Title for the chatbot app
st.title("TaxEase AI Chatbot")

# Sidebar with instructions
st.sidebar.title("Instructions")
st.sidebar.info(
    """
    TaxEase AI helps you navigate the tax filing process.
    - Ask questions about filling out tax forms.
    - Get guidance on deductions and credits.
    - Enjoy conversational, AI-powered assistance.
    """
)

# Sidebar for clearing chat
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = [
        {"role": "system", "content": "Welcome! I am TaxEase AI, your tax assistant. How can I help you today?"}
    ]

# Sidebar example questions
st.sidebar.write("Example Questions:")
example_questions = [
    "How do I file Form 1040?",
    "What are the standard deductions for 2023?",
    "Can I claim tax credits for education?"
]
for question in example_questions:
    if st.sidebar.button(question):
        user_input = question

# Chat input
with st.container():
    user_input = st.text_input("Ask a tax-related question:", placeholder="Type your question here...")

    if user_input:
        # Add user query to the conversation
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate bot response
        bot_response = get_response(user_input)
        st.session_state.messages.append({"role": "bot", "content": bot_response})

# Display conversation history
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=f"user_{i}")
    elif msg["role"] == "bot":
        message(msg["content"], is_user=False, key=f"bot_{i}")
    elif msg["role"] == "system":
        st.info(msg["content"])
