import os
import streamlit as st
from streamlit_chat import message
import pinecone

# Initialize Pinecone
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


import openai

# Set your OpenAI API key
openai.api_key = "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"

def generate_embeddings(texts):
    """Generate embeddings using OpenAI's `text-embedding-ada-002`."""
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return [data["embedding"] for data in response["data"]]

def query_pinecone(query):
    """Query Pinecone index with an embedding of the query."""
    # Step 1: Generate embedding for the query
    query_embedding = generate_embeddings([query])[0]  # Use the same embedding model as used for indexing

    # Step 2: Query Pinecone
    results = index.query(
        namespace="ns1",  # Use the namespace if applicable, otherwise omit
        vector=query_embedding,  # Provide the query embedding
        top_k=5,  # Number of top results to retrieve
        include_metadata=True  # Include metadata in the results
    )

    # Return the retrieved results
    return results["matches"]



def generate_response_with_rag(query, retrieved_results):

    # Introductory context

    introductory_prompt = (
    "   Hi Taxease, you are an assistant who helps users navigate the tax filing process. "
        "The chatbot should be able to answer questions, provide guidance on filling out the tax form, "
        "and offer suggestions for deductions or credits the user may be eligible for. "
        "The chatbot should use natural language processing to understand user queries and respond in a conversational way. "

    )

    # Retrieved context
    context = "\n".join([match["metadata"]["completion"] for match in retrieved_results])

    # Combine the introductory prompt, retrieved context, and user query
    messages = [
        {"role": "system", "content": introductory_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}\n\nAnswer:"}
    ]


    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Or "gpt-4" for higher quality
        messages=messages,
        max_tokens=300,
        temperature=0.7
    )

    return response["choices"][0]["message"]["content"].strip()



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
