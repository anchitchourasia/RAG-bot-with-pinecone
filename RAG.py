import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load Pinecone API key and environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")  # Ensure you have the correct environment

# Initialize Pinecone instance
pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)

# Define your index details
index_name = "census-data-index"
index_host = "https://census-data-index-gkaz17z.svc.aped-4627-b74a.pinecone.io"

# Access the Pinecone index using the host
index = pc.Index(host=index_host)

# Function to handle document embedding and vector creation
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs = st.session_state.loader.load()
        
        if not st.session_state.docs:
            st.error("No documents were loaded. Ensure the directory contains valid PDFs.")
            return

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        
        if not st.session_state.final_documents:
            st.error("Text splitting failed. Check your documents or text splitter settings.")
            return

        # Add chunk_number to metadata
        for i, doc in enumerate(st.session_state.final_documents):
            doc.metadata['chunk_number'] = i + 1  # Adding chunk number to each chunk's metadata

        try:
            # Use Pinecone to store the vectors
            vectors = []
            
            # Check if embed_documents is the correct method
            embeddings = st.session_state.embeddings.embed_documents([doc.page_content for doc in st.session_state.final_documents])

            for i, doc in enumerate(st.session_state.final_documents):
                vectors.append({
                    "id": f"doc_{doc.metadata['source']}#chunk_{doc.metadata['chunk_number']}",
                    "values": embeddings[i],  # Corresponding embedding for the document chunk
                    "metadata": {"source": doc.metadata['source'], "chunk_number": doc.metadata['chunk_number']}
                })
            
            index.upsert(vectors=vectors, namespace="us_census_namespace")
            st.success("Pinecone vector store created successfully!")
        except Exception as e:
            st.error(f"Failed to create Pinecone vector store: {e}")

# Streamlit UI setup
st.title("Gen AI RAG-Based Solution for Incremental Data")
llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct")  # Updated model

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings).as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write(f"Response time: {time.process_time() - start} seconds")
    st.write(response['answer'])

    # Display document similarity search results
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
