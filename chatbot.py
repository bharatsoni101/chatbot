import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from langchain.vectorstores import FAISS
import logging
import configparser

# Read configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')
MODEL = config['groq']['MODEL']
GROQ_API_KEY = config['groq']['GROQ_API_KEY']
GROQ_API_BASE = config['groq']['GROQ_API_BASE']

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler("chatbot.log"), logging.StreamHandler()]
)

# Step#1 upload PDF files
st.header("Upload your PDF files")
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload PDF files and start asking questions", type="pdf")
    if file:
        logging.info(f"File uploaded: {file.name}")

# Step#2 Extract text from PDF files
if file is not None:
    # Read PDF and extract text
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    logging.info(f"Extracted text from {len(pdf.pages)} pages.")
    # Step#3 Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    #st.write(chunks)

    # Ensure chunks is a flat list of strings
    if not isinstance(chunks, list):
        st.error("Text chunks are not in the correct format.")
        logging.error("Text chunks are not in the correct format.")
    else:
        flat_chunks = []
        for chunk in chunks:
            if isinstance(chunk, str):
                flat_chunks.append(chunk)
            elif isinstance(chunk, list):
                flat_chunks.extend(chunk)
        if not all(isinstance(chunk, str) for chunk in flat_chunks):
            st.error("Text chunks are not all strings.")
            logging.error("Text chunks are not all strings.")
        else:
            # Step#4 Create embeddings and store in vector database (using HuggingFaceEmbeddings)
            # Only create vector store if not already present or file changed
            if 'vector_store' not in st.session_state or st.session_state.get('last_file_name') != file.name:
                with st.spinner("Creating embeddings and building vector store. This may take a while..."):
                    logging.info(f"Building vector store for file: {file.name}")
                    embeddings = HuggingFaceEmbeddings()
                    vector_store = FAISS.from_texts(flat_chunks, embedding=embeddings)
                    st.session_state['vector_store'] = vector_store
                    st.session_state['last_file_name'] = file.name
                    logging.info("Vector store created and stored in session state.")
            else:
                vector_store = st.session_state['vector_store']
                logging.info(f"Reusing existing vector store for file: {file.name}")
            # Step#5 Get user query and perform similarity search
            # Initialize response cache if not present
            if 'response_cache' not in st.session_state:
                st.session_state['response_cache'] = {}
            user_query = st.text_input("Ask questions about your documents")
            if user_query:
                logging.info(f"User query: {user_query}")
                cache = st.session_state['response_cache']
                if user_query in cache:
                    # Return cached response
                    st.write(cache[user_query])
                    logging.info("Returned cached response.")
                else:
                    match = vector_store.similarity_search(user_query, k=3)  # Limit to top 3 chunks
                    # Step#6 Generate response using Groq LLM and show output on screen
                    if match:
                        context = "\n".join([doc.page_content for doc in match])
                        client = OpenAI(
                            base_url=GROQ_API_BASE,
                            api_key=GROQ_API_KEY
                        )
                        prompt = f"Answer the following question based on the context below.\nContext: {context}\nQuestion: {user_query}\nAnswer:"
                        with st.spinner("Generating response..."):
                            try:
                                response = client.chat.completions.create(
                                    model=MODEL,
                                    messages=[{"role": "user", "content": prompt}],
                                    timeout=60  # 60 seconds timeout
                                )
                                answer = response.choices[0].message.content
                                st.write(answer)
                                cache[user_query] = answer  # Cache the response
                                logging.info("AI response cached and displayed.")
                            except Exception as e:
                                st.error(f"Error generating response: {e}")
                                logging.error(f"Error generating response: {e}")
                    else:
                        st.info("No relevant context found in the document.")
                        logging.info("No relevant context found for the query.")
