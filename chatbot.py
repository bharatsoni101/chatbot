import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from langchain.vectorstores import FAISS

GROQ_API_KEY = "paste your API key here"
GROQ_API_BASE = "https://api.groq.com/openai/v1"

# Step#1 upload PDF files
st.header("Upload your PDF files")
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload PDF files and start asking questions", type="pdf")

# Step#2 Extract text from PDF files
if file is not None:
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
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
    else:
        flat_chunks = []
        for chunk in chunks:
            if isinstance(chunk, str):
                flat_chunks.append(chunk)
            elif isinstance(chunk, list):
                flat_chunks.extend(chunk)
        if not all(isinstance(chunk, str) for chunk in flat_chunks):
            st.error("Text chunks are not all strings.")
        else:
            # Step#4 Create embeddings and store in vector database (using HuggingFaceEmbeddings)
            with st.spinner("Creating embeddings and building vector store. This may take a while..."):
                embeddings = HuggingFaceEmbeddings()
                vector_store = FAISS.from_texts(flat_chunks, embedding=embeddings)

            # Step#5 Get user query and perform similarity search
            user_query = st.text_input("Ask questions about your documents")
            if user_query:
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
                                model="meta-llama/llama-4-scout-17b-16e-instruct",
                                messages=[{"role": "user", "content": prompt}],
                                timeout=60  # 60 seconds timeout
                            )
                            st.write(response.choices[0].message.content)
                        except Exception as e:
                            st.error(f"Error generating response: {e}")
                else:
                    st.info("No relevant context found in the document.")
