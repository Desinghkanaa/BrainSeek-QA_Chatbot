import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# Load environment variables
load_dotenv()

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="RAG PDF Chatbot", page_icon="🤖")
st.title("Chat with your PDF 📄")

# Initialize Chat History and Vector Store in Streamlit's Session State
# This ensures the app remembers the data between user clicks
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- SIDEBAR: FILE UPLOAD ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if st.button("Process PDF"):
        if uploaded_file is not None:
            with st.spinner("Processing PDF... Please wait."):
                # 1. Save uploaded file to a temporary file (PyPDFLoader needs a file path)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_filepath = temp_file.name

                # 2. Load and Split (Your exact code!)
                loader = PyPDFLoader(temp_filepath)
                documents = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                texts = text_splitter.split_documents(documents)
                
                # 3. Create Embeddings & Vector DB
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                db = FAISS.from_documents(texts, embeddings)
                
                # Save the db to session state so the chat can use it
                st.session_state.vector_store = db
                
                # Clean up the temp file
                os.remove(temp_filepath)
                
                st.success("PDF processed successfully! You can now ask questions.")
        else:
            st.error("Please upload a PDF first.")

# --- MAIN CHAT INTERFACE ---

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_query := st.chat_input("Ask a question about your document..."):
    
    # 1. Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Generate AI Response
    with st.chat_message("assistant"):
        if st.session_state.vector_store is None:
            st.warning("Please upload and process a PDF in the sidebar first!")
        else:
            with st.spinner("Thinking..."):
                # Setup Retriever, LLM, and Chain (Your exact code!)
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash") 
                
                system_prompt = (
                    "You are a helpful assistant answering questions based on the provided document. "
                    "Use the following pieces of retrieved context to answer the question. "
                    "If you don't know the answer, just say that you don't know. "
                    "Keep the answer concise and professional."
                    "\n\n"
                    "Context: {context}"
                )
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])
                
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                # Get the answer
                response = rag_chain.invoke({"input": user_query})
                ai_answer = response["answer"]
                
                # Display and save the answer
                st.markdown(ai_answer)
                st.session_state.messages.append({"role": "assistant", "content": ai_answer})