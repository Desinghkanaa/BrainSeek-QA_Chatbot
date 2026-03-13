import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. NEW IMPORT: Import the Google GenAI Chat Model
from langchain_google_genai import ChatGoogleGenerativeAI 

from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# Load Environment Variables (This automatically finds GOOGLE_API_KEY in your .env)
load_dotenv()

print("Loading PDF...")
loader = PyPDFLoader("Desingh_SDE.pdf")
documents = loader.load()

print("Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

print("Creating Vector Database...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3}) 

# 2. NEW LLM SETUP: Initialize Gemini 
# We are using gemini-1.5-flash as it is fast and highly capable for RAG
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash") 
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

system_prompt = (
    "You are a helpful assistant answering questions based on the provided document. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer or if it's not in the document, just say that you don't know. "
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

print("\n--- System Ready ---")

while True:
    query = input("\nAsk a question (or type 'quit' to exit): ")
    if query.lower() == 'quit':
        break
        
    response = rag_chain.invoke({"input": query})
    
    print("\nAI Answer:\n")
    print(response["answer"])