import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class LangchainEmbeddingWrapper:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True)

    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=True)[0]

    def __call__(self, texts):
        return self.embed_documents(texts)

if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "filenames" not in st.session_state:
    st.session_state.filenames = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query" not in st.session_state:
    st.session_state.query = ""

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192", temperature=0.3)

with st.sidebar:
    st.title("File Upload")
    uploaded_files = st.file_uploader("Upload PDF/Text files", type=["pdf", "txt"], accept_multiple_files=True)

def load_documents(files):
    documents = []
    filenames = []
    for uploaded_file in files:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)
            file_text = ""
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    file_text += text
            documents.append(file_text)
        elif uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
            documents.append(text)
        filenames.append(uploaded_file.name)
    return documents, filenames

def prepare_vectors(documents, filenames):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    all_chunks = []
    
    for doc, filename in zip(documents, filenames):
        chunks = splitter.create_documents([doc])
        for chunk in chunks:
            if 'metadata' not in chunk:
                chunk.metadata = {}
            chunk.metadata['source'] = filename
        all_chunks.extend(chunks) 
    
    model = LangchainEmbeddingWrapper(SentenceTransformer('all-MiniLM-L6-v2'))
    vector_store = FAISS.from_texts(
        texts=[chunk.page_content for chunk in all_chunks], 
        embedding=model, 
        metadatas=[chunk.metadata for chunk in all_chunks]
    )
    return vector_store, all_chunks

def is_relevant(query_embedding, doc_embedding, threshold=0.5):
    similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
    return similarity >= threshold

if st.sidebar.button("Process Files"):
    if not uploaded_files:
        st.warning("Please upload at least one file.")
    else:
        docs, filenames = load_documents(uploaded_files)
        st.session_state.vectors, st.session_state.chunks = prepare_vectors(docs, filenames)
        st.session_state.filenames = filenames 
        st.success("Files processed and vector store created!")

def process_query():
    query = st.session_state.query.strip()
    if not query:
        st.warning("Please enter a query.")
        return

    bot_response = None
    source_documents = []

    if st.session_state.vectors:
        retriever = st.session_state.vectors.as_retriever(search_type="similarity", search_kwargs={"k": 1})
        query_embedding = LangchainEmbeddingWrapper(SentenceTransformer('all-MiniLM-L6-v2')).embed_query(query)
        
        retrieved_docs = retriever.get_relevant_documents(query)
        relevant_docs = [
            doc for doc in retrieved_docs 
            if is_relevant(query_embedding, LangchainEmbeddingWrapper(SentenceTransformer('all-MiniLM-L6-v2')).embed_query(doc.page_content))
        ]
        
        if relevant_docs:
            source_documents = [
                doc.metadata['source'] for doc in relevant_docs if 'source' in doc.metadata
            ]
            context = "\n".join([doc.page_content for doc in relevant_docs])
            prompt_template = """
            Provide a detailed response based on the context. If the context suggests multiple aspects, include all of them.
            If you don't find the answer, just say "I don't know."
            <context>
            {context}
            </context>
            Question: {question}
            """
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": PromptTemplate(
                        input_variables=["context", "question"],
                        template=prompt_template,
                    )
                },
            )
            response = qa_chain.invoke({"query": query})
            bot_response = response["result"]
        else:
            bot_response = "I don't know."
    else:
        bot_response = "I don't know."

    st.session_state.chat_history.append({
        "query": query,
        "response": bot_response,
        "sources": source_documents
    })

    st.session_state.query = ""

st.title("RAG Chatbot")

for chat in st.session_state.chat_history:
    sources_display = ""
    if chat.get("sources"):
        sources_display = "<br><b>Source(s):</b> " + ", ".join(chat["sources"])

    st.markdown(f"""
    <div class="chat-message">
        <b>You:</b> {chat['query']}<br>
        <b>Bot:</b> {chat['response']}{sources_display}
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.text_input(
    "Type your query and press Enter:",
    value=st.session_state.query,
    key="query",
    on_change=process_query,
    placeholder="Type your message here...",
    label_visibility="collapsed",
)
