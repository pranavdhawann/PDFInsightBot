import streamlit as st
import os
import time

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Create necessary directories
if not os.path.exists('pdfFiles'):
    os.makedirs('pdfFiles')
if not os.path.exists('vectorDB'):
    os.makedirs('vectorDB')

# Initialize session state variables
if 'template' not in st.session_state:
    st.session_state.template = """You are a teaching assistant. Your task is to help with questions of the user in regards to their documents. Your tone should be professional and informative.

Context: {context}
History: {history}

User: {question}
Chatbot:"""

if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question",
    )

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(
        base_url="http://localhost:11434",
        model="mistral",
        verbose=True,
    )

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# Streamlit UI
st.title("Talk to your PDF")

uploaded_file = st.file_uploader("Drop Your PDF", type="pdf")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

if uploaded_file is not None:
    st.text("File uploaded successfully")
    if not os.path.exists(f'pdfFiles/{uploaded_file.name}'):
        with st.spinner("Uploading file..."):
            bytes_data = uploaded_file.read()
            with open(f'pdfFiles/{uploaded_file.name}', 'wb') as f:
                f.write(bytes_data)
            st.text("File saved to pdfFiles directory")

    if st.session_state.vectorstore is None:
        # Load PDF data
        loader = PyPDFLoader(f'pdfFiles/{uploaded_file.name}')
        data = loader.load()
        if data:
            st.text("PDF loaded successfully")
        else:
            st.text("Failed to load PDF")

        # Split PDF data
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len
        )
        all_splits = text_splitter.split_documents(data)
        st.text(f"Number of chunks created: {len(all_splits)}")

        # Measure time to create vector store and add progress bar
        progress_bar = st.progress(0)
        total_steps = len(all_splits)  # Total number of steps for progress bar
        start_time = time.time()
        try:
            st.text("Creating vector store...")
            embedding_function = OllamaEmbeddings(model="mistral")

            # Initialize vector store
            vectorstore = Chroma(persist_directory='vectorDB', embedding_function=embedding_function)
            
            # Add chunks to vector store and update progress bar
            for i, doc in enumerate(all_splits):
                vectorstore.add_documents([doc])
                progress_bar.progress((i + 1) / total_steps)

            vectorstore.persist()
            st.session_state.vectorstore = vectorstore
            st.text("Vector store created and persisted")
        except Exception as e:
            st.text(f"Error creating vector store: {e}")
        end_time = time.time()
        st.text(f"Time taken to create vector store: {end_time - start_time:.2f} seconds")

    if st.session_state.vectorstore is not None and st.session_state.qa_chain is None:
        try:
            st.session_state.retriever = st.session_state.vectorstore.as_retriever()
            st.text("Retriever created")
            st.text("Creating QA chain...")
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=st.session_state.llm,
                chain_type='stuff',
                retriever=st.session_state.retriever,
                verbose=True,
                chain_type_kwargs={
                    "verbose": True,
                    "prompt": st.session_state.prompt,
                    "memory": st.session_state.memory,
                }
            )
            st.text("QA chain created")
        except Exception as e:
            st.text(f"Error creating QA chain: {e}")

    if user_input := st.chat_input("You:", key="user_input"):
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)
        with st.chat_message("user"):
            st.markdown(user_input)

        if st.session_state.qa_chain is not None:
            with st.chat_message("assistant"):
                with st.spinner("Typing..."):
                    try:
                        response = st.session_state.qa_chain({"query": user_input, "context": "", "history": []})
                        full_response = response['result']
                        message_placeholder = st.empty()
                        message_placeholder.markdown(full_response)
                    except Exception as e:
                        full_response = f"Error during QA chain execution: {e}"
                        st.text(full_response)
                    
                chatbot_message = {"role": "assistant", "message": full_response}
                st.session_state.chat_history.append(chatbot_message)

else:
    st.write("Please upload a PDF file to start the chatbot")
