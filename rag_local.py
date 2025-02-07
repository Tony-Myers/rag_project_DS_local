import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader, UnstructuredMarkdownLoader  # Updated for Markdown files
from langchain_community.text_splitter import SemanticChunker  # Updated import path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA

st.title("📄 RAG System with DeepSeek R1 & Ollama")

# --- Password Protection ---
password = st.text_input("Enter password", type="password")
if password != st.secrets["password"]:
    st.error("Incorrect password")
    st.stop()

# --- File Uploader: Allow PDFs and Markdown files ---
uploaded_files = st.file_uploader(
    "Upload your PDF or Markdown files here",
    type=["pdf", "md"],
    accept_multiple_files=True
)

if uploaded_files:
    all_docs = []
    for uploaded_file in uploaded_files:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        temp_filename = f"temp_{uploaded_file.name}"
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getvalue())

        if file_ext == "pdf":
            loader = PDFPlumberLoader(temp_filename)
        elif file_ext == "md":
            loader = UnstructuredMarkdownLoader(temp_filename)
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
            continue

        docs = loader.load()
        all_docs.extend(docs)

    # Process the loaded documents
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_splitter.split_documents(all_docs)

    embedder = HuggingFaceEmbeddings()
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = Ollama(model="deepseek-r1:14b")

    prompt = """
    Use the following context to answer the question.
    Context: {context}
    Question: {question}
    Answer:"""
    QA_PROMPT = PromptTemplate.from_template(prompt)

    llm_chain = LLMChain(llm=llm, prompt=QA_PROMPT)
    combine_documents_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

    qa = RetrievalQA(combine_documents_chain=combine_documents_chain, retriever=retriever)

    user_input = st.text_input("Ask a question about your document:")

    if user_input:
        response = qa(user_input)["result"]
        st.write("**Response:**")
        st.write(response)

