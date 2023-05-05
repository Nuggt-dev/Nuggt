import streamlit as st
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS, Pinecone
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from apikey import openai_api_key, pinecone_api_key, pinecone_api_env
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
import pandas as pd
import json

st.set_page_config(page_title="Finance Bot", page_icon=":moneybag:")
st.title("Finance Bot")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Loading PDF..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)

    data = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    texts = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    pinecone.init(
        api_key=pinecone_api_key,
        environment=pinecone_api_env,
    )
    index_name = "nuggt"

    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

    query = st.text_input("Enter your query:")

    if query:
        with st.spinner("Fetching answer..."):
            llm = ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=openai_api_key)
            docs = docsearch.similarity_search(query)
            chain = load_qa_chain(llm, chain_type="stuff")

            answer = chain.run(input_documents=docs, question=query)
        
        st.write(answer)
