from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain 
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings 
from apikey import apikey 

loader = PyPDFLoader("../brk_annual_report.pdf")

data = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
texts = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings(openai_api_key=apikey)

db = FAISS.from_documents(texts, embeddings)

llm = ChatOpenAI(temperature=0, model_name="gpt-4",openai_api_key=apikey)

chain = load_qa_chain(llm, chain_type="stuff")
query = "Give me a summary of Berkshire Hathaway's financial statement in 2022"
docs = db.similarity_search(query)

answer = chain.run(input_documents=docs, question=query)

print(answer)