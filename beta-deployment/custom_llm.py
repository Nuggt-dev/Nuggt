from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
import openai
import json
import os

os.environ["OPENAI_API_KEY"] = "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"
embeddings = OpenAIEmbeddings()

def user_input():
    query = input("Enter your question about the video: ")
    return {"query": query}

def get_response_from_query(db, nuggt, k=4):
    docs = db.similarity_search(nuggt["query"], k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    return {"query": nuggt["query"], "information": docs_page_content}

def get_answer(nuggt):

    prompt = f"""You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {nuggt["query"]}
        By searching the following video transcript: {nuggt["information"]}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed."""
   
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0, 
    )

    return {"query": nuggt["query"], "response": response.choices[0].message["content"], "context": nuggt["information"]}

def output(nuggt):
    print(nuggt["response"])

video_url = "https://www.youtube.com/watch?v=L_Guz73e6fw"
loader = YoutubeLoader.from_youtube_url(video_url)
transcript = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(transcript)

db = FAISS.from_documents(docs, embeddings)

output(get_answer(get_response_from_query(db, user_input())))
    



