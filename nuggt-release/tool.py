from gradio_tools.tools import (StableDiffusionTool, StableDiffusionPromptGeneratorTool, TextToVideoTool)
from langchain.utilities import GoogleSerperAPIWrapper, GoogleSearchAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders import YoutubeLoader
from langchain.tools import SceneXplainTool
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from dotenv import load_dotenv
import python_repl
import openai
import browse
import json
import os 

load_dotenv()
model_name = os.getenv("MODEL_NAME")
openai.api_key = os.environ.get("OPENAI_API_KEY")

def google(query):
    search = GoogleSearchAPIWrapper()
    return str(search.results(query.replace('"', ''), 3))

def browse_website(query):
    data = json.loads(query)
    text = browse.scrape_text(data["url"])
    embeddings = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_text(text)
    db = FAISS.from_texts(docs, embeddings)
    docs = db.similarity_search(data["information"])
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    return chain.run(input_documents=docs, question=data["information"])

def load_video(video_url):
    embeddings = OpenAIEmbeddings()
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db

def video_tool(query):
    data = json.loads(query)
    transcript = load_video(data["video_url"])
    docs = transcript.similarity_search(data["information"])
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    return chain.run(input_documents=docs, question=query)

def document_tool(query):
    data = json.loads(query)
    
    loader = PyMuPDFLoader(tmp_path)
    pages = loader.load_and_split()
    faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
    retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k":2})
    rqa = RetrievalQA.from_chain_type(llm=OpenAI(), 
                                chain_type="stuff", 
                                retriever=retriever, 
                                return_source_documents=True,
                                )
    return str(rqa(data["information"]))
    
def python(code):
    return python_repl.python(code)

def display(code):
    return python_repl.python(code)

def search(query):
    search_api = GoogleSerperAPIWrapper()
    return search_api.run(query)

def custom_llm(query):
    data = json.loads(query)
    messages = [
        {"role": "system", "content": data["prompt"]},
        {"role": "user", "content": data["input"]}
    ]
    response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=0, 
    )

    return response.choices[0].message["content"].strip()

def stable_diffusion(query):
    prompt = StableDiffusionPromptGeneratorTool().langchain.run(query)
    return StableDiffusionTool().run(prompt)

def image_caption(path):
    return SceneXplainTool().run(path)

def generate_video(query):
    return TextToVideoTool().langchain.run(query)

