import json
import os
import openai
from gradio_tools.tools import TextToVideoTool, StableDiffusionTool, StableDiffusionPromptGeneratorTool
from langchain.utilities import GoogleSerperAPIWrapper, GoogleSearchAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader, YoutubeLoader
from langchain.vectorstores import FAISS
from langchain.tools import SceneXplainTool
import python_repl
import browse


class InformationProcessingTool:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    def _get_faiss_from_text(self, text):
        docs = self.text_splitter.split_text(text)
        db = FAISS.from_texts(docs, self.embeddings)
        return db

    def _get_faiss_from_documents(self, documents):
        db = FAISS.from_documents(documents, self.embeddings)
        return db

    def google(self, query):
        search = GoogleSearchAPIWrapper()
        return str(search.results(query.replace('"', ''), 3))

    def browse_website(self, query):
        data = json.loads(query)
        text = browse.scrape_text(data["url"])
        db = self._get_faiss_from_text(text)
        docs = db.similarity_search(data["information"])
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
        return chain.run(input_documents=docs, question=data["information"])

    def load_video(self, video_url):
        loader = YoutubeLoader.from_youtube_url(video_url)
        transcript = loader.load()
        docs = self.text_splitter.split_documents(transcript)
        db = self._get_faiss_from_documents(docs)
        return db

    def video_tool(self, query):
        data = json.loads(query)
        transcript = self.load_video(data["video_url"])
        docs = transcript.similarity_search(data["information"])
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
        return chain.run(input_documents=docs, question=query)

    def document_tool(self, query):
        data = json.loads(query)
        loader = UnstructuredPDFLoader(data["document_name"])
        pages = loader.load_and_split()
        faiss_index = self._get_faiss_from_documents(pages)
        retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        docs = retriever.get_relevant_documents(data["information"])
        return docs

    def python(self, code):
        return python_repl.python(code)

    def search(self, query):
        search_api = GoogleSerperAPIWrapper()
        return search_api.run(query)

    def custom_llm(self, query):
        data = json.loads(query)
        messages = [
            {"role": "system", "content": data["prompt"]},
            {"role": "user", "content": data["input"]}
        ]
        response = openai.ChatCompletion.create(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=os.getenv("MODEL_NAME"),
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message["content"].strip()

    def stable_diffusion(self, query):
        prompt = StableDiffusionPromptGeneratorTool().langchain.run(query)
        return StableDiffusionTool().run(prompt)

    def image_caption(self, path):
        return SceneXplainTool().run(path)

    def generate_video(self, query):
        return TextToVideoTool().langchain.run(query)

