from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.utilities import GoogleSerperAPIWrapper, GoogleSearchAPIWrapper
from gradio_tools.tools import (StableDiffusionTool, ImageCaptioningTool, StableDiffusionPromptGeneratorTool, TextToVideoTool)
from langchain.tools import SceneXplainTool
from langchain.chains import RetrievalQA
from io import StringIO
import traceback
import openai
import sys
import json
import re
import os
import glob
import streamlit as st
import browse
import requests
from colorama import Fore
from langchain.document_loaders import PyPDFLoader, PyMuPDFLoader, SeleniumURLLoader
model_name = "gpt-3.5-turbo"
st.set_page_config(page_title="Nuggt", layout="wide")
count = 0
openai.api_key = "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"
os.environ["OPENAI_API_KEY"] = "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"

class PythonREPLa:
    def __init__(self):
        self.local_vars = {}

    def run(self, code: str) -> str:
        # Redirect stdout and stderr to StringIO
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_output = sys.stdout = StringIO()
        redirected_error = sys.stderr = StringIO()

        try:
            exec(code, self.local_vars)
        except Exception:
            traceback.print_exc()

        # Restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        # Get the output or error message
        output = redirected_output.getvalue()
        error = redirected_error.getvalue()
        if "Traceback" not in error:
            error = False

        return output if not error else "Your code has the following error. Please provide the corrected code.\n" + error

python_repl = PythonREPLa()

def fix_error(code, result):
    while "Your code has the following error." in result:
        error = result.replace("Your code has the following error. Please provide the corrected code.", "")
        user_input = f"Input:\n{code}\nError:\n{error}\nOutput:"

        print(f"I am going to correct: {user_input}")
        print(Fore.RED + "Code needs some correction.")
        messages = [
        {"role": "user", "content": f"""Output the corrected code in the following format:\n```Your code here```\n{user_input}"""},
        #{"role": "user", "content": user_input}
        ]

        response = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                temperature=0, 
        )

        code = response.choices[0].message["content"].strip()
        code = code.replace("`", "")
        print(f"\nOutput of the model: {code}")

        result = python_repl.run(code)
        

    print(Fore.GREEN + "Code has been corrected.")
    return result

code = """
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)

# Countplot of Sex and Survived
st.pyplot(sns.countplot(x='Sex', hue='Survived', data=train_df))
st.write('Figure 1: Survival Count by Sex')

# Barplot of Sex and Survived
st.pyplot(sns.barplot(x='Sex', y='Survived', data=train_df))
st.write('Figure 2: Survival Rate by Sex')
"""

error = """
Your code has the following error.

Traceback (most recent call last):
  File "/Users/shoibloya/Desktop/Playground/nuggt/nuggt.py", line 65, in run
    exec(code, self.local_vars)
  File "<string>", line 6, in <module>
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/streamlit/runtime/metrics_util.py", line 332, in wrapped_func
    result = non_optional_func(*args, **kwargs)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/streamlit/elements/pyplot.py", line 109, in pyplot
    marshall(
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/streamlit/elements/pyplot.py", line 161, in marshall
    fig.savefig(image, **kwargs)
AttributeError: 'AxesSubplot' object has no attribute 'savefig'

"""

#print(fix_error(code, error))

output = """Final Answer:
Content: 
- The recipe and instructions for Creamy Espresso Tiramisu can be found at https://cooking.nytimes.com/recipes/1018684-classic-tiramisu. It involves whipping together egg yolks and sugar, whipping cream and mascarpone, combining espresso, rum, and cocoa powder, and assembling the tiramisu with ladyfingers, cream mixture, and cocoa powder. The recipe can be finished with shaved chocolate, if desired. Finally, the tiramisu should be refrigerated for at least 4 hours before slicing or scooping to serve.
- An alternative recipe for Tiramisu can be found at https://tastesbetterfromscratch.com/easy-tiramisu/. It is made with coffee soaked ladyfingers, sweet and creamy mascarpone (no raw eggs!), and cocoa powder dusted on top. It requires no baking and can be made in advance. 

Source:
- https://cooking.nytimes.com/recipes/1018684-classic-tiramisu
- https://tastesbetterfromscratch.com/easy-tiramisu/
"""

def browse_website(que):
    data = json.loads(query)
    url = [data["url"]]
    loader = SeleniumURLLoader(urls=url)
    pages = loader.load()
    #print(pages)
    faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
    retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k":2})
    rqa = RetrievalQA.from_chain_type(llm=OpenAI(), 
                                chain_type="stuff", 
                                retriever=retriever, 
                                return_source_documents=True,
                                )
    return str(rqa(data["information"]))

print(browse_website({"url": "https://cooking.nytimes.com/recipes/1018684-classic-tiramisu", "information": "How to make tiramisu"}))