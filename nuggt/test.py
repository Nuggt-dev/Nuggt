from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.utilities import GoogleSerperAPIWrapper, GoogleSearchAPIWrapper
from gradio_tools.tools import (StableDiffusionTool, ImageCaptioningTool, StableDiffusionPromptGeneratorTool, TextToVideoTool)
from langchain.tools import SceneXplainTool
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

        return output if not error else "Your code has the following error. Please provide the corrected code.\n" + error

python_repl = PythonREPLa()

def extract_code_from_block(text):
    if "!pip" in text:
        return "The package is successfully installed."
    text = text.replace("`", "")
    text = text.replace("python", "")
    return text

def python(code):
    code = extract_code_from_block(code)
    result = python_repl.run(code) 
    if "Your code has the following error." in result:
        result = fix_error(code, result)

    return result  

def fix_error(code, result):
    while "Your code has the following error." in result:
        error = result.replace("Your code has the following error. Please provide the corrected code.", "")
        user_input = f"Code:\n{code}\nError:\n{error}"

        print(f"I am going to correct: {user_input}")
        print(Fore.RED + "Code needs some correction.")
        messages = [
        {"role": "system", "content": "You are a brilliant programmer. When you are presented with a piece of code and an error, you fix the error and output the corrected code in the format: <corrected_code: The corrected python code>. You do not output anything else but the corrected code."},
        {"role": "user", "content": user_input}
        ]

        response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0, 
        )

        code = response.choices[0].message["content"].strip()
        code = code.replace("<", "").replace("corrected_code: ", "").replace(">", "")

        result = python_repl.run(code)
    
    print(Fore.GREEN + "Code has been corrected.")
    return result

code = """
```python
import pandas as pd

file_path = "imf-dm-export-20230513.xlsx"
data = pd.read_excel(file_path)
data.head()
```
"""
print(python(code))